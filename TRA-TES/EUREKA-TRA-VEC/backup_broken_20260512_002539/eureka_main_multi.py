"""
eureka_main.py (EUREKA-TRA Evolutionary Search 版)
====================================================
EUREKA-TRA 主循环 — 16 候选 evolutionary search + TES 归因 + 4 GPU 并行

与 EUREKA 版本的唯一区别：
  - 对每轮最优候选运行 TES 三层归因分析
  - 将 TES 报告注入下一轮的 LLM prompt
  - 其他完全一致（候选数、训练步数、PPO 配置）

用法：
  cd /root/autodl-tmp/TRA/EUREKA-TRA
  python eureka_main.py --iterations 15 --num_envs 1024 --rl_iters 1500 --num_candidates 16 --num_gpus 4
"""
import os
import sys
import json
import time
import signal
import logging
import argparse
import subprocess
from datetime import datetime
from multiprocessing import Process, Queue

EUREKA_ROOT   = "/root/autodl-tmp/TRA/EUREKA-TRA-VEC"
ISAACLAB_ROOT = "/root/autodl-tmp/TRA/IsaacLab"
PYTHON_BIN    = sys.executable

sys.path.insert(0, EUREKA_ROOT)

from llm.llm_client    import LLMClient
from llm.reward_parser  import extract_code, load_reward_fn, test_reward_fn, set_task
from llm.prompt_builder import build_initial_prompt, build_reflection_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REWARD_FN_PATH = f"{EUREKA_ROOT}/core/current_reward.py"
LOG_DIR        = f"{EUREKA_ROOT}/logs"
RUN_ONE_ITER   = f"{EUREKA_ROOT}/run_one_iter_multi.py"

TOTAL_TIMEOUT = 14400
MAX_RETRIES   = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--resume_run_id", type=str, default="", help="Specific run_id to resume")
    p.add_argument("--iterations",     type=int, default=5)
    p.add_argument("--num_envs",       type=int, default=1024)
    p.add_argument("--rl_iters",       type=int, default=1500)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--num_gpus",       type=int, default=4)
    p.add_argument("--task",           type=str, default="lift", choices=["lift","cabinet","anymal"])
    return p.parse_args()


def _kill_process_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Error killing pgid {pgid}: {e}")


def run_single_candidate(candidate_idx, iter_idx, run_id, reward_code, task,
                         num_envs, rl_iters, seed, gpu_id,
                         collect_trajectory=False):
    """训练单个候选奖励函数。"""
    candidate_reward_path = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_candidate_{candidate_idx}_reward.py"
    with open(candidate_reward_path, "w") as f:
        f.write(reward_code)

    iter_log_dir    = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_candidate_{candidate_idx}"
    stats_path      = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_candidate_{candidate_idx}_stats.json"
    traj_path       = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_candidate_{candidate_idx}_traj.npz"
    tes_report_path = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_candidate_{candidate_idx}_tes.json"

    cmd = [
        PYTHON_BIN, RUN_ONE_ITER,
        "--reward_fn", candidate_reward_path,
        "--output",    stats_path,
        "--num_envs",  str(num_envs),
        "--rl_iters",  str(rl_iters),
        "--seed",      str(seed),
        "--log_dir",   iter_log_dir,
        "--task",      task,
        "--headless",
    ]

    # ★ TRA: 对需要收集轨迹的候选加参数
    if collect_trajectory:
        cmd.extend([
            "--collect_trajectory",
            "--trajectory_path",  traj_path,
            "--tes_report_path",  tes_report_path,
        ])

    iter_log_file = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_candidate_{candidate_idx}_rl.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/miniconda3/lib/python3.10/site-packages/omni/data/Kit/Isaac-Sim/4.5/exts/3/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310:" + env.get("PYTHONPATH", "")
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    returncode = -1
    for attempt in range(1, MAX_RETRIES + 1):
        with open(iter_log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                cwd=EUREKA_ROOT, start_new_session=True, env=env,
            )
            pgid = os.getpgid(proc.pid)
            start_time = time.time()
            try:
                returncode = proc.wait(timeout=TOTAL_TIMEOUT)
                # Ensure all child processes are cleaned up
                _kill_process_group(pgid)
                time.sleep(30)  # [Fix] Vulkan Cooldown
                break
            except subprocess.TimeoutExpired:
                _kill_process_group(pgid)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(30)  # [Fix] Vulkan Cooldown after kill
                if attempt < MAX_RETRIES:
                    time.sleep(30)

    default_stats = {
        "mean_reward": 0.0, "mean_episode_length": 0.0,
        "success_rate": 0.0, "reaching_object": 0.0,
        "lifting_object": 0.0, "object_goal_tracking": 0.0,
        "position_error": 1.0, "orientation_error": 1.0,
        "tes_summary": "",
    }

    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
        except Exception:
            stats = default_stats
    else:
        stats = default_stats

    # ★ TRA: 读取 TES 报告
    if collect_trajectory and os.path.exists(tes_report_path):
        try:
            with open(tes_report_path, "r") as f:
                tes_report = json.load(f)
            if not stats.get("tes_summary"):
                stats["tes_summary"] = tes_report.get("summary", "")
        except Exception as e:
            logger.warning(f"Failed to load TES report: {e}")

    return candidate_idx, stats, reward_code


def _worker(q, *args):
    """多进程 worker。"""
    try:
        result = run_single_candidate(*args)
        q.put(result)
    except Exception as e:
        q.put(None)


def run_candidates_parallel(candidates, iter_idx, run_id, task, num_envs, rl_iters,
                            seed, num_gpus, collect_trajectory=False):
    """并行训练多个候选。每批 num_gpus 个同时跑。"""
    results = []
    batch_size = num_gpus
    num_batches = (len(candidates) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(candidates))
        batch = candidates[batch_start:batch_end]

        logger.info(f"  Batch {batch_idx+1}/{num_batches}: candidates {batch_start+1}-{batch_end}")

        queues = []
        procs = []
        for i, (cand_idx, code) in enumerate(batch):
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
            gpu_id = int(visible_gpus[i % len(visible_gpus)])
            q = Queue()
            p = Process(
                target=_worker,
                args=(q, cand_idx, iter_idx, run_id, code, task,
                      num_envs, rl_iters, seed, gpu_id, collect_trajectory),
            )
            p.start()
            queues.append(q)
            procs.append(p)

        for p, q in zip(procs, queues):
            try:
                result = q.get(timeout=TOTAL_TIMEOUT + 120)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Worker timeout: {e}")
                # ★ 修复：超时后尝试从磁盘读取已完成的 stats
                cand_idx_fallback, code_fallback = batch[procs.index(p)]
                fallback_stats_path = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_candidate_{cand_idx_fallback}_stats.json"
                if os.path.exists(fallback_stats_path):
                    try:
                        with open(fallback_stats_path, "r") as sf:
                            fallback_stats = json.load(sf)
                        results.append((cand_idx_fallback, fallback_stats, code_fallback))
                        logger.info(f"    Recovered candidate {cand_idx_fallback} from disk: pos_err={fallback_stats.get('position_error', 1.0):.4f}")
                    except Exception as re:
                        logger.warning(f"    Failed to recover candidate {cand_idx_fallback}: {re}")
                else:
                    logger.warning(f"    No stats file for candidate {cand_idx_fallback}")
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

        logger.info(f"  Batch {batch_idx+1} complete: {len(results)} total results so far")

        if batch_idx < num_batches - 1:
            time.sleep(30)

    return results



def find_latest_results(log_dir, task, seed):
    """找到最新的results.json"""
    import glob
    pattern = f"{log_dir}/eureka_*_s{seed}_results.json"
    # 兼容不同命名格式
    files = glob.glob(f"{log_dir}/*results.json")
    if not files:
        return None, None
    latest = max(files, key=os.path.getmtime)
    return latest, os.path.basename(latest).split("_results.json")[0].replace("eureka_tra_evo_", "").replace("eureka_evo_", "")

def main():
    args   = parse_args()
    set_task(args.task)
    # --- Redirect LOG_DIR to structured results directory ---
    global LOG_DIR
    _method_map = {"EUREKA-TRA": "eureka-tra", "EUREKA-TRALLM": "eureka-trallm", "EUREKA": "eureka"}
    _method = _method_map.get(os.path.basename(EUREKA_ROOT), os.path.basename(EUREKA_ROOT).lower())
    LOG_DIR = f"/root/autodl-tmp/TRA/results/{args.task}/{_method}/seed{args.seed}/logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_s{args.seed}"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REWARD_FN_PATH), exist_ok=True)

    result_path = f"{LOG_DIR}/eureka_tra_evo_{run_id}_results.json"
    logger.info(f"=== EUREKA-TRA (Evolutionary) iterations={args.iterations} candidates={args.num_candidates} "
                f"envs={args.num_envs} rl_iters={args.rl_iters} gpus={args.num_gpus} run={run_id} ===")

    llm     = LLMClient()
    results = []
    best_code  = ""
    global_best_pos_err = 1.0
    global_best_code = ""
    best_fn    = None

    # ── 续跑逻辑 ──────────────────────────────────────────────
    start_iter = 0
    if args.resume:
        results_file, resume_run = find_latest_results(LOG_DIR, args.task, args.seed)
        if results_file and os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
            start_iter = len(results)
            run_id = resume_run  # 保持同一个run_id
            result_path = results_file
            # 恢复全局最优
            for r in results:
                pe = r["best_stats"].get("position_error", 1.0)
                if pe < global_best_pos_err:
                    global_best_pos_err = pe
                    global_best_code = r["best_reward_code"]
            best_code = global_best_code
            logger.info(f"Resumed from iter {start_iter}, best PE={global_best_pos_err:.4f}, run_id={run_id}")
        else:
            logger.warning("No checkpoint found, starting fresh")

    for it in range(start_iter, args.iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"LLM Iteration {it+1}/{args.iterations}")
        logger.info(f"{'='*60}")

        # ── Step 1: LLM 生成 N 个候选 ─────────────────────────────
        if it == 0:
            system, user = build_initial_prompt(task=args.task)
        else:
            # ★ TRA: 传递 TES 报告
            tes_summary = results[-1]["best_stats"].get("tes_summary", "") if results else ""
            system, user = build_reflection_prompt(
                best_code, results[-1]["best_stats"] if results else {}, it + 1,
                tes_summary=tes_summary, task=args.task,
            )
            if tes_summary:
                logger.info(f"TES summary injected ({len(tes_summary)} chars)")
            else:
                logger.info("No TES summary, using scalar feedback only")

        logger.info(f"Generating {args.num_candidates} candidates from QwenMax...")
        candidates = []
        for cand_idx in range(args.num_candidates):
            try:
                response = llm.complete(system, user)
                code = extract_code(response)
                fn = load_reward_fn(code)
                ok, msg = test_reward_fn(fn)
                if ok:
                    candidates.append((cand_idx, code))
                    logger.info(f"  Candidate {cand_idx+1}: OK ({msg})")
                else:
                    logger.warning(f"  Candidate {cand_idx+1}: test failed ({msg})")
            except Exception as e:
                logger.warning(f"  Candidate {cand_idx+1}: error ({e})")

        if not candidates:
            logger.error("No valid candidates, skip iteration")
            if best_code:
                candidates = [(0, best_code)]
            else:
                continue

        logger.info(f"Valid candidates: {len(candidates)}/{args.num_candidates}")

        # ── Step 2: 并行训练所有候选（开启轨迹收集）──────────────
        logger.info(f"Training {len(candidates)} candidates ({args.num_gpus} GPUs parallel)...")
        train_results = run_candidates_parallel(
            candidates, it + 1, run_id, args.task,
            args.num_envs, args.rl_iters, args.seed, args.num_gpus,
            collect_trajectory=False,  # EUREKA baseline: no trajectory collection
        )

        # ── Step 3: 选最优候选 ────────────────────────────────────
        _threshold = {"lift": 0.99, "cabinet": 0.95, "anymal": 1.95}.get(args.task, 0.99)
        valid_results = [(idx, stats, code) for idx, stats, code in train_results
                         if stats.get("position_error", 1.0) < _threshold]

        if valid_results:
            best_idx, best_stats, best_candidate_code = min(
                valid_results, key=lambda x: x[1].get("position_error", 1.0)
            )
            # 只有超过全局最优才更新反馈代码
            if best_stats.get('position_error', 1.0) < global_best_pos_err:
                global_best_pos_err = best_stats['position_error']
                global_best_code = best_candidate_code
                logger.info(f"New global best! pos_err={global_best_pos_err:.4f}")
            best_code = global_best_code if global_best_code else best_candidate_code
            best_fn = load_reward_fn(best_code)
            with open(REWARD_FN_PATH, "w") as f:
                f.write(best_code)
            logger.info(f"Best candidate: {best_idx}, pos_err={best_stats.get('position_error', 1.0):.4f}, "
                        f"tes={'YES' if best_stats.get('tes_summary') else 'NO'}")
        else:
            logger.warning("All candidates failed, using previous best")
            best_stats = {
                "mean_reward": 0.0, "mean_episode_length": 0.0,
                "success_rate": 0.0, "position_error": 1.0, "orientation_error": 1.0,
                "reaching_object": 0.0, "lifting_object": 0.0, "object_goal_tracking": 0.0,
                "tes_summary": "",
            }
            best_idx = -1

        # ── Step 4: 记录结果 ──────────────────────────────────────
        all_candidate_stats = [
            {"candidate_idx": idx, "position_error": stats.get("position_error", 1.0),
             "mean_reward": stats.get("mean_reward", 0.0),
             "tes": "YES" if stats.get("tes_summary") else "NO"}
            for idx, stats, code in train_results
        ]

        iter_result = {
            "iteration": it + 1,
            "num_candidates": len(candidates),
            "num_valid_results": len(valid_results),
            "best_candidate_idx": best_idx,
            "best_stats": best_stats,
            "best_reward_code": best_code,
            "all_candidate_stats": all_candidate_stats,
        }
        results.append(iter_result)

        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Iter {it+1} complete | "
            f"best={best_idx} | "
            f"pos_err={best_stats.get('position_error', 1.0):.4f} | "
            f"tes={'YES' if best_stats.get('tes_summary') else 'NO'} | "
            f"valid={len(valid_results)}/{len(candidates)}"
        )

        sorted_results = sorted(all_candidate_stats, key=lambda x: x["position_error"])
        for rank, cr in enumerate(sorted_results):
            marker = " ★" if cr["candidate_idx"] == best_idx else ""
            logger.info(f"  Rank {rank+1}: candidate {cr['candidate_idx']} "
                        f"pos_err={cr['position_error']:.4f} tes={cr['tes']}{marker}")

    # ── 汇总 ──────────────────────────────────────────────────────
    logger.info("\n=== EUREKA-TRA (Evolutionary) Training Complete ===")
    for r in results:
        logger.info(
            f"  Iter {r['iteration']}: best={r['best_candidate_idx']} "
            f"pos_err={r['best_stats'].get('position_error', 1.0):.4f} "
            f"({r['num_valid_results']}/{r['num_candidates']} valid)"
        )
    logger.info(f"Full results: {result_path}")


if __name__ == "__main__":
    main()
