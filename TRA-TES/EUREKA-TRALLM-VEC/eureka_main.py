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

EUREKA_ROOT   = "/root/autodl-tmp/TRA/EUREKA-TRA"
ISAACLAB_ROOT = "/root/autodl-tmp/TRA/IsaacLab"
PYTHON_BIN    = "python"

sys.path.insert(0, EUREKA_ROOT)

from llm.llm_client    import LLMClient
from llm.reward_parser  import extract_code, load_reward_fn, test_reward_fn
from llm.prompt_builder import build_initial_prompt, build_reflection_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REWARD_FN_PATH = f"{EUREKA_ROOT}/core/current_reward.py"
LOG_DIR        = f"{EUREKA_ROOT}/logs"
RUN_ONE_ITER   = f"{EUREKA_ROOT}/run_one_iter.py"

TOTAL_TIMEOUT = 43200
MAX_RETRIES   = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations",     type=int, default=5)
    p.add_argument("--num_envs",       type=int, default=1024)
    p.add_argument("--rl_iters",       type=int, default=1500)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--num_gpus",       type=int, default=4)
    return p.parse_args()


def _kill_process_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Error killing pgid {pgid}: {e}")


def run_single_candidate(candidate_idx, iter_idx, run_id, reward_code,
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
                break
            except subprocess.TimeoutExpired:
                _kill_process_group(pgid)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                if attempt < MAX_RETRIES:
                    time.sleep(10)

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


def run_candidates_parallel(candidates, iter_idx, run_id, num_envs, rl_iters,
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
                args=(q, cand_idx, iter_idx, run_id, code,
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
            time.sleep(10)

    return results


def main():
    args   = parse_args()
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

    for it in range(args.iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"LLM Iteration {it+1}/{args.iterations}")
        logger.info(f"{'='*60}")

        # ── Step 1: LLM 生成 N 个候选 ─────────────────────────────
        if it == 0:
            system, user = build_initial_prompt()
        else:
            # ★ TRA: 传递 TES 报告
            tes_summary = results[-1]["best_stats"].get("tes_summary", "") if results else ""
            system, user = build_reflection_prompt(
                best_code, results[-1]["best_stats"] if results else {}, it + 1,
                tes_summary=tes_summary,
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
            candidates, it + 1, run_id,
            args.num_envs, args.rl_iters, args.seed, args.num_gpus,
            collect_trajectory=True,  # ★ TRA: 所有候选都收集轨迹
        )

        # ── Step 3: 选最优候选 ────────────────────────────────────
        valid_results = [(idx, stats, code) for idx, stats, code in train_results
                         if stats.get("position_error", 1.0) < 0.99]

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
