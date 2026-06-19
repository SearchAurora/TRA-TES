"""
eureka_main.py (EUREKA-TRA 版)
===============================
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

TOTAL_TIMEOUT = 2400
MAX_RETRIES   = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--num_envs",   type=int, default=256)
    p.add_argument("--rl_iters",   type=int, default=300)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def _kill_process_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
        logger.info(f"Killed process group {pgid}")
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Error killing pgid {pgid}: {e}")


def _cleanup_isaac_processes():
    try:
        subprocess.run(["pkill", "-9", "-f", "run_one_iter"], capture_output=True, timeout=10)
        time.sleep(5)
    except Exception:
        pass


def run_rl_subprocess(iter_idx, run_id, num_envs, rl_iters, seed):
    iter_log_dir    = f"{LOG_DIR}/iter_{iter_idx}_{run_id}"
    stats_path      = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_stats.json"
    traj_path       = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_traj.npz"
    tes_report_path = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_tes.json"

    cmd = [
        PYTHON_BIN, RUN_ONE_ITER,
        "--reward_fn",        REWARD_FN_PATH,
        "--output",           stats_path,
        "--num_envs",         str(num_envs),
        "--rl_iters",         str(rl_iters),
        "--seed",             str(seed),
        "--log_dir",          iter_log_dir,
        "--headless",
        "--collect_trajectory",
        "--trajectory_path",  traj_path,
        "--tes_report_path",  tes_report_path,
    ]

    iter_log_file = f"{LOG_DIR}/iter_{iter_idx}_{run_id}_rl.log"
    logger.info(f"Launching RL subprocess (TRA mode)")
    logger.info(f"RL log: {iter_log_file}")

    returncode = -1
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Attempt {attempt}/{MAX_RETRIES}")

        with open(iter_log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                cwd=EUREKA_ROOT, start_new_session=True,
            )
            pgid = os.getpgid(proc.pid)
            logger.info(f"RL subprocess PID={proc.pid}, PGID={pgid}")

            start_time = time.time()
            try:
                returncode = proc.wait(timeout=TOTAL_TIMEOUT)
                elapsed = time.time() - start_time
                logger.info(f"RL subprocess finished in {elapsed:.0f}s with code {returncode}")
                break
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                logger.warning(f"Timed out after {elapsed:.0f}s, killing pgid {pgid}")
                _kill_process_group(pgid)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                if attempt < MAX_RETRIES:
                    logger.info("Cleaning up before retry...")
                    _cleanup_isaac_processes()
                    time.sleep(10)

    if returncode != 0:
        logger.warning(f"RL subprocess exited with code {returncode}")

    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        logger.info(f"Stats loaded: reward={stats.get('mean_reward','N/A')}, pos_err={stats.get('position_error','N/A')}")
    else:
        logger.warning(f"Stats file not found: {stats_path}")
        stats = {
            "mean_reward": 0.0, "mean_episode_length": 0.0,
            "success_rate": 0.0, "reaching_object": 0.0,
            "lifting_object": 0.0, "object_goal_tracking": 0.0,
            "position_error": 1.0, "orientation_error": 1.0,
            "tes_summary": "",
        }

    if os.path.exists(tes_report_path):
        try:
            with open(tes_report_path, "r") as f:
                tes_report = json.load(f)
            logger.info(f"TES report loaded ({tes_report.get('n_episodes', 0)} episodes)")
            if not stats.get("tes_summary"):
                stats["tes_summary"] = tes_report.get("summary", "")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load TES report: {e}")
    else:
        logger.info("No TES report file found")

    return stats


def main():
    args   = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REWARD_FN_PATH), exist_ok=True)

    result_path = f"{LOG_DIR}/eureka_tra_{run_id}_results.json"
    logger.info(f"=== EUREKA-TRA  iterations={args.iterations}  envs={args.num_envs}  rl_iters={args.rl_iters}  run={run_id} ===")

    llm          = LLMClient()
    results      = []
    current_code = ""
    reward_fn    = None

    for it in range(args.iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"LLM Iteration {it+1}/{args.iterations}")
        logger.info(f"{'='*60}")

        if it == 0:
            system, user = build_initial_prompt()
        else:
            tes_summary = results[-1]["stats"].get("tes_summary", "")
            system, user = build_reflection_prompt(
                current_code, results[-1]["stats"], it + 1,
                tes_summary=tes_summary,
            )
            if tes_summary:
                logger.info(f"TES summary injected ({len(tes_summary)} chars)")
            else:
                logger.info("No TES summary, using scalar feedback only")

        logger.info("Querying QwenMax...")
        response = llm.complete(system, user)
        code     = extract_code(response)
        logger.info(f"Generated {len(code)} chars of reward code")

        try:
            fn      = load_reward_fn(code)
            ok, msg = test_reward_fn(fn)
            if ok:
                current_code = code
                reward_fn    = fn
                logger.info(f"Reward fn OK: {msg}")
            else:
                logger.warning(f"Reward fn test failed: {msg}")
                if reward_fn is None:
                    logger.error("No valid reward fn, skip iteration")
                    continue
        except Exception as e:
            logger.warning(f"Reward fn load error: {e}")
            if reward_fn is None:
                logger.error("No valid reward fn, skip iteration")
                continue

        with open(REWARD_FN_PATH, "w") as f:
            f.write(current_code)
        logger.info(f"Reward fn saved to {REWARD_FN_PATH}")

        logger.info(f"Starting RL subprocess ({args.rl_iters} iters, {args.num_envs} envs)...")
        stats = run_rl_subprocess(it + 1, run_id, args.num_envs, args.rl_iters, args.seed)

        results.append({"iteration": it + 1, "reward_code": current_code, "stats": stats})
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Iter {it+1} complete | reward={stats['mean_reward']:.4f} | "
            f"pos_err={stats['position_error']:.4f} | tes={'YES' if stats.get('tes_summary') else 'NO'}"
        )

    logger.info("\n=== EUREKA-TRA Training Complete ===")
    for r in results:
        logger.info(f"  Iter {r['iteration']}: reward={r['stats']['mean_reward']:.4f}  pos_err={r['stats']['position_error']:.4f}")
    logger.info(f"Full results: {result_path}")


if __name__ == "__main__":
    main()
