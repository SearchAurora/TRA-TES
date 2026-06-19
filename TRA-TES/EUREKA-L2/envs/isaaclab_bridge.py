"""
IsaacLab Bridge
===============
Writes the LLM-generated reward function into the IsaacLab task,
runs PPO training via subprocess, and parses the training log.

Strategy: inject reward function as a Python file that IsaacLab imports.
We override the reward manager by patching the env config at runtime.
"""
import os
import re
import json
import subprocess
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ISAACLAB_ROOT = "/root/autodl-tmp/TRA/IsaacLab"
PYTHON_BIN    = "python"
TRAIN_SCRIPT  = f"{ISAACLAB_ROOT}/scripts/reinforcement_learning/rsl_rl/train.py"
LOG_DIR       = "/root/autodl-tmp/TRA/EUREKA/logs"

# Path where we drop the generated reward function
REWARD_FN_PATH = "/root/autodl-tmp/TRA/EUREKA/core/current_reward.py"


def write_reward_fn(code: str) -> None:
    """Write generated reward function to disk."""
    os.makedirs(os.path.dirname(REWARD_FN_PATH), exist_ok=True)
    with open(REWARD_FN_PATH, "w") as f:
        f.write(code)
    logger.info(f"Reward function written to {REWARD_FN_PATH}")


def run_training(
    task: str = "Isaac-Lift-Cube-Franka-v0",
    num_envs: int = 256,
    max_iterations: int = 300,
    seed: int = 42,
    experiment_name: str = "eureka_iter",
) -> dict:
    """
    Launch IsaacLab PPO training as subprocess.
    Returns parsed training statistics from the log.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{experiment_name}.log")

    # Set env var so IsaacLab picks up our reward function
    env = os.environ.copy()
    env["EUREKA_REWARD_FN"] = REWARD_FN_PATH

    cmd = [
        PYTHON_BIN, TRAIN_SCRIPT,
        f"--task={task}",
        f"--num_envs={num_envs}",
        f"--seed={seed}",
        "--headless",
        f"--max_iterations={max_iterations}",
        f"--experiment_name={experiment_name}",
    ]

    logger.info(f"Launching training: {' '.join(cmd)}")

    with open(log_file, "w") as lf:
        result = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ISAACLAB_ROOT,
        )

    if result.returncode != 0:
        logger.warning(f"Training exited with code {result.returncode}")

    stats = parse_training_log(log_file)
    logger.info(f"Parsed stats: {stats}")
    return stats


def parse_training_log(log_path: str) -> dict:
    """
    Parse the last iteration's statistics from an IsaacLab training log.
    Returns a dict with key metrics.
    """
    stats = {
        "mean_reward": 0.0,
        "mean_episode_length": 0.0,
        "success_rate": 0.0,
        "reaching_object": 0.0,
        "lifting_object": 0.0,
        "object_goal_tracking": 0.0,
        "position_error": 1.0,
        "orientation_error": 1.0,
    }

    if not os.path.exists(log_path):
        return stats

    content = Path(log_path).read_text()

    # Find last occurrence of each metric
    patterns = {
        "mean_reward":          r"Mean reward:\s*([-\d.]+)",
        "mean_episode_length":  r"Mean episode length:\s*([-\d.]+)",
        "reaching_object":      r"Episode_Reward/reaching_object:\s*([-\d.]+)",
        "lifting_object":       r"Episode_Reward/lifting_object:\s*([-\d.]+)",
        "object_goal_tracking": r"Episode_Reward/object_goal_tracking:\s*([-\d.]+)",
        "position_error":       r"Metrics/object_pose/position_error:\s*([-\d.]+)",
        "orientation_error":    r"Metrics/object_pose/orientation_error:\s*([-\d.]+)",
    }

    for key, pattern in patterns.items():
        matches = re.findall(pattern, content)
        if matches:
            try:
                stats[key] = float(matches[-1])
            except ValueError:
                pass

    # Estimate success rate from episode length (max=500 steps → success)
    if stats["mean_episode_length"] >= 490:
        stats["success_rate"] = 1.0
    elif stats["mean_episode_length"] >= 300:
        stats["success_rate"] = 0.5
    else:
        stats["success_rate"] = 0.0

    return stats
