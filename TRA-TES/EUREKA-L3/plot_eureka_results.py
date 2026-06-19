"""
plot_eureka_results.py
======================
从 EUREKA 各轮训练的 TensorBoard 日志中提取关键指标，
绘制跨 LLM 迭代轮次的训练曲线图。

用法：
    python plot_eureka_results.py \
        --log_dir /root/autodl-tmp/TRA/EUREKA/logs \
        --run_id 20260322_105840 \
        --output /root/autodl-tmp/TRA/EUREKA/logs/eureka_curves.png
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_metrics(log_dir: str, tags: list) -> dict:
    """从单个 TensorBoard 目录加载指定 tag 的最后一个值"""
    tf_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not tf_files:
        return None
    try:
        ea = EventAccumulator(tf_files[0])
        ea.Reload()
        result = {}
        available = ea.Tags().get("scalars", [])
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                if events:
                    # 取最后20步的平均，更稳定
                    vals = [e.value for e in events[-20:]]
                    result[tag] = float(np.mean(vals))
        return result
    except Exception as e:
        print(f"[WARN] Failed to load {log_dir}: {e}")
        return None


def load_tb_curve(log_dir: str, tag: str) -> tuple:
    """从单个 TensorBoard 目录加载完整曲线"""
    tf_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not tf_files:
        return [], []
    try:
        ea = EventAccumulator(tf_files[0])
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        if tag not in available:
            return [], []
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals  = [e.value for e in events]
        return steps, vals
    except Exception:
        return [], []


def find_iter_dirs(log_base: str, run_id: str) -> list:
    """找到指定 run_id 的所有迭代目录，按轮次排序"""
    pattern = os.path.join(log_base, f"iter_*_{run_id}")
    dirs = glob.glob(pattern)
    # 过滤掉 stats.json 文件
    dirs = [d for d in dirs if os.path.isdir(d)]
    # 按 iter 编号排序
    def get_iter_num(d):
        base = os.path.basename(d)
        try:
            return int(base.split("_")[1])
        except:
            return 0
    dirs.sort(key=get_iter_num)
    return dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="/root/autodl-tmp/TRA/EUREKA/logs")
    p.add_argument("--run_id",  required=True, help="e.g. 20260322_105840")
    p.add_argument("--output",  default="/root/autodl-tmp/TRA/EUREKA/logs/eureka_curves.png")
    args = p.parse_args()

    tags = [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/object_pose/position_error",
        "Episode_Reward/lifting_object",
    ]

    iter_dirs = find_iter_dirs(args.log_dir, args.run_id)
    if not iter_dirs:
        print(f"[ERROR] No iter dirs found for run_id={args.run_id} in {args.log_dir}")
        return

    print(f"Found {len(iter_dirs)} iterations")

    # ── 跨轮次汇总数据 ─────────────────────────────────────────────────
    iter_nums   = []
    mean_rewards = []
    pos_errors   = []
    ep_lengths   = []
    lift_rewards = []

    for d in iter_dirs:
        iter_num = int(os.path.basename(d).split("_")[1])
        metrics = load_tb_metrics(d, tags)
        if metrics is None:
            print(f"  Iter {iter_num}: no data")
            continue
        iter_nums.append(iter_num)
        mean_rewards.append(metrics.get("Train/mean_reward", 0.0))
        pos_errors.append(metrics.get("Metrics/object_pose/position_error", 1.0))
        ep_lengths.append(metrics.get("Train/mean_episode_length", 0.0))
        lift_rewards.append(metrics.get("Episode_Reward/lifting_object", 0.0))
        print(f"  Iter {iter_num}: reward={mean_rewards[-1]:.2f}  pos_err={pos_errors[-1]:.4f}  ep_len={ep_lengths[-1]:.1f}")

    if not iter_nums:
        print("[ERROR] No valid data found")
        return

    # ── 绘图 ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"EUREKA Training Results (run: {args.run_id})", fontsize=14, fontweight='bold')

    # Plot 1: Mean reward per LLM iteration
    ax = axes[0, 0]
    ax.plot(iter_nums, mean_rewards, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel("LLM Iteration")
    ax.set_ylabel("Mean Reward (last 20 steps avg)")
    ax.set_title("Mean Reward per LLM Iteration")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Position error per LLM iteration (lower is better)
    ax = axes[0, 1]
    ax.plot(iter_nums, pos_errors, 'rs-', linewidth=2, markersize=8)
    ax.set_xlabel("LLM Iteration")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Object Position Error per LLM Iteration ↓")
    ax.grid(True, alpha=0.3)

    # Plot 3: Episode length per LLM iteration
    ax = axes[1, 0]
    ax.plot(iter_nums, ep_lengths, 'g^-', linewidth=2, markersize=8)
    ax.set_xlabel("LLM Iteration")
    ax.set_ylabel("Mean Episode Length (steps)")
    ax.set_title("Episode Length per LLM Iteration")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Max (success)')
    ax.legend()

    # Plot 4: Reward curves for each iteration overlaid
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(iter_dirs)))
    for i, d in enumerate(iter_dirs):
        iter_num = int(os.path.basename(d).split("_")[1])
        steps, vals = load_tb_curve(d, "Train/mean_reward")
        if steps:
            # Clip for visibility
            vals_clipped = np.clip(vals, -500, 500)
            ax.plot(steps, vals_clipped, color=colors[i], alpha=0.7,
                    linewidth=1.5, label=f"Iter {iter_num}")
    ax.set_xlabel("PPO Steps")
    ax.set_ylabel("Mean Reward (clipped ±500)")
    ax.set_title("Reward Curves (all iterations)")
    ax.grid(True, alpha=0.3)
    if len(iter_dirs) <= 10:
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
