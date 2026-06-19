"""
run_one_iter.py (EUREKA-TRA 版)
================================
单次 RL 训练脚本，增加轨迹收集和 TES 分析功能。

与 EUREKA 版本的区别：
  1. EurekaRewardWrapper 支持分量级奖励分解
  2. 训练过程中收集轨迹数据
  3. 训练结束后运行 TES 三层归因分析
  4. 输出包含 TES 报告的增强版统计 JSON

用法：
  python run_one_iter.py \
    --reward_fn /root/autodl-tmp/TRA/EUREKA-TRA/core/current_reward.py \
    --output    /root/autodl-tmp/TRA/EUREKA-TRA/logs/iter1_stats.json \
    --num_envs  256 \
    --rl_iters  300 \
    --log_dir   /root/autodl-tmp/TRA/EUREKA-TRA/logs/iter1 \
    --collect_trajectory  \
    --trajectory_path /root/autodl-tmp/TRA/EUREKA-TRA/logs/iter1_traj.npz
"""
import os
import sys
import json
import argparse

ISAACLAB_ROOT = "/root/autodl-tmp/TRA/IsaacLab"
EUREKA_ROOT   = "/root/autodl-tmp/TRA/EUREKA-TRA"

sys.path.insert(0, EUREKA_ROOT)
sys.path.insert(0, f"{ISAACLAB_ROOT}/source/isaaclab")
sys.path.insert(0, f"{ISAACLAB_ROOT}/source/isaaclab_tasks")
sys.path.insert(0, f"{ISAACLAB_ROOT}/source/isaaclab_assets")
sys.path.insert(0, f"{ISAACLAB_ROOT}/source/isaaclab_contrib")
sys.path.insert(0, f"{ISAACLAB_ROOT}/source/isaaclab_rl")

# ── AppLauncher 必须最先初始化 ────────────────────────────────────────
from isaaclab.app import AppLauncher

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward_fn", required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--num_envs",  type=int, default=256)
    p.add_argument("--rl_iters",  type=int, default=300)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--log_dir",   default="/tmp/eureka_rl_log")
    # ★ EUREKA-TRA 新参数
    p.add_argument("--collect_trajectory", action="store_true",
                   help="是否收集轨迹数据供 TES 分析")
    p.add_argument("--trajectory_path", default="",
                   help="轨迹数据保存路径 (.npz)")
    p.add_argument("--tes_report_path", default="",
                   help="TES 报告保存路径 (.json)")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()

args = parse_args()
args.headless = True
app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

# ── 之后才能导入 omni/isaaclab/torch ─────────────────────────────────
import torch
import numpy as np
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from importlib.metadata import version as pkg_version
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import (
    LiftCubePPORunnerCfg,
)

from llm.reward_parser import load_reward_fn, test_reward_fn


# ══════════════════════════════════════════════════════════════════════
# Reward Wrapper (EUREKA-TRA 版 — 支持分量级奖励分解)
# ══════════════════════════════════════════════════════════════════════

class EurekaRewardWrapper(gym.Wrapper):
    """
    将 LLM 生成的 compute_reward 注入 IsaacLab 环境。
    EUREKA-TRA 版本额外支持：
      - 分量级奖励分解（每个子奖励项单独记录）
      - 轨迹数据收集（配合 TrajectoryCollector）
    """
    def __init__(self, env, reward_fn, collector=None):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.collector = collector
        self._step_count = np.zeros(env.num_envs, dtype=int)

    def get_observations(self):
        return self.env.get_observations()

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        rewards, reward_components_batch = self._compute_rewards(obs)

        # ★ 收集轨迹数据
        if self.collector is not None:
            self._collect_trajectories(
                obs, rewards, reward_components_batch,
                terminated, truncated
            )

        self._step_count += 1
        # 重置已结束 episode 的计数
        done_mask = (terminated | truncated).cpu().numpy().flatten()
        self._step_count[done_mask] = 0

        return obs, rewards, terminated, truncated, info

    def _compute_rewards(self, obs):
        """计算奖励，同时返回分量级分解。"""
        policy_obs = obs["policy"]  # [num_envs, 36]
        num_envs = policy_obs.shape[0]
        rewards = torch.zeros(num_envs, device=policy_obs.device)
        reward_components_batch = []

        try:
            for i in range(num_envs):
                d = policy_obs[i].cpu().numpy().astype(float)
                obs_dict = self._to_dict(d)

                # 调用奖励函数（可能返回 float 或 dict）
                result = self.reward_fn(obs_dict)

                if isinstance(result, dict):
                    # ★ 新特性：奖励函数返回分量字典
                    total = float(result.get("total", sum(
                        v for v in result.values() if isinstance(v, (int, float))
                    )))
                    components = {
                        k: float(v) for k, v in result.items()
                        if k != "total" and isinstance(v, (int, float))
                    }
                else:
                    total = float(result)
                    # 无分量信息时，用总奖励作为唯一分量
                    components = {"total_reward": total}

                # Clip
                total = max(-50.0, min(50.0, total))
                if not (total == total):  # NaN
                    total = 0.0
                    components = {k: 0.0 for k in components}

                rewards[i] = total
                reward_components_batch.append(components)

        except Exception as e:
            print(f"[WARN] Reward fn error: {e}")
            reward_components_batch = [{"total_reward": 0.0}] * num_envs

        return rewards, reward_components_batch

    def _collect_trajectories(self, obs, rewards, components_batch,
                              terminated, truncated):
        """将当前步数据送入 TrajectoryCollector。"""
        policy_obs = obs["policy"]
        done_mask = (terminated | truncated).cpu().numpy().flatten()
        rewards_np = rewards.cpu().numpy().flatten()

        # 只采样部分环境（避免太多数据）
        sample_envs = min(16, policy_obs.shape[0])

        for i in range(sample_envs):
            d = policy_obs[i].cpu().numpy().astype(float)
            obs_dict = self._to_dict(d)
            components = components_batch[i] if i < len(components_batch) else {"total_reward": 0.0}

            self.collector.record_step(
                env_id=i,
                obs_dict=obs_dict,
                reward_components=components,
                total_reward=float(rewards_np[i]),
                done=bool(done_mask[i]),
            )

    def _to_dict(self, d: np.ndarray) -> dict:
        """obs 索引（共36维）"""
        obj_pos    = d[18:21]
        target_pos = d[21:24]
        goal_dist  = float(np.linalg.norm(obj_pos - target_pos))
        return {
            "eef_pos_x":      float(d[18]),
            "eef_pos_y":      float(d[19]),
            "eef_pos_z":      float(d[20]),
            "obj_pos_x":      float(d[18]),
            "obj_pos_y":      float(d[19]),
            "obj_pos_z":      float(d[20]),
            "obj_goal_dist":  goal_dist,
            "gripper_open":   float(d[7]),
            "joint_vel_norm": float(np.linalg.norm(d[9:18])),
            "action_rate":    float(np.linalg.norm(d[28:36])),
        }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── 加载奖励函数 ──────────────────────────────────────────────────
    with open(args.reward_fn, "r") as f:
        code = f.read()
    reward_fn = load_reward_fn(code)
    ok, msg = test_reward_fn(reward_fn)
    print(f"[INFO] Reward fn: {msg}")

    # ── 初始化轨迹收集器（如果启用） ──────────────────────────────────
    collector = None
    if args.collect_trajectory:
        from core.trajectory_collector import TrajectoryCollector
        collector = TrajectoryCollector(max_episodes=50, max_steps_per_ep=500)
        print("[INFO] Trajectory collection ENABLED")

    # ── 构建环境 ──────────────────────────────────────────────────────
    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    for attr in ["reaching_object", "lifting_object", "object_goal_tracking",
                 "object_goal_tracking_fine_grained", "action_rate", "joint_vel"]:
        if hasattr(env_cfg.rewards, attr):
            getattr(env_cfg.rewards, attr).weight = 0.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = EurekaRewardWrapper(env, reward_fn, collector=collector)

    # ── PPO 配置 ──────────────────────────────────────────────────────
    runner_cfg = LiftCubePPORunnerCfg()
    runner_cfg.max_iterations  = args.rl_iters
    runner_cfg.experiment_name = os.path.basename(args.log_dir)

    installed_version = pkg_version("rsl-rl-lib")
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, installed_version)

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)

    os.makedirs(args.log_dir, exist_ok=True)
    runner = OnPolicyRunner(
        env_wrapped,
        runner_cfg.to_dict(),
        log_dir=args.log_dir,
        device="cuda:0",
    )
    runner.learn(num_learning_iterations=args.rl_iters, init_at_random_ep_len=True)

    # ── 从 TensorBoard events 文件解析真实统计 ────────────────────────
    stats = parse_tb_stats(args.log_dir)

    # ── ★ TES 分析（EUREKA-TRA 核心） ─────────────────────────────────
    tes_summary = ""
    if args.collect_trajectory and collector is not None:
        # 保存轨迹数据
        traj_path = args.trajectory_path or f"{args.log_dir}/trajectory.npz"
        collector.save(traj_path)

        # 运行 TES 分析
        from core.trajectory_collector import TrajectoryCollector
        from core.tes import generate_tes_report, report_to_prompt_text

        episodes = TrajectoryCollector.load(traj_path)
        report = generate_tes_report(episodes)
        tes_summary = report_to_prompt_text(report)

        # 保存 TES 报告
        tes_report_path = args.tes_report_path or f"{args.log_dir}/tes_report.json"
        def _f(v):
            """numpy types -> Python native for JSON serialization"""
            if hasattr(v, 'item'):
                return v.item()
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            return v

        tes_report_dict = {
            "summary": report.summary,
            "n_episodes": int(report.n_episodes),
            "avg_episode_length": _f(report.avg_episode_length),
            "crash_windows": [
                {
                    "start_step": cw.start_step,
                    "end_step": cw.end_step,
                    "severity": _f(cw.severity),
                    "reward_drop": _f(cw.reward_drop),
                }
                for cw in report.crash_windows
            ],
            "component_attributions": [
                {
                    "name": ca.name,
                    "variance_contribution": _f(ca.variance_contribution),
                    "mean_value": _f(ca.mean_value),
                    "trend": ca.trend,
                    "is_problematic": _f(ca.is_problematic),
                }
                for ca in report.component_attributions
            ],
            "phase_stats": [
                {
                    "phase": ps.phase,
                    "mean_reward": _f(ps.mean_reward),
                    "failure_rate": _f(ps.failure_rate),
                    "dominant_component": ps.dominant_component,
                }
                for ps in report.phase_stats
            ],
        }
        with open(tes_report_path, "w") as f:
            json.dump(tes_report_dict, f, indent=2)
        print(f"[INFO] TES report saved to {tes_report_path}")
        print(f"[INFO] TES summary:\n{tes_summary}")

    # ── 保存统计（包含 TES 报告） ─────────────────────────────────────
    stats["tes_summary"] = tes_summary

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[INFO] Stats saved to {args.output}")
    print(f"[INFO] Stats: {stats}")

    env_wrapped.close()
    simulation_app.close()


def parse_tb_stats(log_dir: str) -> dict:
    """从 TensorBoard events 文件解析训练统计。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import glob

    stats = {
        "mean_reward": 0.0, "mean_episode_length": 0.0,
        "success_rate": 0.0, "reaching_object": 0.0,
        "lifting_object": 0.0, "object_goal_tracking": 0.0,
        "position_error": 1.0, "orientation_error": 1.0,
    }
    tf_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not tf_files:
        return stats

    ea = EventAccumulator(tf_files[0])
    ea.Reload()

    tag_map = {
        "Train/mean_reward":                     "mean_reward",
        "Train/mean_episode_length":             "mean_episode_length",
        "Metrics/object_pose/position_error":    "position_error",
        "Metrics/object_pose/orientation_error": "orientation_error",
        "Episode_Reward/reaching_object":        "reaching_object",
        "Episode_Reward/lifting_object":         "lifting_object",
        "Episode_Reward/object_goal_tracking":   "object_goal_tracking",
    }
    scalars = ea.Tags().get("scalars", [])
    for tb_tag, stat_key in tag_map.items():
        if tb_tag in scalars:
            events = ea.Scalars(tb_tag)
            if events:
                stats[stat_key] = float(events[-1].value)

    ep_len = stats["mean_episode_length"]
    stats["success_rate"] = 1.0 if ep_len >= 490 else (0.5 if ep_len >= 300 else 0.0)
    return stats


if __name__ == "__main__":
    main()
