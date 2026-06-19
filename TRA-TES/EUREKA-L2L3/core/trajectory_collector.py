"""
trajectory_collector.py
=======================
在 RL 训练过程中收集轨迹数据，供 TES 三层归因分析使用。

收集内容：
  - 每步的各奖励分量值（reaching, lifting, goal_tracking 等）
  - 每步的关键 obs 值（obj_pos_z, obj_goal_dist, joint_vel_norm 等）
  - episode 边界（done 信号）

设计原则：
  - 只在最后 N 个 episode 收集，避免内存爆炸
  - 收集完成后保存为 .npz 文件，供 TES 离线分析
"""
import numpy as np
from typing import Dict, List, Optional


class TrajectoryCollector:
    """
    收集 RL 训练过程中的轨迹数据。

    用法：
        collector = TrajectoryCollector(max_episodes=50)
        # 在每个 env.step() 后调用
        collector.record_step(obs_dict, reward_components, done)
        # 训练结束后保存
        collector.save("/path/to/trajectory_data.npz")
    """

    def __init__(self, max_episodes: int = 50, max_steps_per_ep: int = 500):
        self.max_episodes = max_episodes
        self.max_steps = max_steps_per_ep

        # 存储已完成的 episode
        self.episodes: List[Dict[str, np.ndarray]] = []

        # 当前正在收集的 episode（按 env_id 索引）
        self._current_eps: Dict[int, Dict[str, List]] = {}

    def record_step(
        self,
        env_id: int,
        obs_dict: dict,
        reward_components: dict,
        total_reward: float,
        done: bool,
    ):
        """
        记录单步数据。

        Args:
            env_id: 环境编号
            obs_dict: 观测字典 (从 _to_dict 输出)
            reward_components: 各奖励分量 {"reaching": 0.5, "lifting": 0.1, ...}
            total_reward: 总奖励值
            done: 是否 episode 结束
        """
        if env_id not in self._current_eps:
            self._current_eps[env_id] = {
                "obs": [],
                "reward_components": [],
                "total_reward": [],
            }

        ep = self._current_eps[env_id]
        ep["obs"].append(obs_dict)
        ep["reward_components"].append(reward_components)
        ep["total_reward"].append(total_reward)

        if done:
            self._finalize_episode(env_id)

    def _finalize_episode(self, env_id: int):
        """将当前 episode 转为 numpy 并存储。"""
        if env_id not in self._current_eps:
            return

        ep = self._current_eps.pop(env_id)
        if len(ep["total_reward"]) < 5:
            return  # 太短的 episode 不要

        # 转换 obs 列表为结构化数组
        obs_keys = list(ep["obs"][0].keys())
        obs_arrays = {
            k: np.array([step[k] for step in ep["obs"]], dtype=np.float32)
            for k in obs_keys
        }

        # 转换 reward_components
        rc_keys = list(ep["reward_components"][0].keys())
        rc_arrays = {
            k: np.array([step[k] for step in ep["reward_components"]], dtype=np.float32)
            for k in rc_keys
        }

        episode_data = {
            "obs": obs_arrays,
            "reward_components": rc_arrays,
            "total_reward": np.array(ep["total_reward"], dtype=np.float32),
            "length": len(ep["total_reward"]),
        }

        self.episodes.append(episode_data)

        # 只保留最近的 max_episodes 个
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def save(self, path: str):
        """保存收集的轨迹数据。"""
        if not self.episodes:
            print("[WARN] No episodes collected, saving empty file")
            np.savez(path, n_episodes=0)
            return

        # 序列化为扁平结构便于 np.savez
        save_dict = {"n_episodes": len(self.episodes)}

        for i, ep in enumerate(self.episodes):
            save_dict[f"ep_{i}_total_reward"] = ep["total_reward"]
            save_dict[f"ep_{i}_length"] = ep["length"]

            for k, v in ep["obs"].items():
                save_dict[f"ep_{i}_obs_{k}"] = v
            for k, v in ep["reward_components"].items():
                save_dict[f"ep_{i}_rc_{k}"] = v

        np.savez_compressed(path, **save_dict)
        print(f"[INFO] Saved {len(self.episodes)} episodes to {path}")

    @staticmethod
    def load(path: str) -> List[Dict]:
        """加载轨迹数据。"""
        data = np.load(path, allow_pickle=True)
        n_episodes = int(data["n_episodes"])
        if n_episodes == 0:
            return []

        episodes = []
        for i in range(n_episodes):
            total_reward = data[f"ep_{i}_total_reward"]
            length = int(data[f"ep_{i}_length"])

            # 恢复 obs
            obs = {}
            rc = {}
            for key in data.files:
                if key.startswith(f"ep_{i}_obs_"):
                    obs_key = key[len(f"ep_{i}_obs_"):]
                    obs[obs_key] = data[key]
                elif key.startswith(f"ep_{i}_rc_"):
                    rc_key = key[len(f"ep_{i}_rc_"):]
                    rc[rc_key] = data[key]

            episodes.append({
                "obs": obs,
                "reward_components": rc,
                "total_reward": total_reward,
                "length": length,
            })

        return episodes
