"""
EUREKA Custom Env Config for FrankaCubeLift
===========================================
继承 IsaacLab 原始 FrankaCubeLiftEnvCfg，
将所有内置奖励权重清零，改用 EUREKA 生成的奖励函数。

注入方式：通过 RewardsCfg 把所有原始 term weight 设为 0，
然后在 wrapper env 里调用 compute_reward。
"""
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

# 导入原始 Franka Lift 配置
import sys
sys.path.insert(0, "/root/autodl-tmp/TRA/IsaacLab/source/isaaclab_tasks")

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift import mdp


@configclass
class EurekaRewardsCfg:
    """所有内置奖励权重清零，EUREKA 奖励通过 wrapper 注入。"""
    reaching_object              = RewTerm(func=mdp.object_ee_distance,   params={"std": 0.1}, weight=0.0)
    lifting_object               = RewTerm(func=mdp.object_is_lifted,     params={"minimal_height": 0.04}, weight=0.0)
    object_goal_tracking         = RewTerm(func=mdp.object_goal_distance, params={"std": 0.3, "command_name": "object_pose", "robot_attr": "ee_frame", "minimal_height": 0.04}, weight=0.0)
    object_goal_tracking_fine_grained = RewTerm(func=mdp.object_goal_distance, params={"std": 0.05, "command_name": "object_pose", "robot_attr": "ee_frame", "minimal_height": 0.04}, weight=0.0)
    action_rate                  = RewTerm(func=mdp.action_rate_l2,       weight=0.0)
    joint_vel                    = RewTerm(func=mdp.joint_vel_l2,         params={"asset_cfg": MISSING}, weight=0.0)


@configclass
class EurekaFrankaCubeLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """EUREKA 版本：内置奖励全部清零。"""

    def __post_init__(self):
        super().__post_init__()
        # 清零所有内置奖励
        self.rewards.reaching_object.weight               = 0.0
        self.rewards.lifting_object.weight                = 0.0
        self.rewards.object_goal_tracking.weight          = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.action_rate.weight                   = 0.0
        self.rewards.joint_vel.weight                     = 0.0
