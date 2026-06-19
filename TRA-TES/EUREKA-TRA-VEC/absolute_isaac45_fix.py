import os

path = "/root/autodl-tmp/TRA/EUREKA-TRA-VEC/run_one_iter_multi.py"
with open(path, "r") as f:
    lines = f.readlines()

# 1. 提取原始文件的所有函数和类定义 (从第一个 def 或 class 开始)
logic_content = []
start_collecting = False
for line in lines:
    if line.startswith("def ") or line.startswith("class ") or line.startswith("if __name__"):
        start_collecting = True
    if start_collecting:
        logic_content.append(line)

# 2. 构造“金字塔顶端”的启动头
# 严禁在 main() 之外 import torch, numpy, isaaclab_rl 等
new_header = """# -*- coding: utf-8 -*-
import os
import sys
import argparse

# [Step 1] 路径初始化
ISAACLAB_ROOT = '/root/autodl-tmp/TRA/IsaacLab'
for p in ['source/isaaclab', 'source/isaaclab_tasks', 'source/isaaclab_assets', 'source/isaaclab_rl']:
    full_p = os.path.join(ISAACLAB_ROOT, p)
    if full_p not in sys.path: sys.path.insert(0, full_p)

# [Step 2] 立即启动 App (在任何 RL 库导入之前)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Eureka Worker")
parser.add_argument("--task", type=str, default="lift")
parser.add_argument("--reward_fn", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--rl_iters", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--collect_trajectory", action="store_true")
parser.add_argument("--trajectory_path", type=str, default=None)
parser.add_argument("--tes_report_path", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 这里的配置必须和你的任务对齐
launcher_config = {"headless": True, "device": "cuda:0"}
simulation_app = AppLauncher(launcher_config).app

# [Step 3] 只有现在才允许导入“重型”库
import torch
import numpy as np
import math
import json
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import DirectRLEnv

# ---------------------------------------------------------
# 原始逻辑开始
# ---------------------------------------------------------
"""

# 3. 物理覆盖
with open(path, "w") as f:
    f.write(new_header)
    f.writelines(logic_content)

print("SUCCESS: run_one_iter_multi.py has been completely isolated for Isaac Sim 4.5.")
