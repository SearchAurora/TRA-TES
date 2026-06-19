import os

path = "/root/autodl-tmp/TRA/EUREKA-TRA-VEC/run_one_iter_multi.py"
with open(path, "r") as f:
    lines = f.readlines()

# 1. 寻找真正的逻辑开始位置 (跳过所有旧的 import)
start_line = 0
for i, line in enumerate(lines):
    if line.startswith("def ") or line.startswith("class "):
        start_line = i
        break

# 2. 构造符合 Isaac Sim 4.5 标准的“防弹”头部
new_header = [
    "import os\n",
    "import sys\n",
    "import argparse\n",
    "\n",
    "# [1] 物理对齐路径\n",
    "ISAACLAB_ROOT = '/root/autodl-tmp/TRA/IsaacLab'\n",
    "sys.path.insert(0, os.path.join(ISAACLAB_ROOT, 'source', 'isaaclab'))\n",
    "sys.path.insert(0, os.path.join(ISAACLAB_ROOT, 'source', 'isaaclab_tasks'))\n",
    "sys.path.insert(0, os.path.join(ISAACLAB_ROOT, 'source', 'isaaclab_assets'))\n",
    "sys.path.insert(0, os.path.join(ISAACLAB_ROOT, 'source', 'isaaclab_rl'))\n",
    "\n",
    "# [2] 必须在导入任何 isaaclab/torch 之前启动 App\n",
    "from isaaclab.app import AppLauncher\n",
    "launcher_config = {'headless': True, 'device': 'cuda:0'}\n",
    "simulation_app = AppLauncher(launcher_config).app\n",
    "\n",
    "# [3] 现在才能安全地导入 torch 和其他 RL 组件\n",
    "import torch\n",
    "import numpy as np\n",
    "from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg\n",
    "from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv\n",
    "\n",
    "import math\n",
    "import json\n",
    "import time\n",
    "\n"
]

# 3. 物理覆盖文件
with open(path, "w") as f:
    f.writelines(new_header)
    f.writelines(lines[start_line:])

print("NUCLEAR FIX: run_one_iter_multi.py has been rebuilt for Isaac Sim 4.5.")
