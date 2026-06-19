"""
Prompt Builder — Multi-task (Lift / Cabinet / Anymal)

Design principles (adapted from EUREKA, ICLR 2024):
  1. Environment interface as context — obs dict is the only contract
  2. Dict output with named components — required by TES attribution
  3. Free-form reward design — LLM names its own components
  4. Reward reflection with per-component stats + TES diagnostics
  5. Component names are NOT fixed — TES dynamically reads any dict keys

Key differences from original EUREKA:
  - We provide a curated obs dict (not raw env source code) because
    our obs pass through _to_dict() which extracts semantic quantities
  - We suggest (but don't mandate) component structure to ensure
    multi-component output for TES L2 variance decomposition
  - TES diagnostic injection replaces EUREKA's scalar-only reflection
"""

import math

# ══════════════════════════════════════════════════════════════════════
#  Per-task configuration
# ══════════════════════════════════════════════════════════════════════

TASK_CONFIGS = {

    # ── Lift ──────────────────────────────────────────────────────────
    "lift": dict(
        task_name="FrankaCubeLift",

        description="""\
A 7-DOF Franka robot arm with a parallel gripper must lift a cube from a \
table and move it to a randomly sampled target position ~0.4 m above the table.
The cube starts at roughly (0.5, 0, 0.055) in the robot root frame.
Episode: 1500 steps, dt=0.02 s (decimation=2). Early termination if obj_pos_z < 0.02.
Success = position_error (L2 cube-to-goal distance) as low as possible.""",

        env_interface="""\
obs = {
    "obj_pos_x":      float,   # cube x in robot root frame
    "obj_pos_y":      float,   # cube y in robot root frame
    "obj_pos_z":      float,   # cube z  (>0.055 = lifted off table)
    "target_pos_x":   float,   # goal x
    "target_pos_y":   float,   # goal y
    "target_pos_z":   float,   # goal z  (~0.4 m above table)
    "obj_goal_dist":  float,   # ‖obj − goal‖₂  (lower = better)
    "eef_obj_dist":   float,   # ‖end-effector − cube‖₂  (lower = closer to grasp)
    "gripper_open":   float,   # gripper aperture  [0 closed … 1 open]
    "joint_vel_norm": float,   # ‖joint velocities‖  (smoothness indicator)
    "action_rate":    float,   # ‖Δaction‖  (jerk indicator)
}""",

        obs_keys="obj_pos_x, obj_pos_y, obj_pos_z, target_pos_x, target_pos_y, target_pos_z, obj_goal_dist, eef_obj_dist, gripper_open, joint_vel_norm, action_rate",

        task_stages="""\
The task has a natural sequential structure:
  Stage 1 — Reach:  move end-effector toward cube  (signal: eef_obj_dist ↓)
  Stage 2 — Grasp:  close gripper when near cube   (signal: gripper_open ↓, conditioned on eef_obj_dist < threshold)
  Stage 3 — Lift:   raise cube off table           (signal: obj_pos_z ↑)
  Stage 4 — Track:  move cube to goal              (signal: obj_goal_dist ↓)
Each stage depends on the previous one succeeding. Design rewards so early-stage
signals are always available (dense), and later-stage signals activate conditionally.""",

        reward_hints="""\
- Use eef_obj_dist for reaching (NOT obj_pos_x/y — that measures cube position, not gripper-to-cube distance).
- Grasping MUST be conditioned on proximity: rewarding a closed gripper far from the cube teaches the agent to close first and never approach.
- Lifting reward should be proportional to height above table, not just a binary threshold.
- Goal tracking can use exponential kernel: exp(-obj_goal_dist / temperature) gated on obj_pos_z > lift_threshold.
- Keep penalties small relative to task rewards — large penalties cause risk-averse freezing.""",

        metric="position_error (lower = better)",
    ),

    # ── Cabinet (Open-Drawer) ─────────────────────────────────────────
    "cabinet": dict(
        task_name="FrankaCabinetOpenDrawer",

        description="""\
A 7-DOF Franka robot arm with a parallel gripper must open the top drawer of \
a cabinet by reaching the handle, grasping it, and pulling it outward.
Drawer joint range: 0.0 (closed) → ~0.39 (fully open). The to_target vector \
points FROM end-effector TO drawer handle — it tells the robot which direction to move.
Episode: 1500 steps. Timeout termination only (no early termination).
Success = drawer_pos as high as possible (position_error = 1 − drawer_pos/0.39).""",

        env_interface="""\
obs = {
    "joint_pos_norm": float,   # ‖robot joint positions‖ (scaled)
    "joint_vel_norm": float,   # ‖robot joint velocities‖
    "to_target_x":    float,   # ee→handle vector x  (direction to move)
    "to_target_y":    float,   # ee→handle vector y
    "to_target_z":    float,   # ee→handle vector z
    "dist_to_handle": float,   # ‖ee − handle‖₂  (lower = closer)
    "drawer_pos":     float,   # drawer joint position  (0=closed, ~0.39=open)
    "drawer_vel":     float,   # drawer joint velocity  (positive = opening)
    "drawer_open":    float,   # 1.0 if drawer_pos > 0.35, else 0.0
}""",

        obs_keys="joint_pos_norm, joint_vel_norm, to_target_x, to_target_y, to_target_z, dist_to_handle, drawer_pos, drawer_vel, drawer_open",

        task_stages="""\
The task has a natural sequential structure:
  Stage 1 — Approach: move end-effector to drawer handle  (signal: dist_to_handle ↓)
  Stage 2 — Align & Grasp: fine-position gripper around handle  (signal: to_target_{x,y,z} → 0)
  Stage 3 — Pull: retract arm to open drawer  (signal: drawer_pos ↑, drawer_vel > 0)
The robot must make physical contact with the handle before pulling is possible.
This means approach reward must dominate early, and pull reward should only
activate when dist_to_handle is small (gated/conditioned).""",

        reward_hints="""\
- Use dist_to_handle for approach (1/(1+d²) works well for dense signal).
- to_target_x/y/z gives a directional signal richer than scalar distance alone — consider rewarding alignment or using individual axis components for fine positioning.
- Pull reward should be gated on proximity: reward drawer_pos only when dist_to_handle < threshold (e.g. 0.1), otherwise the agent gets no gradient toward grasping.
- A bonus for drawer_open=1.0 gives a strong final-stage signal.
- The drawer can only move by physical force — the agent must push/pull, not telekinesis. Approach must succeed before pull can produce any reward.""",

        metric="position_error (lower = better; computed as 1 − drawer_pos/0.39)",
    ),

    # ── Anymal (Velocity Tracking Locomotion) ─────────────────────────
    "anymal": dict(
        task_name="AnymalCFlatLocomotion",

        description="""\
An Anymal-C quadruped robot (12 DOF: 3 joints × 4 legs) must walk on flat \
terrain, tracking randomly commanded linear and angular velocities.
Commands: lin_vel_x/y ∈ [−1, 1] m/s, yaw_rate ∈ [−1, 1] rad/s.
Episode: 1000 steps. Early termination if body contact force exceeds threshold (fall).
Zero command = stand still. Success = low tracking_error (= −(lin_vel_error + yaw_error)).""",

        env_interface="""\
obs = {
    "lin_vel_x":      float,   # body-frame forward velocity
    "lin_vel_y":      float,   # body-frame lateral velocity
    "lin_vel_z":      float,   # body-frame vertical velocity (should be ~0)
    "ang_vel_z":      float,   # body-frame yaw rate
    "cmd_lin_vel_x":  float,   # commanded forward velocity
    "cmd_lin_vel_y":  float,   # commanded lateral velocity
    "cmd_yaw_rate":   float,   # commanded yaw rate
    "lin_vel_error":  float,   # ‖cmd_lin_vel_xy − lin_vel_xy‖₂  (lower = better)
    "yaw_error":      float,   # |cmd_yaw_rate − ang_vel_z|  (lower = better)
    "joint_vel_norm": float,   # ‖all 12 joint velocities‖
    "action_rate":    float,   # ‖Δaction‖
    "base_tilt":      float,   # gravity projection onto xy  (0 = upright)
    "tracking_error": float,   # −(lin_vel_error + yaw_error)  (higher = better)
}""",

        obs_keys="lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_z, cmd_lin_vel_x, cmd_lin_vel_y, cmd_yaw_rate, lin_vel_error, yaw_error, joint_vel_norm, action_rate, base_tilt, tracking_error",

        task_stages="""\
Unlike manipulation tasks, locomotion has no sequential stages. Instead there
are simultaneous objectives with a clear priority hierarchy:
  Priority 1 — Velocity tracking: match commanded lin_vel and yaw_rate (dominant reward)
  Priority 2 — Stability: keep the robot upright (base_tilt near 0)
  Priority 3 — Efficiency: minimize energy and jerk (joint_vel_norm, action_rate)
  Priority 4 — Survival: stay alive (don't fall)
The tracking objective must dominate. If penalties are too large the robot
learns to stand still (zero velocity = zero penalty) and ignores commands.""",

        reward_hints="""\
- Exponential tracking kernels work well: exp(−error / temperature). Temperature ~0.25 is a good start.
- lin_vel tracking should have ~2–3× the weight of yaw tracking (xy velocity is harder to learn).
- base_tilt penalty should be moderate — too large and the robot freezes; too small and it stumbles.
- A small alive bonus (constant per step) helps early exploration before tracking rewards kick in.
- lin_vel_z should be near zero (penalize vertical bounce).
- The robot falls when base contact force is high — this correlates with large base_tilt, so the tilt penalty indirectly prevents falls.""",

        metric="tracking_error (higher = better; negative value, closer to 0 is better)",
    ),
}


# ══════════════════════════════════════════════════════════════════════
#  System prompt components
# ══════════════════════════════════════════════════════════════════════

ROLE = """\
You are an expert reward function engineer for reinforcement learning.
Your goal: write a reward function that makes the agent learn the task as fast as possible."""

FORMAT_RULES = """\
FORMAT RULES (strict):
1. Write: def compute_reward(obs: dict) -> dict
2. obs contains ONLY the keys listed below — any other key crashes at runtime.
3. Available imports: numpy as np, math. Nothing else.
4. Return a dict mapping component names (str) to scalar values (float), plus a "total" key.
   Example: return {"foo": foo, "bar": bar, "total": foo + bar}
5. You may use any number of components with any names — but at least 3 distinct components
   so the diagnostic system can attribute credit. Do NOT collapse everything into "total".
5b. VECTORIZATION RULES (CRITICAL — violating these makes training 5x slower):
   - Every obs value is a numpy array of shape (N,), NOT a scalar. Never index into it.
   - FORBIDDEN: obs["key"][0], obs["key"][i], any_var[idx] — causes "invalid index to scalar variable".
   - FORBIDDEN: for i in range(len(...)): — never loop over environments.
   - REQUIRED: use np.where() for ALL conditional logic.
     Good: reward = np.where(obs["dist"] < 0.05, 10.0, 0.0)
     Bad:  if obs["dist"] < 0.05: reward = 10.0
   - REQUIRED: all arithmetic must work element-wise on arrays.
     Good: bonus = np.exp(-obs["dist"])
     Bad:  bonus = math.exp(-obs["dist"][0])
6. Weights can range from −20.0 to 20.0.
7. Output ONLY the Python function. No explanation, no markdown fences."""


# ══════════════════════════════════════════════════════════════════════
#  Builder functions
# ══════════════════════════════════════════════════════════════════════

def build_initial_prompt(task: str = "lift") -> tuple[str, str]:
    """First iteration: zero-shot reward generation."""
    cfg = TASK_CONFIGS[task]

    system = f"""{ROLE}

{FORMAT_RULES}

AVAILABLE OBS KEYS: {cfg['obs_keys']}

DESIGN GUIDANCE:
{cfg['reward_hints']}"""

    user = f"""\
## Task: {cfg['task_name']}

### Description
{cfg['description']}

### Environment interface
```python
{cfg['env_interface']}
```

### Task structure
{cfg['task_stages']}

### Metric
{cfg['metric']}

Write compute_reward(obs: dict) -> dict for this task.
Use ONLY the obs keys listed above. Output ONLY the Python function."""

    return system, user


def build_reflection_prompt(
    current_code: str,
    training_stats: dict,
    iteration: int,
    tes_summary: str = "",
    task: str = "lift",
) -> tuple[str, str]:
    """Subsequent iterations: improve reward based on training stats + optional TES report."""
    cfg = TASK_CONFIGS[task]

    stats_lines = "\n".join(
        f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
        for k, v in training_stats.items()
        if k != "tes_summary"
    )

    system = f"""{ROLE}

{FORMAT_RULES}

AVAILABLE OBS KEYS: {cfg['obs_keys']}

DESIGN GUIDANCE:
{cfg['reward_hints']}"""

    user_parts = [
        f"## Task: {cfg['task_name']} — Iteration {iteration}",
        f"\n### Description\n{cfg['description']}",
        f"\n### Environment interface\n```python\n{cfg['env_interface']}\n```",
        f"\n### Task structure\n{cfg['task_stages']}",
        f"\n### Current reward function (to be improved)\n```python\n{current_code}\n```",
        f"\n### Training results from current reward\n{stats_lines}",
    ]

    if tes_summary:
        user_parts.append(f"""
### Trajectory Event Analysis
The following diagnostic report was automatically generated by analyzing the
agent's actual rollout trajectories. It identifies WHICH reward component is
problematic, WHEN in the episode the reward collapses, and in WHICH task phase
the failures concentrate. Use this to make targeted, structural fixes — not
blind weight scaling.

{tes_summary}

### Improvement instructions
1. Read the diagnostic report carefully. It tells you exactly what is broken.
2. If a CRASH is detected: the reward drops sharply at a specific timestep.
   Fix the component identified as the local trigger for that crash window.
3. If a component is marked PROBLEMATIC (high variance + negative mean):
   it is hurting training. Reshape its functional form, add gating conditions,
   or reduce its weight.
4. If a phase has high failure rate: the agent fails systematically in that
   episode segment. Strengthen the reward signal for the corresponding behavior.
5. {cfg['metric']} is the ground truth metric. Optimize for it, not mean_reward.
6. Use ONLY these obs keys: {cfg['obs_keys']}
7. Return a dict with named components and a "total" key.
8. Output ONLY the improved Python function.""")
    else:
        user_parts.append(f"""
### Improvement instructions
Improve the reward function based on the training results above.
- Focus on {cfg['metric']} as the primary optimization target.
- If a component appears ineffective (near-zero values), consider reshaping it.
- If the agent is stuck, check whether early-stage rewards provide dense signal.
- Use ONLY these obs keys: {cfg['obs_keys']}
- Return a dict with named components and a "total" key.
- Output ONLY the improved Python function.""")

    user = "\n".join(user_parts)
    return system, user
