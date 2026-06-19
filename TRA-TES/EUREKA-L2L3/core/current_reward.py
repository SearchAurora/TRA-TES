def compute_reward(obs: dict) -> dict:
    # Constants
    REACH_THRESHOLD = 0.1
    GRASP_THRESHOLD = 0.05
    LIFT_THRESHOLD = 0.06
    LIFT_HEIGHT = 0.4
    TEMPERATURE = 0.1

    # Reach reward: Encourage the end-effector to get closer to the cube
    reach_reward = np.exp(-obs["eef_obj_dist"] / REACH_THRESHOLD)

    # Grasp reward: Encourage closing the gripper when near the cube
    grasp_condition = np.where(obs["eef_obj_dist"] < GRASP_THRESHOLD, 1.0, 0.0)
    grasp_reward = grasp_condition * (1.0 - obs["gripper_open"])

    # Lift reward: Encourage lifting the cube off the table
    lift_reward = np.where(obs["obj_pos_z"] > LIFT_THRESHOLD,
                           (obs["obj_pos_z"] - LIFT_THRESHOLD) / (LIFT_HEIGHT - LIFT_THRESHOLD),
                           0.0)

    # Track reward: Encourage moving the cube to the target position
    track_reward = np.where(obs["obj_pos_z"] > LIFT_THRESHOLD,
                            np.exp(-obs["obj_goal_dist"] / TEMPERATURE),
                            0.0)

    # Smoothness penalty: Penalize high joint velocities
    smoothness_penalty = -np.where(obs["joint_vel_norm"] > 0.1, obs["joint_vel_norm"], 0.0)

    # Jerk penalty: Penalize high action rates
    jerk_penalty = -np.where(obs["action_rate"] > 0.1, obs["action_rate"], 0.0)

    # Total reward
    total_reward = reach_reward + 2.0 * grasp_reward + 3.0 * lift_reward + 4.0 * track_reward + smoothness_penalty + jerk_penalty

    return {
        "reach": reach_reward,
        "grasp": grasp_reward,
        "lift": lift_reward,
        "track": track_reward,
        "smoothness_penalty": smoothness_penalty,
        "jerk_penalty": jerk_penalty,
        "total": total_reward
    }