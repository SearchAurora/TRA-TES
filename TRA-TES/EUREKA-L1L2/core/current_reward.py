def compute_reward(obs: dict) -> dict:
    # Constants
    reach_threshold = 0.05
    lift_threshold = 0.06
    temperature = 0.1
    grasp_penalty = -1.0
    lift_reward_scale = 20.0
    track_reward_scale = 10.0
    smoothness_penalty = -0.1
    jerk_penalty = -0.1

    # Reach reward
    reach_reward = np.where(obs["eef_obj_dist"] < reach_threshold, 1.0, 0.0)

    # Grasp reward
    grasp_reward = np.where((obs["eef_obj_dist"] < reach_threshold) & (obs["gripper_open"] < 0.5), 1.0, 0.0)
    grasp_penalty = np.where((obs["eef_obj_dist"] > reach_threshold) & (obs["gripper_open"] < 0.5), grasp_penalty, 0.0)

    # Lift reward
    lift_reward = np.where(obs["obj_pos_z"] > lift_threshold, (obs["obj_pos_z"] - lift_threshold) * lift_reward_scale, 0.0)

    # Track reward
    track_reward = np.where(obs["obj_pos_z"] > lift_threshold, np.exp(-obs["obj_goal_dist"] / temperature) * track_reward_scale, 0.0)

    # Smoothness and jerk penalties
    smoothness_penalty = obs["joint_vel_norm"] * smoothness_penalty
    jerk_penalty = obs["action_rate"] * jerk_penalty

    # Total reward
    total_reward = reach_reward + grasp_reward + lift_reward + track_reward + smoothness_penalty + jerk_penalty

    return {
        "reach": reach_reward,
        "grasp": grasp_reward,
        "lift": lift_reward,
        "track": track_reward,
        "smoothness": smoothness_penalty,
        "jerk": jerk_penalty,
        "total": total_reward
    }