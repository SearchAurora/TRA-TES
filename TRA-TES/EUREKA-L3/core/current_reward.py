def compute_reward(obs: dict) -> dict:
    # Constants
    REACH_THRESHOLD = 0.05
    GRASP_THRESHOLD = 0.03
    LIFT_THRESHOLD = 0.06
    TEMPERATURE = 0.1

    # Reach reward
    reach_reward = np.exp(-obs["eef_obj_dist"] / REACH_THRESHOLD)

    # Grasp reward (conditioned on proximity)
    grasp_condition = np.where(obs["eef_obj_dist"] < GRASP_THRESHOLD, 1.0, 0.0)
    grasp_reward = grasp_condition * (1.0 - obs["gripper_open"])

    # Lift reward
    lift_height = np.maximum(obs["obj_pos_z"] - LIFT_THRESHOLD, 0.0)
    lift_reward = lift_height * 20.0

    # Track reward
    track_reward = np.where(obs["obj_pos_z"] > LIFT_THRESHOLD, 
                             np.exp(-obs["obj_goal_dist"] / TEMPERATURE), 0.0)

    # Smoothness penalty
    smoothness_penalty = -0.1 * obs["joint_vel_norm"]

    # Jerk penalty
    jerk_penalty = -0.05 * obs["action_rate"]

    # Total reward
    total_reward = reach_reward + grasp_reward + lift_reward + track_reward + smoothness_penalty + jerk_penalty

    return {
        "reach": reach_reward,
        "grasp": grasp_reward,
        "lift": lift_reward,
        "track": track_reward,
        "smoothness_penalty": smoothness_penalty,
        "jerk_penalty": jerk_penalty,
        "total": total_reward
    }