def compute_reward(obs: dict) -> dict:
    # Constants
    reach_threshold = 0.05
    grasp_threshold = 0.02
    lift_threshold = 0.055
    temperature = 0.1

    # Stage 1 - Reach
    reach_reward = np.exp(-obs["eef_obj_dist"] / reach_threshold)

    # Stage 2 - Grasp
    grasp_condition = np.where(obs["eef_obj_dist"] < grasp_threshold, 1.0, 0.0)
    grasp_reward = grasp_condition * (1.0 - obs["gripper_open"])

    # Stage 3 - Lift
    lift_reward = np.where(obs["obj_pos_z"] > lift_threshold, 
                           (obs["obj_pos_z"] - lift_threshold) * 10.0, 
                           0.0)

    # Stage 4 - Track
    track_reward = np.where(obs["obj_pos_z"] > lift_threshold,
                            np.exp(-obs["obj_goal_dist"] / temperature),
                            0.0)

    # Smoothness and Jerk Penalties
    smoothness_penalty = -0.1 * obs["joint_vel_norm"]
    jerk_penalty = -0.1 * obs["action_rate"]

    # Total Reward
    total_reward = (reach_reward + grasp_reward + lift_reward + track_reward + 
                    smoothness_penalty + jerk_penalty)

    return {
        "reach": reach_reward,
        "grasp": grasp_reward,
        "lift": lift_reward,
        "track": track_reward,
        "smoothness_penalty": smoothness_penalty,
        "jerk_penalty": jerk_penalty,
        "total": total_reward
    }