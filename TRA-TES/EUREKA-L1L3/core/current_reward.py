def compute_reward(obs: dict) -> dict:
    # Constants
    REACH_THRESHOLD = 0.05
    LIFT_THRESHOLD = 0.055
    GRASP_THRESHOLD = 0.05
    LIFT_HEIGHT_MULTIPLIER = 10.0
    LIFT_HEIGHT_OFFSET = 0.02
    GOAL_TRACKING_TEMPERATURE = 0.1
    SMOOTHNESS_PENALTY = -0.01
    JERK_PENALTY = -0.01

    # Reach reward
    reach_reward = np.where(obs["eef_obj_dist"] < REACH_THRESHOLD, 1.0, 0.0)

    # Grasp reward
    grasp_condition = np.where(obs["eef_obj_dist"] < GRASP_THRESHOLD, 1.0, 0.0)
    grasp_reward = np.where(grasp_condition * (1 - obs["gripper_open"]) > 0, 1.0, 0.0)

    # Lift reward
    lift_height = np.maximum(obs["obj_pos_z"] - LIFT_HEIGHT_OFFSET, 0.0)
    lift_reward = LIFT_HEIGHT_MULTIPLIER * lift_height

    # Goal tracking reward
    goal_tracking_reward = np.where(obs["obj_pos_z"] > LIFT_THRESHOLD, 
                                     np.exp(-obs["obj_goal_dist"] / GOAL_TRACKING_TEMPERATURE), 
                                     0.0)

    # Smoothness and jerk penalties
    smoothness_penalty = SMOOTHNESS_PENALTY * obs["joint_vel_norm"]
    jerk_penalty = JERK_PENALTY * obs["action_rate"]

    # Total reward
    total_reward = reach_reward + grasp_reward + lift_reward + goal_tracking_reward + smoothness_penalty + jerk_penalty

    return {
        "reach": reach_reward,
        "grasp": grasp_reward,
        "lift": lift_reward,
        "goal_tracking": goal_tracking_reward,
        "smoothness_penalty": smoothness_penalty,
        "jerk_penalty": jerk_penalty,
        "total": total_reward
    }