def compute_reward(obs: dict) -> dict:
    # Define reward components
    approach_reward = -5.0 * (np.abs(obs["eef_pos_x"] - obs["obj_pos_x"]) + np.abs(obs["eef_pos_y"] - obs["obj_pos_y"]))
    grasp_reward = -5.0 * (obs["gripper_open"] - 0.1) if obs["gripper_open"] > 0.1 else 0.0
    lift_reward = 5.0 * (obs["obj_pos_z"] - 0.05) if obs["obj_pos_z"] > 0.05 else 0.0
    goal_reward = -5.0 * obs["obj_goal_dist"]
    smoothness_reward = -1.0 * obs["joint_vel_norm"] - 1.0 * obs["action_rate"]

    # Total reward
    total_reward = approach_reward + grasp_reward + lift_reward + goal_reward + smoothness_reward

    return {
        "approach": approach_reward,
        "grasp": grasp_reward,
        "lift": lift_reward,
        "goal": goal_reward,
        "smoothness": smoothness_reward,
        "total": total_reward
    }