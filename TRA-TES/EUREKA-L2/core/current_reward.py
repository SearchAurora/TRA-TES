def compute_reward(obs: dict) -> dict:
    reach_reward = np.where(obs["eef_obj_dist"] < 0.1, 10.0 - 100.0 * obs["eef_obj_dist"], 0.0)
    grasp_reward = np.where((obs["eef_obj_dist"] < 0.05) & (obs["gripper_open"] < 0.3), 20.0, 0.0)
    lift_reward = np.where(obs["obj_pos_z"] > 0.055, (obs["obj_pos_z"] - 0.055) * 100.0, 0.0)
    track_reward = np.where(obs["obj_pos_z"] > 0.055, np.exp(-obs["obj_goal_dist"] / 0.1), 0.0)

    # Reduce the harshness of the smoothness penalty
    smoothness_penalty = np.where(obs["joint_vel_norm"] < 0.1, -0.02 * obs["joint_vel_norm"], -0.05 * obs["joint_vel_norm"])

    # Reduce the harshness of the jerk penalty
    jerk_penalty = np.where(obs["action_rate"] < 0.1, -0.02 * obs["action_rate"], -0.05 * obs["action_rate"])

    total_reward = reach_reward + grasp_reward + lift_reward + track_reward + smoothness_penalty + jerk_penalty

    return {"reach": reach_reward, "grasp": grasp_reward, "lift": lift_reward, "track": track_reward, "smoothness": smoothness_penalty, "jerk": jerk_penalty, "total": total_reward}