def compute_reward(obs: dict) -> dict:
    reach_reward = 10.0 * np.exp(-obs["eef_obj_dist"])
    grasp_condition = np.where(obs["eef_obj_dist"] < 0.05, 1.0, 0.0)
    grasp_reward = 20.0 * (1.0 - obs["gripper_open"]) * grasp_condition
    lift_reward = 15.0 * (obs["obj_pos_z"] - 0.055) * np.where(obs["obj_pos_z"] > 0.055, 1.0, 0.0)
    track_reward = 20.0 * np.exp(-obs["obj_goal_dist"] / 0.1) * np.where(obs["obj_pos_z"] > 0.055, 1.0, 0.0)
    smoothness_penalty = -5.0 * obs["joint_vel_norm"]
    jerk_penalty = -3.0 * obs["action_rate"]

    total_reward = reach_reward + grasp_reward + lift_reward + track_reward + smoothness_penalty + jerk_penalty
    return {"reach": reach_reward, "grasp": grasp_reward, "lift": lift_reward, "track": track_reward, "smoothness": smoothness_penalty, "jerk": jerk_penalty, "total": total_reward}