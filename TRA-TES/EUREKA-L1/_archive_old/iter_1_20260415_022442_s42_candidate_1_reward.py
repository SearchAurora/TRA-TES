def compute_reward(obs: dict) -> dict:
    # Reward for moving closer to the handle
    reward_dist_to_handle = -2.0 * obs["dist_to_handle"]

    # Reward for opening the drawer
    target_open_distance = 0.15  # Example target, adjust as needed
    reward_drawer_open = 3.0 * (obs["drawer_open"] - target_open_distance)
    if obs["drawer_open"] > target_open_distance:
        reward_drawer_open -= 5.0 * (obs["drawer_open"] - target_open_distance)

    # Penalty for high joint velocities
    reward_joint_vel_norm = -1.0 * obs["joint_vel_norm"]

    # Penalty for high joint positions
    reward_joint_pos_norm = -1.0 * obs["joint_pos_norm"]

    # Reward for aligning with the target position
    to_target_vec = np.array([obs["to_target_x"], obs["to_target_y"], obs["to_target_z"]])
    to_target_dist = np.linalg.norm(to_target_vec)
    reward_to_target = -2.0 * to_target_dist

    # Total reward
    total_reward = (
        reward_dist_to_handle +
        reward_drawer_open +
        reward_joint_vel_norm +
        reward_joint_pos_norm +
        reward_to_target
    )

    return {
        "reward_dist_to_handle": reward_dist_to_handle,
        "reward_drawer_open": reward_drawer_open,
        "reward_joint_vel_norm": reward_joint_vel_norm,
        "reward_joint_pos_norm": reward_joint_pos_norm,
        "reward_to_target": reward_to_target,
        "total": total_reward
    }