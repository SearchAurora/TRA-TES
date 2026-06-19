def compute_reward(obs: dict) -> dict:
    # Reward for moving closer to the handle
    dist_to_handle_reward = -2.0 * obs["dist_to_handle"]

    # Reward for opening the drawer
    drawer_open_reward = 3.0 * obs["drawer_open"]

    # Penalize high joint velocities and positions to encourage smooth, controlled motion
    joint_smoothness_penalty = -1.0 * (obs["joint_pos_norm"] + obs["joint_vel_norm"])

    # Reward for reaching the target position
    to_target_distance = np.sqrt(obs["to_target_x"]**2 + obs["to_target_y"]**2 + obs["to_target_z"]**2)
    to_target_reward = -2.0 * to_target_distance

    # Total reward
    total_reward = dist_to_handle_reward + drawer_open_reward + joint_smoothness_penalty + to_target_reward

    return {
        "dist_to_handle_reward": dist_to_handle_reward,
        "drawer_open_reward": drawer_open_reward,
        "joint_smoothness_penalty": joint_smoothness_penalty,
        "to_target_reward": to_target_reward,
        "total": total_reward
    }