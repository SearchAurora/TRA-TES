def compute_reward(obs: dict) -> dict:
    # Reward for moving the end-effector closer to the object
    eef_to_obj_dist = np.linalg.norm([obs["eef_pos_x"] - obs["obj_pos_x"], 
                                      obs["eef_pos_y"] - obs["obj_pos_y"], 
                                      obs["eef_pos_z"] - obs["obj_pos_z"]])
    approach_reward = 2.0 / (1.0 + eef_to_obj_dist)

    # Reward for grasping the object (gripper is closed and close to the object)
    grasp_reward = 3.0 if (obs["gripper_open"] < 0.1 and eef_to_obj_dist < 0.05) else 0.0

    # Reward for lifting the object (object is off the table)
    lift_reward = 4.0 * (obs["obj_pos_z"] - 0.05) if obs["obj_pos_z"] > 0.05 else 0.0

    # Reward for moving the object closer to the goal
    goal_distance_reward = 5.0 / (1.0 + obs["obj_goal_dist"])

    # Penalty for high joint velocities
    joint_vel_penalty = -2.0 * obs["joint_vel_norm"]

    # Penalty for high action rate
    action_rate_penalty = -1.0 * obs["action_rate"]

    # Total reward
    total_reward = (approach_reward + grasp_reward + lift_reward + 
                    goal_distance_reward + joint_vel_penalty + action_rate_penalty)

    return {
        "approach_reward": approach_reward,
        "grasp_reward": grasp_reward,
        "lift_reward": lift_reward,
        "goal_distance_reward": goal_distance_reward,
        "joint_vel_penalty": joint_vel_penalty,
        "action_rate_penalty": action_rate_penalty,
        "total": total_reward
    }