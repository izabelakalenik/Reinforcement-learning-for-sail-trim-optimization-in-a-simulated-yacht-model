
def compute_trim_reward(
    current_boat_speed,
    reference_speed,
    speed_max_coefficient: float = 1.0,
    speed_tracking_coefficient: float = 1.0,
    step_penalty: float = 0.01,
):
    
    current_boat_speed = float(current_boat_speed)
    reference_speed = float(reference_speed)
    speed_max_coefficient = float(speed_max_coefficient)
    speed_tracking_coefficient = float(speed_tracking_coefficient)

    trim_efficiency = current_boat_speed / reference_speed if reference_speed > 1e-6 else 0.0
    speed_gap = abs(current_boat_speed - reference_speed)
    normalized_gap = speed_gap / reference_speed if reference_speed > 1e-6 else 0.0

    # both terms are normalized by v_ref so the reward is scale-free across wind conditions
    speed_reward = speed_max_coefficient * trim_efficiency
    tracking_penalty = speed_tracking_coefficient * normalized_gap

    reward = speed_reward - tracking_penalty - float(step_penalty)

    return float(reward), {
        "speed_reward": float(speed_reward),
        "speed_gap": float(speed_gap),
        "tracking_penalty": float(tracking_penalty),
        "trim_efficiency": float(trim_efficiency),
        "current_boat_speed": float(current_boat_speed),
        "reference_speed": float(reference_speed),
        "step_penalty": float(step_penalty),
    }
