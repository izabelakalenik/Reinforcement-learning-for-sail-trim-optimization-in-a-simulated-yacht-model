from dataclasses import dataclass
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SailTrimMDP:

    wind_limit: float = 20.0
    sail_limit: float = np.pi / 2
    speed_limit: float = 12.0
    max_episode_steps: int = 1500

    speed_max_coefficient: float = 1.0
    speed_tracking_coefficient: float = 1.0
    step_penalty: float = 0.01
    min_ref_speed: float = 0.5  # m/s - exclude no-go zone episodes from training

    initial_speed_low: float = 0.3
    initial_speed_high: float = 1.0

    def observation_space(self):
        low = np.array(
            [0.0, -np.pi, -self.sail_limit, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [self.wind_limit, np.pi, self.sail_limit, self.speed_limit],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def action_space(self):
        return spaces.Box(
            low=-self.sail_limit,
            high=self.sail_limit,
            shape=(1,),
            dtype=np.float32,
        )
