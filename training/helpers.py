import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback


class StaticNormalizeObs(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        low = np.asarray(env.observation_space.low, dtype=np.float64)
        high = np.asarray(env.observation_space.high, dtype=np.float64)
        self._low = low
        self._span = np.where(high > low, high - low, 1.0)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=low.shape, dtype=np.float32
        )

    def observation(self, obs):
        scaled = 2.0 * (np.asarray(obs, dtype=np.float64) - self._low) / self._span - 1.0
        return scaled.astype(np.float32)


class ActionNoiseDecayCallback(BaseCallback):

    def __init__(self, final_sigma: float):
        super().__init__()
        self.final_sigma = final_sigma
        self._initial_sigma = None

    def _on_training_start(self) -> None:
        noise = getattr(self.model, "action_noise", None)
        if noise is not None and hasattr(noise, "_sigma"):
            self._initial_sigma = np.array(noise._sigma, dtype=float).copy()

    def _on_step(self) -> bool:
        noise = getattr(self.model, "action_noise", None)
        if self._initial_sigma is not None and noise is not None:
            progress = min(1.0, self.num_timesteps / self.model._total_timesteps)
            noise._sigma = (
                self._initial_sigma * (1.0 - progress) + self.final_sigma * progress
            )
        return True
