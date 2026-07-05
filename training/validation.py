import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy


class BestModelCheckpoint(BaseCallback):

    def __init__(self, eval_env, model_path, vecnorm_path, n_eval_episodes=12,
                 eval_freq=20000, sync_norm=False, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.sync_norm = sync_norm
        self.best_mean_reward = -np.inf
        self._last_eval = 0

    def _evaluate_and_maybe_save(self) -> None:
        if self.sync_norm:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except Exception:
                pass
        mean_reward, _ = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
            deterministic=True, warn=False,
        )
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self.model_path)
            vec = self.model.get_vec_normalize_env()
            if vec is not None and self.vecnorm_path is not None:
                vec.save(self.vecnorm_path)
            if self.verbose:
                print(f"[BestModelCheckpoint] new best mean_reward={mean_reward:.3f} "
                      f"at step {self.num_timesteps} -> saved {self.model_path}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            self._evaluate_and_maybe_save()
        return True

    def _on_training_end(self) -> None:
        self._evaluate_and_maybe_save()
