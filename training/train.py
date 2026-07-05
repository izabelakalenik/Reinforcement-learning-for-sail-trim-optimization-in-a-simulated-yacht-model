from typing import Optional, Dict
from pathlib import Path
import numpy as np
from simulation.gym_sailing.envs.sail_trim_env import SailTrimEnv
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.logging_csv import CSVLoggingCallback
from training.helpers import StaticNormalizeObs, ActionNoiseDecayCallback
from training.validation import BestModelCheckpoint

DDPG_NOISE_FINAL_SIGMA = 0.10
DDPG_NOISE_INITIAL_SIGMA = 0.30

def linear_schedule(initial_value: float, min_value: float = 1e-5):
    def func(progress_remaining: float) -> float:
        return max(min_value, progress_remaining * initial_value)
    return func

def train(
    algorithm: str,
    total_timesteps: int,
    model_path: str,
    training_metrics_csv_path: str,
    algo_params: Optional[Dict] = None,
    env_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None,
    eval_wind_csv_path: Optional[str] = None,
):
    
    algorithms = {
        "PPO": PPO,
        "SAC": SAC,
        "DDPG": DDPG,
        "TD3": TD3,
    }

    default_params = {
        "PPO": {
            "learning_rate": linear_schedule(3e-4),
            "n_steps": 2048,
            "batch_size": 256,  
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.005,
            "clip_range": 0.2,
            "policy_kwargs": dict(net_arch=[256, 256]),
            "verbose": 1,
        },
        "SAC": {
            "learning_rate": linear_schedule(3e-4),
            "buffer_size": int(1e6),
            "batch_size": 256,
            "gamma": 0.995,
            "tau": 0.005,
            "learning_starts": 5000,
            "gradient_steps": 1,
            "policy_kwargs": dict(net_arch=[256, 256]),
            "verbose": 1,
        },
        "DDPG": {
            "learning_rate": linear_schedule(3e-4),
            "buffer_size": int(1e6),
            "batch_size": 256,
            "gamma": 0.98,
            "tau": 0.005,
            "learning_starts": 10000,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "action_noise": NormalActionNoise(
                mean=np.zeros(1), sigma=DDPG_NOISE_INITIAL_SIGMA * np.ones(1)
            ),
            "policy_kwargs": dict(net_arch=[256, 256]),
            "verbose": 1,
        },
        "TD3": {
            "learning_rate": linear_schedule(3e-4), 
            "buffer_size": int(1e6),
            "batch_size": 256,
            "gamma": 0.98,
            "tau": 0.005,
            "learning_starts": 10000,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "action_noise": NormalActionNoise(
                mean=np.zeros(1), sigma=DDPG_NOISE_INITIAL_SIGMA * np.ones(1)
            ),
            "policy_kwargs": dict(net_arch=[256, 256]),
            "verbose": 1,
        },
    }

    model_algo = algorithms.get(algorithm.upper())
    if model_algo is None:
        raise ValueError("Unsupported algorithm. Use PPO, SAC, DDPG, or TD3.")

    algo_defaults = default_params.get(algorithm.upper(), {})
    algo_kwargs = dict(algo_defaults)
    if algo_params:
        algo_kwargs.update(algo_params)

    if seed is not None:
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)

    # wrap environment for stable training: Monitor + VecNormalize
    # PPO benefits from diverse rollouts across parallel envs; off-policy methods don't
    n_envs = 4 if algorithm.upper() == "PPO" else 1
    is_on_policy = algorithm.upper() == "PPO"

    def make_env(env_seed=None, max_steps=None, wind_csv_path=None):
        kwargs = dict(render_mode=None)
        if env_kwargs:
            kwargs.update(env_kwargs)
        if max_steps is not None:
            kwargs["max_episode_steps"] = max_steps
        if wind_csv_path is not None:
            kwargs["wind_csv_path"] = wind_csv_path
        env = SailTrimEnv(**kwargs)
        
        if not is_on_policy:
            env = StaticNormalizeObs(env)
        env = Monitor(env)
        if env_seed is not None:
            env.reset(seed=env_seed)
        return env

    def wrap_vecnorm(venv, training):
        """
        Running-statistics normalization (VecNormalize) keeps shifting its mean/std
        during training. That is fine for on-policy PPO (fresh rollouts each update)
        but corrupts off-policy DDPG/SAC/TD3: replayed transitions were normalized
        with stale stats, making the critic chase a non-stationary target. So PPO
        uses VecNormalize for both obs and reward, while off-policy methods get
        static, bounds-based obs normalization (StaticNormalizeObs in make_env) and
        unnormalized rewards.
        """
        venv = VecNormalize(venv, norm_obs=is_on_policy, norm_reward=is_on_policy, clip_obs=10.0)
        venv.training = training
        if not training:
            venv.norm_reward = False
        return venv

    # create envs with unique seeds
    env_seeds = None
    if seed is not None:
        env_seeds = [seed + i for i in range(n_envs)]

    env = DummyVecEnv([lambda s=s: make_env(env_seed=s) for s in (env_seeds or [None] * n_envs)])
    env = wrap_vecnorm(env, training=True)

    eval_seed = (seed + 1000) if seed is not None else None
    eval_env = DummyVecEnv([lambda: make_env(env_seed=eval_seed, max_steps=400,
                                             wind_csv_path=eval_wind_csv_path)])
    eval_env = wrap_vecnorm(eval_env, training=False)

    device = "cpu" if algorithm.upper() == "PPO" else "auto"
    model = model_algo("MlpPolicy", env, seed=seed, device=device, **algo_kwargs)

    csv_callback = CSVLoggingCallback(training_metrics_csv_path)
    vecnorm_path = f"{model_path}_vecnormalize_{total_timesteps}.pkl"

    # save the best deterministic policy seen during training
    best_ckpt = BestModelCheckpoint(
        eval_env=eval_env,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        n_eval_episodes=12,
        eval_freq=20000,
        sync_norm=is_on_policy,
        verbose=1,
    )

    callbacks = [csv_callback, best_ckpt]

    if getattr(model, "action_noise", None) is not None:
        callbacks.append(ActionNoiseDecayCallback(DDPG_NOISE_FINAL_SIGMA))
    callback = CallbackList(callbacks)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    if not Path(model_path + ".zip").exists():
        model.save(model_path)
        env.save(vecnorm_path)
    eval_env.close()
    env.close()

