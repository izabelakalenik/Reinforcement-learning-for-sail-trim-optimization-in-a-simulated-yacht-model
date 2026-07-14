import csv
import sys
import time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from simulation.gym_sailing.envs.sail_trim_env import SailTrimEnv
from training.helpers import StaticNormalizeObs
from training.train import default_params
from training.evaluate import ALGOS
from main.config import (TIME_STEPS, MAX_EPISODE_STEPS, WIND_TRAIN, WIND_VAL, ALGORITHMS,
                         VAL_EPISODES, VAL_MAX_STEPS, EVAL_FREQ,
                         BENCH_STEPS, BENCH_SEED, INFER_ACTIONS, RUN_CPU_CONTROL,
                         BENCHMARK_CSV)


def _params(algo):
    params = dict(default_params(algo))
    params["verbose"] = 0   # SB3 logging would pollute the timing
    return params


def _device(algo, override=None):
    if override is not None:
        return override
    return "cpu" if algo == "PPO" else "auto"


def _make_venv(algo):
    is_ppo = algo == "PPO"
    n_envs = 4 if is_ppo else 1

    def make_env(env_seed):
        def _f():
            env = SailTrimEnv(render_mode=None, max_episode_steps=MAX_EPISODE_STEPS,
                              wind_csv_path=WIND_TRAIN)
            if not is_ppo:
                env = StaticNormalizeObs(env)
            env = Monitor(env)
            env.reset(seed=env_seed)
            return env
        return _f

    venv = DummyVecEnv([make_env(BENCH_SEED + i) for i in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=is_ppo, norm_reward=is_ppo, clip_obs=10.0)
    venv.training = True
    return venv


class _Timer(BaseCallback):

    def __init__(self, learning_starts):
        super().__init__()
        self.learning_starts = learning_starts
        self.t_start = None
        self.t_steady = None

    def _on_training_start(self) -> None:
        self.t_start = time.perf_counter()

    def _on_step(self) -> bool:
        if self.t_steady is None and self.num_timesteps >= self.learning_starts:
            self.t_steady = time.perf_counter()
        return True


def benchmark(algo, device_override=None):
    venv = _make_venv(algo)
    params = _params(algo)
    model = ALGOS[algo]("MlpPolicy", venv, seed=BENCH_SEED,
                        device=_device(algo, device_override), **params)

    ls = params.get("learning_starts", 0)   # PPO has none -> 0
    timer = _Timer(ls)
    model.learn(total_timesteps=BENCH_STEPS, callback=timer)
    t_end = time.perf_counter()
    venv.close()

    t_steady_start = timer.t_steady or timer.t_start
    warm_time = t_steady_start - timer.t_start          # 0 for PPO (no learning_starts)
    steady_steps = BENCH_STEPS - ls
    steady_time = t_end - t_steady_start
    sec_per_step = steady_time / steady_steps

    return {
        "algo": algo,
        "warm_steps": ls,
        "warm_time": warm_time,
        "steady_steps": steady_steps,
        "steady_time": steady_time,
        "sec_per_step": sec_per_step,
        "steps_per_sec": 1.0 / sec_per_step,
        # extrapolate the full training run from the measured steady-state cost
        "total_est_s": warm_time + (TIME_STEPS - ls) * sec_per_step,
    }


def _make_val_venv(algo):
    is_ppo = algo == "PPO"

    def _f():
        env = SailTrimEnv(render_mode=None, max_episode_steps=VAL_MAX_STEPS,
                          wind_csv_path=WIND_VAL)
        if not is_ppo:
            env = StaticNormalizeObs(env)
        env = Monitor(env)
        env.reset(seed=BENCH_SEED + 1000)
        return env

    venv = DummyVecEnv([_f])
    venv = VecNormalize(venv, norm_obs=is_ppo, norm_reward=False, clip_obs=10.0)
    venv.training = False
    return venv


def _fresh_model(algo, venv, device_override=None):
    return ALGOS[algo]("MlpPolicy", venv, seed=BENCH_SEED,
                       device=_device(algo, device_override), **_params(algo))


def benchmark_validation(algo):
    venv = _make_val_venv(algo)
    model = _fresh_model(algo, venv)
    obs = venv.reset()

    t0 = time.perf_counter()
    episodes = 0
    while episodes < VAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(action)
        if dones[0]:
            episodes += 1
    elapsed = time.perf_counter() - t0
    venv.close()
    return elapsed


def benchmark_inference(algo):
    venv = _make_val_venv(algo)
    model = _fresh_model(algo, venv)
    obs = venv.reset()

    for _ in range(200):                       # warm-up
        model.predict(obs, deterministic=True)

    t0 = time.perf_counter()
    for _ in range(INFER_ACTIONS):
        model.predict(obs, deterministic=True)
    elapsed = time.perf_counter() - t0
    venv.close()
    return elapsed / INFER_ACTIONS


def _save(rows, n_val):
    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "sec_per_step", "steps_per_sec",
                                          "learning_min", "validation_min", "total_min",
                                          "action_selection_us"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "algo": r["algo"],
                "sec_per_step": round(r["sec_per_step"], 6),
                "steps_per_sec": round(r["steps_per_sec"], 1),
                "learning_min": round(r["total_est_s"] / 60, 2),
                "validation_min": round(n_val * r["val_time"] / 60, 2),
                "total_min": round(r["total_full_s"] / 60, 2),
                "action_selection_us": round(r["infer_s"] * 1e6, 1),
            })
    print(f"\nSaved measurement -> {BENCHMARK_CSV}")


def main():
    n_val = TIME_STEPS // EVAL_FREQ
    print(f"Throughput benchmark: {BENCH_STEPS} steps per algorithm, seed {BENCH_SEED}")
    print("(warm-up = steps before learning_starts; steady = steps with gradient updates)\n")

    rows = []
    for a in ALGORITHMS:
        r = benchmark(a)
        r["val_time"] = benchmark_validation(a)
        r["infer_s"] = benchmark_inference(a)
        # full run = learning + the n_val validations that BestModelCheckpoint performs
        r["total_full_s"] = r["total_est_s"] + n_val * r["val_time"]
        rows.append(r)

    print(f"\n{'algo':6}{'steps/s':>10}{'ms/step':>10}{'learning':>12}"
          f"{'validation':>13}{'TOTAL 500k':>13}")
    for r in rows:
        print(f"{r['algo']:6}{r['steps_per_sec']:10.1f}{r['sec_per_step'] * 1000:10.2f}"
              f"{r['total_est_s'] / 60:9.1f} min{n_val * r['val_time'] / 60:9.1f} min"
              f"{r['total_full_s'] / 60:9.1f} min")

    _save(rows, n_val)

    fastest = min(rows, key=lambda r: r["sec_per_step"])
    print(f"\nCheapest step: {fastest['algo']} ({fastest['sec_per_step'] * 1000:.2f} ms/step)")
    for r in sorted(rows, key=lambda r: r["sec_per_step"]):
        rel = r["sec_per_step"] / fastest["sec_per_step"]
        print(f"  {r['algo']:5} step costs {rel:5.1f}x a {fastest['algo']} step")

    print(f"\nValidation: {n_val} passes per run, {VAL_EPISODES} episodes each "
          f"(a constant overhead, near-identical for every algorithm)")
    print(f"\n{'algo':6}{'action selection':>20}")
    for r in rows:
        print(f"{r['algo']:6}{r['infer_s'] * 1e6:16.0f} us")

    if not RUN_CPU_CONTROL:
        return

    print("\n\n=== Same measurement with every algorithm forced onto the CPU ===")
    cpu_rows = {r["algo"]: r for r in (benchmark(a, device_override="cpu") for a in ALGORITHMS)}

    print(f"\n{'algo':6}{'as trained':>14}{'all on CPU':>14}{'speed-up on CPU':>18}")
    for r in rows:
        c = cpu_rows[r["algo"]]
        speedup = r["sec_per_step"] / c["sec_per_step"]
        print(f"{r['algo']:6}{r['sec_per_step'] * 1000:11.2f} ms{c['sec_per_step'] * 1000:11.2f} ms"
              f"{speedup:16.2f}x")

    cheap_cpu = min(cpu_rows.values(), key=lambda r: r["sec_per_step"])
    print(f"\nOn CPU, relative to {cheap_cpu['algo']}:")
    for c in sorted(cpu_rows.values(), key=lambda r: r["sec_per_step"]):
        print(f"  {c['algo']:5} step costs {c['sec_per_step'] / cheap_cpu['sec_per_step']:5.1f}x")


if __name__ == "__main__":
    main()
