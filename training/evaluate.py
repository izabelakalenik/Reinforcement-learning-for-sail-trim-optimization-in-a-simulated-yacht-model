import sys
import csv
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from simulation.gym_sailing.envs.sail_trim_env import SailTrimEnv
from simulation.gym_sailing.physics.sailboat import SailBoat
from simulation.gym_sailing.utils.angles import norm
from data.polar_data_preprocessing import build_polar_interpolator_from_csv
from training.helpers import StaticNormalizeObs
from main.config import (
    TIME_STEPS, SEEDS, SUFFIX, N_EPISODES, EVAL_SEED, WIND_TEST,
    FIXED_HEADING, BANDS_ORDER, DEFAULT_POLAR_CSV, MODELS_DIR,
)
from utils.paths import results_csv_path

ALGOS = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "TD3": TD3}


def make_eval_venv(algo, vecnorm_path, seed, max_ep=1500, wind_csv_path=None):
    is_ppo = algo == "PPO"

    def make_env():
        env = SailTrimEnv(render_mode=None, max_episode_steps=max_ep, wind_csv_path=wind_csv_path)
        if not is_ppo:
            env = StaticNormalizeObs(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    venv = DummyVecEnv([make_env])
    if is_ppo:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=False)
    venv.seed(seed)
    return venv


def band(twa):
    # TWA bands by point of sail (sailing terminology):
    #   nogo  0-45     (kąt martwy / no-go zone - boat cannot make progress; mostly filtered by min v_ref)
    #   close 45-80    (bajdewind / close-hauled)
    #   beam  80-100   (półwiatr / beam reach)
    #   broad 100-170  (baksztag / broad reach)
    #   run   170-180  (fordewind / running)
    for lo, hi, name in [(0, 45, "nogo"), (45, 80, "close"), (80, 100, "beam"),
                         (100, 170, "broad"), (170, 181, "run")]:
        if lo <= twa < hi:
            return name
    return "?"


def _ceiling_speed(wx, wy, n_steps=900):
    sails = np.deg2rad(np.linspace(-90.0, 90.0, 37))
    v = np.full(sails.shape, 0.5)
    k = SailBoat.TIME_STEP / SailBoat(0, 0, FIXED_HEADING).mass
    sc, cl, cd = SailBoat.SAILCOEFF, SailBoat.CL_MAX, SailBoat.CD_MAX
    dc, wv = SailBoat.DRAGCOEFF, SailBoat.WAVE_DRAGCOEFF
    for _ in range(n_steps):
        app_y = wy - v
        aws = np.hypot(wx, app_y)
        aws = np.where(aws < 0.01, 0.01, aws)
        wind_dir = np.arctan2(app_y, wx)
        aoa = norm(FIXED_HEADING + sails - wind_dir)

        # fold AoA to [-pi/2, pi/2] (symmetric sail chord) - a +-90° bound, not the heading
        aoa = np.where(aoa > np.pi / 2, aoa - np.pi, aoa)
        aoa = np.where(aoa < -np.pi / 2, aoa + np.pi, aoa)
        lift = cl * np.sin(2 * aoa)
        drag_c = cd * np.sin(aoa) ** 2
        aero_y = sc * aws * (lift * wx + drag_c * app_y)
        forward = np.maximum(0.0, aero_y)
        hull = dc * v ** 2 + wv * v ** 4
        v = np.maximum(0.0, v + k * (forward - hull))
    return float(v.max())


def build_ceiling_efficiency_lookup():
    polar = build_polar_interpolator_from_csv(str(DEFAULT_POLAR_CSV), output_unit="m/s")
    twa_grid = np.arange(0.0, 181.0, 10.0)
    tws_grid = np.arange(1.0, 15.1, 1.0)
    table = np.zeros((len(twa_grid), len(tws_grid)))
    for i, twa in enumerate(twa_grid):
        wind_from = FIXED_HEADING + np.deg2rad(twa)  # any side; ceiling is sign-symmetric
        for j, tws in enumerate(tws_grid):
            wx = -tws * np.cos(wind_from)
            wy = -tws * np.sin(wind_from)
            v_ref = float(polar(min(twa, 180.0), tws))
            table[i, j] = _ceiling_speed(wx, wy) / v_ref if v_ref > 1e-6 else np.nan

    def lookup(twa_deg, tws):
        i = int(np.clip(np.round(twa_deg / 10.0), 0, len(twa_grid) - 1))
        j = int(np.clip(np.round(tws - 1.0), 0, len(tws_grid) - 1))
        return table[i, j]

    return lookup


def eval_algo(algo, suffix, n_episodes=N_EPISODES, seed=EVAL_SEED, wind_csv_path=WIND_TEST, ceil_lookup=None):
    mp = MODELS_DIR / f"sail_trim_{algo.lower()}_model_{TIME_STEPS}_{suffix}.zip"
    vps = sorted(MODELS_DIR.glob(f"sail_trim_{algo.lower()}_model_{TIME_STEPS}_{suffix}_vecnormalize_*.pkl"))
    vp = vps[-1] if vps else (MODELS_DIR / "missing.pkl")

    if not mp.exists():
        return None
    if ceil_lookup is None:
        ceil_lookup = build_ceiling_efficiency_lookup()

    model = ALGOS[algo].load(str(mp), device="cpu")
    venv = make_eval_venv(algo, str(vp), seed, wind_csv_path=str(wind_csv_path))
    obs = venv.reset()
    eff_all, effceil_all = [], []
    eff_bands, effceil_bands = {}, {}
    done = 0

    while done < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        if dones[0]:
            info = infos[0]
            if info.get("episode_reference_speed", 0) < 0.5:
                continue
            eff = float(info.get("episode_tail_trim_efficiency", 0.0))
            twa = float(info.get("wind_relative_heading", 0.0))
            tws = float(info.get("wind_speed", 0.0))
            ce = ceil_lookup(twa, tws)
            effceil = eff / ce if ce and ce > 1e-6 else np.nan
            b = band(twa)
            eff_all.append(eff)
            eff_bands.setdefault(b, []).append(eff)
            
            if np.isfinite(effceil):
                effceil_all.append(effceil)
                effceil_bands.setdefault(b, []).append(effceil)
            done += 1
    venv.close()
    return {
        "overall": float(np.mean(eff_all)),
        "bands": {k: float(np.mean(v)) for k, v in eff_bands.items()},
        "overall_effceil": float(np.mean(effceil_all)) if effceil_all else float("nan"),
        "bands_effceil": {k: float(np.mean(v)) for k, v in effceil_bands.items()},
        "n": len(eff_all),
    }


def main(seeds=SEEDS, suffix=SUFFIX, n_episodes=N_EPISODES, wind_csv=WIND_TEST):
    wind_name = Path(wind_csv).name
    ceil_lookup = build_ceiling_efficiency_lookup()
    fields = (["algo", "n", "wind", "overall", "overall_effceil"]
              + BANDS_ORDER + [f"{b}_effceil" for b in BANDS_ORDER])

    for seed in seeds:
        sfx = f"{seed}{suffix}"
        print(f"\n=== Deterministic eval (suffix={sfx}, n={n_episodes}, wind={wind_name}) ===")
        print("    eff = v/v_ref (vs polar);  eff/ceil = v/v_best (vs simulator's achievable optimum)")
        print(f'{"algo":6}{"eff":>8}{"eff/ceil":>10}   per-band eff (nogo/close/beam/broad/run)')

        rows = []
        for algo in ["PPO", "SAC", "DDPG", "TD3"]:
            res = eval_algo(algo, sfx, n_episodes=n_episodes,
                            wind_csv_path=str(wind_csv), ceil_lookup=ceil_lookup)
            if res is None:
                print(f"{algo:6}  [model not found]")
                continue
            bandstr = " ".join(f"{b}:{res['bands'].get(b, float('nan')):.2f}" for b in BANDS_ORDER)
            print(f"{algo:6}{res['overall']:8.3f}{res['overall_effceil']:10.3f}   {bandstr}")
            row = {"algo": algo, "n": res["n"], "wind": wind_name,
                   "overall": round(res["overall"], 4),
                   "overall_effceil": round(res["overall_effceil"], 4)}
            row.update({b: round(res["bands"].get(b, float("nan")), 4) for b in BANDS_ORDER})
            row.update({f"{b}_effceil": round(res["bands_effceil"].get(b, float("nan")), 4)
                        for b in BANDS_ORDER})
            rows.append(row)

        if not rows:
            print(f"  [skipped {sfx}: model not found]")
            continue
        out = results_csv_path(sfx)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results -> {out}")


if __name__ == "__main__":
    main()
