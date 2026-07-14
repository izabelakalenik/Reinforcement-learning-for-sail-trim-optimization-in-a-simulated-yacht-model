import sys
from pathlib import Path
import numpy as np
import pandas as pd
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from main.config import (SEEDS, SUFFIX, MAX_EPISODE_STEPS, ALGORITHMS, SMOOTH, MIN_REF_SPEED,
                         METRICS_DIR, EFF_THRESHOLDS)


def _seed_csvs(algo, seeds):
    paths = []
    for s in seeds:
        hits = sorted(METRICS_DIR.glob(f"results_{algo.lower()}_*_{s}{SUFFIX}_*.csv"))
        if hits:
            paths.append(hits[-1])
    return paths


def _curve_metrics(csv_path, thresholds):
    df = pd.read_csv(csv_path)
    df = df[df["ref_speed"] >= MIN_REF_SPEED]
    eff = df["tail_trim_efficiency"].to_numpy(dtype=float)

    aulc = pd.Series(eff).rolling(SMOOTH, min_periods=1).mean().to_numpy()

    strict = pd.Series(eff).rolling(SMOOTH, min_periods=SMOOTH).mean().to_numpy()
    out = {"AULC": float(np.clip(aulc, 0, None).mean())}
    for thr in thresholds:
        reached = np.where(strict >= thr)[0]
        out[f"steps_to_{thr}"] = float(reached[0] * MAX_EPISODE_STEPS) if len(reached) else float("inf")
    return out


def compute(seeds, thresholds=EFF_THRESHOLDS):
    rows = {}
    for algo in ALGORITHMS:
        csvs = _seed_csvs(algo, seeds)
        if not csvs:
            continue
        per_seed = [_curve_metrics(c, thresholds) for c in csvs]
        agg = {"n_seeds": len(per_seed)}
        for key in per_seed[0]:
            vals = np.array([m[key] for m in per_seed], dtype=float)
            agg[key] = (float(np.mean(vals)), float(np.std(vals)))
        rows[algo] = agg
    return rows


def _fmt_steps(mean_std):
    mean, std = mean_std
    if not np.isfinite(mean):
        return "  never"
    return f"{mean/1000:.0f}k±{std/1000:.0f}k" if std else f"{mean/1000:.0f}k"


def main(seeds=SEEDS, thresholds=EFF_THRESHOLDS):
    res = compute(seeds, tuple(thresholds))
    if not res:
        print("No training CSVs found for those seeds.")
        return
    thr_cols = [f"steps_to_{t}" for t in thresholds]
    print(f"\n=== Learning speed (seeds={list(seeds)}, smooth={SMOOTH}) ===")
    header = f'{"algo":6}{"AULC":>12}' + "".join(f"{c:>14}" for c in thr_cols) + f'{"seeds":>7}'
    print(header)
    for algo in ALGORITHMS:
        if algo not in res:
            continue
        a = res[algo]
        aulc_m, aulc_s = a["AULC"]
        line = f"{algo:6}{aulc_m:7.3f}±{aulc_s:.3f}"
        for c in thr_cols:
            line += f"{_fmt_steps(a[c]):>14}"
        line += f"{a['n_seeds']:>7}"
        print(line)
    print("\nAULC = mean smoothed efficiency over training (higher = faster+higher);"
          " steps_to_X = env steps to first reach X.")


if __name__ == "__main__":
    main()
