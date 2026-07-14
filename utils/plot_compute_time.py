import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from main.config import (SEEDS, ALGORITHMS, COLORS, PLOTS_DIR_TRAIN, BENCHMARK_CSV,
                         TIME_THRESHOLD)
from training.learning_speed import _seed_csvs, _curve_metrics
from utils.plot_deterministic import _scatter_seeds


def _sec_per_step():
    if not BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"Missing {BENCHMARK_CSV}. Run: python -m training.benchmark_time")
    df = pd.read_csv(BENCHMARK_CSV)
    return dict(zip(df["algo"], df["sec_per_step"].astype(float)))


def _steps_per_seed(seeds, threshold=TIME_THRESHOLD):
    out = {}
    for a in ALGORITHMS:
        vals = []
        for s in seeds:
            csvs = _seed_csvs(a, [s])
            if not csvs:
                continue
            steps = _curve_metrics(csvs[0], (threshold,))[f"steps_to_{threshold}"]
            if np.isfinite(steps):
                vals.append(steps)
        out[a] = np.asarray(vals, dtype=float)
    return out


def _bars(ax, names, values_per_algo, ylabel, title, scatter=True, fmt="{:.1f}", legend=False):
    means = [np.mean(values_per_algo[a]) if len(values_per_algo[a]) else np.nan for a in names]
    stds = [np.std(values_per_algo[a], ddof=1) if len(values_per_algo[a]) > 1 else 0.0
            for a in names]
    ax.bar(names, means, yerr=stds if scatter else None, capsize=5,
           color=[COLORS[a] for a in names], alpha=0.85)

    if scatter:
        for i, a in enumerate(names):
            _scatter_seeds(ax, i, values_per_algo[a], jitter=0.13,
                           label="Poszczególne ziarna" if i == 0 else None)

    tops = [max(m + s, float(np.max(values_per_algo[a])) if len(values_per_algo[a]) else m)
            for a, m, s in zip(names, means, stds)]
    ax.set_ylim(0, max(tops) * 1.28)
    for i, (m, t) in enumerate(zip(means, tops)):
        ax.text(i, t + max(tops) * 0.04, fmt.format(m), ha="center", va="bottom", fontsize=10)

    if legend and scatter:
        ax.legend(fontsize=8, loc="upper left")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")


def main(seeds=SEEDS):
    PLOTS_DIR_TRAIN.mkdir(parents=True, exist_ok=True)
    cost = _sec_per_step()
    steps = _steps_per_seed(seeds)
    names = [a for a in ALGORITHMS if a in cost and len(steps.get(a, []))]

    # one env step, in milliseconds (a single measured value per algorithm)
    ms = {a: np.array([cost[a] * 1000.0]) for a in names}

    # steps to threshold, in thousands
    ksteps = {a: steps[a] / 1000.0 for a in names}
    
    # time to threshold, in minutes
    minutes = {a: steps[a] * cost[a] / 60.0 for a in names}

    print(f"\nComputation cost (threshold {TIME_THRESHOLD}):")
    for a in names:
        print(f"  {a:5} {ms[a][0]:7.2f} ms/step   {np.mean(ksteps[a]):7.0f} k steps"
              f"   {np.mean(minutes[a]):7.1f} min")

    thr = str(TIME_THRESHOLD).replace(".", ",")
    
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax_cost = fig.add_subplot(gs[0, 0:2])
    ax_steps = fig.add_subplot(gs[0, 2:4])
    ax_time = fig.add_subplot(gs[1, 1:3])

    _bars(ax_cost, names, ms, "Czas jednego kroku  (ms)",
          "Koszt kroku uczenia", scatter=False, fmt="{:.2f}")
    _bars(ax_steps, names, ksteps, "Kroki  (tys.)",
          f"Kroki do progu {thr}", fmt="{:.0f}", legend=True)
    _bars(ax_time, names, minutes, "Czas  (min)",
          f"Czas obliczeń do progu {thr}", fmt="{:.1f}")

    fig.suptitle("Koszt obliczeniowy uczenia", fontsize=14)
    out = PLOTS_DIR_TRAIN / "08_compute_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
