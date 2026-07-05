import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config import SEEDS, SUFFIX, ALGORITHMS, COLORS, PLOTS_DIR_EVAL, BANDS_ORDER, BAND_LABELS
from utils.paths import results_csv_path


def _load(seeds):
    frames = []
    for s in seeds:
        path = results_csv_path(f"{s}{SUFFIX}")
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run training/evaluate.py first.")
        df = pd.read_csv(path)
        df["seed"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _mean_std(df, col):
    g = df.groupby("algo")[col]
    return g.mean(), g.std().fillna(0.0)


def _seed_points(df, algo, col):
    v = df.loc[df["algo"] == algo, col].to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _scatter_seeds(ax, x_center, values, jitter=0.0, label=None):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0:
        return
    offs = np.linspace(-jitter, jitter, n) if (n > 1 and jitter > 0) else np.zeros(n)
    ax.scatter(x_center + offs, vals, s=24, color="black", edgecolor="white",
               linewidth=0.6, zorder=5, label=label)


def write_summary(df, names, nseed):
    summary = []
    eff_m, eff_s = _mean_std(df, "overall")
    ceil_m, ceil_s = _mean_std(df, "overall_effceil") if "overall_effceil" in df else (None, None)
    for a in names:
        row = {"algo": a, "n_seeds": nseed,
               "eff_mean": round(eff_m[a], 4), "eff_std": round(eff_s[a], 4)}
        if ceil_m is not None:
            row["effceil_mean"] = round(ceil_m[a], 4)
            row["effceil_std"] = round(ceil_s[a], 4)
        summary.append(row)
    summary_path = _ROOT / "results/metrics/eval_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")
    for r in summary:
        line = f"  {r['algo']:5} eff={r['eff_mean']:.3f}±{r['eff_std']:.3f}"
        if "effceil_mean" in r:
            line += f"  eff/ceil={r['effceil_mean']:.3f}±{r['effceil_std']:.3f}"
        print(line)


def plot_overall_efficiency(df, names, suffix_note):
    eff_m, eff_s = _mean_std(df, "overall")
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [eff_m[a] for a in names]
    errs = [eff_s[a] for a in names]
    bars = ax.bar(names, vals, yerr=errs, capsize=5, color=[COLORS[a] for a in names], alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=8, fontsize=11)
    for i, a in enumerate(names):  # overlay individual seed values
        _scatter_seeds(ax, i, _seed_points(df, a, "overall"), jitter=0.13,
                       label="Poszczególne ziarna" if i == 0 else None)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Prędkość referencyjna (1,0)")
    ax.axhline(0.97, color="gray", linestyle=":", linewidth=1, label="Osiągalny pułap (~0,97)")
    ax.set_ylabel("Sprawność trymu  (v / v_ref)")
    ax.set_title(f"Końcowa skuteczność - zbiór testowy {suffix_note}")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(PLOTS_DIR_EVAL / "08_deterministic_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR_EVAL / '08_deterministic_efficiency.png'}")


def plot_by_twa(df, names, suffix_note):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(BANDS_ORDER))
    width = 0.2
    offsets = {a: (i - (len(names) - 1) / 2.0) * width for i, a in enumerate(names)}
    band_m = {b: _mean_std(df, b) for b in BANDS_ORDER}
    seed_label_used = False
    for a in names:
        means = [band_m[b][0][a] for b in BANDS_ORDER]
        stds = [band_m[b][1][a] for b in BANDS_ORDER]
        ax.bar(x + offsets[a], means, width=width, yerr=stds, capsize=3,
               color=COLORS[a], label=a, alpha=0.85)
        for bi, b in enumerate(BANDS_ORDER):  # overlay individual seed values per band
            pts = _seed_points(df, a, b)
            lbl = None
            if not seed_label_used and len(pts):
                lbl, seed_label_used = "Poszczególne ziarna", True
            _scatter_seeds(ax, x[bi] + offsets[a], pts, jitter=width * 0.28, label=lbl)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_ylabel("Sprawność trymu  (v / v_ref)")
    ax.set_title(f"Sprawność według kąta do wiatru - zbiór testowy {suffix_note}")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(PLOTS_DIR_EVAL / "09_deterministic_by_twa.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR_EVAL / '09_deterministic_by_twa.png'}")


def plot_by_twa_effceil(df, names, suffix_note):
    eff_cols = [f"{b}_effceil" for b in BANDS_ORDER]
    if not all(c in df.columns for c in eff_cols):
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(BANDS_ORDER))
    width = 0.2
    offsets = {a: (i - (len(names) - 1) / 2.0) * width for i, a in enumerate(names)}
    band_m = {b: _mean_std(df, f"{b}_effceil") for b in BANDS_ORDER}
    seed_label_used = False
    allpts = []
    for a in names:
        means = [band_m[b][0][a] for b in BANDS_ORDER]
        stds = [band_m[b][1][a] for b in BANDS_ORDER]
        ax.bar(x + offsets[a], means, width=width, yerr=stds, capsize=3,
               color=COLORS[a], label=a, alpha=0.85)
        for bi, b in enumerate(BANDS_ORDER):  # overlay individual seed values per band
            pts = _seed_points(df, a, f"{b}_effceil")
            allpts.extend(pts.tolist())
            lbl = None
            if not seed_label_used and len(pts):
                lbl, seed_label_used = "Poszczególne ziarna", True
            _scatter_seeds(ax, x[bi] + offsets[a], pts, jitter=width * 0.28, label=lbl)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_ylabel("Sprawność względem pułapu  (v / v_best)")
    ax.set_title(f"Sprawność względem pułapu według kąta do wiatru - zbiór testowy {suffix_note}")
    ax.set_ylim(0, max(1.05, (max(allpts) if allpts else 1.0) * 1.05))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(PLOTS_DIR_EVAL / "13_deterministic_effceil_by_twa.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR_EVAL / '13_deterministic_effceil_by_twa.png'}")


def plot_eff_ceil(df, names, suffix_note):
    if "overall_effceil" not in df.columns:
        return
    ceil_m, ceil_s = _mean_std(df, "overall_effceil")
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [ceil_m[a] for a in names]
    errs = [ceil_s[a] for a in names]
    bars = ax.bar(names, vals, yerr=errs, capsize=5, color=[COLORS[a] for a in names], alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=8, fontsize=11)
    for i, a in enumerate(names):  # overlay individual seed values
        _scatter_seeds(ax, i, _seed_points(df, a, "overall_effceil"), jitter=0.13,
                       label="Poszczególne ziarna" if i == 0 else None)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Osiągalny pułap (1,0)")
    ax.set_ylabel("Sprawność względem pułapu  (v / v_best)")
    ax.set_title(f"Sprawność względem pułapu symulatora - zbiór testowy {suffix_note}")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(PLOTS_DIR_EVAL / "10_deterministic_eff_ceil.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR_EVAL / '10_deterministic_eff_ceil.png'}")


def main(seeds=SEEDS):
    PLOTS_DIR_EVAL.mkdir(parents=True, exist_ok=True)
    df = _load(seeds)
    names = [a for a in ALGORITHMS if a in set(df["algo"])]
    nseed = len(seeds)
    print(f"Aggregating {nseed} seed(s): {list(seeds)}")
    suffix_note = f"(średnia ± odch. std z {nseed} ziaren)" if nseed > 1 else "(jedno ziarno)"

    write_summary(df, names, nseed)
    plot_overall_efficiency(df, names, suffix_note)
    plot_by_twa(df, names, suffix_note)
    plot_eff_ceil(df, names, suffix_note)
    plot_by_twa_effceil(df, names, suffix_note)


if __name__ == "__main__":
    main()
