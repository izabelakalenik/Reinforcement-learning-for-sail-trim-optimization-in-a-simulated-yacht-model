import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main.config import (SEEDS, SUFFIX, ALGORITHMS, COLORS, SMOOTH,
                    MIN_REF_SPEED, METRICS_DIR, PLOTS_DIR_TRAIN)
PLOTS_DIR_TRAIN.mkdir(parents=True, exist_ok=True)

MODE = "all"  # "all" = per-seed + cumulative; or "perseed" / "cumulative" / "single"


def _read(path):
    df = pd.read_csv(path)
    if "tail_trim_efficiency" not in df.columns and "avg_trim_efficiency" in df.columns:
        df["tail_trim_efficiency"] = df["avg_trim_efficiency"]
    return df


def load_all():
    dfs = {}
    for alg in ALGORITHMS:
        candidates = sorted(METRICS_DIR.glob(f"results_{alg.lower()}*.csv"))
        candidates = [p for p in candidates if "test" not in p.stem]
        if not candidates:
            print(f"[skip] no results CSV found for {alg}")
            continue
        path = candidates[-1]  # highest timestep/seed 
        print(f"[{alg}] loading {path.name}")
        dfs[alg] = _read(path)
    return dfs


def load_seed(seed):
    dfs = {}
    for alg in ALGORITHMS:
        hits = sorted(METRICS_DIR.glob(f"results_{alg.lower()}_*_{seed}{SUFFIX}_*.csv"))
        if not hits:
            print(f"[skip] no CSV for {alg} seed {seed}")
            continue
        dfs[alg] = _read(hits[-1])
    return dfs


def load_per_alg_seeds(seeds):
    out = {}
    for alg in ALGORITHMS:
        per = {}
        for s in seeds:
            hits = sorted(METRICS_DIR.glob(f"results_{alg.lower()}_*_{s}{SUFFIX}_*.csv"))
            if hits:
                per[s] = _read(hits[-1])
        if per:
            out[alg] = per
    return out


def _filter(dfs):
    return {alg: df[df["ref_speed"] >= MIN_REF_SPEED].copy() for alg, df in dfs.items()}


def _bar_offsets(keys, width):
    keys = list(keys)
    return {k: (i - (len(keys) - 1) / 2.0) * width for i, k in enumerate(keys)}


def save(fig, name, suffix=""):
    out = PLOTS_DIR_TRAIN / f"{name}{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_trim_efficiency(dfs, suffix=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    for alg, df in dfs.items():
        raw = df["tail_trim_efficiency"].to_numpy(dtype=float)
        episodes = df["episode"].to_numpy(dtype=float)
        smooth = pd.Series(raw).rolling(SMOOTH, min_periods=1).mean().to_numpy(dtype=float)
        ax.plot(episodes, raw, color=COLORS[alg], alpha=0.25, linewidth=1)
        ax.plot(episodes, smooth, color=COLORS[alg], label=alg, linewidth=2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Prędkość referencyjna (1,0)")
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Sprawność trymu  (v / v_ref)")
    ax.set_title("Krzywe uczenia - sprawność trymu")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "01_trim_efficiency", suffix)


def plot_speed_error(dfs, suffix=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    for alg, df in dfs.items():
        smooth = df["tail_avg_speed_error"].rolling(SMOOTH, min_periods=1).mean()
        ax.plot(df["episode"], smooth, color=COLORS[alg], label=alg, linewidth=2)
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Błąd prędkości  |v - v_ref|  (m/s)")
    ax.set_title("Błąd śledzenia prędkości")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "02_speed_error", suffix)


def plot_avg_reward(dfs, suffix=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    for alg, df in dfs.items():
        smooth = df["avg_reward"].rolling(SMOOTH, min_periods=1).mean()
        ax.plot(df["episode"], smooth, color=COLORS[alg], label=alg, linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Średnia nagroda")
    ax.set_title("Średnia nagroda w epizodzie")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "03_avg_reward", suffix)


def plot_speed_vs_reference(dfs, suffix=""):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (alg, df) in zip(axes, dfs.items()):
        episodes = df["episode"].to_numpy(dtype=float)
        achieved = pd.Series(df["tail_avg_boat_speed"].to_numpy(dtype=float))
        reference = pd.Series(df["ref_speed"].to_numpy(dtype=float))
        achieved_smooth = achieved.rolling(SMOOTH, min_periods=1).mean().to_numpy(dtype=float)
        reference_smooth = reference.rolling(SMOOTH, min_periods=1).mean().to_numpy(dtype=float)
        ax.plot(episodes, achieved.to_numpy(), color=COLORS[alg], alpha=0.2, linewidth=1)
        ax.plot(episodes, achieved_smooth, color=COLORS[alg], linewidth=2, label="Osiągnięta prędkość")
        ax.plot(episodes, reference.to_numpy(), color="gray", alpha=0.2, linewidth=1)
        ax.plot(episodes, reference_smooth, color="gray", linewidth=2, linestyle="--", label="Prędkość referencyjna")
        ax.set_title(alg)
        ax.set_xlabel("Epizod")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Prędkość  (m/s)")
    fig.suptitle("Osiągnięta prędkość a prędkość referencyjna", fontsize=13)
    save(fig, "04_speed_vs_reference", suffix)


def plot_efficiency_by_twa(dfs, suffix=""):
    bins = np.arange(0, 190, 10)
    labels = [f"{b}" for b in bins[:-1]]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 2.0
    offsets = _bar_offsets(dfs.keys(), width)
    for alg, df in dfs.items():
        df = df[df["tail_trim_efficiency"] < 2.0]  # filter degenerate near-zero TWA entries
        df = df.copy()
        df["twa_bin"] = pd.cut(df["twa"], bins=bins, labels=labels, right=False)
        grouped = df.groupby("twa_bin", observed=True)["tail_trim_efficiency"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)
        x = np.arange(len(means)) * 10 + offsets[alg]
        ax.bar(x, means, width=width, color=COLORS[alg], label=alg, alpha=0.8, yerr=stds, capsize=2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Kąt do wiatru rzeczywistego  (stopnie)")
    ax.set_ylabel("Średnia sprawność trymu")
    ax.set_title("Sprawność trymu według kąta do wiatru")
    ax.set_xticks(np.arange(len(labels)) * 10)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "05_efficiency_by_twa", suffix)


def plot_efficiency_by_wind_speed(dfs, suffix=""):
    bins = [0, 2, 4, 6, 8, 10, 12, 15]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-15"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    offsets = _bar_offsets(dfs.keys(), width)
    for alg, df in dfs.items():
        df = df[df["tail_trim_efficiency"] < 2.0].copy()
        df["ws_bin"] = pd.cut(df["wind_speed"], bins=bins, labels=labels, right=False)
        grouped = df.groupby("ws_bin", observed=True)["tail_trim_efficiency"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)
        x = np.arange(len(means)) + offsets[alg]
        ax.bar(x, means, width=width, color=COLORS[alg], label=alg, alpha=0.8, yerr=stds, capsize=3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prędkość wiatru  (m/s)")
    ax.set_ylabel("Średnia sprawność trymu")
    ax.set_title("Sprawność trymu według prędkości wiatru")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "06_efficiency_by_wind_speed", suffix)


def plot_final_summary(dfs, suffix=""):
    metrics = {
        "tail_trim_efficiency": "Sprawność trymu",
        "tail_avg_speed_error": "Błąd prędkości (m/s)",
        "avg_reward": "Średnia nagroda",
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))
    for ax, (col, title) in zip(axes, metrics.items()):
        values, colors, names = [], [], []
        for alg, df in dfs.items():
            quarter = max(1, len(df) // 4)
            val = df[col].iloc[-quarter:].mean()
            if col == "tail_trim_efficiency":
                val = min(val, 2.0)  # cap display for degenerate cases
            values.append(val)
            colors.append(COLORS[alg])
            names.append(alg)
        bars = ax.bar(names, values, color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
        if col == "tail_trim_efficiency":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Podsumowanie z ostatniej ćwiartki treningu", fontsize=13)
    save(fig, "07_final_summary", suffix)


def _seed_band(ax, per_seed_dfs, alg, ycol, xcol="episode", show_seeds=True):
    xmax = min(df[xcol].max() for df in per_seed_dfs)
    grid = np.linspace(0, xmax, int(xmax) + 1)
    curves = []
    for df in per_seed_dfs:
        x = df[xcol].to_numpy(dtype=float)
        y = pd.Series(df[ycol].to_numpy(dtype=float)).rolling(SMOOTH, min_periods=1).mean().to_numpy()
        order = np.argsort(x)
        curves.append(np.interp(grid, x[order], y[order]))
    mat = np.vstack(curves)
    mean, std = mat.mean(axis=0), mat.std(axis=0)
    if show_seeds:
        for c in curves:
            ax.plot(grid, c, color=COLORS[alg], alpha=0.15, linewidth=1)
    ax.plot(grid, mean, color=COLORS[alg], linewidth=2, label=alg)
    ax.fill_between(grid, mean - std, mean + std, color=COLORS[alg], alpha=0.18)
    return grid, mean, std


def _nseeds(per_alg):
    return max((len(d) for d in per_alg.values()), default=0)


def cumulative_trim_efficiency(per_alg):
    fig, ax = plt.subplots(figsize=(10, 5))
    for alg, per_seed in per_alg.items():
        _seed_band(ax, list(per_seed.values()), alg, "tail_trim_efficiency")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Prędkość referencyjna (1,0)")
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Sprawność trymu  (v / v_ref)")
    ax.set_title(f"Krzywe uczenia - sprawność trymu (średnia ± odch. std z {_nseeds(per_alg)} ziaren)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "01_trim_efficiency", "_cumulative")


def cumulative_speed_error(per_alg):
    fig, ax = plt.subplots(figsize=(10, 5))
    for alg, per_seed in per_alg.items():
        _seed_band(ax, list(per_seed.values()), alg, "tail_avg_speed_error")
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Błąd prędkości  |v - v_ref|  (m/s)")
    ax.set_title(f"Błąd śledzenia prędkości (średnia ± odch. std z {_nseeds(per_alg)} ziaren)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "02_speed_error", "_cumulative")


def cumulative_avg_reward(per_alg):
    fig, ax = plt.subplots(figsize=(10, 5))
    for alg, per_seed in per_alg.items():
        _seed_band(ax, list(per_seed.values()), alg, "avg_reward")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Epizod")
    ax.set_ylabel("Średnia nagroda")
    ax.set_title(f"Średnia nagroda w epizodzie (średnia ± odch. std z {_nseeds(per_alg)} ziaren)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "03_avg_reward", "_cumulative")


def cumulative_speed_vs_reference(per_alg):
    n = len(per_alg)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for i, (ax, (alg, per_seed)) in enumerate(zip(axes, per_alg.items())):
        dfs = list(per_seed.values())
        _seed_band(ax, dfs, alg, "tail_avg_boat_speed", show_seeds=False)

        xmax = min(df["episode"].max() for df in dfs)
        grid = np.linspace(0, xmax, int(xmax) + 1)
        refs = []
        for df in dfs:
            x = df["episode"].to_numpy(dtype=float)
            y = pd.Series(df["ref_speed"].to_numpy(dtype=float)).rolling(SMOOTH, min_periods=1).mean().to_numpy()
            order = np.argsort(x)
            refs.append(np.interp(grid, x[order], y[order]))
        ax.plot(grid, np.vstack(refs).mean(axis=0), color="gray", linewidth=2,
                linestyle="--", label="Prędkość referencyjna")
        ax.set_title(alg)
        ax.set_xlabel("Epizod")
        if i % ncols == 0:
            ax.set_ylabel("Prędkość  (m/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Osiągnięta prędkość a prędkość referencyjna (średnia z {_nseeds(per_alg)} ziaren)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "04_speed_vs_reference", "_cumulative")


def cumulative_final_summary(per_alg):
    metrics = {
        "tail_trim_efficiency": "Sprawność trymu",
        "tail_avg_speed_error": "Błąd prędkości (m/s)",
        "avg_reward": "Średnia nagroda",
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))
    for ax, (col, title) in zip(axes, metrics.items()):
        names, means, errs, colors = [], [], [], []
        for alg, per_seed in per_alg.items():
            per_seed_vals = []
            for df in per_seed.values():
                quarter = max(1, len(df) // 4)
                v = df[col].iloc[-quarter:].mean()
                if col == "tail_trim_efficiency":
                    v = min(v, 2.0)
                per_seed_vals.append(v)
            names.append(alg)
            means.append(float(np.mean(per_seed_vals)))
            errs.append(float(np.std(per_seed_vals)))
            colors.append(COLORS[alg])
        bars = ax.bar(names, means, yerr=errs, capsize=4, color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
        if col == "tail_trim_efficiency":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Podsumowanie z ostatniej ćwiartki treningu (średnia ± odch. std z {_nseeds(per_alg)} ziaren)",
                 fontsize=13)
    save(fig, "07_final_summary", "_cumulative")


def run_per_seed(seeds):
    for seed in seeds:
        dfs = load_seed(seed)
        if not dfs:
            continue
        dfs = _filter(dfs)
        sfx = f"_seed{seed}"
        print(f"\n--- per-seed plots: seed {seed} ({list(dfs.keys())}) ---")
        plot_trim_efficiency(dfs, sfx)
        plot_speed_error(dfs, sfx)
        plot_avg_reward(dfs, sfx)
        plot_speed_vs_reference(dfs, sfx)
        plot_efficiency_by_twa(dfs, sfx)
        plot_efficiency_by_wind_speed(dfs, sfx)
        plot_final_summary(dfs, sfx)


def run_cumulative(seeds):
    per_alg_raw = load_per_alg_seeds(seeds)
    if not per_alg_raw:
        print("No per-seed CSVs found for cumulative plots.")
        return
    per_alg = {alg: {s: df[df["ref_speed"] >= MIN_REF_SPEED].copy() for s, df in d.items()}
               for alg, d in per_alg_raw.items()}
    print(f"\n--- cumulative plots (mean ± std over seeds {seeds}) ---")
    cumulative_trim_efficiency(per_alg)
    cumulative_speed_error(per_alg)
    cumulative_avg_reward(per_alg)
    cumulative_speed_vs_reference(per_alg)

    pooled = {alg: pd.concat(d.values(), ignore_index=True) for alg, d in per_alg.items()}
    plot_efficiency_by_twa(pooled, "_cumulative")
    plot_efficiency_by_wind_speed(pooled, "_cumulative")
    cumulative_final_summary(per_alg)


def main(seeds=SEEDS, mode=MODE):
    if mode == "single":
        dfs = load_all()
        if not dfs:
            print("No result CSVs found in results/metrics/. Run main.py first.")
            return
        run = _filter(dfs)
        for fn in (plot_trim_efficiency, plot_speed_error, plot_avg_reward,
                   plot_speed_vs_reference, plot_efficiency_by_twa,
                   plot_efficiency_by_wind_speed, plot_final_summary):
            fn(run)
    else:
        if mode in ("perseed", "all"):
            run_per_seed(seeds)
        if mode in ("cumulative", "all"):
            run_cumulative(seeds)

    print(f"\nAll plots saved to {PLOTS_DIR_TRAIN.resolve()}")


if __name__ == "__main__":
    main()
