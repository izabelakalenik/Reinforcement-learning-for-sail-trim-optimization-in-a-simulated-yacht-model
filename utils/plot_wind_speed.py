import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from training.evaluate import make_eval_venv, ALGOS, build_ceiling_efficiency_lookup
from utils.plot_deterministic import _scatter_seeds
from main.config import (TIME_STEPS, SEEDS, SUFFIX, N_EPISODES, EVAL_SEED, ALGORITHMS,
                    COLORS, WIND_TEST, METRICS_DIR, PLOTS_DIR_EVAL, TWA_BINS, BAND_LABELS,
                    MODELS_DIR)

BINS = [0, 2, 4, 6, 8, 10, 12]
BIN_LABELS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
_CACHE = METRICS_DIR / "eval_by_wind.csv"
MIN_CELL = 5


def _per_episode(algo, suffix, ceil_lookup, n_episodes=N_EPISODES):
    mp = MODELS_DIR / f"sail_trim_{algo.lower()}_model_{TIME_STEPS}_{suffix}.zip"
    if not mp.exists():
        return None
    vps = sorted(MODELS_DIR.glob(f"sail_trim_{algo.lower()}_model_{TIME_STEPS}_{suffix}_vecnormalize_*.pkl"))
    vp = vps[-1] if vps else (MODELS_DIR / "missing.pkl")
    model = ALGOS[algo].load(str(mp), device="cpu")
    venv = make_eval_venv(algo, str(vp), EVAL_SEED, wind_csv_path=str(WIND_TEST))
    obs = venv.reset()
    ws, twas, effs, effceils, done = [], [], [], [], 0
    while done < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        if dones[0]:
            info = infos[0]
            if info.get("episode_reference_speed", 0) < 0.5:
                continue
            tws = float(info.get("wind_speed", 0.0))
            twa = float(info.get("wind_relative_heading", 0.0))
            eff = float(info.get("episode_tail_trim_efficiency", 0.0))
            ce = ceil_lookup(twa, tws)                      
            effceil = eff / ce if ce and ce > 1e-6 else float("nan")  
            ws.append(tws)
            twas.append(twa)
            effs.append(eff)
            effceils.append(effceil)
            done += 1
    venv.close()
    return ws, twas, effs, effceils


def collect(seeds=SEEDS, suffix=SUFFIX):
    if _CACHE.exists():
        cached = pd.read_csv(_CACHE)
        if {"seed", "effceil"}.issubset(cached.columns):
            print(f"Loading cached per-episode data from {_CACHE}")
            return cached
        print(f"Cache {_CACHE} missing 'seed'/'effceil' - recomputing.")
    print(f"Collecting deterministic test episodes over seeds {list(seeds)} ...")
    ceil_lookup = build_ceiling_efficiency_lookup()
    frames = []
    for algo in ALGORITHMS:
        for s in seeds:
            res = _per_episode(algo, f"{s}{suffix}", ceil_lookup)
            if res:
                frames.append(pd.DataFrame({"algo": algo, "seed": s, "wind_speed": res[0],
                                            "twa": res[1], "eff": res[2], "effceil": res[3]}))
    df = pd.concat(frames, ignore_index=True)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE, index=False)
    print(f"Saved per-episode data -> {_CACHE}")
    return df


def _binned_means(df, names, col="eff"):
    d = df[(df[col] < 2.0) & (df["algo"].isin(names))].copy()
    d["b"] = pd.cut(d["wind_speed"], bins=BINS, labels=BIN_LABELS, right=False)
    table = d.groupby(["b", "algo"], observed=True)[col].mean().unstack("algo")
    return table.reindex(BIN_LABELS)[names]


def _binned_per_seed(df, names, col="eff"):
    d = df[(df[col] < 2.0) & (df["algo"].isin(names))].copy()
    d["b"] = pd.cut(d["wind_speed"], bins=BINS, labels=BIN_LABELS, right=False)
    g = d.groupby(["algo", "b", "seed"], observed=True)[col].mean().reset_index()
    out = {}
    for a in names:
        sub = g[g["algo"] == a]
        out[a] = sub.pivot(index="b", columns="seed", values=col).reindex(BIN_LABELS)
    return out


def _plot_grouped(df, names, col, title, ylabel, out_name):
    per_seed = _binned_per_seed(df, names, col)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(BIN_LABELS))
    width = 0.2
    seed_label_used = False
    for i, a in enumerate(names):
        offs = (i - (len(names) - 1) / 2.0) * width
        means = per_seed[a].mean(axis=1)                     # across-seed mean per bin
        stds = per_seed[a].std(axis=1, ddof=1).fillna(0.0)   # across-seed std per bin
        ax.bar(x + offs, [means.loc[b] for b in BIN_LABELS], width=width,
               yerr=[stds.loc[b] for b in BIN_LABELS], capsize=3,
               color=COLORS[a], label=a, alpha=0.85)
        for bi, b in enumerate(BIN_LABELS):  # overlay individual seed values per bin
            pts = per_seed[a].loc[b].to_numpy(dtype=float)
            pts = pts[np.isfinite(pts)]
            lbl = None
            if not seed_label_used and len(pts):
                lbl, seed_label_used = "Poszczególne ziarna", True
            _scatter_seeds(ax, x[bi] + offs, pts, jitter=width * 0.28, label=lbl)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Prędkość wiatru  (m/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    allpts = np.concatenate([per_seed[a].to_numpy(dtype=float).ravel() for a in names])
    ax.set_ylim(0, max(1.15, float(np.nanmax(allpts)) * 1.05))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    out = PLOTS_DIR_EVAL / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _grid(df, algo, col="eff"):
    d = df[df["algo"] == algo].copy()
    d["tb"] = pd.cut(d["twa"], bins=TWA_BINS, labels=BAND_LABELS, right=False)
    d["wb"] = pd.cut(d["wind_speed"], bins=BINS, labels=BIN_LABELS, right=False)
    grp = d.groupby(["tb", "wb"], observed=False)[col]
    mean = grp.mean().unstack("wb").reindex(index=BAND_LABELS, columns=BIN_LABELS)
    cnt = grp.count().unstack("wb").reindex(index=BAND_LABELS, columns=BIN_LABELS)
    return mean.mask(cnt < MIN_CELL)


def plot_heatmap(df, names):
    ncols = 2
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5.0 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("lightgrey")
    im = None
    for ax, a in zip(axes, names):
        g = _grid(df, a, "eff")
        im = ax.imshow(np.ma.masked_invalid(g.values), aspect="auto", cmap=cmap,
                       vmin=0.2, vmax=1.1, origin="upper")
        ax.set_title(a)
        ax.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
        ax.set_yticks(range(len(BAND_LABELS)), BAND_LABELS, fontsize=8)
        ax.set_xlabel("Prędkość wiatru  (m/s)")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")
    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=list(axes), shrink=0.6, label="Sprawność trymu  (v / v_ref)")
    fig.suptitle("Sprawność trymu w zależności od kursu i prędkości wiatru "
                 "(zbiór testowy, średnia z pięciu ziaren)", fontsize=13)
    out = PLOTS_DIR_EVAL / "14_deterministic_eff_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main(seeds=SEEDS):
    PLOTS_DIR_EVAL.mkdir(parents=True, exist_ok=True)
    df = collect(seeds)
    names = [a for a in ALGORITHMS if a in set(df["algo"])]

    print("\nTrim efficiency (v/v_ref) by wind speed:")
    print(_binned_means(df, names, "eff").round(3).to_string())
    _plot_grouped(df, names, "eff",
                  "Sprawność według prędkości wiatru - zbiór testowy "
                  "(średnia ± odch. std z pięciu ziaren)",
                  "Sprawność trymu  (v / v_ref)",
                  "11_deterministic_by_wind_speed.png")

    print("\nEfficiency vs ceiling (v/v_best) by wind speed:")
    print(_binned_means(df, names, "effceil").round(3).to_string())
    _plot_grouped(df, names, "effceil",
                  "Sprawność względem pułapu wg prędkości wiatru - zbiór testowy "
                  "(średnia ± odch. std z pięciu ziaren)",
                  "Sprawność względem pułapu  (v / v_best)",
                  "12_deterministic_effceil_by_wind_speed.png")

    plot_heatmap(df, names)


if __name__ == "__main__":
    main()
