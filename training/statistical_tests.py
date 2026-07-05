import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from main.config import SEEDS, SUFFIX, ALGORITHMS, METRICS_DIR
from training.learning_speed import _seed_csvs, _curve_metrics

ALPHA = 0.05


def _eta_per_seed(seeds):
    data = {a: [] for a in ALGORITHMS}
    for s in seeds:
        path = METRICS_DIR / f"eval_{s}{SUFFIX}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if row["algo"] in data:
                data[row["algo"]].append(float(row["overall"]))
    return data


def _aulc_per_seed(seeds):
    data = {}
    for a in ALGORITHMS:
        vals = []
        for s in seeds:
            csvs = _seed_csvs(a, [s])
            if csvs:
                vals.append(_curve_metrics(csvs[0], ())["AULC"])
        data[a] = vals
    return data


def _welch_ci(a, b, alpha=ALPHA):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    diff = a.mean() - b.mean()
    if se == 0:
        return diff, diff
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = stats.t.ppf(1 - alpha / 2, df)
    return diff - t * se, diff + t * se


def _pairwise(data, label):
    n = len(next(iter(data.values())))
    print(f"\n=== {label}  (n = {n} seeds) ===")
    print(f"{'pair':13}{'meanA':>8}{'meanB':>8}{'diff':>10}{'Welch p':>10}{'MW p':>9}"
          f"   {'95% CI of diff':<20}")
    for i, a in enumerate(ALGORITHMS):
        for b in ALGORITHMS[i + 1:]:
            xa, xb = np.asarray(data[a], float), np.asarray(data[b], float)
            if len(xa) < 2 or len(xb) < 2:
                continue
            p_welch = float(stats.ttest_ind(xa, xb, equal_var=False).pvalue)
            p_mw = float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
            lo, hi = _welch_ci(xa, xb)
            diff = xa.mean() - xb.mean()
            sig = "*" if (p_welch < ALPHA and p_mw < ALPHA) else ""
            print(f"{a + ' vs ' + b:13}{xa.mean():8.3f}{xb.mean():8.3f}{diff:+10.3f}"
                  f"{p_welch:10.4f}{p_mw:9.4f}   [{lo:+.3f}, {hi:+.3f}] {sig}")
    print(f"  * = significant in both tests (p < {ALPHA});  MW = Mann-Whitney U")


def main(seeds=SEEDS):
    _pairwise(_eta_per_seed(seeds), "Final efficiency  η = v/v_ref")
    _pairwise(_aulc_per_seed(seeds), "Learning speed  (AULC)")


if __name__ == "__main__":
    main()
