import sys
from pathlib import Path
import numpy as np
import pandas as pd
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from main.config import DEFAULT_WIND_CSV

CYCLE = 7
TRAIN_WEEKS = {0, 1, 2, 3, 4}
VAL_WEEKS = {5}
TEST_WEEKS = {6}


def assign_split(df: pd.DataFrame) -> pd.Series:
    if "valid_time" in df.columns:
        t = pd.to_datetime(df["valid_time"])
        week_index = (t.dt.dayofyear - 1) // 7
    else:
        # fallback: assume hourly rows, 24*7 rows per week
        week_index = pd.Series(np.arange(len(df)) // (24 * 7), index=df.index)

    bucket = week_index % CYCLE

    def label(b):
        if b in VAL_WEEKS:
            return "val"
        if b in TEST_WEEKS:
            return "test"
        return "train"

    return bucket.map(label)


def report(name: str, df: pd.DataFrame, total: int) -> None:
    pct = 100.0 * len(df) / total if total else 0.0
    line = f"  {name:5}: {len(df):5d} rows ({pct:4.1f}%)"
    if "valid_time" in df.columns and len(df):
        t = pd.to_datetime(df["valid_time"])
        line += f"  | {t.min().date()} -> {t.max().date()}  | months covered: {sorted(t.dt.month.unique())}"
    if "wind_speed" in df.columns and len(df):
        line += f"  | wind mean={df['wind_speed'].mean():.2f} std={df['wind_speed'].std():.2f} m/s"
    print(line)


def main(input_path: Path):
    df = pd.read_csv(input_path)
    if "valid_time" in df.columns:
        df = df.sort_values("valid_time").reset_index(drop=True)

    labels = assign_split(df)
    total = len(df)
    print(f"Splitting {input_path.name}  ({total} rows)  by whole-week temporal blocks "
          f"({len(TRAIN_WEEKS)} train : {len(VAL_WEEKS)} val : {len(TEST_WEEKS)} test per {CYCLE}-week cycle)\n")

    stem = input_path.stem  # e.g. wind_data_gdansk
    out_dir = input_path.parent
    for name in ("train", "val", "test"):
        part = df[labels == name].reset_index(drop=True)
        out_path = out_dir / f"{stem}_{name}.csv"
        part.to_csv(out_path, index=False)
        report(name, part, total)
        print(f"         -> {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    INPUT = DEFAULT_WIND_CSV
    main(INPUT)
