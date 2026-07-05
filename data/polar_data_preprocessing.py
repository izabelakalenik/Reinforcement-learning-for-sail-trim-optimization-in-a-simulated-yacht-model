from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from config import DEFAULT_POLAR_CSV, POLAR_RAW_POL

"""
Assumptions:
- TWS in knots
- TWA in degrees (0-180 relative to bow)
- BSP (boat speed) in knots
"""

_KNOTS_TO_MS = 0.514444  # conversion factor from knots to m/s


def parse_pol(pol_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = Path(pol_path)
    if not p.exists():
        raise FileNotFoundError(pol_path)

    with p.open('r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    header_tokens = lines[0].split()

    tws = np.array([float(x) for x in header_tokens[1:]], dtype=float)

    twa_list = []
    speeds = []
    for ln in lines[1:]:
        tokens = ln.split()

        # first token TWA, rest speeds
        twa = float(tokens[0])
        vals = [float(x) for x in tokens[1:1+len(tws)]]
        twa_list.append(twa)
        speeds.append(vals)

    twa_arr = np.array(twa_list, dtype=float)
    speeds_arr = np.array(speeds, dtype=float)

    return twa_arr, tws, speeds_arr


def save_pol_as_csv(pol_path: str, csv_path: str, to_m_s: bool = True) -> pd.DataFrame:
    twa, tws, speeds = parse_pol(pol_path)

    rows = []
    for i, twa_val in enumerate(twa):
        for j, tws_val in enumerate(tws):
            bsp = speeds[i, j]
            rows.append((float(twa_val), float(tws_val), float(bsp)))

    df = pd.DataFrame(rows, columns=['TWA_deg', 'TWS_knots', 'BSP_knots'])

    if to_m_s:
        df = df.assign(
            TWS_ms=df['TWS_knots'] * _KNOTS_TO_MS,
            BSP_ms=df['BSP_knots'] * _KNOTS_TO_MS,
        )
    df.to_csv(csv_path, index=False)
    return df


def build_polar_interpolator(twa: np.ndarray, tws: np.ndarray, speeds: np.ndarray, output_unit: str = 'm/s'):
    """Builds a 2D interpolator for (TWA, TWS) -> BSP.

    Parameters:
        twa: 1D array of TWA (degrees)
        tws: 1D array of TWS - units must match query units (m/s when called from build_polar_interpolator_from_csv)
        speeds: 2D array of BSP in knots with shape (len(twa), len(tws))
        output_unit: 'm/s' or 'knots'

    Returns:
        callable(twa_deg, tws) -> BSP in the requested output_unit
    """

    # ensure increasing order for grid axes
    # RegularGridInterpolator expects axis arrays in increasing order
    sort_twa_idx = np.argsort(twa)
    sort_tws_idx = np.argsort(tws)
    twa_sorted = twa[sort_twa_idx]
    tws_sorted = tws[sort_tws_idx]
    speeds_sorted = speeds[np.ix_(sort_twa_idx, sort_tws_idx)]

    interpolator = RegularGridInterpolator((twa_sorted, tws_sorted), speeds_sorted, bounds_error=False, fill_value=None)

    def query(point_twa: float, point_tws: float) -> float:
        ptwa = float(point_twa)
        ptws = float(point_tws)
        # for TWA, polars are given 0-180 relative to bow; if given >180, mirror it
        # but here wrap angles into [0,180] by using abs and mod 360 shortcut
        # reflect angles >180: twa = 360 - twa for twa>180, then if >180 still reduce
        if ptwa < 0:
            ptwa = abs(ptwa)
        if ptwa > 360:
            ptwa = ptwa % 360
        if ptwa > 180:
            ptwa = 360 - ptwa

        ptwa = float(np.clip(ptwa, twa_sorted[0], twa_sorted[-1]))
        ptws = float(np.clip(ptws, tws_sorted[0], tws_sorted[-1]))

        val_arr = interpolator([ptwa, ptws])
        val_knots = float(val_arr[0])
        if output_unit == 'knots':
            return val_knots
        elif output_unit == 'm/s':
            return val_knots * _KNOTS_TO_MS
        else:
            raise ValueError("output_unit must be 'm/s' or 'knots'")

    return query


def build_polar_interpolator_from_csv(csv_path: str, output_unit: str = 'm/s'):
    df = pd.read_csv(csv_path)

    # always use knots columns so build_polar_interpolator performs exactly one conversion
    twa_col = 'TWA_deg'
    tws_col = 'TWS_knots'
    speed_col = 'BSP_knots'

    if twa_col not in df.columns or tws_col not in df.columns or speed_col not in df.columns:
        missing = [col for col in (twa_col, tws_col, speed_col) if col not in df.columns]
        raise ValueError(f"Polar CSV is missing required columns: {', '.join(missing)}")

    polar_grid = (
        df.pivot_table(index=twa_col, columns=tws_col, values=speed_col, aggfunc='mean')
        .sort_index()
        .sort_index(axis=1)
    )

    twa = polar_grid.index.to_numpy(dtype=float)
    tws = polar_grid.columns.to_numpy(dtype=float) * _KNOTS_TO_MS  
    speeds = polar_grid.to_numpy(dtype=float)  
    return build_polar_interpolator(twa, tws, speeds, output_unit=output_unit)


if __name__ == '__main__':
    pol = Path(POLAR_RAW_POL)
    csv_out = Path(DEFAULT_POLAR_CSV)

    print(f"Parsing {pol} and saving CSV to {csv_out}")
    df = save_pol_as_csv(str(pol), str(csv_out), to_m_s=True)
    print(f"Saved {len(df)} rows. Sample:\n", df.head())
