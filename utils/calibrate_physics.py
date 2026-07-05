import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from data.polar_data_preprocessing import build_polar_interpolator_from_csv
from simulation.gym_sailing.utils.angles import norm
from main.config import FIXED_HEADING, DEFAULT_POLAR_CSV, DEFAULT_WIND_CSV

H = FIXED_HEADING 
CL_MAX, CD_MAX = 1.5, 2.5  
MASS = 6970.0              
DT = 0.1                  
N_STEPS = 1500             
V0 = 0.5                   
polar = build_polar_interpolator_from_csv(str(DEFAULT_POLAR_CSV), output_unit="m/s")


def terminal_speeds(wx, wy, sails, sailcoeff, dragcoeff, wave):
    v = np.full((wx.shape[0], sails.shape[1]), V0, dtype=float)  # (C, S)
    k = DT / MASS
    for _ in range(N_STEPS):
        app_y = wy - v                       # (C, S)
        aws = np.hypot(wx, app_y)
        aws = np.where(aws < 0.01, 0.01, aws)
        wind_dir = np.arctan2(app_y, wx)
        aoa = norm(H + sails - wind_dir)
        aoa = np.where(aoa > H, aoa - np.pi, aoa)
        aoa = np.where(aoa < -H, aoa + np.pi, aoa)
        lift = CL_MAX * np.sin(2 * aoa)
        drag_c = CD_MAX * np.sin(aoa) ** 2
        aero_y = sailcoeff * aws * (lift * wx + drag_c * app_y)   # heading-projected aero
        forward = np.maximum(0.0, aero_y)
        hull = dragcoeff * v ** 2 + wave * v ** 4
        v = v + k * (forward - hull)
        v = np.maximum(0.0, v)
    return v


def ceiling(wind_vec, sailcoeff, dragcoeff, wave, sails):
    wx = np.array([[float(wind_vec[0])]])
    wy = np.array([[float(wind_vec[1])]])
    s = np.asarray(sails)[None, :]
    return float(terminal_speeds(wx, wy, s, sailcoeff, dragcoeff, wave).max())


def wind_vec_from(ws, wd):
    r = np.deg2rad(wd)
    return np.array([-ws * np.sin(r), -ws * np.cos(r)])


def build_calibration_set(n=160, seed=0):
    df = pd.read_csv(DEFAULT_WIND_CSV).sample(n, random_state=seed)
    items = []
    for _, r in df.iterrows():
        wv = wind_vec_from(r.wind_speed, r.wind_direction)
        ws = np.hypot(wv[0], wv[1])
        wfrom = np.arctan2(-wv[1], -wv[0])
        twad = np.degrees(abs(norm(wfrom - H)))
        vref = float(polar(min(twad, 180), ws))
        if vref >= 0.5:
            items.append((wv, vref, twad, ws))
    return items


CALSET = build_calibration_set()
SAILS = np.deg2rad(np.linspace(-90, 90, 37))[None, :]           # (1, S)
WX = np.array([[c[0][0]] for c in CALSET])                      # (C, 1)
WY = np.array([[c[0][1]] for c in CALSET])                      # (C, 1)
VREF = np.array([c[1] for c in CALSET])                         # (C,)
TWAD = np.array([c[2] for c in CALSET])                         # (C,)


def all_effs(sc, dc, wave):
    term = terminal_speeds(WX, WY, SAILS, sc, dc, wave)         # (C, S)
    return term.max(axis=1) / VREF                              # (C,)


def objective(params):
    sc, dc, wave = params
    if sc <= 0 or dc <= 0 or wave < 0:
        return 1e6
    e = all_effs(sc, dc, wave) - 1.0
    # target ceiling == 1.0; penalise overshoot a bit harder than undershoot
    return float(np.mean(np.where(e < 0, e ** 2, 1.3 * e ** 2)))


def report(label, sc, dc, wv_coeff):
    bands = [(0, 45, "nogo"), (45, 80, "close"), (80, 100, "beam"), (100, 170, "broad"), (170, 181, "run")]
    effs = all_effs(sc, dc, wv_coeff)
    bv = {}
    for c, twad in zip(effs, TWAD):
        for lo, hi, nm in bands:
            if lo <= twad < hi:
                bv.setdefault(nm, []).append(c)
    print(f"\n[{label}]  SAILCOEFF={sc:.2f}  DRAGCOEFF={dc:.2f}  WAVE={wv_coeff:.4f}")
    print(f"   mean achievable ceiling = {effs.mean():.3f}  (median {np.median(effs):.3f}, "
          f"p25 {np.percentile(effs,25):.3f}, p75 {np.percentile(effs,75):.3f})")
    print(f"   frac in [0.9,1.1]: {((effs>=0.9)&(effs<=1.1)).mean():.2f}   frac<0.7: {(effs<0.7).mean():.2f}   frac>1.2: {(effs>1.2).mean():.2f}")
    print("   per-band:", {k: round(float(np.mean(v)), 3) for k, v in bv.items()})


if __name__ == "__main__":
    report("PRE-RECALIBRATION (v²-only drag)", 31.4, 65.0, 0.0)

    res = minimize(objective, x0=np.array([60.0, 60.0, 0.5]), method="Nelder-Mead",
                   options={"xatol": 0.05, "fatol": 1e-4, "maxiter": 400})
    sc, dc, wv = res.x
    report("RECALIBRATED (v²+v⁴ drag)", sc, dc, wv)
    print(f"\nfit objective={res.fun:.4f}  success={res.success}")
    print(f"COEFFS: SAILCOEFF={sc:.3f}  DRAGCOEFF={dc:.3f}  WAVE={wv:.4f}")
