import math
from pathlib import Path
_ROOT = Path(__file__).resolve().parent

# --- simulation ---
FIXED_HEADING = math.pi / 2

# simulator run mode - change here to switch how the environment runs:
#   None        -> headless, no rendering (default; use for training - fastest)
#   "human"     -> live pygame window (watch the agent trim in real time)
#   "rgb_array" -> return frames as RGB arrays (e.g. to record a video)
RENDER_MODE = None

# --- training / experiment setup ---
TIME_STEPS = 500_000                          # steps per training run
MAX_EPISODE_STEPS = 1500                      # steps per episode
SEEDS = (42, 43, 44, 45, 46)                  
SUFFIX = "v2"
ALGORITHMS = ("SAC", "DDPG", "TD3", "PPO")       

# --- evaluation ---
N_EPISODES = 120                              # deterministic test episodes per algorithm
EVAL_SEED = 4242                              # fixed wind draws during evaluation
MIN_REF_SPEED = 0.5                           # m/s; below this a condition is a no-go zone

BANDS_ORDER = ["nogo", "close", "beam", "broad", "run"]

# --- plotting ---
SMOOTH = 25                                   # rolling-average window for learning curves
COLORS = {"PPO": "#2196F3", "SAC": "#4CAF50", "DDPG": "#FF5722", "TD3": "#9C27B0"}

TWA_BINS = [0, 45, 80, 100, 170, 181]
BAND_LABELS = ["Kąt martwy\n(0-45°)", "Bajdewind\n(45-80°)", "Półwiatr\n(80-100°)",
               "Baksztag\n(100-170°)", "Fordewind\n(170-180°)"]

# --- data paths ---
_DATASETS = _ROOT / "data" / "datasets"

# --- raw source files ---
WIND_RAW_NC = _DATASETS / "era5_dataset.nc"
POLAR_RAW_POL = _DATASETS / "First40_7.pol"

# --- default polar and wind data files ---
DEFAULT_POLAR_CSV = _DATASETS / "First40_7.csv"
DEFAULT_WIND_CSV = _DATASETS / "wind_data_gdansk.csv"

# --- temporal train / val / test wind splits ---
WIND_TRAIN = _DATASETS / "wind_data_gdansk_train.csv"
WIND_VAL = _DATASETS / "wind_data_gdansk_val.csv"
WIND_TEST = _DATASETS / "wind_data_gdansk_test.csv"

# --- output directories ---
MODELS_DIR = _ROOT / "results" / "models"
METRICS_DIR = _ROOT / "results" / "metrics"
PLOTS_DIR_EVAL = _ROOT / f"results/plots/{TIME_STEPS}/evaluation"
PLOTS_DIR_TRAIN = _ROOT / f"results/plots/{TIME_STEPS}/training"
