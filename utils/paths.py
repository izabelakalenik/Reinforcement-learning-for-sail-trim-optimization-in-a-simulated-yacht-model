from main.config import METRICS_DIR


def results_csv_path(suffix):
    return METRICS_DIR / f"eval_{suffix}.csv"
