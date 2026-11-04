from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = ROOT_DIR / "deliverables" / "figures"
REPORT_FIGURES_DIR = FIGURES_DIR / "report"
PPT_FIGURES_DIR = FIGURES_DIR / "ppt"
TABLES_DIR = ROOT_DIR / "deliverables" / "tables"
NOTES_DIR = ROOT_DIR / "deliverables" / "notes"

# Keep Matplotlib's writable cache inside the project instead of the user profile.
MPL_CACHE_DIR = ROOT_DIR / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

RANDOM_STATE = 42
TEST_SIZE = 0.30
VALIDATION_FRACTION_OF_TEMP = 0.50
TARGET_COLUMN = "MedHouseVal"
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
ALL_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
TARGET_UNIT_LABEL = "100,000 USD"
CV_FOLDS = 5
RIDGE_ALPHAS = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
]


def ensure_directories() -> None:
    """Create all output directories used by the experiment."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        REPORT_FIGURES_DIR,
        PPT_FIGURES_DIR,
        TABLES_DIR,
        NOTES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
