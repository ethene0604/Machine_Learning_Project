from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_FRACTION_OF_TEMP,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the locally prepared California Housing dataset."""
    frame = pd.read_csv(path)
    missing_columns = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if len(frame) != 20_640:
        raise ValueError(f"Expected 20,640 rows, found {len(frame):,}.")

    frame = frame[ALL_COLUMNS].copy()
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in ALL_COLUMNS):
        raise TypeError("All expected California Housing columns must be numeric.")
    return frame


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[int], list[int], list[int]]:
    """Create a reproducible 70/15/15 split while preserving original row ids."""
    row_ids = np.arange(len(frame), dtype=int)
    train_ids, temp_ids = train_test_split(
        row_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    validation_ids, test_ids = train_test_split(
        temp_ids,
        test_size=VALIDATION_FRACTION_OF_TEMP,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    split_frames: dict[str, pd.DataFrame] = {}
    for name, indices in (
        ("train", train_ids),
        ("validation", validation_ids),
        ("test", test_ids),
    ):
        subset = frame.iloc[indices].copy()
        subset.insert(0, "row_id", indices)
        split_frames[name] = subset.reset_index(drop=True)
    return split_frames, train_ids.tolist(), validation_ids.tolist(), test_ids.tolist()


def save_splits(
    split_frames: dict[str, pd.DataFrame],
    train_ids: list[int],
    validation_ids: list[int],
    test_ids: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_frame in split_frames.items():
        split_frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")

    index_rows = [
        *(("train", row_id) for row_id in train_ids),
        *(("validation", row_id) for row_id in validation_ids),
        *(("test", row_id) for row_id in test_ids),
    ]
    pd.DataFrame(index_rows, columns=["split", "row_id"]).to_csv(
        output_dir / "split_indices.csv", index=False, encoding="utf-8"
    )

    config = {
        "random_state": RANDOM_STATE,
        "target_test_size": TEST_SIZE,
        "validation_fraction_of_temp": VALIDATION_FRACTION_OF_TEMP,
        "counts": {name: int(len(value)) for name, value in split_frames.items()},
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "train_ids_sha256": hashlib.sha256(
            json.dumps(train_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "validation_ids_sha256": hashlib.sha256(
            json.dumps(validation_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "test_ids_sha256": hashlib.sha256(
            json.dumps(test_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / "split_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return config


def build_data_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize data quality and distribution for the report."""
    rows: list[dict[str, Any]] = []
    for column in ALL_COLUMNS:
        series = frame[column]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q1": q1,
                "median": float(series.median()),
                "q3": q3,
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "iqr_lower_bound": lower,
                "iqr_upper_bound": upper,
                "iqr_outlier_count": int(((series < lower) | (series > upper)).sum()),
            }
        )
    return pd.DataFrame(rows)
