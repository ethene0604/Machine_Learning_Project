from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_predictions(
    output_dir: Path,
    model_key: str,
    split_name: str,
    row_ids: np.ndarray | pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "row_id": np.asarray(row_ids),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    predictions.to_csv(
        output_dir / f"{model_key}_{split_name}.csv", index=False, encoding="utf-8"
    )
    return predictions


def metrics_rows(
    model_key: str,
    model_label: str,
    feature_set: str,
    split_metrics: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, values in split_metrics.items():
        rows.append(
            {
                "model_key": model_key,
                "model": model_label,
                "feature_set": feature_set,
                "split": split_name,
                **values,
            }
        )
    return rows
