"""Random Forest experiments on A's fixed California Housing splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN
from .features import AddRatioFeatures, feature_names_for_set


SEARCH_SPACE = {
    "model__n_estimators": [80, 160],
    "model__max_depth": [12, 20, None],
    "model__min_samples_leaf": [1, 2, 4],
    "model__min_samples_split": [2, 5],
    "model__max_features": [0.7, 1.0],
}


def build_rf_pipeline(feature_set: str, *, n_estimators: int = 100) -> Pipeline:
    """Use A's deterministic ratios and fold-local imputation without scaling trees."""
    if feature_set not in ("raw", "ratio"):
        raise ValueError(f"Unknown feature set: {feature_set}")
    steps = []
    if feature_set == "ratio":
        steps.append(("ratios", AddRatioFeatures()))
    steps.extend(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def make_search(feature_set: str, *, n_iter: int = 6, n_jobs: int = 2) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=build_rf_pipeline(feature_set),
        param_distributions=SEARCH_SPACE,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        refit=True,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        return_train_score=True,
        error_score="raise",
    )


def check_split_files(directory: Path) -> dict[str, pd.DataFrame]:
    """Refuse changed or overlapping fixed splits before any modeling."""
    config = json.loads((directory / "split_config.json").read_text(encoding="utf-8"))
    frames = {}
    all_ids = []
    for name in ("train", "validation", "test"):
        frame = pd.read_csv(directory / f"{name}.csv")
        required = {"row_id", TARGET_COLUMN, *FEATURE_COLUMNS}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} split is missing required columns")
        if len(frame) != config["counts"][name]:
            raise ValueError(f"{name} split count changed")
        ids = frame["row_id"].astype(int).tolist()
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if digest != config[f"{name}_ids_sha256"]:
            raise ValueError(f"{name} split row-id hash changed")
        if frame[[*FEATURE_COLUMNS, TARGET_COLUMN]].isna().any().any():
            raise ValueError(f"{name} split contains missing values")
        frames[name] = frame
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Split row ids overlap")
    return frames


def feature_importance_table(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray | pd.Series, *, n_repeats: int = 5
) -> pd.DataFrame:
    """Impurity importance and validation-set permutation RMSE increase."""
    feature_set = "ratio" if "ratios" in pipeline.named_steps else "raw"
    feature_names = feature_names_for_set(feature_set)
    X_features = pipeline.named_steps["ratios"].transform(X) if feature_set == "ratio" else X
    # Permute the model's actual input columns, including derived ratios.
    X_imputed = pipeline.named_steps["imputer"].transform(X_features)
    forest = pipeline.named_steps["model"]
    permutation = permutation_importance(
        forest,
        X_imputed,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "impurity_importance": forest.feature_importances_,
            "permutation_rmse_increase": permutation.importances_mean,
            "permutation_std": permutation.importances_std,
        }
    ).sort_values("permutation_rmse_increase", ascending=False).reset_index(drop=True)


def select_by_validation(candidates: list[dict]) -> dict:
    """Lock the lowest validation RMSE without consulting test results."""
    return min(candidates, key=lambda candidate: candidate["validation_metrics"]["rmse"])
