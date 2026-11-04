from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, FEATURE_COLUMNS, RANDOM_STATE, RIDGE_ALPHAS, TARGET_COLUMN
from .evaluation import regression_metrics
from .features import build_preprocessor, feature_names_for_set


MODEL_SPECS = [
    {
        "key": "linear_raw",
        "label": "Linear Regression",
        "family": "linear",
        "feature_set": "raw",
    },
    {
        "key": "ridge_raw",
        "label": "Ridge Regression",
        "family": "ridge",
        "feature_set": "raw",
    },
    {
        "key": "linear_ratio",
        "label": "Linear Regression",
        "family": "linear",
        "feature_set": "ratio",
    },
    {
        "key": "ridge_ratio",
        "label": "Ridge Regression",
        "family": "ridge",
        "feature_set": "ratio",
    },
]


def _make_pipeline(feature_set: str, estimator: object) -> Pipeline:
    return Pipeline(
        [
            *build_preprocessor(feature_set).steps,
            ("model", estimator),
        ]
    )


def fit_candidate_models(
    split_frames: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Fit locked Linear/Ridge candidates and select using validation RMSE."""
    X_train = split_frames["train"].loc[:, FEATURE_COLUMNS]
    y_train = split_frames["train"][TARGET_COLUMN].to_numpy()
    X_validation = split_frames["validation"].loc[:, FEATURE_COLUMNS]
    y_validation = split_frames["validation"][TARGET_COLUMN].to_numpy()

    candidates: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for spec in MODEL_SPECS:
        if spec["family"] == "linear":
            pipeline = _make_pipeline(
                spec["feature_set"],
                LinearRegression(fit_intercept=True),
            )
            started = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            best_alpha = None
            cv_score = None
            cv_std = None
        else:
            pipeline = _make_pipeline(
                spec["feature_set"],
                Ridge(fit_intercept=True),
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid={"model__alpha": RIDGE_ALPHAS},
                scoring="neg_root_mean_squared_error",
                cv=kfold,
                refit=True,
                n_jobs=-1,
                return_train_score=True,
                error_score="raise",
            )
            started = time.perf_counter()
            search.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            pipeline = search.best_estimator_
            best_alpha = float(search.best_params_["model__alpha"])
            cv_score = float(-search.best_score_)
            cv_std = float(
                search.cv_results_["std_test_score"][search.best_index_]
            )
            for index, alpha in enumerate(search.cv_results_["param_model__alpha"]):
                cv_rows.append(
                    {
                        "model_key": spec["key"],
                        "model": spec["label"],
                        "feature_set": spec["feature_set"],
                        "alpha": float(alpha),
                        "mean_train_rmse": float(
                            -search.cv_results_["mean_train_score"][index]
                        ),
                        "mean_validation_rmse": float(
                            -search.cv_results_["mean_test_score"][index]
                        ),
                        "std_validation_rmse": float(
                            search.cv_results_["std_test_score"][index]
                        ),
                        "rank_validation_rmse": int(search.cv_results_["rank_test_score"][index]),
                    }
                )

        prediction_train = pipeline.predict(X_train)
        prediction_validation = pipeline.predict(X_validation)
        train_metrics = regression_metrics(y_train, prediction_train)
        validation_metrics = regression_metrics(y_validation, prediction_validation)
        model = pipeline.named_steps["model"]

        candidates.append(
            {
                **spec,
                "pipeline": pipeline,
                "fit_seconds": float(fit_seconds),
                "best_alpha": best_alpha,
                "cv_rmse": cv_score,
                "cv_rmse_std": cv_std,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "coefficients": np.asarray(model.coef_, dtype=float),
                "intercept": float(model.intercept_),
                "feature_names": feature_names_for_set(spec["feature_set"]),
            }
        )

    selected = min(candidates, key=lambda item: item["validation_metrics"]["rmse"])
    for candidate in candidates:
        candidate["selected"] = candidate["key"] == selected["key"]

    cv_frame = pd.DataFrame(cv_rows)
    ranking = pd.DataFrame(
        [
            {
                "model_key": candidate["key"],
                "model": candidate["label"],
                "feature_set": candidate["feature_set"],
                "validation_mae": candidate["validation_metrics"]["mae"],
                "validation_rmse": candidate["validation_metrics"]["rmse"],
                "validation_r2": candidate["validation_metrics"]["r2"],
                "best_alpha": candidate["best_alpha"],
                "cv_rmse": candidate["cv_rmse"],
                "cv_rmse_std": candidate["cv_rmse_std"],
                "fit_seconds": candidate["fit_seconds"],
                "selected_by_validation": candidate["selected"],
            }
            for candidate in candidates
        ]
    ).sort_values("validation_rmse")
    return candidates, ranking, cv_frame


def fitted_preprocessor(candidate: dict[str, Any]) -> Pipeline:
    """Return a fitted preprocessing-only pipeline cloned from a candidate."""
    preprocessor = Pipeline(candidate["pipeline"].steps[:-1])
    return clone(preprocessor)


def best_ridge_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ridge_candidates = [item for item in candidates if item["family"] == "ridge"]
    return min(ridge_candidates, key=lambda item: item["validation_metrics"]["rmse"])
