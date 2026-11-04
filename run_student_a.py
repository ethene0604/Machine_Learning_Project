from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CV_FOLDS,
    DATA_DIR,
    FEATURE_COLUMNS,
    METRICS_DIR,
    MODELS_DIR,
    NOTES_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    ensure_directories,
)
from src.data import (
    build_data_profile,
    load_dataset,
    save_splits,
    sha256_file,
    split_dataset,
)
from src.evaluation import metrics_rows, regression_metrics, save_predictions
from src.features import build_preprocessor, feature_names_for_set
from src.linear_models import best_ridge_candidate, fit_candidate_models
from src.plots import (
    plot_correlation_heatmap,
    plot_feature_boxplots,
    plot_feature_distributions,
    plot_feature_engineering_ablation,
    plot_error_by_price_band,
    plot_geographic_prices,
    plot_model_comparison,
    plot_prediction_diagnostics,
    plot_ridge_alpha_curves,
    plot_scratch_comparison,
    plot_scratch_loss,
    plot_split_metrics,
    plot_standardized_coefficients,
)
from src.ridge_scratch import RidgeGradientDescent


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _feature_engineering_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        for model_label in ("Linear Regression", "Ridge Regression"):
            raw = metrics[
                (metrics["split"] == split_name)
                & (metrics["model"] == model_label)
                & (metrics["feature_set"] == "raw")
            ].iloc[0]
            ratio = metrics[
                (metrics["split"] == split_name)
                & (metrics["model"] == model_label)
                & (metrics["feature_set"] == "ratio")
            ].iloc[0]
            rows.append(
                {
                    "model": model_label,
                    "split": split_name,
                    "raw_mae": raw["mae"],
                    "ratio_mae": ratio["mae"],
                    "delta_mae": ratio["mae"] - raw["mae"],
                    "raw_rmse": raw["rmse"],
                    "ratio_rmse": ratio["rmse"],
                    "delta_rmse": ratio["rmse"] - raw["rmse"],
                    "delta_rmse_percent": 100
                    * (ratio["rmse"] - raw["rmse"])
                    / raw["rmse"],
                    "raw_r2": raw["r2"],
                    "ratio_r2": ratio["r2"],
                    "delta_r2": ratio["r2"] - raw["r2"],
                }
            )
    return pd.DataFrame(rows)


def _error_by_price_band(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["price_band"] = pd.cut(
        frame["y_true"],
        bins=[-np.inf, 1.5, 3.0, 4.99, np.inf],
        labels=["<1.5", "1.5-3.0", "3.0-4.99", ">=4.99 (capped)"],
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for band, group in frame.groupby("price_band", observed=True):
        rows.append(
            {
                "price_band": str(band),
                "n_samples": int(len(group)),
                "mae": float(np.mean(np.abs(group["residual"]))),
                "rmse": float(np.sqrt(np.mean(group["residual"] ** 2))),
                "mean_bias": float(np.mean(group["y_pred"] - group["y_true"])),
            }
        )
    return pd.DataFrame(rows)


def _fit_from_scratch(
    best_ridge: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit and select the NumPy implementation using validation data only."""
    feature_set = best_ridge["feature_set"]
    preprocessor = build_preprocessor(feature_set)
    X_train = split_frames["train"].loc[:, FEATURE_COLUMNS]
    y_train = split_frames["train"][TARGET_COLUMN].to_numpy()
    X_validation = split_frames["validation"].loc[:, FEATURE_COLUMNS]
    y_validation = split_frames["validation"][TARGET_COLUMN].to_numpy()
    preprocessor.fit(X_train, y_train)
    X_train_scaled = np.asarray(preprocessor.transform(X_train), dtype=float)
    X_validation_scaled = np.asarray(
        preprocessor.transform(X_validation), dtype=float
    )

    alpha_sklearn = float(best_ridge["best_alpha"])
    lambda_custom = alpha_sklearn / len(y_train)
    learning_rates = [0.01, 0.02, 0.05]
    candidates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for learning_rate in learning_rates:
        model = RidgeGradientDescent(
            lambda_=lambda_custom,
            learning_rate=learning_rate,
            max_iter=5_000,
            tolerance=1e-7,
            record_every=10,
        )
        started = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        fit_seconds = time.perf_counter() - started
        train_prediction = model.predict(X_train_scaled)
        validation_prediction = model.predict(X_validation_scaled)
        train_metrics = regression_metrics(y_train, train_prediction)
        validation_metrics = regression_metrics(
            y_validation, validation_prediction
        )
        candidate = {
            "model": model,
            "preprocessor": preprocessor,
            "feature_set": feature_set,
            "alpha_sklearn": alpha_sklearn,
            "lambda_custom": lambda_custom,
            "learning_rate": learning_rate,
            "fit_seconds": fit_seconds,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "X_train_scaled": X_train_scaled,
            "y_train": y_train,
            "X_validation_scaled": X_validation_scaled,
            "y_validation": y_validation,
        }
        candidates.append(candidate)
        rows.append(
            {
                "feature_set": feature_set,
                "alpha_sklearn": alpha_sklearn,
                "lambda_custom": lambda_custom,
                "learning_rate": learning_rate,
                "iterations": model.n_iter_,
                "converged": model.converged_,
                "train_rmse": train_metrics["rmse"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
                "fit_seconds": fit_seconds,
            }
        )

    selected = min(
        candidates,
        key=lambda item: item["validation_metrics"]["rmse"],
    )
    selected["selection_table"] = pd.DataFrame(rows).sort_values(
        "validation_rmse"
    )
    return selected, selected["selection_table"]


def main() -> None:
    started = time.perf_counter()
    ensure_directories()
    dataset_path = DATA_DIR / "california_housing.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing local dataset: {dataset_path}. Prepare the CSV before running."
        )

    frame = load_dataset(dataset_path)
    split_frames, train_ids, validation_ids, test_ids = split_dataset(frame)
    split_config = save_splits(
        split_frames,
        train_ids,
        validation_ids,
        test_ids,
        PROCESSED_DATA_DIR,
    )

    profile = build_data_profile(frame)
    profile.to_csv(
        METRICS_DIR / "data_profile.csv", index=False, encoding="utf-8"
    )
    profile.to_csv(
        TABLES_DIR / "table_01_data_profile.csv",
        index=False,
        encoding="utf-8",
    )

    print("Generating EDA figures...")
    plot_feature_distributions(split_frames["train"])
    plot_feature_boxplots(split_frames["train"])
    plot_correlation_heatmap(split_frames["train"])
    plot_geographic_prices(split_frames["train"])

    print("Fitting Linear Regression and Ridge candidates...")
    candidates, ranking, cv_results = fit_candidate_models(split_frames)
    ranking.to_csv(
        METRICS_DIR / "linear_model_validation_ranking.csv",
        index=False,
        encoding="utf-8",
    )
    ranking.to_csv(
        TABLES_DIR / "table_02_linear_validation_ranking.csv",
        index=False,
        encoding="utf-8",
    )
    cv_results.to_csv(
        METRICS_DIR / "ridge_cv_results.csv",
        index=False,
        encoding="utf-8",
    )

    selected = next(candidate for candidate in candidates if candidate["selected"])
    ridge_selected = best_ridge_candidate(candidates)
    coefficient_frames = []
    for candidate in candidates:
        coefficient_frames.append(
            pd.DataFrame(
                {
                    "model_key": candidate["key"],
                    "model": candidate["label"],
                    "feature_set": candidate["feature_set"],
                    "feature": candidate["feature_names"],
                    "coefficient": candidate["coefficients"],
                }
            )
        )
    pd.concat(coefficient_frames, ignore_index=True).to_csv(
        METRICS_DIR / "linear_model_coefficients.csv",
        index=False,
        encoding="utf-8",
    )

    print("Fitting NumPy Ridge extension...")
    scratch_result, scratch_selection = _fit_from_scratch(
        ridge_selected, split_frames
    )
    scratch_selection.to_csv(
        METRICS_DIR / "from_scratch_learning_rate_selection.csv",
        index=False,
        encoding="utf-8",
    )

    # All choices above use training and validation data only. The test set is
    # evaluated only after the candidates and the scratch learning rate are locked.
    print("Final test-set evaluation of locked candidates...")
    prediction_frames: dict[str, dict[str, pd.DataFrame]] = {}
    metric_rows: list[dict[str, Any]] = []
    X_tests = {
        "train": split_frames["train"].loc[:, FEATURE_COLUMNS],
        "validation": split_frames["validation"].loc[:, FEATURE_COLUMNS],
        "test": split_frames["test"].loc[:, FEATURE_COLUMNS],
    }
    y_sets = {
        split_name: split_frames[split_name][TARGET_COLUMN].to_numpy()
        for split_name in ("train", "validation", "test")
    }
    row_id_sets = {
        split_name: split_frames[split_name]["row_id"].to_numpy()
        for split_name in ("train", "validation", "test")
    }

    for candidate in candidates:
        prediction_frames[candidate["key"]] = {}
        split_metrics: dict[str, dict[str, float]] = {}
        for split_name in ("train", "validation", "test"):
            prediction = candidate["pipeline"].predict(X_tests[split_name])
            split_metrics[split_name] = regression_metrics(
                y_sets[split_name], prediction
            )
            prediction_frames[candidate["key"]][split_name] = save_predictions(
                PREDICTIONS_DIR,
                candidate["key"],
                split_name,
                row_id_sets[split_name],
                y_sets[split_name],
                prediction,
            )
        metric_rows.extend(
            metrics_rows(
                candidate["key"],
                candidate["label"],
                candidate["feature_set"],
                split_metrics,
            )
        )

    scratch_model = scratch_result["model"]
    scratch_preprocessor = scratch_result["preprocessor"]
    X_test_scaled = np.asarray(
        scratch_preprocessor.transform(X_tests["test"]), dtype=float
    )
    scratch_predictions: dict[str, pd.DataFrame] = {}
    scratch_metrics: dict[str, dict[str, float]] = {}
    for split_name, prediction in (
        ("train", scratch_model.predict(scratch_result["X_train_scaled"])),
        (
            "validation",
            scratch_model.predict(scratch_result["X_validation_scaled"]),
        ),
        ("test", scratch_model.predict(X_test_scaled)),
    ):
        scratch_metrics[split_name] = regression_metrics(
            y_sets[split_name], prediction
        )
        scratch_predictions[split_name] = save_predictions(
            PREDICTIONS_DIR,
            "numpy_ridge_gd",
            split_name,
            row_id_sets[split_name],
            y_sets[split_name],
            prediction,
        )
    metric_rows.extend(
        metrics_rows(
            "numpy_ridge_gd",
            "NumPy Ridge GD",
            scratch_result["feature_set"],
            scratch_metrics,
        )
    )

    metrics = pd.DataFrame(metric_rows)
    metrics["selected_by_validation"] = metrics["model_key"].eq(selected["key"])
    metrics.to_csv(
        METRICS_DIR / "student_a_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    metrics[metrics["split"] == "validation"].to_csv(
        TABLES_DIR / "table_03_linear_validation_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    metrics[metrics["split"] == "test"].to_csv(
        TABLES_DIR / "table_04_linear_test_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    metrics[
        (metrics["model_key"] == selected["key"])
        & (metrics["split"].isin(["train", "validation", "test"]))
    ].to_csv(
        TABLES_DIR / "table_05_selected_model_split_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    ablation = _feature_engineering_ablation(metrics)
    ablation.to_csv(
        METRICS_DIR / "feature_engineering_ablation.csv",
        index=False,
        encoding="utf-8",
    )
    ablation.to_csv(
        TABLES_DIR / "table_06_feature_engineering_ablation.csv",
        index=False,
        encoding="utf-8",
    )

    scratch_comparison = pd.DataFrame(
        [
            {
                "split": split_name,
                "sklearn_rmse": ridge_selected["validation_metrics"]["rmse"]
                if split_name == "validation"
                else metrics[
                    (metrics["model_key"] == ridge_selected["key"])
                    & (metrics["split"] == split_name)
                ]["rmse"].iloc[0],
                "scratch_rmse": scratch_metrics[split_name]["rmse"],
                "sklearn_mae": ridge_selected["validation_metrics"]["mae"]
                if split_name == "validation"
                else metrics[
                    (metrics["model_key"] == ridge_selected["key"])
                    & (metrics["split"] == split_name)
                ]["mae"].iloc[0],
                "scratch_mae": scratch_metrics[split_name]["mae"],
                "sklearn_r2": ridge_selected["validation_metrics"]["r2"]
                if split_name == "validation"
                else metrics[
                    (metrics["model_key"] == ridge_selected["key"])
                    & (metrics["split"] == split_name)
                ]["r2"].iloc[0],
                "scratch_r2": scratch_metrics[split_name]["r2"],
            }
            for split_name in ("train", "validation", "test")
        ]
    )
    scratch_comparison.to_csv(
        METRICS_DIR / "from_scratch_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    scratch_comparison.to_csv(
        TABLES_DIR / "table_07_from_scratch_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    error_band = _error_by_price_band(
        prediction_frames[selected["key"]]["test"]
    )
    error_band.to_csv(
        METRICS_DIR / "selected_model_test_error_by_price_band.csv",
        index=False,
        encoding="utf-8",
    )
    error_band.to_csv(
        TABLES_DIR / "table_08_test_error_by_price_band.csv",
        index=False,
        encoding="utf-8",
    )

    plot_model_comparison(ranking)
    plot_ridge_alpha_curves(cv_results)
    plot_prediction_diagnostics(
        prediction_frames[selected["key"]]["validation"],
        "validation",
        selected["label"],
        "12_selected_model_validation_diagnostics",
        include_ppt=True,
    )
    plot_prediction_diagnostics(
        prediction_frames[selected["key"]]["test"],
        "test",
        selected["label"],
        "13_selected_model_test_diagnostics",
        include_ppt=True,
    )
    plot_prediction_diagnostics(
        scratch_predictions["test"],
        "test",
        "NumPy Ridge Gradient Descent",
        "14_from_scratch_test_diagnostics",
        include_ppt=False,
    )
    plot_feature_engineering_ablation(metrics)
    plot_standardized_coefficients(selected)
    plot_split_metrics(metrics, selected["key"], selected["label"])
    plot_scratch_loss(scratch_result)
    plot_scratch_comparison(scratch_comparison)
    plot_error_by_price_band(error_band)

    joblib.dump(
        selected["pipeline"],
        MODELS_DIR / "selected_linear_model.joblib",
    )
    joblib.dump(
        ridge_selected["pipeline"],
        MODELS_DIR / "selected_ridge_model.joblib",
    )

    runtime = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: version(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "seaborn",
                "joblib",
            )
        },
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "split": split_config,
        "selection_rule": "lowest validation RMSE; test set evaluated once afterward",
        "selected_linear_model": {
            "model_key": selected["key"],
            "model": selected["label"],
            "feature_set": selected["feature_set"],
            "alpha": selected["best_alpha"],
            "cv_rmse": selected["cv_rmse"],
            "cv_rmse_std": selected["cv_rmse_std"],
            "validation_rmse": selected["validation_metrics"]["rmse"],
        },
        "selected_ridge_for_extension": {
            "model_key": ridge_selected["key"],
            "feature_set": ridge_selected["feature_set"],
            "alpha_sklearn": ridge_selected["best_alpha"],
            "validation_rmse": ridge_selected["validation_metrics"]["rmse"],
        },
        "num_by_scratch": {
            "feature_set": scratch_result["feature_set"],
            "alpha_sklearn": scratch_result["alpha_sklearn"],
            "lambda_custom": scratch_result["lambda_custom"],
            "learning_rate": scratch_result["learning_rate"],
            "iterations": scratch_model.n_iter_,
            "converged": scratch_model.converged_,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(RESULTS_DIR / "run_metadata.json", runtime)
    _write_json(NOTES_DIR / "student_a_run_summary.json", runtime)

    print("\nValidation ranking")
    print(ranking.to_string(index=False))
    print("\nLocked test metrics")
    print(
        metrics[metrics["split"] == "test"].to_string(index=False)
    )
    print("\nFeature-engineering ablation")
    print(ablation.to_string(index=False))
    print("\nNumPy Ridge comparison")
    print(scratch_comparison.to_string(index=False))
    print("\nTest error by true-price band")
    print(error_band.to_string(index=False))
    print(f"\nCompleted in {runtime['elapsed_seconds']:.1f} seconds.")


if __name__ == "__main__":
    main()
