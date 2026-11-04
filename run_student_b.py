"""Reproduce Student B's Random Forest work on Student A's fixed splits."""

from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import (
    FEATURE_COLUMNS, METRICS_DIR, MODELS_DIR, NOTES_DIR, PREDICTIONS_DIR,
    PROCESSED_DATA_DIR, RANDOM_STATE, RESULTS_DIR, TABLES_DIR, TARGET_COLUMN,
    ensure_directories,
)
from src.evaluation import metrics_rows, regression_metrics, save_predictions
from src.plots import plot_prediction_diagnostics
from src.random_forest import (
    SEARCH_SPACE, build_rf_pipeline, check_split_files, feature_importance_table,
    make_search, select_by_validation,
)
from src.rf_plots import plot_rf_complexity, plot_rf_importance, plot_rf_validation


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate(candidate, frames, splits):
    rows = []
    saved = {}
    for split in splits:
        frame = frames[split]
        prediction = candidate["pipeline"].predict(frame[FEATURE_COLUMNS])
        metric = regression_metrics(frame[TARGET_COLUMN].to_numpy(), prediction)
        rows.extend(metrics_rows(candidate["key"], candidate["label"], candidate["feature_set"], {split: metric}))
        saved[split] = save_predictions(
            PREDICTIONS_DIR, candidate["key"], split,
            frame["row_id"].to_numpy(), frame[TARGET_COLUMN].to_numpy(), prediction,
        )
    return rows, saved


def main() -> None:
    ensure_directories()
    frames = check_split_files(PROCESSED_DATA_DIR)
    split_config = json.loads((PROCESSED_DATA_DIR / "split_config.json").read_text(encoding="utf-8"))
    train, validation = frames["train"], frames["validation"]
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_valid, y_valid = validation[FEATURE_COLUMNS], validation[TARGET_COLUMN]

    candidates = []
    cv_rows = []
    for feature_set in ("raw", "ratio"):
        print(f"Fitting RF baseline: {feature_set}", flush=True)
        baseline = build_rf_pipeline(feature_set)
        started = time.perf_counter()
        baseline.fit(X_train, y_train)
        baseline_seconds = time.perf_counter() - started
        candidates.append({
            "key": f"rf_{feature_set}", "label": "Random Forest baseline",
            "feature_set": feature_set, "pipeline": baseline,
            "fit_seconds": baseline_seconds,
            "validation_metrics": regression_metrics(y_valid, baseline.predict(X_valid)),
            "cv_rmse": None, "cv_rmse_std": None,
            "parameters": baseline.named_steps["model"].get_params(),
        })

        print(f"Searching RF: {feature_set}, 6 candidates x 5 folds", flush=True)
        search = make_search(feature_set)
        started = time.perf_counter()
        search.fit(X_train, y_train)
        search_seconds = time.perf_counter() - started
        optimized = search.best_estimator_
        candidates.append({
            "key": f"rf_{feature_set}_optimized", "label": "Random Forest optimized",
            "feature_set": feature_set, "pipeline": optimized,
            "fit_seconds": search_seconds,
            "validation_metrics": regression_metrics(y_valid, optimized.predict(X_valid)),
            "cv_rmse": -float(search.best_score_),
            "cv_rmse_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "parameters": search.best_params_,
        })
        for index, params in enumerate(search.cv_results_["params"]):
            cv_rows.append({
                "feature_set": feature_set, "candidate": index + 1,
                **params,
                "mean_train_rmse": -float(search.cv_results_["mean_train_score"][index]),
                "mean_cv_rmse": -float(search.cv_results_["mean_test_score"][index]),
                "std_cv_rmse": float(search.cv_results_["std_test_score"][index]),
                "rank_cv_rmse": int(search.cv_results_["rank_test_score"][index]),
                "mean_fit_seconds": float(search.cv_results_["mean_fit_time"][index]),
            })

    metric_rows_all = []
    predictions = {}
    for candidate in candidates:
        rows, saved = _evaluate(candidate, frames, ("train", "validation"))
        metric_rows_all.extend(rows)
        predictions[candidate["key"]] = saved
    selected = select_by_validation(candidates)

    # The locked selection record is written before the first RF test prediction.
    lock = {
        "selection_rule": "minimum validation RMSE among RF candidates; test unused",
        "selected_model_key": selected["key"],
        "selected_feature_set": selected["feature_set"],
        "selected_parameters": selected["parameters"],
        "selected_validation_rmse": selected["validation_metrics"]["rmse"],
        "split_index_hashes": {k: v for k, v in split_config.items() if k.endswith("_ids_sha256")},
    }
    _write_json(RESULTS_DIR / "rf_selection_lock.json", lock)
    print(f"Locked {selected['key']} at validation RMSE {selected['validation_metrics']['rmse']:.4f}", flush=True)

    # One final test evaluation, only for the locked RF candidate.
    test_rows, test_predictions = _evaluate(selected, frames, ("test",))
    metric_rows_all.extend(test_rows)
    predictions[selected["key"]].update(test_predictions)
    metrics = pd.DataFrame(metric_rows_all)
    metrics["selected_by_validation"] = metrics["model_key"].eq(selected["key"])
    metrics.to_csv(METRICS_DIR / "student_b_rf_metrics.csv", index=False, encoding="utf-8")
    metrics[metrics["split"] == "validation"].to_csv(TABLES_DIR / "table_09_rf_validation_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(cv_rows).to_csv(METRICS_DIR / "rf_cv_results.csv", index=False, encoding="utf-8")

    ablation = []
    for suffix in ("", "_optimized"):
        raw = metrics[(metrics.model_key == f"rf_raw{suffix}") & (metrics.split == "validation")].iloc[0]
        ratio = metrics[(metrics.model_key == f"rf_ratio{suffix}") & (metrics.split == "validation")].iloc[0]
        ablation.append({
            "model": "RF baseline" if not suffix else "RF optimized",
            "split": "validation", "raw_rmse": raw.rmse, "ratio_rmse": ratio.rmse,
            "delta_rmse": ratio.rmse - raw.rmse,
            "delta_rmse_percent": 100 * (ratio.rmse - raw.rmse) / raw.rmse,
            "raw_mae": raw.mae, "ratio_mae": ratio.mae,
            "raw_r2": raw.r2, "ratio_r2": ratio.r2,
        })
    pd.DataFrame(ablation).to_csv(TABLES_DIR / "table_10_rf_feature_ablation.csv", index=False, encoding="utf-8")

    # Complexity: hold selected feature set and the other model settings fixed.
    complexity_rows = []
    for depth in (5, 10, 20, None):
        pipeline = clone(selected["pipeline"])
        pipeline.set_params(model__max_depth=depth)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        complexity_rows.append({
            "feature_set": selected["feature_set"], "max_depth": depth,
            "depth_label": str(depth) if depth is not None else "None",
            "train_rmse": regression_metrics(y_train, pipeline.predict(X_train))["rmse"],
            "validation_rmse": regression_metrics(y_valid, pipeline.predict(X_valid))["rmse"],
            "fit_seconds": time.perf_counter() - started,
        })
    complexity = pd.DataFrame(complexity_rows)
    complexity.to_csv(METRICS_DIR / "rf_depth_complexity.csv", index=False, encoding="utf-8")
    complexity.to_csv(TABLES_DIR / "table_11_rf_depth_complexity.csv", index=False, encoding="utf-8")

    print("Computing validation feature importance", flush=True)
    importance = feature_importance_table(selected["pipeline"], X_valid, y_valid, n_repeats=5)
    importance.to_csv(METRICS_DIR / "rf_feature_importance.csv", index=False, encoding="utf-8")
    importance.to_csv(TABLES_DIR / "table_12_rf_feature_importance.csv", index=False, encoding="utf-8")
    joblib.dump(selected["pipeline"], MODELS_DIR / "selected_rf_model.joblib")

    best_params = {
        "random_state": RANDOM_STATE,
        "cv_folds": 5, "cv_shuffle": True, "scoring": "neg_root_mean_squared_error",
        "n_iter_per_feature_set": 6, "search_space": SEARCH_SPACE,
        "baselines": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1,
                      "min_samples_split": 2, "max_features": 1.0},
        "best_by_feature_set": {
            feature_set: next(candidate for candidate in candidates if candidate["key"] == f"rf_{feature_set}_optimized")["parameters"]
            for feature_set in ("raw", "ratio")
        },
        "selection_lock": lock,
    }
    _write_json(RESULTS_DIR / "best_params.json", best_params)
    _write_json(NOTES_DIR / "student_b_run_summary.json", {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "joblib")},
        "selection_lock": lock,
        "rf_test_metrics": test_rows[0],
    })

    plot_rf_validation(metrics)
    plot_rf_complexity(complexity)
    plot_rf_importance(importance, selected["feature_set"])
    for split in ("validation", "test"):
        plot_prediction_diagnostics(
            predictions[selected["key"]][split], split, selected["key"],
            f"19_rf_{split}_diagnostics" if split == "validation" else "20_rf_test_diagnostics",
            include_ppt=True,
        )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
