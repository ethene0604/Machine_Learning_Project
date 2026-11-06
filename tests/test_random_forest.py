import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.random_forest import (
    build_rf_pipeline,
    check_split_files,
    feature_importance_table,
    make_search,
    select_by_validation,
)


def _small_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.uniform(1, 10, (60, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = 2 * X["MedInc"] + rng.normal(0, 0.1, len(X))
    return X, y


def test_rf_pipeline_uses_same_feature_definitions_and_no_scaler():
    X, y = _small_data()
    raw = build_rf_pipeline("raw", n_estimators=5)
    ratio = build_rf_pipeline("ratio", n_estimators=5)
    assert list(raw.named_steps) == ["imputer", "model"]
    assert list(ratio.named_steps) == ["ratios", "imputer", "model"]
    assert isinstance(raw.named_steps["model"], RandomForestRegressor)
    raw.fit(X, y)
    ratio.fit(X, y)
    assert raw.named_steps["model"].n_features_in_ == 8
    assert ratio.named_steps["model"].n_features_in_ == 11


def test_search_uses_five_training_folds_and_bounded_candidates():
    search = make_search("raw", n_iter=3, n_jobs=1)
    assert search.n_iter == 3
    assert search.cv.n_splits == 5
    assert search.cv.shuffle
    assert search.scoring == "neg_root_mean_squared_error"
    assert search.estimator.named_steps["model"].n_jobs == 1
    assert set(search.param_distributions) == {
        "model__n_estimators", "model__max_depth", "model__min_samples_leaf",
        "model__min_samples_split", "model__max_features",
    }


def test_split_check_rejects_overlapping_rows_and_wrong_hash(tmp_path):
    columns = ["row_id", *FEATURE_COLUMNS, TARGET_COLUMN]
    for name, ids in (("train", [0, 1]), ("validation", [2]), ("test", [3])):
        data = pd.DataFrame([[row_id, *([1.0] * 8), 2.0] for row_id in ids], columns=columns)
        data.to_csv(tmp_path / f"{name}.csv", index=False)
    config = {
        "counts": {"train": 2, "validation": 1, "test": 1},
        **{
            f"{name}_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()
            for name, ids in (("train", [0, 1]), ("validation", [2]), ("test", [3]))
        },
    }
    (tmp_path / "split_config.json").write_text(json.dumps(config), encoding="utf-8")
    frames = check_split_files(tmp_path)
    assert list(frames) == ["train", "validation", "test"]
    frames["test"].loc[0, "row_id"] = 1
    frames["test"].to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="hash|overlap"):
        check_split_files(tmp_path)


def test_importance_table_has_aligned_features_and_normalized_impurity():
    X, y = _small_data()
    model = build_rf_pipeline("ratio", n_estimators=8).fit(X, y)
    table = feature_importance_table(model, X, y, n_repeats=2)
    assert len(table) == 11
    assert set(table["feature"]) == set(FEATURE_COLUMNS) | {
        "BedroomRatio", "RoomsPerPerson", "HouseholdsApprox"
    }
    assert table["impurity_importance"].sum() == pytest.approx(1.0)
    assert table["permutation_rmse_increase"].notna().all()


def test_selection_uses_validation_rmse_only():
    candidates = [
        {"key": "rf_raw", "validation_metrics": {"rmse": 0.6}},
        {"key": "rf_ratio", "validation_metrics": {"rmse": 0.5}},
    ]
    assert select_by_validation(candidates)["key"] == "rf_ratio"
