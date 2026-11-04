import json

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, TABLES_DIR
from src.evaluation import regression_metrics
from src.random_forest import check_split_files


def test_locked_rf_result_matches_saved_predictions_and_fixed_split():
    splits = check_split_files(PROCESSED_DATA_DIR)
    lock = json.loads((RESULTS_DIR / "rf_selection_lock.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS_DIR / "student_b_rf_metrics.csv")
    selected = lock["selected_model_key"]
    validation = metrics[metrics.split == "validation"]
    assert selected == validation.loc[validation.rmse.idxmin(), "model_key"]

    test_rows = metrics[metrics.split == "test"]
    assert len(test_rows) == 1
    assert test_rows.iloc[0].model_key == selected
    predictions = pd.read_csv(PREDICTIONS_DIR / f"{selected}_test.csv")
    assert np.array_equal(predictions.row_id.to_numpy(), splits["test"].row_id.to_numpy())
    assert np.allclose(predictions.y_true, splits["test"].MedHouseVal)
    recomputed = regression_metrics(predictions.y_true, predictions.y_pred)
    for metric in ("mae", "mse", "rmse", "r2"):
        assert np.isclose(recomputed[metric], test_rows.iloc[0][metric])


def test_combined_handoff_selects_by_validation_not_test():
    validation = pd.read_csv(TABLES_DIR / "table_13_all_validation_metrics.csv")
    final = pd.read_csv(TABLES_DIR / "final_metrics.csv")
    selected = validation.loc[validation.rmse.idxmin(), "model_key"]
    assert selected == "rf_raw_optimized"
    assert final.loc[final.selected_overall_by_validation, "model_key"].tolist() == [selected]
