"""Run the complete California Housing regression project in one environment."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.config import NOTES_DIR, RESULTS_DIR
from src.forest_experiment import main as run_forest
from src.linear_experiment import main as run_linear
from src.reporting import main as build_handoff


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    linear = run_linear()
    forest = run_forest()
    overall_selection = build_handoff()
    summary = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "python": linear.get("python"),
        "platform": linear.get("platform"),
        "packages": linear.get("packages"),
        "dataset_path": linear.get("dataset_path"),
        "dataset_sha256": linear.get("dataset_sha256"),
        "split": linear.get("split"),
        "selection_rule": "minimum validation RMSE; test results are not used for selection",
        "linear_selection": linear.get("selected_linear_model"),
        "ridge_extension": linear.get("selected_ridge_for_extension"),
        "numpy_ridge": linear.get("num_by_scratch"),
        "forest_selection": forest.get("selection_lock"),
        "forest_test_metrics": forest.get("rf_test_metrics"),
        "overall_selection": overall_selection,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(NOTES_DIR / "run_summary.json", summary)
    _write_json(RESULTS_DIR / "run_metadata.json", summary)
    print(f"Full experiment completed in {summary['elapsed_seconds']:.1f} seconds.")


if __name__ == "__main__":
    main()
