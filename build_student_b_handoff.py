"""Combine already-recorded A and B results without rerunning the test set."""

from __future__ import annotations

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import METRICS_DIR, PPT_FIGURES_DIR, PREDICTIONS_DIR, REPORT_FIGURES_DIR, RESULTS_DIR, TABLES_DIR


def main() -> None:
    lock = json.loads((RESULTS_DIR / "rf_selection_lock.json").read_text(encoding="utf-8"))
    selected_key = lock["selected_model_key"]
    a = pd.read_csv(METRICS_DIR / "student_a_metrics.csv")
    b = pd.read_csv(METRICS_DIR / "student_b_rf_metrics.csv")
    a = a[a.model_key != "numpy_ridge_gd"].copy()
    b_selected = b[b.model_key == selected_key].copy()

    validation = pd.concat(
        [a[a.split == "validation"], b[b.split == "validation"]], ignore_index=True
    )
    if b.loc[b[b.split == "validation"].rmse.idxmin(), "model_key"] != selected_key:
        raise ValueError("RF lock is inconsistent with the RF validation ranking")
    overall_key = validation.loc[validation.rmse.idxmin(), "model_key"]
    validation["selected_overall_by_validation"] = validation.model_key.eq(overall_key)
    validation.to_csv(TABLES_DIR / "table_13_all_validation_metrics.csv", index=False, encoding="utf-8")

    final = pd.concat(
        [a[a.split == "test"], b_selected[b_selected.split == "test"]], ignore_index=True
    )
    final["selected_overall_by_validation"] = final.model_key.eq(overall_key)
    final.to_csv(TABLES_DIR / "final_metrics.csv", index=False, encoding="utf-8")

    a_ablation = pd.read_csv(METRICS_DIR / "feature_engineering_ablation.csv")
    a_ablation = a_ablation[a_ablation.split == "validation"]
    b_ablation = pd.read_csv(TABLES_DIR / "table_10_rf_feature_ablation.csv")
    b_ablation["delta_mae"] = b_ablation.ratio_mae - b_ablation.raw_mae
    b_ablation["delta_r2"] = b_ablation.ratio_r2 - b_ablation.raw_r2
    combined = pd.concat([a_ablation, b_ablation], ignore_index=True)
    combined.to_csv(TABLES_DIR / "table_14_all_feature_ablation.csv", index=False, encoding="utf-8")

    predictions = pd.read_csv(PREDICTIONS_DIR / f"{selected_key}_test.csv")
    predictions["price_band"] = pd.cut(
        predictions.y_true,
        bins=[-np.inf, 1.5, 3.0, 4.99, np.inf],
        labels=["<1.5", "1.5-3.0", "3.0-4.99", ">=4.99 (capped)"],
        right=False,
    )
    error_rows = []
    for band, group in predictions.groupby("price_band", observed=True):
        error_rows.append({
            "price_band": str(band), "n_samples": len(group),
            "mae": np.abs(group.residual).mean(),
            "rmse": np.sqrt(np.square(group.residual).mean()),
            "mean_bias": (group.y_pred - group.y_true).mean(),
        })
    pd.DataFrame(error_rows).to_csv(
        TABLES_DIR / "table_15_rf_test_error_by_price_band.csv", index=False, encoding="utf-8"
    )

    plot = validation.copy()
    plot["label"] = plot.model_key.str.replace("_", " ")
    plot = plot.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot.label, plot.rmse, color=["#D98250" if key == overall_key else "#648BBC" for key in plot.model_key])
    ax.set(xlabel="Validation RMSE (100,000 USD)", title="All model candidates on the same validation split")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for directory, dpi in ((REPORT_FIGURES_DIR, 300), (PPT_FIGURES_DIR, 220)):
        fig.savefig(directory / "21_all_model_validation_comparison.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
