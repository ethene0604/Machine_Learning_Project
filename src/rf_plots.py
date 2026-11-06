"""Report and slide figures for Random Forest experiments."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import PPT_FIGURES_DIR, REPORT_FIGURES_DIR, TARGET_UNIT_LABEL


def _save(fig, stem: str) -> None:
    for directory, dpi in ((REPORT_FIGURES_DIR, 300), (PPT_FIGURES_DIR, 220)):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rf_validation(metrics: pd.DataFrame) -> None:
    frame = metrics[metrics["split"] == "validation"].copy()
    frame["label"] = frame["model_key"].str.replace("rf_", "RF ", regex=False).str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(frame["label"], frame["rmse"], color=["#648BBC" if "optimized" not in x else "#D98250" for x in frame["model_key"]])
    ax.set_ylabel(f"Validation RMSE ({TARGET_UNIT_LABEL})")
    ax.set_title("Random Forest: fixed validation set")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "16_rf_validation_comparison")


def plot_rf_complexity(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["depth_label"], frame["train_rmse"], marker="o", label="Train")
    ax.plot(frame["depth_label"], frame["validation_rmse"], marker="o", label="Validation")
    ax.set(xlabel="Maximum tree depth", ylabel=f"RMSE ({TARGET_UNIT_LABEL})", title="Random Forest complexity (other settings fixed)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    _save(fig, "17_rf_depth_complexity")


def plot_rf_importance(frame: pd.DataFrame, feature_set: str) -> None:
    frame = frame.sort_values("permutation_rmse_increase")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(frame["feature"], frame["impurity_importance"], color="#648BBC")
    axes[0].set_xlabel("Impurity importance (sum = 1)")
    axes[0].set_title("Impurity-based importance")
    axes[1].barh(frame["feature"], frame["permutation_rmse_increase"], xerr=frame["permutation_std"], color="#D98250")
    axes[1].set_xlabel(f"Validation RMSE increase ({TARGET_UNIT_LABEL})")
    axes[1].set_title("Permutation importance")
    fig.suptitle(f"Selected Random Forest ({feature_set} features): validation set")
    fig.tight_layout()
    _save(fig, "18_rf_feature_importance")
