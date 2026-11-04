from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .config import (
    FEATURE_COLUMNS,
    PPT_FIGURES_DIR,
    REPORT_FIGURES_DIR,
    TARGET_COLUMN,
    TARGET_UNIT_LABEL,
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    }
)

FEATURE_LABELS = {
    "MedInc": "Median income",
    "HouseAge": "House age",
    "AveRooms": "Average rooms",
    "AveBedrms": "Average bedrooms",
    "Population": "Population",
    "AveOccup": "Average occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "MedHouseVal": "Median house value",
    "BedroomRatio": "Bedroom ratio",
    "RoomsPerPerson": "Rooms per person",
    "HouseholdsApprox": "Approx. households",
}


def _save(figure: Figure, stem: str, include_ppt: bool = True) -> None:
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        REPORT_FIGURES_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    if include_ppt:
        PPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            PPT_FIGURES_DIR / f"{stem}.png",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def plot_feature_distributions(train_frame: pd.DataFrame) -> None:
    columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    ncols = 3
    nrows = math.ceil(len(columns) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for axis, column in zip(axes, columns):
        sns.histplot(
            train_frame[column],
            bins=40,
            color="#3572A5",
            edgecolor="white",
            linewidth=0.3,
            ax=axis,
        )
        axis.set_title(FEATURE_LABELS[column])
        axis.set_xlabel(FEATURE_LABELS[column])
        axis.set_ylabel("Count")
    for axis in axes[len(columns) :]:
        axis.axis("off")
    figure.suptitle("Training-set feature and target distributions", x=0.01, ha="left")
    figure.tight_layout()
    _save(figure, "01_feature_distributions")


def plot_feature_boxplots(train_frame: pd.DataFrame) -> None:
    ncols = 4
    nrows = math.ceil(len(FEATURE_COLUMNS) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for axis, column in zip(axes, FEATURE_COLUMNS):
        sns.boxplot(x=train_frame[column], color="#5BA3D0", width=0.45, ax=axis)
        axis.set_title(FEATURE_LABELS[column])
        axis.set_xlabel(FEATURE_LABELS[column])
    for axis in axes[len(FEATURE_COLUMNS) :]:
        axis.axis("off")
    figure.suptitle("Training-set feature boxplots", x=0.01, ha="left")
    figure.tight_layout()
    _save(figure, "02_feature_boxplots", include_ppt=False)


def plot_correlation_heatmap(train_frame: pd.DataFrame) -> None:
    correlation = train_frame[[*FEATURE_COLUMNS, TARGET_COLUMN]].corr()
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,
    )
    axis.set_title("Training-set correlation matrix")
    figure.tight_layout()
    _save(figure, "03_correlation_heatmap")


def plot_geographic_prices(train_frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    scatter = axis.scatter(
        train_frame["Longitude"],
        train_frame["Latitude"],
        c=train_frame[TARGET_COLUMN],
        s=5,
        alpha=0.55,
        cmap="viridis",
        linewidths=0,
    )
    axis.set_title("Training-set geographic distribution of house values")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_aspect("equal", adjustable="box")
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.85)
    colorbar.set_label(f"Median house value ({TARGET_UNIT_LABEL})")
    figure.tight_layout()
    _save(figure, "04_geographic_price_scatter")


def plot_model_comparison(ranking: pd.DataFrame) -> None:
    frame = ranking.copy()
    frame["display_name"] = (
        frame["model"]
        .str.replace(" Regression", "", regex=False)
        + "\n"
        + frame["feature_set"].map({"raw": "Raw", "ratio": "Ratio features"})
    )
    frame = frame.sort_values("validation_rmse")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.barplot(
        data=frame,
        y="display_name",
        x="validation_rmse",
        hue="model",
        dodge=False,
        palette="Blues_d",
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Validation RMSE (lower is better)")
    axes[0].set_xlabel(f"RMSE ({TARGET_UNIT_LABEL})")
    axes[0].set_ylabel("")

    sns.barplot(
        data=frame,
        y="display_name",
        x="validation_r2",
        hue="model",
        dodge=False,
        palette="Greens_d",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Validation R² (higher is better)")
    axes[1].set_xlabel("R²")
    axes[1].set_ylabel("")
    figure.suptitle("Linear-model validation comparison")
    figure.tight_layout()
    _save(figure, "05_model_validation_comparison")


def plot_prediction_diagnostics(
    predictions: pd.DataFrame,
    split_name: str,
    model_label: str,
    stem: str,
    include_ppt: bool,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    lower = float(
        min(predictions["y_true"].min(), predictions["y_pred"].min())
    )
    upper = float(
        max(predictions["y_true"].max(), predictions["y_pred"].max())
    )
    axes[0].scatter(
        predictions["y_true"],
        predictions["y_pred"],
        s=8,
        alpha=0.35,
        color="#2878B5",
        edgecolors="none",
    )
    axes[0].plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
        color="#D1495B",
        label="y = x",
    )
    axes[0].set_title("Actual vs. predicted")
    axes[0].set_xlabel(f"Actual ({TARGET_UNIT_LABEL})")
    axes[0].set_ylabel(f"Predicted ({TARGET_UNIT_LABEL})")
    axes[0].legend()

    axes[1].scatter(
        predictions["y_pred"],
        predictions["residual"],
        s=8,
        alpha=0.35,
        color="#5A8F3D",
        edgecolors="none",
    )
    axes[1].axhline(0, linestyle="--", linewidth=1.5, color="#D1495B")
    axes[1].set_title("Residuals vs. predicted")
    axes[1].set_xlabel(f"Predicted ({TARGET_UNIT_LABEL})")
    axes[1].set_ylabel(f"Residual = actual - predicted ({TARGET_UNIT_LABEL})")

    figure.suptitle(f"{model_label}: {split_name.capitalize()}-set diagnostics")
    figure.tight_layout()
    _save(figure, stem, include_ppt=include_ppt)


def plot_ridge_alpha_curves(cv_results: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for (model_key, feature_set), group in cv_results.groupby(
        ["model_key", "feature_set"], sort=False
    ):
        axis.errorbar(
            group["alpha"],
            group["mean_validation_rmse"],
            yerr=group["std_validation_rmse"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=f"{feature_set}: {model_key}",
        )
    axis.set_xscale("log")
    axis.set_title("Ridge alpha selection with 5-fold cross-validation")
    axis.set_xlabel("Alpha")
    axis.set_ylabel(f"Cross-validation RMSE ({TARGET_UNIT_LABEL})")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save(figure, "06_ridge_alpha_curves")


def plot_feature_engineering_ablation(metrics: pd.DataFrame) -> None:
    data = metrics[metrics["split"].isin(["validation", "test"])].copy()
    raw = data[data["feature_set"] == "raw"].set_index(
        ["model", "split"]
    )["rmse"]
    ratio = data[data["feature_set"] == "ratio"].set_index(
        ["model", "split"]
    )["rmse"]
    comparison = (
        (ratio - raw)
        .dropna()
        .rename("delta_rmse")
        .reset_index()
    )
    comparison["split"] = comparison["split"].map(
        {"validation": "Validation", "test": "Test"}
    )
    figure, axis = plt.subplots(figsize=(9, 5.2))
    sns.barplot(
        data=comparison,
        x="model",
        y="delta_rmse",
        hue="split",
        palette="Set2",
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_title("Effect of ratio features on RMSE")
    axis.set_xlabel("")
    axis.set_ylabel(f"Ratio-feature RMSE minus raw RMSE ({TARGET_UNIT_LABEL})")
    axis.text(
        0.01,
        -0.16,
        "Negative values indicate improvement from engineered features.",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout()
    _save(figure, "07_feature_engineering_ablation")


def plot_standardized_coefficients(candidate: dict[str, object]) -> None:
    names = list(candidate["feature_names"])
    coefficients = np.asarray(candidate["coefficients"], dtype=float)
    frame = (
        pd.DataFrame({"feature": names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda value: value["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=True)
    )
    colors = np.where(frame["coefficient"] >= 0, "#2878B5", "#D1495B")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        frame["feature"].map(FEATURE_LABELS),
        frame["coefficient"],
        color=colors,
    )
    axis.axvline(0, linewidth=1, color="black")
    axis.set_title(f"Standardized coefficients: {candidate['label']} ({candidate['feature_set']})")
    axis.set_xlabel("Coefficient on standardized feature")
    axis.set_ylabel("")
    figure.tight_layout()
    _save(figure, "08_selected_model_coefficients", include_ppt=False)


def plot_split_metrics(
    metrics: pd.DataFrame, selected_key: str, selected_label: str
) -> None:
    data = metrics[metrics["model_key"] == selected_key].copy()
    data["split"] = pd.Categorical(
        data["split"], categories=["train", "validation", "test"], ordered=True
    )
    data = data.sort_values("split")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sns.barplot(
        data=data,
        x="split",
        y="rmse",
        color="#2878B5",
        ax=axes[0],
    )
    axes[0].set_title("RMSE by split")
    axes[0].set_xlabel("")
    axes[0].set_ylabel(f"RMSE ({TARGET_UNIT_LABEL})")
    sns.barplot(
        data=data,
        x="split",
        y="r2",
        color="#5A8F3D",
        ax=axes[1],
    )
    axes[1].set_title("R² by split")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R²")
    figure.suptitle(f"Selected validation model: {selected_label}")
    figure.tight_layout()
    _save(figure, "09_selected_model_split_metrics", include_ppt=False)


def plot_scratch_loss(scratch_result: dict[str, object]) -> None:
    history = np.asarray(scratch_result["model"].loss_history_, dtype=float)
    steps = np.arange(1, len(history) + 1) * scratch_result["model"].record_every
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(steps, history, color="#2878B5", linewidth=2)
    axis.set_yscale("log")
    axis.set_title("NumPy Ridge gradient-descent training objective")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Training objective (log scale)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save(figure, "10_from_scratch_training_loss")


def plot_scratch_comparison(comparison: pd.DataFrame) -> None:
    data = comparison.melt(
        id_vars=["split"],
        value_vars=["sklearn_rmse", "scratch_rmse"],
        var_name="implementation",
        value_name="rmse",
    )
    data["implementation"] = data["implementation"].map(
        {"sklearn_rmse": "Scikit-learn", "scratch_rmse": "NumPy GD"}
    )
    figure, axis = plt.subplots(figsize=(8.5, 5))
    sns.barplot(
        data=data,
        x="split",
        y="rmse",
        hue="implementation",
        palette="Set1",
        ax=axis,
    )
    axis.set_title("Scikit-learn and NumPy Ridge RMSE comparison")
    axis.set_xlabel("")
    axis.set_ylabel(f"RMSE ({TARGET_UNIT_LABEL})")
    figure.tight_layout()
    _save(figure, "11_from_scratch_comparison")


def plot_error_by_price_band(error_band: pd.DataFrame) -> None:
    data = error_band.copy()
    order = ["<1.5", "1.5-3.0", "3.0-4.99", ">=4.99 (capped)"]
    data["price_band"] = pd.Categorical(data["price_band"], categories=order, ordered=True)
    data = data.sort_values("price_band")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(
        data=data,
        x="price_band",
        y="rmse",
        color="#2878B5",
        ax=axes[0],
    )
    axes[0].set_title("Test RMSE by true-price band")
    axes[0].set_xlabel("True house-value band")
    axes[0].set_ylabel(f"RMSE ({TARGET_UNIT_LABEL})")
    axes[0].tick_params(axis="x", rotation=15)

    colors = np.where(data["mean_bias"] >= 0, "#D1495B", "#2878B5")
    axes[1].bar(
        data["price_band"].astype(str),
        data["mean_bias"],
        color=colors,
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Mean prediction bias by true-price band")
    axes[1].set_xlabel("True house-value band")
    axes[1].set_ylabel(f"Mean bias = predicted - actual ({TARGET_UNIT_LABEL})")
    axes[1].tick_params(axis="x", rotation=15)
    figure.tight_layout()
    _save(figure, "15_error_by_price_band", include_ppt=False)
