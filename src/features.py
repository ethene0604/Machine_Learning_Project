from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS


class AddRatioFeatures(BaseEstimator, TransformerMixin):
    """Add deterministic housing-density and room-ratio features.

    The transformer has no learned parameters, so it is leakage-safe. It returns
    a DataFrame to keep feature names available for coefficient analysis.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def fit(self, X: pd.DataFrame, y: object = None) -> "AddRatioFeatures":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AddRatioFeatures expects a pandas DataFrame.")
        missing = [column for column in FEATURE_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"Missing source columns for ratio features: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in FEATURE_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"Missing source columns for ratio features: {missing}")
        transformed = X.loc[:, FEATURE_COLUMNS].copy()
        transformed["BedroomRatio"] = (
            transformed["AveBedrms"] / (transformed["AveRooms"] + self.eps)
        )
        transformed["RoomsPerPerson"] = (
            transformed["AveRooms"] / (transformed["AveOccup"] + self.eps)
        )
        transformed["HouseholdsApprox"] = (
            transformed["Population"] / (transformed["AveOccup"] + self.eps)
        )
        return transformed.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(
            [*FEATURE_COLUMNS, "BedroomRatio", "RoomsPerPerson", "HouseholdsApprox"],
            dtype=object,
        )


def feature_names_for_set(feature_set: str) -> list[str]:
    if feature_set == "raw":
        return list(FEATURE_COLUMNS)
    if feature_set == "ratio":
        return [
            *FEATURE_COLUMNS,
            "BedroomRatio",
            "RoomsPerPerson",
            "HouseholdsApprox",
        ]
    raise ValueError(f"Unknown feature set: {feature_set}")


def build_preprocessor(feature_set: str) -> Pipeline:
    """Build a leakage-safe preprocessing pipeline for one feature set."""
    steps: list[tuple[str, object]] = []
    if feature_set == "ratio":
        steps.append(("ratios", AddRatioFeatures()))
    elif feature_set != "raw":
        raise ValueError(f"Unknown feature set: {feature_set}")
    steps.extend(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return Pipeline(steps)
