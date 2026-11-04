from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeGradientDescent:
    """Ridge regression fitted with explicit full-batch gradient descent.

    Objective:
        J(w, b) = (1/m) ||Xw + b - y||_2^2 + lambda ||w||_2^2

    The intercept is not regularized. Scikit-learn's Ridge objective uses a
    residual sum of squares, so lambda = alpha / m for the same feature values.
    """

    lambda_: float
    learning_rate: float = 0.05
    max_iter: int = 5_000
    tolerance: float = 1e-7
    record_every: int = 10

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeGradientDescent":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be 2D and have the same row count as y.")
        if self.lambda_ < 0 or self.learning_rate <= 0:
            raise ValueError("lambda_ must be non-negative and learning_rate positive.")

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = float(np.mean(y))
        self.loss_history_: list[float] = []
        self.gradient_norm_history_: list[float] = []
        self.n_iter_ = 0
        converged = False

        for iteration in range(1, self.max_iter + 1):
            residual = X @ self.coef_ + self.intercept_ - y
            gradient_coef = (
                (2.0 / n_samples) * (X.T @ residual)
                + 2.0 * self.lambda_ * self.coef_
            )
            gradient_intercept = (2.0 / n_samples) * residual.sum()

            self.coef_ -= self.learning_rate * gradient_coef
            self.intercept_ -= self.learning_rate * gradient_intercept

            if iteration % self.record_every == 0 or iteration == 1:
                loss = float(
                    np.mean(residual**2)
                    + self.lambda_ * np.dot(self.coef_, self.coef_)
                )
                gradient_norm = float(
                    np.sqrt(
                        np.dot(gradient_coef, gradient_coef)
                        + gradient_intercept**2
                    )
                )
                self.loss_history_.append(loss)
                self.gradient_norm_history_.append(gradient_norm)

            self.n_iter_ = iteration
            if max(
                float(np.max(np.abs(gradient_coef))),
                abs(float(gradient_intercept)),
            ) < self.tolerance:
                converged = True
                break

        self.converged_ = converged
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def objective(self, X: np.ndarray, y: np.ndarray) -> float:
        residual = self.predict(X) - np.asarray(y, dtype=float)
        return float(
            np.mean(residual**2)
            + self.lambda_ * np.dot(self.coef_, self.coef_)
        )
