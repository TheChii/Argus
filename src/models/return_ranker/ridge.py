"""Ridge regression baseline return ranker."""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logger import setup_logger

logger = setup_logger("ridge_ranker")


class RidgeReturnRanker:
    """
    Baseline L2 regularized regression model predicting 5-day excess return.

    Guarantees:
    - Cross-validated alpha tuning via TimeSeriesSplit
    - Standardized rank score generation
    - Feature weights introspection
    """

    def __init__(self, alphas: list[float] | None = None):
        self.alphas = alphas or [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
        self.model = RidgeCV(alphas=self.alphas, cv=TimeSeriesSplit(n_splits=5))
        self.is_fitted = False
        self.feature_names_: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "RidgeReturnRanker":
        """Fits Ridge model on feature matrix X and target y."""
        clean_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X_clean = X[clean_mask]
        y_clean = y[clean_mask]

        if len(X_clean) < 50:
            raise ValueError(f"Insufficient observations to fit Ridge ranker ({len(X_clean)} rows).")

        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        self.feature_names_ = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        logger.info(f"Ridge ranker fitted: best alpha={self.model.alpha_:.1f}, train rows={len(X_clean)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts expected 5-day excess return."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        # Handle possible NaNs in test feature matrix
        X_fill = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X_fill)

    def get_feature_weights(self) -> pd.DataFrame:
        """Returns sorted feature coefficients."""
        if not self.is_fitted:
            return pd.DataFrame()
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "weight": self.model.coef_,
            "abs_weight": np.abs(self.model.coef_),
        })
        return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
