"""LightGBM return ranker with constrained trees and multi-seed ensembling."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("lgbm_ranker")


class LightGBMReturnRanker:
    """
    Constrained LightGBM Return Ranker with Multi-Seed Averaging.

    Guarantees:
    - Shallow trees (max_depth=3, num_leaves=8) to prevent overfitting on noisy returns
    - High leaf threshold (min_child_samples=250) reflecting overlapping 5-day return labels
    - Conservative learning rate (0.03) with row subsampling (0.7) and column subsampling (0.8)
    - Ensembles 5-10 distinct random seeds to dampen single-tree variance
    """

    def __init__(
        self,
        max_depth: int = 3,
        num_leaves: int = 8,
        min_child_samples: int = 250,
        learning_rate: float = 0.03,
        subsample: float = 0.7,
        colsample_bytree: float = 0.8,
        n_estimators: int = 300,
        n_seeds: int = 5,
        random_seeds: list[int] | None = None,
    ):
        self.max_depth = max_depth
        self.num_leaves = min(num_leaves, 2 ** max_depth)
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_estimators = n_estimators
        self.n_seeds = n_seeds
        self.random_seeds = random_seeds or [42 + i * 17 for i in range(n_seeds)]

        self.models_: list[lgb.LGBMRegressor] = []
        self.is_fitted = False
        self.feature_names_: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "LightGBMReturnRanker":
        """
        Fits multi-seed LightGBM models on feature matrix X and target y.
        """
        clean_mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X_clean = X[clean_mask]
        y_clean = y[clean_mask]

        if len(X_clean) < self.min_child_samples * 2:
            raise ValueError(f"Insufficient training observations ({len(X_clean)} rows) for min_child_samples={self.min_child_samples}.")

        self.feature_names_ = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        self.models_ = []

        for seed in self.random_seeds:
            model = lgb.LGBMRegressor(
                objective="regression",
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                subsample_freq=1,
                colsample_bytree=self.colsample_bytree,
                n_estimators=self.n_estimators,
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_clean, y_clean)
            self.models_.append(model)

        self.is_fitted = True
        logger.info(f"LightGBM fitted with {len(self.models_)} seeds (train rows={len(X_clean)}, max_depth={self.max_depth}).")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts expected 5-day excess return by averaging predictions across all seed models.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        X_fill = np.nan_to_num(X, nan=0.0)
        seed_predictions = np.zeros((len(X_fill), len(self.models_)), dtype=float)

        for i, model in enumerate(self.models_):
            seed_predictions[:, i] = model.predict(X_fill)

        # Average predictions across seeds
        return np.mean(seed_predictions, axis=1)

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Returns average feature importance (split and gain) across all seed models.
        """
        if not self.is_fitted:
            return pd.DataFrame()

        split_importances = np.mean([m.feature_importances_ for m in self.models_], axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": split_importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
