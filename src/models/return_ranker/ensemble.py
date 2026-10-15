"""Ensemble return ranker blending Ridge and multi-seed LightGBM 50/50."""
import numpy as np
import pandas as pd

from src.features.normalization import to_percentile_rank
from src.models.return_ranker.lgbm import LightGBMReturnRanker
from src.models.return_ranker.ridge import RidgeReturnRanker
from src.utils.logger import setup_logger

logger = setup_logger("blended_ranker")


class BlendedReturnRanker:
    """
    Blends Ridge baseline and multi-seed LightGBM predictions 50/50.

    Combines cross-sectional rank percentiles to prevent scale mismatch:
    Rank_final = 0.5 * Rank(Ridge) + 0.5 * Rank(LightGBM)
    """

    def __init__(
        self,
        ridge_ranker: RidgeReturnRanker | None = None,
        lgbm_ranker: LightGBMReturnRanker | None = None,
        ridge_weight: float = 0.5,
        lgbm_weight: float = 0.5,
    ):
        self.ridge_ranker = ridge_ranker or RidgeReturnRanker()
        self.lgbm_ranker = lgbm_ranker or LightGBMReturnRanker()
        self.ridge_weight = ridge_weight
        self.lgbm_weight = lgbm_weight
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "BlendedReturnRanker":
        """Fits both Ridge and LightGBM models on training data."""
        logger.info("Fitting Ridge baseline in ensemble...")
        self.ridge_ranker.fit(X, y, feature_names=feature_names)

        logger.info("Fitting Multi-Seed LightGBM in ensemble...")
        self.lgbm_ranker.fit(X, y, feature_names=feature_names)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, dates: np.ndarray | None = None) -> np.ndarray:
        """
        Generates blended prediction scores using percentile rank weighting.
        If `dates` is provided, computes percentile ranks strictly cross-sectionally per date.
        """
        if not self.is_fitted:
            raise ValueError("Blended model is not fitted yet.")

        pred_ridge = self.ridge_ranker.predict(X)
        pred_lgbm = self.lgbm_ranker.predict(X)

        if dates is not None and len(dates) == len(X):
            # Per-date cross-sectional ranking
            rank_ridge = np.zeros_like(pred_ridge, dtype=float)
            rank_lgbm = np.zeros_like(pred_lgbm, dtype=float)
            unique_d = np.unique(dates)
            for d in unique_d:
                idx = np.where(dates == d)[0]
                rank_ridge[idx] = to_percentile_rank(pred_ridge[idx])
                rank_lgbm[idx] = to_percentile_rank(pred_lgbm[idx])
        else:
            rank_ridge = to_percentile_rank(pred_ridge)
            rank_lgbm = to_percentile_rank(pred_lgbm)

        return self.ridge_weight * rank_ridge + self.lgbm_weight * rank_lgbm

    def predict_panel(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Generates date-by-date cross-sectional predictions for Ridge, LightGBM, and the 50/50 Blend.
        """
        out_df = df.copy()
        X = df[feature_cols].values

        pred_ridge = self.ridge_ranker.predict(X)
        pred_lgbm = self.lgbm_ranker.predict(X)

        out_df["pred_ridge"] = pred_ridge
        out_df["pred_lgbm"] = pred_lgbm

        # Group by date to perform date-level cross-sectional rank blending
        def blend_group(grp: pd.DataFrame) -> pd.Series:
            r_rank = to_percentile_rank(grp["pred_ridge"].values)
            l_rank = to_percentile_rank(grp["pred_lgbm"].values)
            return pd.Series(self.ridge_weight * r_rank + self.lgbm_weight * l_rank, index=grp.index)

        out_df["pred_blend"] = out_df.groupby(date_col, group_keys=False).apply(blend_group)
        return out_df
