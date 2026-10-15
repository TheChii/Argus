import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.utils.logger import setup_logger

logger = setup_logger("normalization")


def to_percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Scales array values to uniform percentile ranks in [-0.5, 0.5]."""
    n = len(arr)
    if n <= 1:
        return np.zeros_like(arr, dtype=float)
    ranks = rankdata(arr, method="average")
    return (ranks - 1.0) / (n - 1.0) - 0.5


def rank_normalize_series(s: pd.Series) -> pd.Series:
    """
    Ranks values in series into uniform percentile values in [-0.5, 0.5].
    Ties are assigned average rank.
    """
    valid = s.dropna()
    n = len(valid)
    if n <= 1:
        return pd.Series(0.0, index=s.index)

    ranks = to_percentile_rank(valid.values)
    return pd.Series(ranks, index=valid.index).reindex(s.index).fillna(0.0)


class CrossSectionalNormalizer:
    """
    Applies daily cross-sectional rank normalization across tickers.

    Only applies to ticker-varying features.
    Market-wide macro and regime features are preserved in their native or time-standardized scale.
    """

    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns

    def normalize(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes specified feature columns date-by-date across all tickers active on that date.
        """
        if panel_df is None or panel_df.empty or "date" not in panel_df.columns:
            return pd.DataFrame() if panel_df is None else panel_df.copy()

        df = panel_df.copy()
        try:
            from src.features.fast_features import fast_cross_sectional_normalize, is_cpp_available
            if is_cpp_available():
                return fast_cross_sectional_normalize(df, self.feature_columns)
        except (ImportError, OSError, ValueError) as e:
            logger.debug("C++ fast cross-sectional normalization unavailable (%s); using Python fallback.", e)

        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df.groupby("date")[col].transform(rank_normalize_series)

        return df
