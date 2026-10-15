"""Models package containing sentiment, return rankers, volatility, sanity checks, and validation."""
from .metrics import (
    compute_daily_spearman_ic,
    compute_quintile_spread_performance,
    summarize_ic_metrics,
)
from .return_ranker.ensemble import BlendedReturnRanker, to_percentile_rank
from .return_ranker.lgbm import LightGBMReturnRanker
from .return_ranker.ridge import RidgeReturnRanker
from .sanity_checks import ModelSanityChecker
from .validation import WalkForwardValidator
from .vol_model.har import HARVolatilityModel


def __getattr__(name: str):
    if name == "FinBERTScorer":
        from .sentiment.scorer import FinBERTScorer
        return FinBERTScorer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "FinBERTScorer",
    "HARVolatilityModel",
    "RidgeReturnRanker",
    "LightGBMReturnRanker",
    "BlendedReturnRanker",
    "to_percentile_rank",
    "compute_daily_spearman_ic",
    "summarize_ic_metrics",
    "compute_quintile_spread_performance",
    "WalkForwardValidator",
    "ModelSanityChecker",
]
