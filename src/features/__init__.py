"""Feature extraction modules."""
from .normalization import CrossSectionalNormalizer, rank_normalize_series
from .regime import MarketRegimeDetector
from .relative import RelativeFeatureExtractor
from .technical import TechnicalFeatureExtractor, compute_ticker_technical_features


def __getattr__(name: str):
    if name == "SentimentFeatureExtractor":
        from .sentiment import SentimentFeatureExtractor
        return SentimentFeatureExtractor
    if name == "DatasetBuilder":
        from .dataset import DatasetBuilder
        return DatasetBuilder
    if name == "BarraFactorModel":
        from .barra import BarraFactorModel
        return BarraFactorModel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "TechnicalFeatureExtractor",
    "compute_ticker_technical_features",
    "RelativeFeatureExtractor",
    "MarketRegimeDetector",
    "SentimentFeatureExtractor",
    "CrossSectionalNormalizer",
    "rank_normalize_series",
    "DatasetBuilder",
    "BarraFactorModel",
]
