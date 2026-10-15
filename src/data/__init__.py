"""Data ingestion pipelines for prices, macro, and news."""
from .fundamentals import FundamentalPipeline
from .macro import MacroPipeline
from .news.archive import HeadlineArchive
from .news.finnhub import FinnhubNewsClient
from .news.gdelt import GDELTNewsClient
from .prices import PricePipeline

__all__ = [
    "PricePipeline",
    "MacroPipeline",
    "FundamentalPipeline",
    "HeadlineArchive",
    "FinnhubNewsClient",
    "GDELTNewsClient",
]

