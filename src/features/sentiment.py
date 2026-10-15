"""Sentiment feature extraction from scored headlines."""
import numpy as np
import pandas as pd

from src.data.news.archive import HeadlineArchive
from src.models.sentiment.scorer import FinBERTScorer
from src.utils.logger import setup_logger

logger = setup_logger("sentiment_features")


class SentimentFeatureExtractor:
    """
    Extracts daily ticker-level sentiment features from archived headlines and FinBERT scores.

    Standardized columns produced:
    - sentiment_mean: daily mean of (P(pos) - P(neg))
    - sentiment_count: number of headlines for that ticker on that trading day
    - sentiment_dispersion: standard deviation of headline sentiments
    - sentiment_3d_mean: 3-day rolling average of sentiment_mean
    """

    FEATURE_COLUMNS = [
        "sentiment_mean",
        "sentiment_count",
        "sentiment_dispersion",
        "sentiment_3d_mean",
    ]

    def __init__(
        self,
        archive: HeadlineArchive | None = None,
        scorer: FinBERTScorer | None = None
    ):
        self.archive = archive or HeadlineArchive()
        self.scorer = scorer or FinBERTScorer()

    def extract_features(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Computes sentiment features across all tickers and trading dates.

        Returns:
            DataFrame with columns: ['date', 'ticker', 'sentiment_mean', 'sentiment_count', 'sentiment_dispersion', 'sentiment_3d_mean']
        """
        headlines_df = self.archive.load_headlines(start_date=start_date, end_date=end_date, tickers=tickers)
        if headlines_df.empty:
            return pd.DataFrame(columns=["date", "ticker"] + self.FEATURE_COLUMNS)

        scores_df = self.scorer.load_scores()
        if scores_df.empty:
            logger.warning("No sentiment scores found in cache. Run scoring first.")
            return pd.DataFrame(columns=["date", "ticker"] + self.FEATURE_COLUMNS)

        # Merge headline metadata with sentiment scores
        merged = pd.merge(
            headlines_df[["article_hash", "ticker", "effective_trading_date"]],
            scores_df[["article_hash", "sentiment_score"]],
            on="article_hash",
            how="inner"
        )

        if merged.empty:
            return pd.DataFrame(columns=["date", "ticker"] + self.FEATURE_COLUMNS)

        # Group by ticker and effective_trading_date
        def agg_sentiment(group: pd.DataFrame) -> pd.Series:
            scores = group["sentiment_score"].values
            cnt = len(scores)
            mean_val = float(np.mean(scores)) if cnt > 0 else 0.0
            disp_val = float(np.std(scores)) if cnt > 1 else 0.0
            return pd.Series({
                "sentiment_mean": mean_val,
                "sentiment_count": cnt,
                "sentiment_dispersion": disp_val,
            })

        daily_agg = merged.groupby(["effective_trading_date", "ticker"]).apply(agg_sentiment, include_groups=False).reset_index()
        daily_agg = daily_agg.rename(columns={"effective_trading_date": "date"})
        daily_agg["sentiment_3d_mean"] = daily_agg["sentiment_mean"]
        return daily_agg[["date", "ticker"] + self.FEATURE_COLUMNS]

    def align_with_panel(
        self,
        panel_df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Left-joins sentiment features onto a continuous panel DataFrame (containing 'date' and 'ticker').
        Fills missing news days with neutral sentiment (count=0, mean=0.0, dispersion=0.0).
        Computes 3-day rolling sentiment on the continuous panel so sentiment decays to 0.0 after 3 trading days.
        """
        if sentiment_df is None:
            sentiment_df = self.extract_features()

        result = panel_df.sort_values(["ticker", "date"]).copy()

        if sentiment_df.empty:
            for col in self.FEATURE_COLUMNS:
                result[col] = 0.0
            return result.sort_values(["date", "ticker"]).reset_index(drop=True)

        base_cols = ["sentiment_mean", "sentiment_count", "sentiment_dispersion"]
        merged = pd.merge(result, sentiment_df[["date", "ticker"] + base_cols], on=["date", "ticker"], how="left")

        # Days without news receive 0 count, 0 mean, 0 dispersion
        merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
        merged["sentiment_count"] = merged["sentiment_count"].fillna(0.0)
        merged["sentiment_dispersion"] = merged["sentiment_dispersion"].fillna(0.0)

        # 3-trading-day rolling average per ticker on the continuous panel timeline
        merged["sentiment_3d_mean"] = merged.groupby("ticker")["sentiment_mean"].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )

        return merged.sort_values(["date", "ticker"]).reset_index(drop=True)
