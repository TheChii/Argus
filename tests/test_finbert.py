"""Unit tests for FinBERT sentiment scoring and feature extraction."""
import tempfile
from pathlib import Path

import pandas as pd

from src.data.news.archive import HeadlineArchive
from src.features.sentiment import SentimentFeatureExtractor
from src.models.sentiment.scorer import FinBERTScorer


def test_finbert_scorer_caching():
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_file = Path(tmpdir) / "scores.parquet"
        scorer = FinBERTScorer(scores_path=scores_file)

        articles = [
            {"article_hash": "hash_1", "headline": "Apple reports record breaking quarterly revenue and profit."},
            {"article_hash": "hash_2", "headline": "Tech company faces severe regulatory penalties and declining sales."},
        ]

        # First run: performs inference and caches
        df1 = scorer.score_articles(articles)
        assert len(df1) == 2
        assert "sentiment_score" in df1.columns
        # Positive headline should have positive score
        row1 = df1[df1["article_hash"] == "hash_1"].iloc[0]
        row2 = df1[df1["article_hash"] == "hash_2"].iloc[0]
        assert row1["sentiment_score"] > 0.0
        assert row2["sentiment_score"] < 0.0

        # Second run with exact same articles: must read from cache
        df2 = scorer.score_articles(articles)
        assert len(df2) == 2


def test_sentiment_feature_extraction():
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_dir = Path(tmpdir) / "archive"
        scores_file = Path(tmpdir) / "scores.parquet"

        archive = HeadlineArchive(archive_dir=archive_dir)
        scorer = FinBERTScorer(scores_path=scores_file)

        # Ingest mock headline
        archive.append_articles([{
            "ticker": "AAPL",
            "headline": "Apple reports record quarterly revenue and profit beating all estimates",
            "publish_time": "2026-10-07T14:00:00Z",
            "source": "Finnhub",
        }])

        # Score headlines
        headlines = archive.load_headlines()
        scorer.score_articles(headlines.to_dict(orient="records"))

        # Extract sentiment features
        extractor = SentimentFeatureExtractor(archive=archive, scorer=scorer)
        panel_df = pd.DataFrame({
            "date": ["2026-10-07", "2026-10-07"],
            "ticker": ["AAPL", "MSFT"],
        })

        aligned = extractor.align_with_panel(panel_df)
        assert "sentiment_mean" in aligned.columns
        assert "sentiment_count" in aligned.columns

        # AAPL had 1 headline
        aapl_row = aligned[aligned["ticker"] == "AAPL"].iloc[0]
        assert aapl_row["sentiment_count"] == 1
        assert aapl_row["sentiment_mean"] > 0.0

        # MSFT had 0 headlines -> neutral
        msft_row = aligned[aligned["ticker"] == "MSFT"].iloc[0]
        assert msft_row["sentiment_count"] == 0
        assert msft_row["sentiment_mean"] == 0.0
