"""Tests for Raw Headline Archive and Deduplication."""
import tempfile
from pathlib import Path

from src.data.news.archive import HeadlineArchive, compute_article_hash


def test_article_hash_deterministic():
    h1 = compute_article_hash("AAPL", "2026-10-08T15:00:00Z", "Apple releases new chip")
    h2 = compute_article_hash("AAPL", "2026-10-08T15:00:00Z", "Apple releases new chip")
    h3 = compute_article_hash("AAPL", "2026-10-08T15:00:00Z", "Apple releases different chip")

    assert h1 == h2
    assert h1 != h3


def test_archive_append_and_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = HeadlineArchive(archive_dir=Path(tmpdir))

        sample_articles = [
            {
                "ticker": "AAPL",
                "headline": "Apple reaches new highs",
                "summary": "Shares rallied in afternoon trading.",
                "publish_time": "2026-10-07T14:00:00Z",
                "source": "Finnhub",
                "url": "https://example.com/1",
            },
            {
                "ticker": "MSFT",
                "headline": "Microsoft expands cloud infrastructure",
                "summary": "New data centers announced.",
                "publish_time": "2026-10-07T18:00:00Z",  # After 4pm ET
                "source": "Finnhub",
                "url": "https://example.com/2",
            },
        ]

        # First ingestion
        res1 = archive.append_articles(sample_articles)
        assert res1["inserted"] == 2
        assert res1["duplicates"] == 0

        # Second ingestion with exact same articles -> should be skipped as duplicates
        res2 = archive.append_articles(sample_articles)
        assert res2["inserted"] == 0
        assert res2["duplicates"] == 2

        # Ingestion with one new article and one existing
        mixed_articles = [
            sample_articles[0],
            {
                "ticker": "NVDA",
                "headline": "Nvidia announces new architecture",
                "summary": "Next-generation GPU revealed.",
                "publish_time": "2026-10-08T10:00:00Z",
                "source": "Finnhub",
                "url": "https://example.com/3",
            },
        ]
        res3 = archive.append_articles(mixed_articles)
        assert res3["inserted"] == 1
        assert res3["duplicates"] == 1

        # Check retrieval
        df_all = archive.load_headlines()
        assert len(df_all) == 3
        assert set(df_all["ticker"]) == {"AAPL", "MSFT", "NVDA"}

        # Check filter by ticker
        df_aapl = archive.load_headlines(tickers=["AAPL"])
        assert len(df_aapl) == 1
        assert df_aapl.iloc[0]["headline"] == "Apple reaches new highs"

        # Check stats
        stats = archive.get_archive_stats()
        assert stats["total_headlines"] == 3
        assert stats["unique_tickers"] == 3
