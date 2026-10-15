"""Unit tests for Historical News Backfiller."""
from datetime import datetime
from unittest.mock import MagicMock

from src.data.news.archive import HeadlineArchive
from src.data.news.backfill import HistoricalNewsBackfiller


def test_window_generation():
    """Verifies generation of contiguous historical windows."""
    backfiller = HistoricalNewsBackfiller()
    windows = backfiller._generate_windows(days_back=365, num_windows=2)

    assert len(windows) == 2
    # Verify format YYYY-MM-DD
    for start, end in windows:
        assert len(start) == 10
        assert len(end) == 10
        assert start < end

    # Window 0 end should match today
    today_str = datetime.now().strftime("%Y-%m-%d")
    assert windows[0][1] == today_str


def test_backfill_tickers_ingestion(tmp_path):
    """Verifies that backfill fetches from client and writes into archive."""
    archive = HeadlineArchive(archive_dir=tmp_path / "archive")
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True

    # Mock return articles for AAPL across 2 windows
    now_epoch = int(datetime.now().timestamp())
    mock_client.fetch_company_news.side_effect = [
        [
            {"ticker": "AAPL", "headline": "Apple Unveils New AI Features", "publish_time": now_epoch - 1000},
            {"ticker": "AAPL", "headline": "Apple Beats Quarterly Estimates", "publish_time": now_epoch - 2000},
        ],
        [
            # Includes duplicate from earlier plus older article
            {"ticker": "AAPL", "headline": "Apple Beats Quarterly Estimates", "publish_time": now_epoch - 2000},
            {"ticker": "AAPL", "headline": "Apple Suppliers Report Strong Demand", "publish_time": now_epoch - 500000},
        ]
    ]

    backfiller = HistoricalNewsBackfiller(archive=archive, finnhub_client=mock_client)
    res = backfiller.backfill_tickers(tickers=["AAPL"], days_back=180, num_windows=2)

    assert res["tickers_processed"] == 1
    assert res["total_articles_fetched"] == 4
    # 3 unique articles inserted, 1 duplicate skipped
    assert res["total_inserted"] == 3
    assert res["total_duplicates"] == 1
    assert res["archive_total_headlines"] == 3
