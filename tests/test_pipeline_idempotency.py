"""Tests for DailyPipelineOrchestrator idempotency and safety."""
import tempfile
from pathlib import Path

import pandas as pd

from src.pipeline import DailyPipelineOrchestrator


def test_orchestrator_idempotency():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        orchestrator = DailyPipelineOrchestrator(data_dir=data_dir)

        # Pre-seed one ticker's prices
        prices_dir = data_dir / "prices"
        prices_dir.mkdir(parents=True, exist_ok=True)
        mock_price = pd.DataFrame({
            "date": ["2026-10-06", "2026-10-07"],
            "ticker": ["SPY", "SPY"],
            "open": [400.0, 401.0],
            "high": [402.0, 403.0],
            "low": [399.0, 400.0],
            "close": [401.5, 402.5],
            "volume": [50000000, 45000000],
        })
        mock_price.to_parquet(prices_dir / "SPY.parquet", index=False)

        # Pre-seed macro data
        macro_dir = data_dir / "macro"
        macro_dir.mkdir(parents=True, exist_ok=True)
        mock_macro = pd.DataFrame({
            "date": ["2026-10-06", "2026-10-07"],
            "T10Y2Y": [0.45, 0.44],
            "DFF": [5.33, 5.33],
            "VIXCLS": [15.2, 15.0],
            "BAMLH0A0HYM2": [3.15, 3.12],
        })
        mock_macro.to_parquet(macro_dir / "fred_macro.parquet", index=False)

        # Pre-seed headlines
        news_dir = data_dir / "news" / "archive"
        news_dir.mkdir(parents=True, exist_ok=True)
        sample_articles = [{
            "ticker": "AAPL",
            "headline": "Sample Apple Headline",
            "publish_time": "2026-10-07T12:00:00Z",
            "source": "Finnhub",
        }]
        orchestrator.headline_archive.append_articles(sample_articles)

        # Initial archive size
        initial_news_count = orchestrator.headline_archive.get_archive_stats()["total_headlines"]
        assert initial_news_count == 1

        # Check status summary
        status1 = orchestrator.get_status_summary()
        assert status1["macro"]["cached"] is True
        assert status1["news_archive"]["total_headlines"] == 1

        # Re-append the exact same article
        res_reappend = orchestrator.headline_archive.append_articles(sample_articles)
        assert res_reappend["inserted"] == 0
        assert res_reappend["duplicates"] == 1

        # Verify archive size did not change
        status2 = orchestrator.get_status_summary()
        assert status2["news_archive"]["total_headlines"] == 1
