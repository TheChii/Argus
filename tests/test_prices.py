"""Tests for Price pipeline caching and incremental updates."""
import tempfile
from pathlib import Path

import pandas as pd

from src.data.prices import PricePipeline


def test_price_pipeline_cache_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = PricePipeline(prices_dir=Path(tmpdir))

        # Create mock price dataframe
        mock_df = pd.DataFrame({
            "date": ["2026-10-05", "2026-10-06", "2026-10-07"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "open": [150.0, 151.0, 152.0],
            "high": [153.0, 154.0, 155.0],
            "low": [149.0, 150.0, 151.0],
            "close": [152.0, 153.0, 154.0],
            "volume": [1000000, 1100000, 1200000],
        })

        target_path = pipeline.get_ticker_parquet_path("AAPL")
        mock_df.to_parquet(target_path, index=False)

        # Verify load_ticker_prices
        loaded = pipeline.load_ticker_prices("AAPL")
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded.columns) == pipeline.REQUIRED_COLUMNS
        assert pipeline.get_latest_cached_date("AAPL") == "2026-10-07"

        # Verify load_panel
        panel = pipeline.load_panel(["AAPL"])
        assert len(panel) == 3
        assert panel["ticker"].iloc[0] == "AAPL"
