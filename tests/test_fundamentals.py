"""Unit tests for Fundamental Valuations Pipeline."""
import numpy as np

from src.data.fundamentals import FundamentalPipeline


def test_fallback_fundamentals_deterministic():
    pipeline = FundamentalPipeline()
    f1 = pipeline.get_fallback_fundamentals("AAPL")
    f2 = pipeline.get_fallback_fundamentals("AAPL")

    assert f1["ticker"] == "AAPL"
    assert f1["sector"] == "Information Technology"
    assert f1 == f2, "Fallback fundamentals must be strictly deterministic"

    # Verify valuation columns
    for col in [
        "earnings_yield",
        "book_to_price",
        "sales_to_price",
        "fcf_yield",
        "dividend_yield",
        "ev_to_ebitda",
        "market_cap",
        "market_cap_log",
    ]:
        assert col in f1, f"Missing fundamental metric {col}"
        assert f1[col] > 0, f"Expected positive metric for {col}"

    # Verify different tickers in same sector have distinct perturbed values
    msft_f = pipeline.get_fallback_fundamentals("MSFT")
    assert msft_f["sector"] == "Information Technology"
    assert msft_f["earnings_yield"] != f1["earnings_yield"]


def test_fundamental_pipeline_caching(tmp_path):
    pipeline = FundamentalPipeline(fundamentals_dir=tmp_path)

    # Sync a subset of tickers
    test_tickers = ["AAPL", "MSFT", "JPM"]
    df = pipeline.sync_universe(tickers=test_tickers, force=True)

    assert len(df) == 3
    assert set(df["ticker"]) == set(test_tickers)
    assert pipeline.get_panel_parquet_path().exists()

    # Load from parquet cache
    loaded_df = pipeline.load_fundamentals(tickers=test_tickers)
    assert len(loaded_df) == 3
    np.testing.assert_allclose(
        df["earnings_yield"].values,
        loaded_df["earnings_yield"].values,
        atol=1e-5,
    )


def test_fundamental_pipeline_offline_fallback(tmp_path, monkeypatch):
    pipeline = FundamentalPipeline(fundamentals_dir=tmp_path)

    # Force live fetch to raise an exception
    def broken_fetch(ticker):
        raise ConnectionError("Simulated offline environment")

    monkeypatch.setattr(pipeline, "fetch_ticker_fundamentals", broken_fetch)

    # sync_ticker should gracefully fall back to deterministic sector benchmarks
    res = pipeline.sync_ticker("NVDA", force=True)
    assert res["ticker"] == "NVDA"
    assert res["sector"] == "Information Technology"
    assert res["earnings_yield"] > 0
