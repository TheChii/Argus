"""Tests for Universe and Ticker Normalization."""
from src.config.universe import (
    BENCHMARK_TICKER,
    denormalize_ticker,
    get_all_tickers,
    get_sector_etfs,
    get_sp100_constituents,
    normalize_ticker_for_yfinance,
)


def test_sp100_constituents_count():
    constituents = get_sp100_constituents()
    assert len(constituents) >= 100, f"Expected at least 100 constituents, got {len(constituents)}"

    # Check sample ticker fields
    sample = constituents[0]
    assert "symbol" in sample
    assert "yf_symbol" in sample
    assert "sector" in sample
    assert "sector_etf" in sample


def test_sector_etfs_mapping():
    etfs = get_sector_etfs()
    assert len(etfs) == 11
    assert "XLK" in etfs
    assert "XLF" in etfs
    assert "XLV" in etfs
    assert "XLE" in etfs


def test_ticker_normalization():
    assert normalize_ticker_for_yfinance("BRK.B") == "BRK-B"
    assert normalize_ticker_for_yfinance("BF.B") == "BF-B"
    assert normalize_ticker_for_yfinance("AAPL") == "AAPL"

    assert denormalize_ticker("BRK-B") == "BRK.B"
    assert denormalize_ticker("BF-B") == "BF.B"
    assert denormalize_ticker("AAPL") == "AAPL"


def test_all_tickers_contains_benchmarks():
    all_tickers = get_all_tickers()
    assert BENCHMARK_TICKER in all_tickers
    for etf in get_sector_etfs():
        assert etf in all_tickers
    assert "AAPL" in all_tickers
    assert "MSFT" in all_tickers
