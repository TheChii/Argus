"""Unit tests for C++ accelerated feature extraction and normalization."""
import numpy as np
import pandas as pd
import pytest

from src.features.fast_features import (
    fast_compute_panel_technical,
    fast_compute_ticker_technical,
    fast_cross_sectional_normalize,
    is_cpp_available,
)
from src.features.normalization import CrossSectionalNormalizer
from src.features.technical import TechnicalFeatureExtractor, compute_ticker_technical_features


def test_cpp_library_loaded():
    """Verify that libfast_features.so is compiled and loaded."""
    assert is_cpp_available(), "libfast_features.so should be available and loaded"


def test_fast_compute_ticker_technical_matches_python():
    """Verify numerical parity between C++ single-ticker technical features and Python."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
    volumes = np.random.uniform(500000, 5000000, n)

    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "open": prices * 0.995,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": volumes,
    })

    py_res = compute_ticker_technical_features(df)
    cpp_res = fast_compute_ticker_technical(df)

    cols_to_check = [
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "rolling_vol_5d",
        "rolling_vol_20d",
        "volume_zscore_20d",
        "dist_from_ma50",
        "rsi_14d",
    ]

    for col in cols_to_check:
        py_vals = py_res[col].values
        cpp_vals = cpp_res[col].values

        assert np.array_equal(np.isnan(py_vals), np.isnan(cpp_vals)), f"NaN mismatch in {col}"

        valid = ~np.isnan(py_vals)
        assert np.allclose(py_vals[valid], cpp_vals[valid], atol=1e-5, rtol=1e-5), f"Value mismatch in {col}"


def test_fast_compute_panel_technical_matches_python():
    """Verify panel-level parallel C++ extractor matches Python extractor."""
    np.random.seed(123)
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]
    n_days = 150
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()

    rows = []
    for ticker in tickers:
        prices = 150.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, n_days)))
        vols = np.random.uniform(1e6, 1e7, n_days)
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": prices[i] * 0.99,
                "high": prices[i] * 1.02,
                "low": prices[i] * 0.98,
                "close": prices[i],
                "volume": vols[i],
            })

    panel_df = pd.DataFrame(rows)

    extractor = TechnicalFeatureExtractor()
    py_panel = extractor.extract_panel(panel_df)
    cpp_panel = fast_compute_panel_technical(panel_df)

    assert len(py_panel) == len(cpp_panel)
    for col in extractor.FEATURE_COLUMNS:
        py_vals = py_panel[col].values
        cpp_vals = cpp_panel[col].values
        valid = ~np.isnan(py_vals)
        assert np.array_equal(np.isnan(py_vals), np.isnan(cpp_vals)), f"NaN mismatch in {col}"
        assert np.allclose(py_vals[valid], cpp_vals[valid], atol=1e-5, rtol=1e-5), f"Value mismatch in {col}"


def test_fast_cross_sectional_normalize_matches_python():
    """Verify C++ cross-sectional rank normalization matches Python."""
    np.random.seed(999)
    n_tickers = 50
    n_dates = 40
    tickers = [f"TICK_{i}" for i in range(n_tickers)]
    dates = [f"2023-01-{i+1:02d}" for i in range(n_dates)]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "feat_a": np.random.normal(0, 1),
                "feat_b": np.random.exponential(2),
                # Add duplicate/tied values and occasional NaNs to test edge cases
                "feat_c": np.random.choice([1.0, 2.0, 2.0, 5.0, np.nan], p=[0.2, 0.3, 0.3, 0.1, 0.1]),
            })

    test_df = pd.DataFrame(rows)
    feature_cols = ["feat_a", "feat_b", "feat_c"]

    normalizer = CrossSectionalNormalizer(feature_columns=feature_cols)
    py_norm = normalizer.normalize(test_df)
    cpp_norm = fast_cross_sectional_normalize(test_df, feature_cols=feature_cols)

    for col in feature_cols:
        py_vals = py_norm[col].values
        cpp_vals = cpp_norm[col].values
        assert np.allclose(py_vals, cpp_vals, atol=1e-6, rtol=1e-6), f"Mismatch in normalized {col}"


def test_flat_price_rsi():
    """Verify RSI handles flat prices (all 50.0) identically to Python."""
    n = 50
    dates = pd.date_range("2021-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({
        "date": dates,
        "ticker": "FLAT",
        "open": 100.0,
        "high": 100.0,
        "low": 100.0,
        "close": 100.0,
        "volume": 1000.0,
    })

    py_res = compute_ticker_technical_features(df)
    cpp_res = fast_compute_ticker_technical(df)

    assert np.allclose(py_res["rsi_14d"].dropna().values, cpp_res["rsi_14d"].dropna().values, atol=1e-6)
    assert np.all(cpp_res["rsi_14d"].dropna().values == 50.0)


def test_empty_and_none_inputs():
    """Verify safe handling of empty and None inputs without crashing."""
    assert fast_compute_ticker_technical(None).empty
    assert fast_compute_ticker_technical(pd.DataFrame()).empty
    assert fast_compute_panel_technical(None).empty
    assert fast_compute_panel_technical(pd.DataFrame()).empty
    assert fast_cross_sectional_normalize(None, ["col_a"]).empty
    assert fast_cross_sectional_normalize(pd.DataFrame(), ["col_a"]).empty


def test_missing_required_columns():
    """Verify safe fallback when required columns are absent."""
    df_missing_close = pd.DataFrame({"date": ["2023-01-01"], "volume": [1000.0]})
    res = fast_compute_ticker_technical(df_missing_close)
    assert isinstance(res, pd.DataFrame)


def test_corrupt_data_and_strings():
    """Verify non-numeric corrupt values and strings are safely coerced without crash or segfault."""
    df_corrupt = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "ticker": ["ABC"] * 4,
        "close": ["100.0", "corrupt_text", "105.0", None],
        "volume": [1000.0, -500.0, "bad_vol", 2000.0],
    })
    res = fast_compute_ticker_technical(df_corrupt)
    assert len(res) == 4
    assert "return_1d" in res.columns
    assert "rsi_14d" in res.columns


def test_infinities_and_negative_prices():
    """Verify handling of infs, -infs, zero, and negative prices."""
    df_bad_prices = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "ticker": ["BAD"] * 5,
        "close": [100.0, 0.0, -10.0, np.inf, 105.0],
        "volume": [1000.0, 0.0, -200.0, np.inf, 1500.0],
    })
    res = fast_compute_ticker_technical(df_bad_prices)
    assert len(res) == 5
    assert np.isnan(res["return_1d"].iloc[1]) or res["return_1d"].iloc[1] == -1.0


def test_duplicate_dataframe_indices_preserved():
    """Verify cross-sectional normalization preserves rows correctly even with non-unique indices."""
    df_dup = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-01", "2023-01-02", "2023-01-01"],
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "score": [50.0, 10.0, 80.0, 20.0],
    }, index=[99, 99, 42, 42])

    normed = fast_cross_sectional_normalize(df_dup, feature_cols=["score"])
    assert len(normed) == len(df_dup)
    assert list(normed.index) == list(df_dup.index)
    row_aapl_day1 = normed[(normed["ticker"] == "AAPL") & (normed["date"] == "2023-01-01")]
    row_msft_day1 = normed[(normed["ticker"] == "MSFT") & (normed["date"] == "2023-01-01")]
    assert np.isclose(row_aapl_day1["score"].iloc[0], -0.5)
    assert np.isclose(row_msft_day1["score"].iloc[0], 0.5)


def test_c_abi_null_pointer_and_bounds_safety():
    """Directly test that C++ entry points reject NULL pointers and corrupted offsets with error codes."""
    import ctypes

    from src.features.fast_features import _CPP_LIB
    if _CPP_LIB is None:
        pytest.skip("C++ library not loaded")

    ret = _CPP_LIB.compute_ticker_technical_cpp(
        None, None, 10, None, None, None, None, None, None, None, None, None
    )
    assert ret == -1, f"Expected -1 for NULL pointers, got {ret}"

    bad_offsets = (ctypes.c_int * 3)(0, 15, 10)
    dummy_double = (ctypes.c_double * 10)()
    ret_batch = _CPP_LIB.compute_panel_technical_cpp(
        dummy_double, dummy_double, bad_offsets, 2, 10,
        dummy_double, dummy_double, dummy_double, dummy_double,
        dummy_double, dummy_double, dummy_double, dummy_double, dummy_double
    )
    assert ret_batch == -2, f"Expected -2 for corrupt offsets, got {ret_batch}"


def test_relative_features_missing_benchmark_safety():
    """Verify RelativeFeatureExtractor does not fail if SPY is absent."""
    from src.features.relative import RelativeFeatureExtractor
    df_no_spy = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "ticker": ["AAPL", "AAPL"],
        "return_5d": [0.02, 0.03],
        "return_20d": [0.05, 0.06],
    })
    extractor = RelativeFeatureExtractor()
    out = extractor.extract_relative_features(df_no_spy)
    assert "rel_return_spy_5d" in out.columns
    assert not out.empty

