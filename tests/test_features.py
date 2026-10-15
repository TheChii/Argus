"""Unit tests for Technical, Relative, Regime, and Normalization features."""
import numpy as np
import pandas as pd

from src.features.normalization import rank_normalize_series
from src.features.regime import MarketRegimeDetector
from src.features.technical import compute_rsi, compute_ticker_technical_features


def test_rsi_bounds():
    prices = pd.Series([100.0 + i for i in range(30)])
    rsi = compute_rsi(prices, period=14)
    # Monotonically increasing prices should have RSI close to 100
    assert rsi.dropna().iloc[-1] > 90.0

    prices_down = pd.Series([100.0 - i for i in range(30)])
    rsi_down = compute_rsi(prices_down, period=14)
    assert rsi_down.dropna().iloc[-1] < 10.0

    # Flat price series should produce neutral 50.0 RSI
    prices_flat = pd.Series([100.0] * 30)
    rsi_flat = compute_rsi(prices_flat, period=14)
    assert abs(rsi_flat.dropna().iloc[-1] - 50.0) < 1e-3


def test_ticker_technical_features():
    dates = pd.date_range("2026-01-01", periods=65, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * 65,
        "open": np.linspace(150, 180, 65),
        "high": np.linspace(151, 181, 65),
        "low": np.linspace(149, 179, 65),
        "close": np.linspace(150, 180, 65),
        "volume": np.random.uniform(1e6, 2e6, 65),
    })

    feat_df = compute_ticker_technical_features(df)
    assert "return_1d" in feat_df.columns
    assert "return_5d" in feat_df.columns
    assert "return_20d" in feat_df.columns
    assert "return_60d" in feat_df.columns
    assert "rolling_vol_5d" in feat_df.columns
    assert "rolling_vol_20d" in feat_df.columns
    assert "volume_zscore_20d" in feat_df.columns
    assert "dist_from_ma50" in feat_df.columns
    assert "rsi_14d" in feat_df.columns

    # 60d return should be populated at row 60
    assert not np.isnan(feat_df["return_60d"].iloc[60])


def test_cross_sectional_rank_normalization():
    s = pd.Series([10.0, 50.0, 20.0, 100.0, 5.0])
    normed = rank_normalize_series(s)
    # Scaled to [-0.5, 0.5]
    assert abs(normed.min() - (-0.5)) < 1e-6
    assert abs(normed.max() - 0.5) < 1e-6
    assert abs(normed.mean()) < 1e-6  # Centered at zero


def test_market_regime_detector():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100, freq="B").strftime("%Y-%m-%d").tolist()
    spy_ret = np.random.normal(0.0005, 0.01, 100)
    vix = np.random.uniform(12, 35, 100)

    detector = MarketRegimeDetector(n_states=3, random_state=42)
    res = detector.fit_predict_regimes(dates, spy_ret, vix)

    assert len(res) == 100
    assert "regime_prob_0" in res.columns
    assert "regime_prob_1" in res.columns
    assert "regime_prob_2" in res.columns
    # Probabilities should sum to 1.0 on each row
    prob_sums = res[["regime_prob_0", "regime_prob_1", "regime_prob_2"]].sum(axis=1)
    np.testing.assert_allclose(prob_sums.values, 1.0, atol=1e-5)
