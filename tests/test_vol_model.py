"""Unit tests for HAR Volatility Model."""
import numpy as np
import pandas as pd

from src.models.vol_model.har import HARVolatilityModel


def test_har_volatility_model():
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2026-01-01", periods=n, freq="B").strftime("%Y-%m-%d")

    # Generate synthetic realistic volatility inputs
    ret1 = np.random.normal(0.0005, 0.015, n)
    roll_vol = np.random.uniform(0.12, 0.35, n)
    vix = np.random.uniform(14.0, 30.0, n)
    # Synthetic target vol with positive relationship to past vol and VIX
    target_vol = 0.5 * roll_vol + 0.5 * (vix / 100.0) + np.random.normal(0, 0.02, n)
    target_vol = np.clip(target_vol, 0.05, 0.80)

    df = pd.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * n,
        "return_1d": ret1,
        "return_60d": np.random.normal(0.03, 0.08, n),
        "rolling_vol_20d": roll_vol,
        "vix": vix,
        "target_realized_vol_5d": target_vol,
    })

    model = HARVolatilityModel()
    stats = model.fit(df)

    assert model.is_fitted
    assert stats["r2"] > 0.0  # Positive predictive power
    assert stats["rmse"] < 0.20

    # Predictions check
    preds = model.predict(df)
    assert len(preds) == n
    # Must be strictly positive and bounded
    assert (preds > 0.0).all()
    assert (preds >= 0.05).all()
    assert (preds <= 1.50).all()


def test_har_volatility_model_multi_ticker():
    # Multi-ticker stacked panel test: verifies no cross-ticker rolling contamination
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    tickers = ["AAPL", "MSFT", "GOOGL"]

    records = []
    for d in dates:
        for t in tickers:
            records.append({
                "date": d,
                "ticker": t,
                "return_1d": 0.01,
                "rolling_vol_5d": 0.18,
                "rolling_vol_20d": 0.22,
                "vix": 16.0,
            })
    panel_df = pd.DataFrame(records)

    model = HARVolatilityModel()
    har_features = model.extract_har_features(panel_df)
    assert len(har_features) == len(panel_df)
    assert "log_rv_weekly" in har_features.columns
    # All values should be finite and close to log(0.18)
    assert np.isfinite(har_features["log_rv_weekly"].values).all()
    np.testing.assert_allclose(har_features["log_rv_weekly"].values, np.log(0.18), atol=1e-3)
