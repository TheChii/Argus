"""Comprehensive unit tests for Quantitative Portfolio Rules.

Tests cover:
1. Turnover buffer (hysteresis): Retain positions in [0.60, 0.90), buy only >= 0.90, sell < 0.60.
2. 8% Position Cap Redistribution across 15 names.
3. Exposure scaling under elevated VIX.
4. Confidence filter triggering cash liquidation when rolling 60d IC <= 0.
"""
import numpy as np
import pandas as pd

from src.portfolio.allocator import PortfolioAllocator


def test_turnover_buffer_hysteresis(tmp_path):
    allocator = PortfolioAllocator(
        portfolio_dir=tmp_path,
        target_holdings=5,
        buy_percentile=0.80,   # top 20% for test
        sell_percentile=0.50,  # top 50% for test
    )

    # 10 tickers SYM_0 (highest) to SYM_9 (lowest)
    # Percentiles: SYM_0 = 1.0, SYM_1 = 0.89, SYM_2 = 0.78, SYM_3 = 0.67, SYM_4 = 0.56, SYM_5 = 0.44...
    tickers = [f"SYM_{i}" for i in range(10)]
    scores = np.linspace(1.0, 0.0, 10)
    scores_df = pd.DataFrame({"ticker": tickers, "pred_score": scores})

    # Current holdings: SYM_2 (in buffer zone: pct ~0.78), SYM_8 (below sell: pct ~0.11)
    current_holdings = {"SYM_2": 0.50, "SYM_8": 0.50}

    target_df, meta = allocator.compute_target_weights(
        scores_df,
        current_holdings=current_holdings,
        use_turnover_buffer=True,
        use_exposure_scaling=False,
        use_confidence_filter=False,
        return_metadata=True,
    )

    selected = target_df["ticker"].tolist()

    # SYM_2 must be retained (buffer zone [0.50, 0.80))
    assert "SYM_2" in selected
    assert "SYM_2" in meta["retained_tickers"]

    # SYM_8 must be dropped (pct < 0.50)
    assert "SYM_8" not in selected
    assert "SYM_8" in meta["dropped_tickers"]

    # SYM_0 and SYM_1 must enter (pct >= 0.80)
    assert "SYM_0" in selected
    assert "SYM_1" in selected

    # Verify trade list produces SELL for SYM_8 first
    orders, trade_summary = allocator.generate_trade_list(target_df, current_holdings=current_holdings)
    assert trade_summary["sells_count"] >= 1
    assert orders[0]["ticker"] == "SYM_8"
    assert orders[0]["action"] == "SELL"


def test_eight_percent_cap_redistribution(tmp_path):
    allocator = PortfolioAllocator(
        portfolio_dir=tmp_path,
        target_holdings=15,
        max_weight=0.08,
    )

    # 20 tickers
    tickers = [f"SYM_{i}" for i in range(20)]
    scores = np.linspace(1.0, 0.0, 20)
    scores_df = pd.DataFrame({"ticker": tickers, "pred_score": scores})

    # Give SYM_0 very low vol (would get high raw weight without cap)
    vols = [0.05 if i == 0 else 0.30 for i in range(20)]
    vol_df = pd.DataFrame({"ticker": tickers, "predicted_vol": vols})

    target_df, meta = allocator.compute_target_weights(
        scores_df,
        vol_df=vol_df,
        top_n=15,
        use_turnover_buffer=False,
        use_exposure_scaling=False,
        return_metadata=True,
    )

    assert len(target_df) == 15
    # Strict 8% cap on all individual holdings
    for w in target_df["target_weight"]:
        assert w <= 0.0801, f"Weight {w} exceeded 8% cap"

    # Total weights should sum to 1.0 (within rounding tolerance)
    assert abs(target_df["target_weight"].sum() - 1.0) < 0.01


def test_exposure_scaling_under_high_vix(tmp_path):
    allocator = PortfolioAllocator(
        portfolio_dir=tmp_path,
        target_holdings=15,
        vix_baseline=20.0,
    )

    tickers = [f"SYM_{i}" for i in range(20)]
    scores_df = pd.DataFrame({"ticker": tickers, "pred_score": np.linspace(1.0, 0.0, 20)})

    # Test with elevated VIX = 30.0 -> exposure scale should be 20 / 30 = 0.6667
    target_df, meta = allocator.compute_target_weights(
        scores_df,
        current_vix=30.0,
        top_n=15,
        use_turnover_buffer=False,
        use_exposure_scaling=True,
        return_metadata=True,
    )

    expected_scale = 20.0 / 30.0
    assert abs(meta["exposure_scale"] - expected_scale) < 1e-3
    assert abs(target_df["target_weight"].sum() - expected_scale) < 0.02
    assert abs(meta["cash_weight"] - (1.0 - expected_scale)) < 0.02


def test_confidence_filter_triggers_cash(tmp_path):
    allocator = PortfolioAllocator(
        portfolio_dir=tmp_path,
        target_holdings=15,
        rolling_ic_threshold=0.0,
        confidence_fallback="cash",
    )

    tickers = [f"SYM_{i}" for i in range(20)]
    scores_df = pd.DataFrame({"ticker": tickers, "pred_score": np.linspace(1.0, 0.0, 20)})
    current_holdings = {"SYM_0": 0.50, "SYM_1": 0.50}

    # Rolling 60d IC is negative (-0.015)
    target_df, meta = allocator.compute_target_weights(
        scores_df,
        current_holdings=current_holdings,
        rolling_60d_ic=-0.015,
        use_confidence_filter=True,
        return_metadata=True,
    )

    assert meta["confidence_filter_triggered"] is True
    assert meta["exposure_scale"] == 0.0
    assert meta["cash_weight"] == 1.0

    # Orders must liquidate existing holdings
    orders, trade_summary = allocator.generate_trade_list(target_df, current_holdings=current_holdings)
    assert len(orders) == 2
    assert all(o["action"] == "SELL" for o in orders)
    assert all(o["target_weight"] == 0.0 for o in orders)


def test_allocator_empty_slice_return_contract(tmp_path):
    allocator = PortfolioAllocator(portfolio_dir=tmp_path)
    empty_scores = pd.DataFrame(columns=["ticker", "pred_score"])

    # Test return_metadata=False returns DataFrame, not tuple
    res_df = allocator.compute_target_weights(empty_scores, return_metadata=False)
    assert isinstance(res_df, pd.DataFrame)
    assert "metadata" in res_df.attrs
    assert res_df.attrs["metadata"]["cash_weight"] == 1.0

    # Test return_metadata=True returns (DataFrame, dict) tuple
    res_tuple = allocator.compute_target_weights(empty_scores, return_metadata=True)
    assert isinstance(res_tuple, tuple)
    assert len(res_tuple) == 2
    assert isinstance(res_tuple[0], pd.DataFrame)
    assert isinstance(res_tuple[1], dict)
    assert res_tuple[1]["cash_weight"] == 1.0
