"""Unit tests for Portfolio Allocator and Position Sizing."""
import numpy as np
import pandas as pd

from src.portfolio.allocator import PortfolioAllocator


def test_inverse_vol_target_weights():
    allocator = PortfolioAllocator()
    tickers = [f"SYM_{i}" for i in range(25)]
    scores = np.linspace(0.10, -0.10, 25)

    scores_df = pd.DataFrame({
        "ticker": tickers,
        "pred_score": scores,
    })

    # High vol for SYM_0 (0.40), Low vol for SYM_1 (0.10)
    vols = [0.40 if i == 0 else (0.10 if i == 1 else 0.20) for i in range(25)]
    vol_df = pd.DataFrame({
        "ticker": tickers,
        "predicted_vol": vols,
    })

    # 1. Inverse volatility sizing (top 5 positions)
    inv_res = allocator.compute_target_weights(scores_df, vol_df=vol_df, method="inv_vol", top_n=5)
    assert len(inv_res) == 5
    assert inv_res["rank"].tolist() == [1, 2, 3, 4, 5]
    # Total target weights should sum to ~1.0
    assert abs(inv_res["target_weight"].sum() - 1.0) < 1e-3

    # Lower volatility stock (SYM_1 with 0.10 vol) must get higher weight than higher vol stock (SYM_0 with 0.40 vol)
    sym0_weight = inv_res[inv_res["ticker"] == "SYM_0"]["target_weight"].iloc[0]
    sym1_weight = inv_res[inv_res["ticker"] == "SYM_1"]["target_weight"].iloc[0]
    assert sym1_weight > sym0_weight

    # 2. Equal weight sizing
    eq_res = allocator.compute_target_weights(scores_df, method="equal", top_n=5)
    assert len(eq_res) == 5
    for w in eq_res["target_weight"]:
        assert abs(w - 0.20) < 1e-4


def test_generate_trade_list_and_holdings(tmp_path):
    allocator = PortfolioAllocator(portfolio_dir=tmp_path)

    # Initial target weights
    target_df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "NVDA"],
        "target_weight": [0.40, 0.35, 0.25],
    })

    # Case 1: Initial creation from 100% cash
    orders_1, summary_1 = allocator.generate_trade_list(target_df, current_holdings={})
    assert len(orders_1) == 3
    assert all(o["action"] == "BUY" for o in orders_1)
    assert summary_1["buys_count"] == 3
    assert summary_1["sells_count"] == 0
    assert abs(summary_1["one_way_turnover"] - 0.5) < 1e-4

    # Save new holdings
    allocator.save_holdings({"AAPL": 0.40, "MSFT": 0.35, "NVDA": 0.25})
    loaded = allocator.load_holdings()
    assert loaded["AAPL"] == 0.40

    # Case 2: Rebalance - NVDA drops out, GOOGL enters, AAPL trims
    new_target_df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "target_weight": [0.30, 0.40, 0.30],
    })

    orders_2, summary_2 = allocator.generate_trade_list(new_target_df)
    # Expected orders:
    # SELL NVDA (0.25 -> 0.0)
    # SELL AAPL (0.40 -> 0.30)
    # BUY MSFT (0.35 -> 0.40)
    # BUY GOOGL (0.0 -> 0.30)
    actions = [o["action"] for o in orders_2]
    # Sells must come first
    assert actions[:2] == ["SELL", "SELL"]
    assert actions[2:] == ["BUY", "BUY"]

    sell_tickers = [o["ticker"] for o in orders_2 if o["action"] == "SELL"]
    assert "NVDA" in sell_tickers
    assert "AAPL" in sell_tickers
