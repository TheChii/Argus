"""Unit tests for Cost-Adjusted Backtest Engine."""
import pandas as pd

from src.backtest.cost_adjusted import CostAdjustedBacktest


def test_cost_adjusted_backtest_turnover_and_metrics():
    # 30 trading dates (6 weekly rebalances)
    dates = [f"2026-W{i:02d}" for i in range(30)]
    tickers = [f"TICK_{j}" for j in range(25)]

    records = []
    for d in dates:
        for j, t in enumerate(tickers):
            # Deterministic scores so quintiles are stable or rotate
            score = float(j + (1 if "05" in d else 0))
            # Fwd return: positive for high scorers, negative for low scorers
            fwd_ret = (j - 12.0) * 0.005
            records.append({
                "date": d,
                "ticker": t,
                "pred_score": score,
                "target_excess_return_5d": fwd_ret,
            })

    df = pd.DataFrame(records)

    bt = CostAdjustedBacktest(cost_bps=10.0, rebalance_interval=5)
    res = bt.run(df, pred_col="pred_score", target_col="target_excess_return_5d")

    assert res["rebalance_periods"] >= 5
    assert res["cost_bps"] == 10.0
    assert "annualized_return_gross" in res
    assert "annualized_return_net" in res
    assert "sharpe_ratio_net" in res
    assert "max_drawdown" in res
    # Net return should be less than gross return due to transaction costs
    assert res["annualized_return_net"] <= res["annualized_return_gross"]
