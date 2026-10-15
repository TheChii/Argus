"""Cost-adjusted backtest engine with weekly rebalancing and turnover modeling."""
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("cost_backtest")


class CostAdjustedBacktest:
    """
    Weekly rebalanced quintile backtest engine with realistic turnover and transaction costs.

    Guarantees:
    - Top Quintile (Q5) vs Bottom Quintile (Q1) selection
    - Precise two-way turnover calculation across rebalance dates
    - Configurable transaction costs (5-10 bps per trade)
    - Comprehensive performance analytics (Gross vs Net Sharpe, Max Drawdown, Calmar)
    """

    def __init__(
        self,
        cost_bps: float = 7.5,        # 7.5 bps per one-way turnover
        rebalance_interval: int = 5,  # Rebalance weekly (every 5 trading days)
    ):
        self.cost_bps = cost_bps
        self.cost_decimal = cost_bps * 1e-4
        self.rebalance_interval = rebalance_interval

    def run(
        self,
        oos_df: pd.DataFrame,
        pred_col: str = "pred_score",
        target_col: str = "target_excess_return_5d",
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> dict[str, Any]:
        """
        Executes weekly cost-adjusted backtest across out-of-sample prediction periods.
        """
        unique_dates = sorted(oos_df[date_col].unique())
        if len(unique_dates) < self.rebalance_interval * 2:
            raise ValueError(f"Insufficient out-of-sample dates ({len(unique_dates)}) for weekly backtest.")

        # Rebalance every `rebalance_interval` trading days
        rebal_dates = unique_dates[::self.rebalance_interval]

        period_records = []
        prev_q5_weights: dict[str, float] = {}
        prev_q1_weights: dict[str, float] = {}

        for d in rebal_dates:
            day_slice = oos_df[oos_df[date_col] == d].dropna(subset=[pred_col, target_col]).copy()
            if len(day_slice) < 20:
                continue

            # Form quintiles
            day_slice["quintile"] = pd.qcut(day_slice[pred_col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            q5_tickers = set(day_slice[day_slice["quintile"] == 5][ticker_col])
            q1_tickers = set(day_slice[day_slice["quintile"] == 1][ticker_col])

            # Equal weights
            q5_weights = {t: 1.0 / len(q5_tickers) for t in q5_tickers}
            q1_weights = {t: 1.0 / len(q1_tickers) for t in q1_tickers}

            # Two-way turnover: 0.5 * sum(|w_new - w_old|) (1.0 on initial creation from cash)
            if not prev_q5_weights:
                turnover_q5 = 1.0
            else:
                all_q5_syms = set(q5_weights.keys()) | set(prev_q5_weights.keys())
                turnover_q5 = 0.5 * sum(abs(q5_weights.get(t, 0.0) - prev_q5_weights.get(t, 0.0)) for t in all_q5_syms)

            if not prev_q1_weights:
                turnover_q1 = 1.0
            else:
                all_q1_syms = set(q1_weights.keys()) | set(prev_q1_weights.keys())
                turnover_q1 = 0.5 * sum(abs(q1_weights.get(t, 0.0) - prev_q1_weights.get(t, 0.0)) for t in all_q1_syms)

            total_turnover = turnover_q5 + turnover_q1
            # Cost = turnover * cost_bps
            transaction_cost = total_turnover * self.cost_decimal

            # Realized 5-day forward excess returns
            ret_map = dict(zip(day_slice[ticker_col], day_slice[target_col]))
            q5_return = float(np.mean([ret_map[t] for t in q5_tickers if t in ret_map]))
            q1_return = float(np.mean([ret_map[t] for t in q1_tickers if t in ret_map]))

            gross_spread_ret = q5_return - q1_return
            net_spread_ret = gross_spread_ret - transaction_cost

            # Long-only Q5 net return
            long_only_cost = turnover_q5 * self.cost_decimal
            net_q5_return = q5_return - long_only_cost

            period_records.append({
                "date": d,
                "q5_gross_return": q5_return,
                "q1_gross_return": q1_return,
                "gross_spread": gross_spread_ret,
                "net_spread": net_spread_ret,
                "net_q5_return": net_q5_return,
                "turnover": total_turnover,
                "cost": transaction_cost,
                "long_holdings_count": len(q5_tickers),
            })

            prev_q5_weights = q5_weights
            prev_q1_weights = q1_weights

        if not period_records:
            return {"error": "No rebalancing periods executed."}

        history_df = pd.DataFrame(period_records)
        return self._compute_performance_summary(history_df)

    def _compute_performance_summary(self, history_df: pd.DataFrame) -> dict[str, Any]:
        """Calculates annualized metrics, Sharpe, drawdowns, and Calmar ratios."""
        n_periods = len(history_df)
        periods_per_year = 52.0  # Weekly rebalancing

        gross_returns = history_df["gross_spread"].values
        net_returns = history_df["net_spread"].values
        turnover_vals = history_df["turnover"].values

        # Cumulative equity curves
        np.cumprod(1.0 + gross_returns)
        cum_net = np.cumprod(1.0 + net_returns)

        # Max drawdown
        peaks = np.maximum.accumulate(cum_net)
        drawdowns = (peaks - cum_net) / peaks
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Annualized statistics
        ann_gross_ret = float(np.mean(gross_returns) * periods_per_year)
        ann_net_ret = float(np.mean(net_returns) * periods_per_year)
        ann_vol = float(np.std(net_returns, ddof=1) * np.sqrt(periods_per_year)) if len(net_returns) > 1 else 1e-6

        gross_vol = float(np.std(gross_returns, ddof=1) * np.sqrt(periods_per_year)) if len(gross_returns) > 1 else 1e-6
        gross_sharpe = float(ann_gross_ret / gross_vol) if gross_vol > 1e-8 else 0.0
        net_sharpe = float(ann_net_ret / ann_vol) if ann_vol > 1e-8 else 0.0

        calmar = float(ann_net_ret / max_drawdown) if max_drawdown > 1e-8 else 0.0
        win_rate = float(np.mean(net_returns > 0.0) * 100.0)
        mean_turnover = float(np.mean(turnover_vals))
        total_costs_paid = float(np.sum(history_df["cost"]))

        return {
            "rebalance_periods": n_periods,
            "cost_bps": self.cost_bps,
            "annualized_return_gross": round(ann_gross_ret * 100.0, 2),
            "annualized_return_net": round(ann_net_ret * 100.0, 2),
            "annualized_volatility": round(ann_vol * 100.0, 2),
            "sharpe_ratio_gross": round(gross_sharpe, 2),
            "sharpe_ratio_net": round(net_sharpe, 2),
            "max_drawdown": round(max_drawdown * 100.0, 2),
            "calmar_ratio": round(calmar, 2),
            "win_rate_pct": round(win_rate, 1),
            "mean_weekly_turnover": round(mean_turnover * 100.0, 1),
            "total_transaction_cost_pct": round(total_costs_paid * 100.0, 2),
            "history_df": history_df,
        }
