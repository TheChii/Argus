"""Evaluation metrics: Daily Spearman IC, t-stat, and Quintile spread Sharpe."""
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_daily_spearman_ic(
    df: pd.DataFrame,
    pred_col: str = "pred_score",
    target_col: str = "target_excess_return_5d",
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Computes daily cross-sectional Spearman rank correlation (IC) between predictions and actual forward returns.
    """
    daily_records = []

    for d, grp in df.groupby(date_col):
        valid = grp[[pred_col, target_col]].dropna()
        if len(valid) >= 10:
            rho, pval = spearmanr(valid[pred_col], valid[target_col])
            if not np.isnan(rho):
                daily_records.append({
                    "date": d,
                    "ic": float(rho),
                    "pval": float(pval),
                    "n_tickers": len(valid),
                })

    if not daily_records:
        return pd.DataFrame(columns=["date", "ic", "pval", "n_tickers"])

    return pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)


def summarize_ic_metrics(daily_ic_df: pd.DataFrame) -> dict[str, Any]:
    """
    Computes summary IC statistics: Mean IC, IC Std, IC IR, t-statistic, and % positive.
    """
    if daily_ic_df.empty or len(daily_ic_df) < 2:
        return {
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "ic_ir": 0.0,
            "t_stat": 0.0,
            "pct_positive": 0.0,
            "n_days": len(daily_ic_df),
        }

    ics = daily_ic_df["ic"].values
    n = len(ics)
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1))
    ic_ir = float(mean_ic / std_ic) if std_ic > 1e-8 else 0.0
    se = std_ic / np.sqrt(n)
    t_stat = float(mean_ic / se) if se > 1e-8 else 0.0
    pct_pos = float(np.mean(ics > 0.0) * 100.0)

    return {
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "ic_ir": round(ic_ir, 3),
        "t_stat": round(t_stat, 2),
        "pct_positive": round(pct_pos, 1),
        "n_days": n,
    }


def compute_quintile_spread_performance(
    df: pd.DataFrame,
    pred_col: str = "pred_score",
    fwd_ret_col: str = "target_excess_return_5d",
    date_col: str = "date",
    cost_bps: float = 7.5,
) -> dict[str, Any]:
    """
    Calculates Top Quintile (Q5) minus Bottom Quintile (Q1) spread returns.
    Cost is deducted per period (e.g. 7.5 bps).
    """
    unique_dates = sorted(df[date_col].unique())
    rebal_dates = set(unique_dates[::5])
    daily_spreads = []

    for d in unique_dates:
        if d not in rebal_dates:
            continue
        grp = df[df[date_col] == d]
        valid = grp[[pred_col, fwd_ret_col]].dropna()
        if len(valid) >= 20:
            q_ranks = pd.qcut(valid[pred_col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            valid["quintile"] = q_ranks
            q1_ret = valid[valid["quintile"] == 1][fwd_ret_col].mean()
            q5_ret = valid[valid["quintile"] == 5][fwd_ret_col].mean()

            # Cost deduction for weekly rebalance (e.g. 7.5 bps * 2-way)
            spread_ret = (q5_ret - q1_ret) - (cost_bps * 1e-4 * 2)
            daily_spreads.append(spread_ret)

    if not daily_spreads:
        return {
            "mean_period_spread": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe_ratio": 0.0,
            "periods_count": 0,
        }

    spread_arr = np.array(daily_spreads)
    # Assuming weekly rebalancing (approx 52 periods per year)
    mean_p = float(np.mean(spread_arr))
    std_p = float(np.std(spread_arr, ddof=1)) if len(spread_arr) > 1 else 1e-6
    ann_return = mean_p * 52.0
    ann_vol = std_p * np.sqrt(52.0)
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-8 else 0.0

    return {
        "mean_period_spread": round(mean_p, 4),
        "annualized_return": round(ann_return, 4),
        "annualized_vol": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 2),
        "periods_count": len(spread_arr),
    }
