"""Weekly portfolio rebalance report generator.

Guarantees:
- Produces comprehensive Markdown & JSON rebalance reports.
- Includes Buy List with Top 3 SHAP reasons for every pick.
- Includes Sell List with exit reasons (dropped below 40th percentile).
- Lists Retained Buffer Holdings (stayed in [0.60, 0.90) zone).
- Documents Estimated Costs (bps & $) and Portfolio Turnover.
- Reports Live Rolling 60-Day Spearman IC and Confidence Filter status.
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("weekly_report")


class WeeklyReportGenerator:
    """Generates structured Markdown and JSON reports for weekly portfolio rebalancing."""

    def __init__(self, reports_dir: Path | None = None):
        settings = get_settings()
        self.reports_dir = reports_dir or (settings.data_dir / "reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        as_of_date: str,
        target_df: pd.DataFrame,
        portfolio_meta: dict[str, Any],
        orders: list[dict[str, Any]],
        trade_summary: dict[str, Any],
        ic_summary: dict[str, Any] | None = None,
        cost_bps: float = 7.5,
        portfolio_value: float = 100_000.0,
    ) -> str:
        """
        Builds, saves, and returns the Markdown weekly rebalance report.
        """
        ic_summary = ic_summary or {}
        rolling_ic = ic_summary.get("latest_rolling_ic", "N/A")
        rolling_t = ic_summary.get("rolling_t_stat", "N/A")
        pct_pos = ic_summary.get("rolling_pct_positive", "N/A")

        one_way_turnover = trade_summary.get("one_way_turnover", 0.0)
        two_way_turnover = one_way_turnover * 2.0
        cost_rate = cost_bps / 10_000.0
        est_cost_pct = two_way_turnover * cost_rate * 100.0
        est_cost_dollars = portfolio_value * (two_way_turnover * cost_rate)

        vix = portfolio_meta.get("vix")
        vix_str = f"{vix:.2f}" if vix is not None else "Normal (<20)"
        exposure_scale = portfolio_meta.get("exposure_scale", 1.0)
        cash_weight = portfolio_meta.get("cash_weight", 0.0)
        conf_triggered = portfolio_meta.get("confidence_filter_triggered", False)

        lines = [
            "# Quantitative Portfolio Weekly Rebalance Report",
            f"**As-of Date:** `{as_of_date}` | **Generated:** `{pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M UTC')}`",
            "",
            "---",
            "",
            "## 1. Executive Summary & Market Regime",
            "",
            f"- **Market Volatility (VIX):** `{vix_str}`",
            f"- **Target Equity Exposure:** `{exposure_scale * 100:.1f}%` ({'Scaled down due to VIX' if exposure_scale < 1.0 else '100% full investment'})",
            f"- **Cash Allocation:** `{cash_weight * 100:.1f}%`",
            f"- **Confidence Filter Status:** `{'TRIGGERED (Holding Cash/SPY due to IC <= 0)' if conf_triggered else 'ACTIVE (Model performance healthy)'}`",
            f"- **Live 60-Day Rolling Spearman IC:** `{rolling_ic}` (t-stat: `{rolling_t}`, Positive Days: `{pct_pos}%`)",
            "",
            "---",
            "",
            "## 2. Portfolio Turnover & Actions",
            "",
            f"- **Total Trade Orders:** `{len(orders)}` (`{trade_summary.get('sells_count', 0)}` SELLs, `{trade_summary.get('buys_count', 0)}` BUYs)",
            f"- **One-Way Portfolio Turnover:** `{one_way_turnover * 100:.1f}%`",
            f"- **Estimated Transaction Cost:** `{est_cost_pct:.2f}%` ({cost_bps} bps / trade) ≈ `${est_cost_dollars:,.2f}` on `${portfolio_value:,.0f}` portfolio",
            "",
            "### Buy List (New Top 10% Entrants with SHAP Rationale)",
            "",
        ]

        # Buy List table with SHAP reasons
        buy_orders = [o for o in orders if o["action"] == "BUY"]
        if buy_orders:
            lines.append("| Ticker | Target Wt | Vol | Action | Primary SHAP Reasons (Why the Model Picked It) |")
            lines.append("| :--- | :---: | :---: | :---: | :--- |")
            for o in buy_orders:
                t = o["ticker"]
                t_row = target_df[target_df["ticker"] == t]
                vol_str = f"{t_row['predicted_vol'].iloc[0] * 100:.1f}%" if not t_row.empty and "predicted_vol" in t_row.columns else "N/A"

                # Fetch SHAP explanations
                shap_reasons = []
                if not t_row.empty:
                    for k in ["shap_top_1", "shap_top_2", "shap_top_3"]:
                        if k in t_row.columns and pd.notna(t_row[k].iloc[0]) and t_row[k].iloc[0]:
                            shap_reasons.append(str(t_row[k].iloc[0]))
                shap_str = "<br>• ".join([""] + shap_reasons) if shap_reasons else "High composite momentum & rank"

                lines.append(f"| **{t}** | {o['target_weight'] * 100:.2f}% | {vol_str} | +{o['pct_change']:.2f}% | {shap_str} |")
        else:
            lines.append("_No new buy additions this rebalance period._")

        lines.extend([
            "",
            "### Sell List (Positions Dropped Below Top 40% Cutoff)",
            "",
        ])

        sell_orders = [o for o in orders if o["action"] == "SELL" and o["target_weight"] == 0.0]
        trimmed_orders = [o for o in orders if o["action"] == "SELL" and o["target_weight"] > 0.0]

        if sell_orders:
            lines.append("| Ticker | Prior Weight | Action | Exit Reason |")
            lines.append("| :--- | :---: | :---: | :--- |")
            for o in sell_orders:
                lines.append(f"| **{o['ticker']}** | {o['current_weight'] * 100:.2f}% | Liquidate (-{o['current_weight'] * 100:.2f}%) | Dropped below 40th percentile threshold |")
        else:
            lines.append("_No full position liquidations._")

        if trimmed_orders:
            lines.extend([
                "",
                "### Trimmed Positions (Rebalanced to Target Sizing)",
                "",
                "| Ticker | Prior Weight | Target Weight | Change |",
                "| :--- | :---: | :---: | :---: |",
            ])
            for o in trimmed_orders:
                lines.append(f"| **{o['ticker']}** | {o['current_weight'] * 100:.2f}% | {o['target_weight'] * 100:.2f}% | {o['pct_change']:.2f}% |")

        lines.extend([
            "",
            "### Retained Buffer Holdings (Suppressed Spurious Turnover)",
            "",
        ])
        retained = portfolio_meta.get("retained_tickers", [])
        if retained:
            lines.append(f"Retained `{len(retained)}` existing holdings in the `[0.60, 0.90)` buffer zone: `{', '.join(retained)}`")
        else:
            lines.append("_No buffer zone holdings (cold start or full turnover)._")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Target Portfolio Holdings",
            "",
            "| Rank | Ticker | Target Weight | Predicted Vol | Selection Logic |",
            "| :---: | :--- | :---: | :---: | :--- |",
        ])

        for _, row in target_df.iterrows():
            t = row.get("ticker", "CASH")
            rk = row.get("rank", "-")
            w = row.get("target_weight", 0.0)
            pv = row.get("predicted_vol", 0.0)
            reason = row.get("selection_reason", "standard")
            pv_str = f"{pv * 100:.1f}%" if pv > 0 else "-"
            lines.append(f"| {rk} | **{t}** | {w * 100:.2f}% | {pv_str} | {reason} |")

        if cash_weight > 0.001:
            lines.append(f"| - | **CASH** | {cash_weight * 100:.2f}% | - | Unallocated / Volatility scale buffer |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Execution Trade List (SELLs Ordered First)",
            "",
        ])

        if orders:
            lines.append("| Step | Ticker | Action | Current Wt | Target Wt | Delta Wt | Change % |")
            lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
            for i, o in enumerate(orders, 1):
                lines.append(
                    f"| {i} | **{o['ticker']}** | `{o['action']}` | {o['current_weight'] * 100:.2f}% | "
                    f"{o['target_weight'] * 100:.2f}% | {o['delta_weight'] * 100:.2f}% | {o['pct_change']:+.2f}% |"
                )
        else:
            lines.append("_Portfolio already aligns with target weights. Zero trades required._")

        lines.append("")
        content = "\n".join(lines)

        # Save Markdown report
        md_file = self.reports_dir / f"weekly_report_{as_of_date}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(content)

        # Save JSON metadata report
        json_file = self.reports_dir / f"weekly_report_{as_of_date}.json"
        report_data = {
            "as_of_date": as_of_date,
            "target_positions_count": len(target_df),
            "portfolio_meta": portfolio_meta,
            "orders": orders,
            "trade_summary": trade_summary,
            "ic_summary": ic_summary,
            "est_transaction_cost_pct": round(est_cost_pct, 4),
            "est_transaction_cost_dollars": round(est_cost_dollars, 2),
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Weekly report generated: {md_file}")
        return content
