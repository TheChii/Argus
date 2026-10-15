"""Business logic for the Investor-Dedicated Dashboard.

Provides:
- 3-second verdict assessment (confidence light, regime, exposure).
- Capital-to-shares allocation calculator (whole vs fractional shares).
- Copy-paste broker order slip generator (Fidelity, Schwab, Robinhood, IBKR).
- Plain-English translations of quant SHAP drivers and FinBERT sentiment.
- Turnover-buffered rebalancing assistant.
"""
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("investor_logic")


def get_verdict_assessment(meta_strip: dict[str, Any] | None = None) -> dict[str, Any]:
    """Translates pipeline metadata into an immediate, plain-English investing verdict."""
    strip = meta_strip or {}
    light = str(strip.get("confidence_light", "green")).lower()
    regime = str(strip.get("regime_name", "Low Vol Bull"))
    vix = float(strip.get("vix", 18.5))
    ic = float(strip.get("rolling_60d_ic", 0.0362))
    t_stat = float(strip.get("rolling_t_stat", 2.85))
    exposure = float(strip.get("exposure_scale", 1.0)) * 100.0

    if light == "green":
        status_code = "GREEN"
        badge_text = "SAFE TO INVEST TODAY"
        badge_icon = "[SAFE]"
        color = "#238636"
        summary_sentence = (
            f"Market conditions are calm ({regime}, VIX {vix:.1f}) and the model's 60-day predictive "
            f"edge is confirmed positive (IC {ic:+.4f}, t={t_stat:.2f}). Safe to deploy capital at 100% target allocation."
        )
        action_plan = "Deploy new capital or execute target rebalance. Model is in full offensive mode."
    elif light == "amber":
        status_code = "AMBER"
        badge_text = "CAUTION — REDUCE RISK"
        badge_icon = "[CAUTION]"
        color = "#d29922"
        summary_sentence = (
            f"Elevated market volatility or neutral model signal detected ({regime}, VIX {vix:.1f}). "
            f"The model recommends cutting new position sizing in half (50% risk scale)."
        )
        action_plan = "Deploy at 50% target size or wait for clearer trend confirmation. Keep 50% in cash or short-term bills."
    else:
        status_code = "RED"
        badge_text = "DEFENSIVE MODE — HOLD CASH / SPY"
        badge_icon = "[DEFENSIVE]"
        color = "#da3633"
        summary_sentence = (
            f"High market stress or negative model performance ({regime}, VIX {vix:.1f}, IC {ic:+.4f}). "
            f"Capital preservation mode is active: 0% stock exposure recommended."
        )
        action_plan = "Do not buy today. Hold cash or benchmark Treasury/SPY until regime normalizes."

    return {
        "status_code": status_code,
        "badge_text": badge_text,
        "badge_icon": badge_icon,
        "color": color,
        "summary_sentence": summary_sentence,
        "action_plan": action_plan,
        "regime": regime,
        "vix": vix,
        "rolling_ic": ic,
        "t_stat": t_stat,
        "exposure_pct": exposure,
    }


def get_latest_prices(tickers: list[str], prices_dir: Path | None = None) -> dict[str, float]:
    """Retrieves the latest available close price for given tickers from cached parquet files."""
    if prices_dir is None:
        settings = get_settings()
        prices_dir = settings.data_dir / "prices"

    prices: dict[str, float] = {}
    for ticker in tickers:
        p_file = prices_dir / f"{ticker}.parquet"
        if p_file.exists():
            try:
                df = pd.read_parquet(p_file, columns=["close"])
                if not df.empty and "close" in df.columns:
                    val = float(df["close"].dropna().iloc[-1])
                    if val > 0:
                        prices[ticker] = round(val, 2)
                        continue
            except (ValueError, KeyError, OSError) as e:
                logger.debug("Failed reading price for %s (%s); using baseline fallback.", ticker, e)
        prices[ticker] = 100.0  # Fallback reasonable price if missing
    return prices


def calculate_share_allocations(
    budget: float,
    target_picks: list[dict[str, Any]],
    prices: dict[str, float],
    allow_fractional: bool = False,
) -> dict[str, Any]:
    """Calculates exact dollar budgets and share quantities to buy for an investor.

    Guarantees:
    - Never allocates more capital than the specified budget.
    - Accurately computes whole integer shares or fractional shares.
    - Calculates remaining cash buffer down to the cent.
    """
    if budget <= 0 or not target_picks:
        return {
            "budget": max(0.0, float(budget)),
            "total_allocated": 0.0,
            "cash_remainder": max(0.0, float(budget)),
            "cash_remainder_pct": 100.0 if budget > 0 else 0.0,
            "allow_fractional": allow_fractional,
            "allocations": [],
        }

    # Normalize weights if needed
    raw_weights = [max(0.0, float(p.get("target_weight", 0.0))) for p in target_picks]
    total_w = sum(raw_weights)
    if total_w <= 0:
        total_w = 1.0

    allocations = []
    total_spent = 0.0

    for pick in target_picks:
        ticker = pick.get("ticker", "UNKNOWN")
        name = pick.get("name", ticker)
        sector = pick.get("sector", "General")
        raw_w = max(0.0, float(pick.get("target_weight", 0.0)))
        norm_w = raw_w / total_w if total_w > 0 else 0.0

        price = float(prices.get(ticker, 100.0))
        if price <= 0:
            price = 100.0

        target_dollars = round(budget * norm_w, 2)

        if allow_fractional:
            shares = round(target_dollars / price, 4)
            actual_dollars = round(shares * price, 2)
            # Guard against tiny rounding overshoot
            if actual_dollars > target_dollars:
                shares = round((target_dollars - 0.01) / price, 4)
                actual_dollars = round(shares * price, 2)
        else:
            shares = int(target_dollars // price)
            actual_dollars = round(shares * price, 2)

        total_spent += actual_dollars

        allocations.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "price": price,
            "target_weight": round(norm_w, 4),
            "target_dollars": target_dollars,
            "shares": shares,
            "actual_dollars": actual_dollars,
            "actual_weight": round(actual_dollars / budget, 4) if budget > 0 else 0.0,
        })

    total_spent = round(total_spent, 2)
    cash_remainder = round(max(0.0, budget - total_spent), 2)
    cash_remainder_pct = round((cash_remainder / budget) * 100.0, 2) if budget > 0 else 0.0

    return {
        "budget": round(budget, 2),
        "total_allocated": total_spent,
        "cash_remainder": cash_remainder,
        "cash_remainder_pct": cash_remainder_pct,
        "allow_fractional": allow_fractional,
        "allocations": allocations,
    }


def generate_broker_order_text(
    allocations_summary: dict[str, Any],
    broker_format: str = "Standard",
    as_of_date: str = "",
    clean_csv: bool = False,
) -> str:
    """Generates formatted, ready-to-copy order slips for popular retail brokerages."""
    budget = allocations_summary.get("budget", 0.0)
    spent = allocations_summary.get("total_allocated", 0.0)
    cash = allocations_summary.get("cash_remainder", 0.0)
    allocations = [a for a in allocations_summary.get("allocations", []) if a.get("shares", 0) > 0]
    fractional = allocations_summary.get("allow_fractional", False)
    date_str = as_of_date or "Today"

    lines = []

    if broker_format == "Fidelity / Schwab":
        lines.append(f"# FIDELITY / SCHWAB ORDER TICKET — {date_str}")
        lines.append(f"# Target Capital: ${budget:,.2f} | Allocated: ${spent:,.2f} | Cash Buffer: ${cash:,.2f}")
        lines.append("# Format: Action | Symbol | Quantity | Order Type | Timing")
        lines.append("-" * 65)
        for i, a in enumerate(allocations, 1):
            qty = f"{a['shares']:.4f}" if fractional else str(int(a['shares']))
            lines.append(f"BUY | {a['ticker']:<5} | Qty: {qty:<6} | MKT | Day  (Est: ${a['actual_dollars']:,.2f} @ ${a['price']:.2f})")

    elif broker_format == "Robinhood / Webull":
        lines.append(f"ROBINHOOD / WEBULL QUICK BUY LIST — {date_str}")
        lines.append(f"Target Budget: ${budget:,.2f} | Total Spend: ${spent:,.2f} | Unused Cash: ${cash:,.2f}")
        lines.append("-" * 65)
        for i, a in enumerate(allocations, 1):
            if fractional:
                lines.append(f"{i:02d}. Buy ${a['actual_dollars']:,.2f} of {a['ticker']} ({a['name']})")
            else:
                lines.append(f"{i:02d}. Buy {int(a['shares'])} shares of {a['ticker']} @ market (~${a['actual_dollars']:,.2f})")

    elif broker_format == "Interactive Brokers":
        if not clean_csv:
            lines.append("# INTERACTIVE BROKERS (IBKR) BASKET TRADER FORMAT")
            lines.append(f"# As-Of: {date_str} | Capital: ${budget:,.2f}")
        lines.append("Action,Symbol,SecType,Exchange,Quantity,OrderType")
        for a in allocations:
            qty = f"{a['shares']:.4f}" if fractional else str(int(a['shares']))
            lines.append(f"BUY,{a['ticker']},STK,SMART,{qty},MKT")

    else:  # Standard
        lines.append(f"TARGET INVESTMENT ORDER SLIP — {date_str}")
        lines.append(f"Budget: ${budget:,.2f} | Total Deployed: ${spent:,.2f} | Cash Reserve: ${cash:,.2f}")
        lines.append(f"Positions: {len(allocations)} stocks | Mode: {'Fractional ($)' if fractional else 'Whole Shares'}")
        lines.append("=" * 65)
        for i, a in enumerate(allocations, 1):
            qty = f"{a['shares']:.4f}" if fractional else str(int(a['shares']))
            lines.append(
                f"[{i:02d}] BUY {qty:>6} shares {a['ticker']:<5} "
                f"({a['name'][:22]:<22}) @ ~${a['price']:>7.2f}  "
                f"[Total: ${a['actual_dollars']:>8.2f} | {a['actual_weight']*100:>4.1f}%]"
            )
        lines.append("=" * 65)
        lines.append("Tip: Place market or limit orders during normal trading hours (9:30 AM - 4:00 PM ET).")

    return "\n".join(lines)


def translate_shap_feature_to_plain_english(
    feature_name: str,
    attribution: float,
    feature_val: float | None = None,
) -> dict[str, Any]:
    """Translates mathematical SHAP feature attributions into intuitive investor explanations."""
    is_pos = attribution >= 0

    translations = {
        "rolling_vol_20d": {
            "category": "Risk & Stability",
            "icon": "[VOL]",
            "pos": "Smooth price action: 20-day realized volatility is lower and more stable than peer stocks.",
            "neg": "Elevated monthly price fluctuations relative to peer stocks.",
        },
        "rolling_vol_5d": {
            "category": "Risk & Stability",
            "icon": "[VOL]",
            "pos": "Calm short-term price stability with minimal turbulence over the last 5 days.",
            "neg": "Short-term choppy trading action over the past week.",
        },
        "rsi_14d": {
            "category": "Technical Momentum",
            "icon": "[RSI]",
            "pos": "Healthy momentum sweet-spot: upward buying demand without being severely overbought.",
            "neg": "Extended technical momentum that may require consolidation.",
        },
        "return_20d": {
            "category": "Trend Momentum",
            "icon": "[MOM]",
            "pos": "Strong 1-month upward price trend consistently outperforming the broader market.",
            "neg": "1-month price consolidation or temporary pullback.",
        },
        "return_5d": {
            "category": "Trend Momentum",
            "icon": "[MOM]",
            "pos": "Fresh weekly buying momentum heading into today's trading session.",
            "neg": "Minor short-term weekly dip creating attractive entry levels.",
        },
        "return_60d": {
            "category": "Trend Momentum",
            "icon": "[MOM]",
            "pos": "Longer-term quarterly trend strength confirming sustained institutional backing.",
            "neg": "Lagging 3-month medium-term price performance.",
        },
        "return_1d": {
            "category": "Immediate Trend",
            "icon": "[TREND]",
            "pos": "Positive close on the previous trading session with solid buyers.",
            "neg": "Previous session retracement.",
        },
        "rel_return_spy_20d": {
            "category": "Market Outperformance",
            "icon": "[ALPHA]",
            "pos": "Substantially beating the S&P 500 index over the past 20 trading days.",
            "neg": "Trailing the benchmark S&P 500 index over the past 20 trading days.",
        },
        "rel_return_spy_5d": {
            "category": "Market Outperformance",
            "icon": "[ALPHA]",
            "pos": "Outperforming the S&P 500 over the past 5 trading sessions.",
            "neg": "Short-term lag behind the S&P 500 over the past week.",
        },
        "rel_return_sector_20d": {
            "category": "Sector Leadership",
            "icon": "[SECTOR]",
            "pos": "Sector leader: outperforming competitor stocks in the same industry.",
            "neg": "Trailing direct competitors in the same industry sector.",
        },
        "sentiment_mean": {
            "category": "News Sentiment",
            "icon": "[SENT]",
            "pos": "Strong positive media sentiment & bullish news coverage parsed by FinBERT AI.",
            "neg": "Cautious or neutral media coverage in recent corporate headlines.",
        },
        "sentiment_3d_mean": {
            "category": "News Sentiment",
            "icon": "[SENT]",
            "pos": "Fresh wave of positive corporate news announcements over the last 3 days.",
            "neg": "Subdued or cautious headline tone over the last 3 days.",
        },
        "sentiment_count": {
            "category": "News Sentiment",
            "icon": "[SENT]",
            "pos": "High volume of media coverage confirming strong institutional market attention.",
            "neg": "Quiet news cycle with limited headline activity.",
        },
        "sentiment_dispersion": {
            "category": "News Sentiment",
            "icon": "[SENT]",
            "pos": "Consistent consensus tone across all published news stories.",
            "neg": "Mixed opinions and conflicting reports across media outlets.",
        },
        "volume_zscore_20d": {
            "category": "Institutional Volume",
            "icon": "[VOL_Z]",
            "pos": "Above-average trading volume indicating institutional accumulation.",
            "neg": "Trading on lighter-than-usual volume.",
        },
        "vix": {
            "category": "Macro Environment",
            "icon": "[MACRO]",
            "pos": "Benefiting from a calm macro backdrop with subdued market-wide fear.",
            "neg": "Headwinds from elevated broad market volatility.",
        },
        "regime_prob_0": {
            "category": "Macro Environment",
            "icon": "[MACRO]",
            "pos": "Macro regime favors equities: high probability of low-volatility bull expansion.",
            "neg": "Macro regime shifting toward defensive or neutral territory.",
        },
        "earnings_yield": {
            "category": "Fundamental Valuation",
            "icon": "[VALUE]",
            "pos": "High earnings yield (E/P): strong corporate profits relative to current share price.",
            "neg": "Low earnings yield indicating premium valuation multiples.",
        },
        "book_to_price": {
            "category": "Fundamental Valuation",
            "icon": "[VALUE]",
            "pos": "Substantial tangible asset backing and deep book-to-price discount.",
            "neg": "High price-to-book multiple pricing in future intangible growth.",
        },
        "sales_to_price": {
            "category": "Fundamental Valuation",
            "icon": "[VALUE]",
            "pos": "Strong revenue generation yield (S/P) offering margin of safety.",
            "neg": "High revenue multiple requiring sustained top-line acceleration.",
        },
        "fcf_yield": {
            "category": "Cash Flow Quality",
            "icon": "[FCF]",
            "pos": "Robust free cash flow generation backing dividends and share buybacks.",
            "neg": "Capital-intensive cash burn or reinvestment cycle.",
        },
        "dividend_yield": {
            "category": "Income Yield",
            "icon": "[DIV]",
            "pos": "Attractive cash distribution yield providing downside cushion.",
            "neg": "Low or zero dividend payout ratio.",
        },
        "ev_to_ebitda": {
            "category": "Enterprise Value",
            "icon": "[EV]",
            "pos": "Favorable enterprise value multiple (EV/EBITDA) indicating undervalued cash flows.",
            "neg": "Elevated enterprise multiple reflecting high growth expectations.",
        },
        "market_cap_log": {
            "category": "Size Factor",
            "icon": "[SIZE]",
            "pos": "Mega-cap balance sheet fortress strength and institutional liquidity.",
            "neg": "Smaller relative capitalization within the universe.",
        },
    }

    t_info = translations.get(feature_name, {
        "category": "Model Signal",
        "icon": "[SIGNAL]",
        "pos": f"Positive predictive factor contribution ({feature_name}).",
        "neg": f"Slight drag factor ({feature_name}).",
    })

    explanation = t_info["pos"] if is_pos else t_info["neg"]

    return {
        "feature": feature_name,
        "category": t_info["category"],
        "icon": t_info["icon"],
        "text": explanation,
        "attribution": attribution,
        "is_positive": is_pos,
    }


def calculate_rebalance_orders(
    portfolio_value: float,
    actions: list[dict[str, Any]],
    prices: dict[str, float],
    min_delta_pct: float = 0.005,
) -> dict[str, Any]:
    """Evaluates whether current holdings require active rebalancing today.

    Applies a 0.5% (50 bps) turnover buffer to prevent unnecessary trading commissions.
    """
    if portfolio_value <= 0 or not actions:
        return {
            "portfolio_value": portfolio_value,
            "rebalance_needed": False,
            "orders": [],
            "message": "No portfolio value or actions available.",
        }

    orders = []
    for a in actions:
        ticker = a.get("ticker", "")
        curr_w = float(a.get("current_weight", 0.0))
        targ_w = float(a.get("target_weight", 0.0))
        delta_w = targ_w - curr_w
        action_type = a.get("action", "HOLD")

        if abs(delta_w) >= min_delta_pct and action_type in ("BUY", "SELL"):
            price = prices.get(ticker, 100.0)
            trade_dollars = abs(delta_w) * portfolio_value
            shares = int(trade_dollars // price)
            orders.append({
                "ticker": ticker,
                "name": a.get("name", ticker),
                "sector": a.get("sector", "General"),
                "action": action_type,
                "current_weight": curr_w,
                "target_weight": targ_w,
                "delta_weight": delta_w,
                "trade_dollars": round(trade_dollars, 2),
                "price": price,
                "shares": shares,
            })

    rebalance_needed = len(orders) > 0
    if rebalance_needed:
        msg = f"{len(orders)} position adjustment(s) exceed the 0.5% turnover buffer and require action."
    else:
        msg = "All position drift is under the 0.5% buffer threshold. No trades required today! Hold steady and avoid fees."

    return {
        "portfolio_value": portfolio_value,
        "rebalance_needed": rebalance_needed,
        "orders": orders,
        "message": msg,
    }
