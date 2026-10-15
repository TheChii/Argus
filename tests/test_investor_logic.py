"""Unit tests for the Investor-Dedicated Dashboard logic."""

import pytest

from src.dashboard.investor_logic import (
    calculate_rebalance_orders,
    calculate_share_allocations,
    generate_broker_order_text,
    get_latest_prices,
    get_verdict_assessment,
    translate_shap_feature_to_plain_english,
)


def test_verdict_assessment_green():
    meta = {
        "confidence_light": "green",
        "regime_name": "Low Vol Bull",
        "vix": 17.5,
        "rolling_60d_ic": 0.042,
        "rolling_t_stat": 3.1,
        "exposure_scale": 1.0,
    }
    v = get_verdict_assessment(meta)
    assert v["status_code"] == "GREEN"
    assert "SAFE TO INVEST" in v["badge_text"]
    assert v["exposure_pct"] == 100.0
    assert "100%" in v["summary_sentence"]
    assert v["badge_icon"] == "[SAFE]"


def test_verdict_assessment_amber():
    meta = {
        "confidence_light": "amber",
        "regime_name": "Normal",
        "vix": 23.0,
        "rolling_60d_ic": 0.008,
        "exposure_scale": 0.5,
    }
    v = get_verdict_assessment(meta)
    assert v["status_code"] == "AMBER"
    assert "CAUTION" in v["badge_text"]
    assert v["exposure_pct"] == 50.0
    assert v["badge_icon"] == "[CAUTION]"


def test_verdict_assessment_red():
    meta = {
        "confidence_light": "red",
        "regime_name": "High Vol Bear",
        "vix": 35.0,
        "rolling_60d_ic": -0.015,
        "exposure_scale": 0.0,
    }
    v = get_verdict_assessment(meta)
    assert v["status_code"] == "RED"
    assert "DEFENSIVE" in v["badge_text"]
    assert v["exposure_pct"] == 0.0
    assert v["badge_icon"] == "[DEFENSIVE]"


def test_calculate_share_allocations_whole_shares():
    target_picks = [
        {"ticker": "AAPL", "name": "Apple", "sector": "Technology", "target_weight": 0.40},
        {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology", "target_weight": 0.35},
        {"ticker": "NVDA", "name": "Nvidia", "sector": "Technology", "target_weight": 0.25},
    ]
    prices = {"AAPL": 200.0, "MSFT": 400.0, "NVDA": 120.0}
    budget = 10000.0

    res = calculate_share_allocations(budget, target_picks, prices, allow_fractional=False)

    assert res["budget"] == 10000.0
    assert res["total_allocated"] <= budget
    assert res["cash_remainder"] >= 0.0
    assert abs((res["total_allocated"] + res["cash_remainder"]) - budget) < 0.05
    assert len(res["allocations"]) == 3

    # Check integer shares
    for a in res["allocations"]:
        assert isinstance(a["shares"], int)
        assert a["actual_dollars"] <= a["target_dollars"] + 0.01

    # AAPL: 40% of 10000 = 4000 -> 4000 // 200 = 20 shares ($4000)
    # MSFT: 35% of 10000 = 3500 -> 3500 // 400 = 8 shares ($3200)
    # NVDA: 25% of 10000 = 2500 -> 2500 // 120 = 20 shares ($2400)
    # Total = 4000 + 3200 + 2400 = 9600, Cash = 400
    aapl = next(a for a in res["allocations"] if a["ticker"] == "AAPL")
    assert aapl["shares"] == 20
    assert aapl["actual_dollars"] == 4000.0

    msft = next(a for a in res["allocations"] if a["ticker"] == "MSFT")
    assert msft["shares"] == 8
    assert msft["actual_dollars"] == 3200.0


def test_calculate_share_allocations_fractional():
    target_picks = [
        {"ticker": "AAPL", "name": "Apple", "sector": "Technology", "target_weight": 0.50},
        {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology", "target_weight": 0.50},
    ]
    prices = {"AAPL": 230.50, "MSFT": 420.75}
    budget = 5000.0

    res = calculate_share_allocations(budget, target_picks, prices, allow_fractional=True)

    assert res["budget"] == 5000.0
    assert res["total_allocated"] <= budget + 0.05
    assert res["cash_remainder"] >= 0.0
    # In fractional mode, unallocated cash should be minimal (< $1.00)
    assert res["cash_remainder"] < 1.0


def test_calculate_share_allocations_edge_cases():
    # Empty picks
    res_empty = calculate_share_allocations(10000.0, [], {})
    assert res_empty["total_allocated"] == 0.0
    assert res_empty["allocations"] == []

    # Zero or negative budget
    res_zero = calculate_share_allocations(0.0, [{"ticker": "AAPL", "target_weight": 1.0}], {"AAPL": 100.0})
    assert res_zero["total_allocated"] == 0.0
    assert res_zero["allocations"] == []

    res_neg = calculate_share_allocations(-500.0, [{"ticker": "AAPL", "target_weight": 1.0}], {"AAPL": 100.0})
    assert res_neg["total_allocated"] == 0.0


def test_generate_broker_order_text():
    target_picks = [
        {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Technology", "target_weight": 0.60},
        {"ticker": "MU", "name": "Micron", "sector": "Technology", "target_weight": 0.40},
    ]
    prices = {"AMD": 150.0, "MU": 100.0}
    summary = calculate_share_allocations(1000.0, target_picks, prices, allow_fractional=False)

    # Standard format
    std_slip = generate_broker_order_text(summary, broker_format="Standard", as_of_date="2026-10-12")
    assert "TARGET INVESTMENT ORDER SLIP" in std_slip
    assert "AMD" in std_slip
    assert "MU" in std_slip
    assert "BUY" in std_slip

    # Fidelity format
    fid_slip = generate_broker_order_text(summary, broker_format="Fidelity / Schwab", as_of_date="2026-10-12")
    assert "FIDELITY / SCHWAB" in fid_slip
    assert "BUY | AMD" in fid_slip

    # Robinhood format
    rh_slip = generate_broker_order_text(summary, broker_format="Robinhood / Webull", as_of_date="2026-10-12")
    assert "ROBINHOOD / WEBULL" in rh_slip
    assert "AMD" in rh_slip

    # IBKR format
    ibkr_slip = generate_broker_order_text(summary, broker_format="Interactive Brokers", as_of_date="2026-10-12")
    assert "INTERACTIVE BROKERS" in ibkr_slip
    assert "BUY,AMD,STK,SMART" in ibkr_slip

    # IBKR clean CSV format (direct TWS BasketTrader load without comments)
    ibkr_clean = generate_broker_order_text(summary, broker_format="Interactive Brokers", as_of_date="2026-10-12", clean_csv=True)
    assert not any(line.startswith("#") for line in ibkr_clean.splitlines())
    assert ibkr_clean.startswith("Action,Symbol,SecType,Exchange,Quantity,OrderType")
    assert "BUY,AMD,STK,SMART" in ibkr_clean


def test_translate_shap_feature_to_plain_english():
    # Volatility
    trans_vol = translate_shap_feature_to_plain_english("rolling_vol_20d", attribution=2.5)
    assert trans_vol["category"] == "Risk & Stability"
    assert "Smooth price action" in trans_vol["text"]
    assert trans_vol["icon"] == "[VOL]"

    # Momentum
    trans_mom = translate_shap_feature_to_plain_english("return_20d", attribution=1.2)
    assert trans_mom["category"] == "Trend Momentum"
    assert "1-month upward price trend" in trans_mom["text"]
    assert trans_mom["icon"] == "[MOM]"

    # FinBERT news sentiment
    trans_sent = translate_shap_feature_to_plain_english("sentiment_mean", attribution=0.8)
    assert trans_sent["category"] == "News Sentiment"
    assert "FinBERT" in trans_sent["text"]

    # Unknown fallback
    trans_unk = translate_shap_feature_to_plain_english("custom_new_feature_xyz", attribution=0.5)
    assert trans_unk["category"] == "Model Signal"
    assert "custom_new_feature_xyz" in trans_unk["text"]


def test_calculate_rebalance_orders():
    actions = [
        # Small drift (< 0.5% threshold)
        {"ticker": "AMD", "current_weight": 0.065, "target_weight": 0.068, "action": "HOLD"},
        # Large trim (> 0.5%)
        {"ticker": "META", "current_weight": 0.080, "target_weight": 0.065, "action": "SELL"},
        # Large add (> 0.5%)
        {"ticker": "INTC", "current_weight": 0.050, "target_weight": 0.065, "action": "BUY"},
    ]
    prices = {"AMD": 150.0, "META": 700.0, "INTC": 100.0}

    # Threshold 0.5% (0.005)
    res = calculate_rebalance_orders(portfolio_value=50000.0, actions=actions, prices=prices, min_delta_pct=0.005)
    assert res["rebalance_needed"] is True
    assert len(res["orders"]) == 2  # META and INTC

    meta_order = next(o for o in res["orders"] if o["ticker"] == "META")
    assert meta_order["action"] == "SELL"
    assert meta_order["trade_dollars"] == pytest.approx(750.0, abs=1.0)  # 0.015 * 50000 = 750

    # Test when all drifts are tiny
    tiny_actions = [
        {"ticker": "AMD", "current_weight": 0.065, "target_weight": 0.066, "action": "HOLD"},
    ]
    res_tiny = calculate_rebalance_orders(portfolio_value=50000.0, actions=tiny_actions, prices=prices, min_delta_pct=0.005)
    assert res_tiny["rebalance_needed"] is False
    assert len(res_tiny["orders"]) == 0
    assert "No trades required today" in res_tiny["message"]


def test_get_latest_prices():
    # Test retrieving known existing tickers
    prices = get_latest_prices(["SPY", "AMAT", "DE"])
    assert "SPY" in prices
    assert prices["SPY"] > 0
    assert "AMAT" in prices
    assert prices["AMAT"] > 0
    assert "DE" in prices
    assert prices["DE"] > 0
