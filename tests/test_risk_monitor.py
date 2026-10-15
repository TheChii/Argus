"""Comprehensive unit and integration tests for Intraday Risk Circuit Breakers & Stop-Losses."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.dashboard.alerts import AlertDispatcher
from src.dashboard.database import DashboardDB
from src.dashboard.server import app
from src.features.technical import compute_atr
from src.portfolio.broker import DryRunBrokerClient
from src.portfolio.risk_monitor import IntradayRiskMonitor


@pytest.fixture
def temp_db():
    """Provides a fresh isolated SQLite database for test runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_risk_ledger.db"
        db = DashboardDB(db_path=db_path)
        yield db


@pytest.fixture
def client():
    """Provides a FastAPI test client for dashboard server endpoints."""
    return TestClient(app)


def test_atr_computation():
    """Validates Wilder's Average True Range calculation on synthetic price bars."""
    dates = pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "open": np.linspace(150, 160, 30),
        "high": np.linspace(152, 165, 30),
        "low": np.linspace(148, 158, 30),
        "close": np.linspace(150, 162, 30),
        "volume": 1000000,
    })

    atr = compute_atr(df, period=14)
    assert not atr.empty
    assert len(atr) == 30
    latest_atr = atr.iloc[-1]
    assert latest_atr > 0.0
    assert not np.isnan(latest_atr)

    # Empty and degenerate dataframe handling
    empty_atr = compute_atr(pd.DataFrame(), period=14)
    assert empty_atr.empty


def test_hard_stop_trigger(temp_db):
    """Validates that a -7.0% drawdown from entry cost basis flags position as BREACHED."""
    temp_db.record_fill(
        fill_date="2026-03-01",
        ticker="AAPL",
        action="BUY",
        shares=100.0,
        price=100.0,
        notes="Initial entry",
        broker="dry_run",
    )

    broker = DryRunBrokerClient()
    broker.db = temp_db

    monitor = IntradayRiskMonitor(
        db=temp_db,
        broker=broker,
        hard_stop_pct=0.07,
        trailing_stop_pct=0.07,
        atr_multiplier=2.5,
    )

    # Current price drops to $92.00 (-8.0% from $100 entry)
    checked = monitor.check_positions(live_prices={"AAPL": 92.0})
    assert len(checked) == 1
    pos = checked[0]

    assert pos["ticker"] == "AAPL"
    assert pos["cost_basis"] == 100.0
    assert pos["current_price"] == 92.0
    assert pos["drawdown_from_entry_pct"] == -8.0
    assert pos["status"] == "BREACHED"
    assert "HARD_STOP" in pos["trigger_type"]


def test_trailing_stop_trigger(temp_db):
    """Validates that a -7.0% pullback from peak high-water mark breaches trailing stop even if above entry."""
    temp_db.record_fill(
        fill_date="2026-03-01",
        ticker="MSFT",
        action="BUY",
        shares=50.0,
        price=100.0,
        notes="Initial entry",
        broker="dry_run",
    )

    broker = DryRunBrokerClient()
    broker.db = temp_db

    monitor = IntradayRiskMonitor(
        db=temp_db,
        broker=broker,
        hard_stop_pct=0.07,
        trailing_stop_pct=0.07,
        atr_multiplier=2.5,
    )

    # 1. Price runs up to $150 (peak high-water mark registered)
    checked_runup = monitor.check_positions(live_prices={"MSFT": 150.0})
    assert checked_runup[0]["peak_price"] == 150.0
    assert checked_runup[0]["status"] == "HEALTHY"

    # 2. Price falls to $138.00 (-8.0% from $150 peak, still +38% above entry)
    checked_pullback = monitor.check_positions(live_prices={"MSFT": 138.0})
    assert len(checked_pullback) == 1
    pos = checked_pullback[0]

    assert pos["peak_price"] == 150.0
    assert pos["current_price"] == 138.0
    assert pos["drawdown_from_peak_pct"] == -8.0
    assert pos["status"] == "BREACHED"
    assert "TRAILING_STOP" in pos["trigger_type"]


def test_atr_volatility_stop_trigger(temp_db):
    """Validates that a drop below Peak - 2.5 * ATR triggers ATR_STOP breach."""
    temp_db.record_fill(
        fill_date="2026-03-01",
        ticker="NVDA",
        action="BUY",
        shares=20.0,
        price=200.0,
        notes="Initial entry",
        broker="dry_run",
    )

    broker = DryRunBrokerClient()
    broker.db = temp_db

    monitor = IntradayRiskMonitor(
        db=temp_db,
        broker=broker,
        hard_stop_pct=0.15,  # Relaxed hard stop to isolate ATR trigger
        trailing_stop_pct=0.15,  # Relaxed trailing stop to isolate ATR trigger
        atr_multiplier=2.5,
    )

    # Mock ATR to return $4.00
    with patch.object(monitor, "get_atr", return_value=4.0):
        # Peak = $200. ATR Stop floor = 200 - (2.5 * 4.0) = $190.00
        # If price drops to $188.00, ATR stop floor is breached
        checked = monitor.check_positions(live_prices={"NVDA": 188.0})
        pos = checked[0]

        assert pos["atr_14"] == 4.0
        assert pos["atr_stop_price"] == 190.0
        assert pos["current_price"] == 188.0
        assert pos["status"] == "BREACHED"
        assert "ATR_STOP" in pos["trigger_type"]


def test_healthy_and_caution_states(temp_db):
    """Verifies HEALTHY vs CAUTION status transitions based on drawdown."""
    temp_db.record_fill(fill_date="2026-03-01", ticker="GOOGL", action="BUY", shares=10.0, price=100.0, broker="dry_run")
    temp_db.record_fill(fill_date="2026-03-01", ticker="AMZN", action="BUY", shares=10.0, price=100.0, broker="dry_run")

    broker = DryRunBrokerClient()
    broker.db = temp_db

    monitor = IntradayRiskMonitor(db=temp_db, broker=broker)

    with patch.object(monitor, "get_atr", return_value=5.0):
        # GOOGL at $99.0 (-1.0% -> HEALTHY), AMZN at $95.0 (-5.0% -> CAUTION)
        # With ATR=5.0, 2.5 * ATR = 12.5 (stop at $87.50). 7% stop is at $93.0.
        # At $95.0, drawdown is -5.0% (between -4% and -7%), so CAUTION.
        checked = monitor.check_positions(live_prices={"GOOGL": 99.0, "AMZN": 95.0})
        status_map = {p["ticker"]: p["status"] for p in checked}

        assert status_map["GOOGL"] == "HEALTHY"
        assert status_map["AMZN"] == "CAUTION"



def test_execute_circuit_breaker_cash(temp_db):
    """Verifies complete liquidation pipeline to cash upon circuit breaker breach."""
    temp_db.record_fill(fill_date="2026-03-01", ticker="TSLA", action="BUY", shares=50.0, price=200.0, broker="dry_run")

    broker = DryRunBrokerClient()
    broker.db = temp_db

    # Mock alerts dispatcher to avoid real Telegram HTTP calls
    mock_alerts = MagicMock(spec=AlertDispatcher)
    monitor = IntradayRiskMonitor(
        db=temp_db,
        broker=broker,
        alerts=mock_alerts,
        hard_stop_pct=0.07,
    )

    # TSLA drops to $180 (-10.0% breach)
    live_prices = {"TSLA": 180.0}
    executions = monitor.execute_circuit_breaker(liquidate_to="cash", live_prices=live_prices)

    assert len(executions) == 1
    exec_item = executions[0]
    assert exec_item["ticker"] == "TSLA"
    assert exec_item["shares_liquidated"] == 50.0
    assert exec_item["exit_price"] == 180.0
    assert exec_item["realized_proceeds"] == 9000.0
    assert exec_item["status"] == "EXECUTED"

    # Verify sell fill in DB ledger
    fills_df = temp_db.get_fills()
    sells = fills_df[fills_df["action"] == "SELL"]
    assert len(sells) == 1
    assert sells.iloc[0]["ticker"] == "TSLA"
    assert sells.iloc[0]["shares"] == 50.0
    assert "EMERGENCY_CIRCUIT_BREAKER" in sells.iloc[0]["notes"]

    # Verify open positions is now empty
    open_pos = temp_db.get_open_positions_from_fills()
    assert open_pos.get("TSLA", {}).get("shares", 0.0) == 0.0

    # Verify circuit breaker event log
    events_df = temp_db.get_circuit_breaker_events()
    assert len(events_df) == 1
    assert events_df.iloc[0]["ticker"] == "TSLA"
    assert events_df.iloc[0]["hedge_action"] == "cash"

    # Verify Telegram notification called
    assert mock_alerts.notify_circuit_breaker_triggered.called


def test_execute_circuit_breaker_hedged_spy(temp_db):
    """Verifies circuit breaker liquidation and immediate reinvestment into hedged SPY."""
    temp_db.record_fill(fill_date="2026-03-01", ticker="META", action="BUY", shares=20.0, price=500.0, broker="dry_run")

    broker = DryRunBrokerClient()
    broker.db = temp_db

    mock_alerts = MagicMock(spec=AlertDispatcher)
    monitor = IntradayRiskMonitor(
        db=temp_db,
        broker=broker,
        alerts=mock_alerts,
        hard_stop_pct=0.07,
    )

    # META drops to $450 (-10.0% breach), SPY is quoted at $500
    live_prices = {"META": 450.0, "SPY": 500.0}
    executions = monitor.execute_circuit_breaker(liquidate_to="hedged_spy", live_prices=live_prices)

    assert len(executions) == 1
    exec_item = executions[0]
    assert exec_item["ticker"] == "META"
    assert exec_item["hedge"]["hedged_ticker"] == "SPY"
    # Proceeds = 20 * 450 = $9,000. At SPY $500 -> 18.0 shares
    assert exec_item["hedge"]["shares"] == 18.0

    # Verify both SELL META and BUY SPY recorded in fills
    fills_df = temp_db.get_fills()
    assert len(fills_df[fills_df["ticker"] == "META"]) == 2  # 1 buy, 1 sell
    assert len(fills_df[fills_df["ticker"] == "SPY"]) == 1   # 1 hedge buy


def test_fastapi_risk_monitor_endpoints(client):
    """Tests /api/risk/monitor and /api/risk/circuit-breaker REST endpoints."""
    # GET /api/risk/monitor
    res = client.get("/api/risk/monitor")
    assert res.status_code == 200
    data = res.json()
    assert "positions" in data
    assert "total_open" in data
    assert "breached_count" in data
    assert "system_status" in data
    assert "recent_events" in data

    # POST /api/risk/circuit-breaker (empty request executes without errors)
    cb_res = client.post("/api/risk/circuit-breaker", json={"liquidate_to": "cash"})
    assert cb_res.status_code == 200
    cb_data = cb_res.json()
    assert cb_data["success"] is True
    assert "liquidations_count" in cb_data
