"""Unit and integration tests for the FastAPI/HTML5 Argus Quant Trading Terminal."""
import pytest
from fastapi.testclient import TestClient

from src.dashboard.investor_logic import calculate_rebalance_orders, get_latest_prices
from src.dashboard.server import app
from src.dashboard.snapshot import load_today_snapshot
from src.portfolio.broker import DryRunBrokerClient, get_broker_client


@pytest.fixture
def client():
    return TestClient(app)


def test_unified_terminal_snapshot_data():
    """Validates that today_snapshot provides all required fields for the XTB terminal."""
    snapshot = load_today_snapshot()
    assert "today_cockpit" in snapshot
    assert "actions" in snapshot["today_cockpit"]
    assert len(snapshot["today_cockpit"]["actions"]) > 0

    first_act = snapshot["today_cockpit"]["actions"][0]
    assert "ticker" in first_act
    assert "action" in first_act
    assert "target_weight" in first_act
    assert "predicted_vol" in first_act


def test_unified_terminal_broker_integration():
    """Validates default broker client returns proper account and order structure."""
    broker = get_broker_client("dry_run")
    assert isinstance(broker, DryRunBrokerClient)

    summary = broker.get_account_summary()
    assert "portfolio_value" in summary
    assert "cash" in summary
    assert "buying_power" in summary

    receipt = broker.submit_order("AAPL", quantity=10, side="buy", estimated_price=180.0)
    assert receipt["status"] == "FILLED"
    assert receipt["ticker"] == "AAPL"


def test_unified_terminal_rebalance_orders():
    """Validates turnover buffer calculation for rebalance orders."""
    snapshot = load_today_snapshot()
    actions = snapshot["today_cockpit"]["actions"]
    tickers = [a["ticker"] for a in actions]
    prices = get_latest_prices(tickers)

    reb = calculate_rebalance_orders(100000.0, actions, prices, min_delta_pct=0.005)
    assert "rebalance_needed" in reb
    assert "orders" in reb
    assert isinstance(reb["orders"], list)


def test_web_index_serves_html(client):
    """Verifies that root URL / serves the HTML5 single page app."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "ARGUS" in res.text
    assert "QUANT" in res.text
    assert "tailwindcss" in res.text


def test_api_snapshot_endpoint(client):
    """Verifies /api/snapshot returns precomputed payload."""
    res = client.get("/api/snapshot")
    assert res.status_code == 200
    data = res.json()
    assert "today_cockpit" in data
    assert "meta" in data


def test_api_account_endpoint(client):
    """Verifies /api/account returns real-time broker balances."""
    res = client.get("/api/account")
    assert res.status_code == 200
    data = res.json()
    assert "portfolio_value" in data
    assert "cash" in data
    assert "buying_power" in data


def test_api_order_submission(client):
    """Verifies /api/order executes single order with broker and logs to SQLite."""
    payload = {
        "ticker": "AAPL",
        "quantity": 10.0,
        "side": "buy",
        "price": 185.00,
    }
    res = client.post("/api/order", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["status"] == "FILLED"
    assert data["ticker"] == "AAPL"
    assert "fill_id" in data


def test_api_sizing_endpoint(client):
    """Verifies /api/size computes dollar allocations and broker slips."""
    payload = {
        "budget": 25000.0,
        "allow_fractional": False,
    }
    res = client.post("/api/size", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "slips" in data
    assert "standard" in data["slips"]
    assert "fidelity" in data["slips"]
    assert "ibkr" in data["slips"]
    assert "ibkr_csv" in data["slips"]
    assert data["slips"]["ibkr_csv"].startswith("Action,Symbol,SecType,Exchange,Quantity,OrderType")


def test_api_positions_and_fills(client):
    """Verifies /api/positions and /api/fills retrieve ledger entries."""
    res_pos = client.get("/api/positions")
    assert res_pos.status_code == 200
    pos_data = res_pos.json()
    assert "positions" in pos_data

    res_fills = client.get("/api/fills")
    assert res_fills.status_code == 200
    fills_data = res_fills.json()
    assert "fills" in fills_data


def test_api_broker_switch(client):
    """Verifies /api/settings/broker updates active execution engine."""
    res = client.post("/api/settings/broker", json={"broker": "dry_run"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["active_broker"] == "dry_run"


def test_telegram_alert_html_escaping():
    """Verifies that AlertDispatcher escapes HTML special characters to prevent Telegram 400 errors."""
    from unittest.mock import patch

    from src.dashboard.alerts import AlertDispatcher

    dispatcher = AlertDispatcher()
    dispatcher.bot_token = "dummy_token"
    dispatcher.chat_id = "123456"

    with patch.object(dispatcher, "_send_telegram", return_value=True) as mock_send:
        # Error message with raw unescaped HTML characters (<module>, &, <class 'ValueError'>)
        raw_error = "Failure in <module>: variable 'x' < 10 & type <class 'ValueError'>"
        dispatcher.notify_pipeline_failure("finbert_scoring", raw_error)

        mock_send.assert_called_once()
        sent_html = mock_send.call_args[0][0]
        # Must contain escaped &lt; and &gt; and &amp;
        assert "&lt;module&gt;" in sent_html
        assert "&lt;class 'ValueError'&gt;" in sent_html
        assert "variable 'x' &lt; 10 &amp; type" in sent_html
        # Must not contain unescaped raw angle brackets inside error content
        assert "<module>" not in sent_html
        assert "<class 'ValueError'>" not in sent_html


def test_api_batch_rebalance_success(client):
    """Verifies /api/rebalance validates RebalanceOrderItem models and executes batch orders."""
    payload = {
        "orders": [
            {"ticker": "NVDA", "shares": 10.0, "side": "buy", "price": 120.0},
            {"ticker": "MSFT", "shares": 5.0, "side": "sell", "price": 400.0},
        ]
    }
    res = client.post("/api/rebalance", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["executed_count"] == 2
    assert len(data["results"]) == 2


def test_api_batch_rebalance_validation_failure(client):
    """Verifies Pydantic schema validation rejects malformed order payloads with HTTP 422."""
    # Malformed: shares is a string that cannot be cast to float
    payload = {
        "orders": [
            {"ticker": "AAPL", "shares": "not_a_number", "side": "buy"}
        ]
    }
    res = client.post("/api/rebalance", json=payload)
    assert res.status_code == 422


def test_api_circuit_breaker_endpoint(client):
    """Verifies /api/risk/circuit-breaker endpoint executes emergency liquidation."""
    payload = {
        "tickers": ["AAPL"],
        "liquidate_to": "cash"
    }
    res = client.post("/api/risk/circuit-breaker", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "liquidations_count" in data
    assert data["liquidated_to"] == "cash"

