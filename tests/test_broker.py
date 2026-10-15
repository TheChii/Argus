"""Tests for Argus Quant broker execution clients (Alpaca, Trading 212, Dry Run)."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dashboard.database import DashboardDB
from src.portfolio.broker import (
    AlpacaBrokerClient,
    DryRunBrokerClient,
    Trading212BrokerClient,
    get_broker_client,
)


@pytest.fixture
def temp_db():
    """Provides a clean temporary SQLite database for broker test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_ledger.db"
        db = DashboardDB(db_path=db_path)
        yield db


def test_dry_run_broker_lifecycle(temp_db):
    """Verifies simulated dry run broker operations, balance calculations, and fill logging."""
    broker = DryRunBrokerClient(initial_cash=50000.0)
    broker.db = temp_db

    acct = broker.get_account_summary()
    assert acct["configured"] is True
    assert acct["status"] == "SIMULATION_ACTIVE"
    assert acct["cash"] == 50000.0

    # Submit single order
    receipt = broker.submit_order("AAPL", quantity=10, side="buy", estimated_price=150.0)
    assert receipt["status"] == "FILLED"
    assert receipt["ticker"] == "AAPL"
    assert receipt["shares"] == 10
    assert receipt["filled_price"] == 150.0
    assert receipt["order_id"].startswith("dry_")

    # Submit batch orders and persist to ledger
    batch_orders = [
        {"ticker": "MSFT", "shares": 5, "price": 400.0, "side": "buy"},
        {"ticker": "NVDA", "shares": 8, "price": 120.0, "side": "buy"},
    ]
    results = broker.submit_batch_orders(batch_orders, record_to_ledger=True)
    assert len(results) == 2
    assert all(r["status"] == "FILLED" for r in results)

    # Verify ledger records
    fills_df = temp_db.get_fills()
    assert len(fills_df) == 2
    assert "broker" in fills_df.columns
    assert "broker_order_id" in fills_df.columns
    assert fills_df.iloc[0]["broker"] == "dry_run_simulation"


def test_alpaca_broker_unconfigured():
    """Verifies Alpaca client gracefully handles missing API credentials."""
    client = AlpacaBrokerClient(api_key="", secret_key="", paper=True)
    assert client.is_configured() is False

    acct = client.get_account_summary()
    assert acct["configured"] is False
    assert acct["status"] == "UNCONFIGURED"

    positions = client.get_open_positions()
    assert positions == []

    with pytest.raises(ValueError, match="not configured"):
        client.submit_order("AAPL", 5)


@patch("requests.get")
@patch("requests.post")
def test_alpaca_broker_mocked_execution(mock_post, mock_get, temp_db):
    """Verifies Alpaca REST API request payload formatting and response parsing."""
    client = AlpacaBrokerClient(api_key="test_key", secret_key="test_secret", paper=True)
    client.db = temp_db

    # Mock account response
    mock_acct_resp = MagicMock()
    mock_acct_resp.status_code = 200
    mock_acct_resp.json.return_value = {
        "status": "ACTIVE",
        "portfolio_value": "105200.50",
        "cash": "25000.00",
        "buying_power": "50000.00",
        "currency": "USD",
        "daytrade_count": 0,
    }

    # Mock positions response
    mock_pos_resp = MagicMock()
    mock_pos_resp.status_code = 200
    mock_pos_resp.json.return_value = [
        {
            "symbol": "AAPL",
            "qty": "15",
            "market_value": "3450.0",
            "current_price": "230.0",
            "avg_entry_price": "220.0",
            "unrealized_pl": "150.0",
        }
    ]

    mock_get.side_effect = [mock_acct_resp, mock_pos_resp, mock_acct_resp]

    # Test account summary
    acct = client.get_account_summary()
    assert acct["configured"] is True
    assert acct["portfolio_value"] == 105200.50
    assert acct["cash"] == 25000.00

    # Test positions
    positions = client.get_open_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "AAPL"
    assert positions[0]["shares"] == 15.0

    # Mock order submission
    mock_order_resp = MagicMock()
    mock_order_resp.status_code = 200
    mock_order_resp.json.return_value = {
        "id": "alpaca_ord_99812",
        "symbol": "MSFT",
        "status": "accepted",
        "filled_avg_price": None,
    }
    mock_post.return_value = mock_order_resp

    receipt = client.submit_order("MSFT", quantity=4, side="buy", estimated_price=425.0)
    assert receipt["order_id"] == "alpaca_ord_99812"
    assert receipt["status"] == "ACCEPTED"
    assert receipt["ticker"] == "MSFT"
    assert receipt["filled_price"] == 425.0

    # Verify payload format sent to Alpaca
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["json"] == {
        "symbol": "MSFT",
        "qty": 4,
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }


def test_trading212_symbol_mapping():
    """Verifies symbol translation to/from Trading 212 format."""
    assert Trading212BrokerClient.map_to_t212_ticker("AAPL") == "AAPL_US_EQ"
    assert Trading212BrokerClient.map_to_t212_ticker("BRK-B") == "BRK_B_US_EQ"
    assert Trading212BrokerClient.map_to_t212_ticker("BRK.B") == "BRK_B_US_EQ"
    assert Trading212BrokerClient.map_to_t212_ticker("BRK/B") == "BRK_B_US_EQ"
    assert Trading212BrokerClient.map_to_t212_ticker("GOOG.L") == "GOOG_L_US_EQ"

    assert Trading212BrokerClient.map_from_t212_ticker("AAPL_US_EQ") == "AAPL"
    assert Trading212BrokerClient.map_from_t212_ticker("BRK_B_US_EQ") == "BRK-B"


def test_trading212_broker_unconfigured():
    """Verifies Trading 212 client safely reports unconfigured state."""
    client = Trading212BrokerClient(api_key="")
    assert client.is_configured() is False

    acct = client.get_account_summary()
    assert acct["configured"] is False
    assert acct["status"] == "UNCONFIGURED"

    with pytest.raises(ValueError, match="not configured"):
        client.submit_order("AAPL", 5)


@patch("requests.get")
@patch("requests.post")
def test_trading212_broker_mocked_execution(mock_post, mock_get, temp_db):
    """Verifies Trading 212 REST API signed quantity and endpoint parsing."""
    client = Trading212BrokerClient(api_key="t212_test_token", env="demo")
    client.db = temp_db

    # Mock cash endpoint
    mock_cash = MagicMock()
    mock_cash.status_code = 200
    mock_cash.json.return_value = {
        "free": 12500.0,
        "total": 35000.0,
        "ppl": 150.0,
        "invested": 22500.0,
    }
    mock_get.return_value = mock_cash

    acct = client.get_account_summary()
    assert acct["configured"] is True
    assert acct["cash"] == 12500.0
    assert acct["portfolio_value"] == 35000.0

    # Mock market buy order
    mock_order = MagicMock()
    mock_order.status_code = 200
    mock_order.json.return_value = {
        "id": 88471234,
        "status": "LOCAL",
    }
    mock_post.return_value = mock_order

    receipt = client.submit_order("AMZN", quantity=10, side="buy", estimated_price=180.0)
    assert receipt["order_id"] == "88471234"
    assert receipt["ticker"] == "AMZN"
    assert receipt["status"] == "ACCEPTED"

    # Verify Trading 212 signed quantity (+10 for buy)
    mock_post.assert_called_with(
        "https://demo.trading212.com/api/v0/equity/orders/market",
        json={"quantity": 10, "ticker": "AMZN_US_EQ"},
        headers={"Authorization": "t212_test_token", "Content-Type": "application/json"},
        timeout=10,
    )

    # Test SELL order signed quantity (-5 for sell)
    mock_post.reset_mock()
    client.submit_order("AMZN", quantity=5, side="sell", estimated_price=185.0)
    mock_post.assert_called_with(
        "https://demo.trading212.com/api/v0/equity/orders/market",
        json={"quantity": -5, "ticker": "AMZN_US_EQ"},
        headers={"Authorization": "t212_test_token", "Content-Type": "application/json"},
        timeout=10,
    )


def test_batch_execution_resilience(temp_db):
    """Verifies that an error on one order does not crash the remaining batch."""
    broker = DryRunBrokerClient()
    broker.db = temp_db

    orders = [
        {"ticker": "AAPL", "shares": 5, "price": 150.0, "side": "buy"},
        {"ticker": "", "shares": 10, "price": 50.0, "side": "buy"},  # Invalid
        {"ticker": "MSFT", "shares": 2, "price": 400.0, "side": "buy"},
    ]

    results = broker.submit_batch_orders(orders, record_to_ledger=True)
    assert len(results) == 2  # The 2 valid ones executed
    assert results[0]["ticker"] == "AAPL"
    assert results[1]["ticker"] == "MSFT"


def test_broker_factory_fallback():
    """Verifies get_broker_client factory falls back gracefully."""
    client = get_broker_client("dry_run")
    assert isinstance(client, DryRunBrokerClient)

    # Unconfigured alpaca should fall back to DryRun
    with patch.object(AlpacaBrokerClient, "is_configured", return_value=False):
        client = get_broker_client("alpaca")
        assert isinstance(client, DryRunBrokerClient)


def test_database_broker_snapshot_persistence(temp_db):
    """Verifies broker account snapshots and fill schemas in SQLite."""
    snap_id = temp_db.record_broker_account_snapshot(
        broker="alpaca_paper",
        portfolio_value=105000.0,
        cash=25000.0,
        buying_power=50000.0,
        currency="USD",
    )
    assert snap_id > 0

    latest = temp_db.get_latest_broker_snapshot("alpaca_paper")
    assert latest is not None
    assert latest["broker"] == "alpaca_paper"
    assert latest["portfolio_value"] == 105000.0
    assert latest["cash"] == 25000.0


def test_trading212_basic_auth_and_fill_price(temp_db):
    """Verifies Trading 212 HTTP Basic Auth header encoding and fill price calculation."""
    client = Trading212BrokerClient(api_key="my_key", api_secret="my_secret")
    headers = client._headers()
    assert headers["Authorization"] == "Basic bXlfa2V5Om15X3NlY3JldA=="

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": 998877,
            "status": "FILLED",
            "filledQuantity": 10.0,
            "filledValue": 1825.50,
        }
        mock_post.return_value = mock_resp
        receipt = client.submit_order("AAPL", quantity=10, side="buy", estimated_price=180.0)
        assert receipt["status"] == "FILLED"
        assert receipt["filled_price"] == 182.55


def test_alpaca_fractional_order_constraints(temp_db):
    """Verifies Alpaca enforces day time_in_force and dot notation for fractional orders."""
    client = AlpacaBrokerClient(api_key="test_key", secret_key="test_secret", paper=True)
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "alpaca_frac_123",
            "symbol": "BRK.B",
            "status": "filled",
            "filled_avg_price": "450.25",
        }
        mock_post.return_value = mock_resp
        receipt = client.submit_order("BRK-B", quantity=2.5, side="buy", time_in_force="gtc", order_type="limit")
        assert receipt["status"] == "FILLED"
        assert receipt["ticker"] == "BRK-B"
        assert receipt["filled_price"] == 450.25

        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["symbol"] == "BRK.B"  # Normalized to dot notation
        assert payload["time_in_force"] == "day"  # Forced to day for fractional
        assert payload["type"] == "market"  # Forced to market for fractional
        assert payload["qty"] == 2.5
