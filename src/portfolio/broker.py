"""Broker execution clients for Argus Quant.

Supported Platforms:
1. Alpaca Markets (Paper & Live REST API)
2. Trading 212 (Demo & Live REST API)
3. Dry Run Simulator (Zero-key local paper sandbox)
"""
import base64
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import requests

from src.config.settings import get_settings
from src.config.universe import denormalize_ticker
from src.dashboard.database import DashboardDB
from src.utils.logger import setup_logger

logger = setup_logger("broker")


class BaseBrokerClient(ABC):
    """Abstract interface defining required execution operations for brokers."""

    def __init__(self, name: str, environment: str = "paper"):
        self.name = name
        self.environment = environment
        self.db = DashboardDB()

    @abstractmethod
    def get_account_summary(self) -> dict[str, Any]:
        """Returns account equity, free cash, and buying power."""
        pass

    @abstractmethod
    def get_open_positions(self) -> list[dict[str, Any]]:
        """Returns list of open positions currently held at the broker."""
        pass

    @abstractmethod
    def submit_order(
        self,
        ticker: str,
        quantity: float,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        estimated_price: float = 0.0,
    ) -> dict[str, Any]:
        """Submits an order to the broker and returns execution result."""
        pass

    def submit_batch_orders(
        self,
        orders: list[dict[str, Any]],
        record_to_ledger: bool = True,
    ) -> list[dict[str, Any]]:
        """Executes a list of orders in sequence and persists fills into SQLite ledger."""
        results: list[dict[str, Any]] = []
        fill_date = datetime.now(UTC).strftime("%Y-%m-%d")

        for ord_spec in orders:
            ticker = ord_spec.get("ticker", "").upper()
            shares = float(ord_spec.get("shares", 0.0))
            side = ord_spec.get("side", "buy").lower()
            price = float(ord_spec.get("price", 0.0))

            if not ticker or shares <= 0:
                continue

            try:
                receipt = self.submit_order(
                    ticker=ticker,
                    quantity=shares,
                    side=side,
                    order_type="market",
                    time_in_force="day",
                    estimated_price=price,
                )
                results.append(receipt)

                if record_to_ledger and receipt.get("status") in ("FILLED", "ACCEPTED"):
                    actual_price = receipt.get("filled_price") or price
                    self.db.record_fill(
                        fill_date=fill_date,
                        ticker=ticker,
                        action=side.upper(),
                        shares=shares,
                        price=actual_price,
                        commission=0.0,
                        notes=f"Auto-routed via {self.name} ({self.environment})",
                        broker=f"{self.name}_{self.environment}",
                        broker_order_id=receipt.get("order_id", ""),
                        status=receipt.get("status", "FILLED"),
                    )
            except Exception as e:
                err_msg = str(e)
                logger.error(f"Failed to submit order for {ticker}: {err_msg}")
                results.append({
                    "order_id": "",
                    "ticker": ticker,
                    "side": side,
                    "shares": shares,
                    "status": "REJECTED",
                    "error": err_msg,
                    "filled_price": 0.0,
                })

        # Persist updated account snapshot after batch execution
        try:
            summary = self.get_account_summary()
            if summary and "portfolio_value" in summary:
                self.db.record_broker_account_snapshot(
                    broker=f"{self.name}_{self.environment}",
                    portfolio_value=float(summary.get("portfolio_value", 0.0)),
                    cash=float(summary.get("cash", 0.0)),
                    buying_power=float(summary.get("buying_power", 0.0)),
                    currency=str(summary.get("currency", "USD")),
                )
        except Exception as e:
            logger.warning(f"Could not record broker balance snapshot: {e}")

        return results


class AlpacaBrokerClient(BaseBrokerClient):
    """Alpaca Markets Paper & Live Trading Client via REST API v2."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        base_url: str | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.alpaca_api_key or ""
        self.secret_key = secret_key or settings.alpaca_secret_key or ""
        self.paper = paper if api_key is not None else settings.alpaca_paper
        env_name = "paper" if self.paper else "live"
        super().__init__(name="alpaca", environment=env_name)

        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = (
                "https://paper-api.alpaca.markets/v2" if self.paper
                else "https://api.alpaca.markets/v2"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def get_account_summary(self) -> dict[str, Any]:
        if not self.is_configured():
            return {
                "configured": False,
                "status": "UNCONFIGURED",
                "portfolio_value": 0.0,
                "cash": 0.0,
                "buying_power": 0.0,
                "currency": "USD",
                "message": "Alpaca API credentials missing in .env",
            }

        url = f"{self.base_url}/account"
        resp = requests.get(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return {
            "configured": True,
            "status": data.get("status", "ACTIVE"),
            "portfolio_value": float(data.get("portfolio_value", 0.0)),
            "cash": float(data.get("cash", 0.0)),
            "buying_power": float(data.get("buying_power", 0.0)),
            "currency": data.get("currency", "USD"),
            "daytrade_count": int(data.get("daytrade_count", 0)),
        }

    def get_open_positions(self) -> list[dict[str, Any]]:
        if not self.is_configured():
            return []

        url = f"{self.base_url}/positions"
        resp = requests.get(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        positions = []

        for p in resp.json():
            positions.append({
                "ticker": p.get("symbol", "").upper(),
                "shares": float(p.get("qty", 0.0)),
                "market_value": float(p.get("market_value", 0.0)),
                "current_price": float(p.get("current_price", 0.0)),
                "avg_entry_price": float(p.get("avg_entry_price", 0.0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0.0)),
            })
        return positions

    def submit_order(
        self,
        ticker: str,
        quantity: float,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        estimated_price: float = 0.0,
    ) -> dict[str, Any]:
        if not self.is_configured():
            raise ValueError("Alpaca API credentials are not configured in .env")

        url = f"{self.base_url}/orders"
        # Alpaca expects dot notation for share classes (e.g. BRK.B, not BRK-B or BRK/B)
        alpaca_symbol = denormalize_ticker(ticker.upper()).replace("/", ".")
        is_fractional = not float(quantity).is_integer()
        qty_val = int(quantity) if not is_fractional else round(float(quantity), 4)

        # Alpaca API rule: Fractional orders strictly require time_in_force='day' and type='market'
        effective_tif = "day" if is_fractional else time_in_force.lower()
        effective_type = "market" if is_fractional else order_type.lower()

        payload = {
            "symbol": alpaca_symbol,
            "qty": qty_val,
            "side": side.lower(),
            "type": effective_type,
            "time_in_force": effective_tif,
        }

        resp = requests.post(url, json=payload, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        filled_price = float(data.get("filled_avg_price") or estimated_price)
        status = data.get("status", "accepted").upper()
        if status in ("FILLED", "PARTIALLY_FILLED"):
            canonical_status = "FILLED"
        elif status in ("ACCEPTED", "NEW", "PENDING_NEW"):
            canonical_status = "ACCEPTED"
        else:
            canonical_status = status

        return {
            "order_id": str(data.get("id")),
            "ticker": ticker.upper(),
            "side": side.lower(),
            "shares": float(quantity),
            "status": canonical_status,
            "filled_price": filled_price,
            "broker": self.name,
            "raw_response": data,
        }


class Trading212BrokerClient(BaseBrokerClient):
    """Trading 212 Demo & Live Trading Client via REST API v0."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        env: str | None = None,
        base_url: str | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.trading212_api_key or ""
        self.api_secret = api_secret or settings.trading212_api_secret or ""
        self.env = (env or settings.trading212_env or "demo").lower()
        super().__init__(name="trading212", environment=self.env)

        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = (
                "https://demo.trading212.com/api/v0" if self.env == "demo"
                else "https://live.trading212.com/api/v0"
            )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_secret:
            # Trading 212 HTTP Basic Auth: base64(API_KEY:API_SECRET)
            raw_creds = f"{self.api_key}:{self.api_secret}"
            b64_creds = base64.b64encode(raw_creds.encode("utf-8")).decode("utf-8")
            headers["Authorization"] = f"Basic {b64_creds}"
        elif self.api_key:
            headers["Authorization"] = self.api_key
        return headers

    def is_configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def map_to_t212_ticker(ticker: str) -> str:
        """Maps standard US ticker symbol to Trading 212 instrument format (e.g. AAPL -> AAPL_US_EQ, BRK.B -> BRK_B_US_EQ)."""
        clean = ticker.upper().replace(".", "_").replace("-", "_").replace("/", "_")
        return f"{clean}_US_EQ"

    @staticmethod
    def map_from_t212_ticker(t212_ticker: str) -> str:
        """Strips Trading 212 suffix back to standard symbol."""
        if t212_ticker.endswith("_US_EQ"):
            return t212_ticker[:-6].replace("_", "-")
        return t212_ticker

    def get_account_summary(self) -> dict[str, Any]:
        if not self.is_configured():
            return {
                "configured": False,
                "status": "UNCONFIGURED",
                "portfolio_value": 0.0,
                "cash": 0.0,
                "buying_power": 0.0,
                "currency": "EUR/USD",
                "message": "Trading 212 API key missing in .env",
            }

        url = f"{self.base_url}/equity/account/cash"
        resp = requests.get(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        free_cash = float(data.get("free", 0.0))
        total_equity = float(data.get("total", 0.0))

        return {
            "configured": True,
            "status": "ACTIVE",
            "portfolio_value": total_equity,
            "cash": free_cash,
            "buying_power": free_cash,
            "invested": float(data.get("invested", 0.0)),
            "ppl": float(data.get("ppl", 0.0)),
            "currency": "USD",
        }

    def get_open_positions(self) -> list[dict[str, Any]]:
        if not self.is_configured():
            return []

        url = f"{self.base_url}/equity/portfolio"
        resp = requests.get(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        positions = []

        for p in resp.json():
            raw_t = p.get("ticker", "")
            positions.append({
                "ticker": self.map_from_t212_ticker(raw_t),
                "t212_ticker": raw_t,
                "shares": float(p.get("quantity", 0.0)),
                "current_price": float(p.get("currentPrice", 0.0)),
                "avg_entry_price": float(p.get("averagePrice", 0.0)),
                "ppl": float(p.get("ppl", 0.0)),
            })
        return positions

    def submit_order(
        self,
        ticker: str,
        quantity: float,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        estimated_price: float = 0.0,
    ) -> dict[str, Any]:
        if not self.is_configured():
            raise ValueError("Trading 212 API key is not configured in .env")

        url = f"{self.base_url}/equity/orders/market"
        # In Trading 212 market orders: positive quantity for BUY, negative for SELL
        qty_signed = abs(quantity) if side.lower() == "buy" else -abs(quantity)
        t212_sym = self.map_to_t212_ticker(ticker)

        payload = {
            "quantity": qty_signed,
            "ticker": t212_sym,
        }

        resp = requests.post(url, json=payload, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        order_id = str(data.get("id") or data.get("orderId") or uuid.uuid4().hex[:8])
        raw_status = str(data.get("status", "ACCEPTED")).upper()
        if raw_status in ("FILLED", "PARTIALLY_FILLED"):
            status = "FILLED"
        elif raw_status in ("LOCAL", "NEW", "CONFIRMED", "ACCEPTED"):
            status = "ACCEPTED"
        else:
            status = raw_status

        # If order was filled and filledValue / filledQuantity are available, compute realized price
        filled_qty = float(data.get("filledQuantity", 0.0) or 0.0)
        filled_val = float(data.get("filledValue", 0.0) or 0.0)
        if filled_qty > 0 and filled_val > 0:
            realized_price = round(filled_val / filled_qty, 4)
        else:
            realized_price = estimated_price

        return {
            "order_id": order_id,
            "ticker": ticker.upper(),
            "side": side.lower(),
            "shares": float(quantity),
            "status": status,
            "filled_price": realized_price,
            "broker": self.name,
            "raw_response": data,
        }


class DryRunBrokerClient(BaseBrokerClient):
    """Local high-fidelity simulated paper trading sandbox requiring zero API keys."""

    def __init__(self, initial_cash: float = 100000.0):
        super().__init__(name="dry_run", environment="simulation")
        self.initial_cash = initial_cash
        self._positions: dict[str, dict[str, float]] = {}

    def get_account_summary(self) -> dict[str, Any]:
        # Calculate cash and equity based on fills in SQLite ledger
        open_pos = self.db.get_open_positions_from_fills()
        pos_value = sum(p.get("shares", 0.0) * p.get("cost_basis", 100.0) for p in open_pos.values())
        cash = max(0.0, self.initial_cash - pos_value)

        return {
            "configured": True,
            "status": "SIMULATION_ACTIVE",
            "portfolio_value": self.initial_cash,
            "cash": cash,
            "buying_power": cash,
            "currency": "USD",
            "simulated": True,
        }

    def get_open_positions(self) -> list[dict[str, Any]]:
        open_pos = self.db.get_open_positions_from_fills()
        res = []
        for sym, d in open_pos.items():
            cost = float(d.get("cost_basis", 100.0))
            shares = float(d.get("shares", 0.0))
            if shares <= 0:
                continue
            res.append({
                "ticker": sym,
                "shares": shares,
                "cost_basis": cost,
                "avg_entry_price": cost,
                "current_price": cost,
                "market_value": round(shares * cost, 2),
            })
        return res

    def submit_order(
        self,
        ticker: str,
        quantity: float,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        estimated_price: float = 0.0,
    ) -> dict[str, Any]:
        fill_price = estimated_price if estimated_price > 0 else 100.0
        order_id = f"dry_{uuid.uuid4().hex[:8]}"

        return {
            "order_id": order_id,
            "ticker": ticker.upper(),
            "side": side.lower(),
            "shares": float(quantity),
            "status": "FILLED",
            "filled_price": fill_price,
            "broker": "dry_run",
            "simulated": True,
        }


def get_broker_client(broker_name: str | None = None) -> BaseBrokerClient:
    """Factory creating the appropriate broker client with graceful fallback."""
    settings = get_settings()
    choice = (broker_name or settings.default_broker or "dry_run").lower()

    if choice == "alpaca":
        client = AlpacaBrokerClient()
        if client.is_configured():
            return client
        logger.warning("Alpaca credentials missing. Falling back to DryRun simulation.")
        return DryRunBrokerClient()

    elif choice in ("trading212", "t212"):
        client = Trading212BrokerClient()
        if client.is_configured():
            return client
        logger.warning("Trading 212 credentials missing. Falling back to DryRun simulation.")
        return DryRunBrokerClient()

    return DryRunBrokerClient()
