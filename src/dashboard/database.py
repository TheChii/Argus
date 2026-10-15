"""Database and storage layer for the operations dashboard.

Guarantees:
- SQLite database for persistent fill tracking, trades, settings, and alert logs.
- DuckDB analytical queries over parquet files (prices, news, scores).
- Thread-safe and atomic transactions.
"""
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("dashboard_db")


class DashboardDB:
    """Manages SQLite storage for operational ledger, fills, and alert history."""

    def __init__(self, db_path: Path | None = None):
        settings = get_settings()
        self.dashboard_dir = settings.data_dir / "dashboard"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (self.dashboard_dir / "portfolio_ledger.db")
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError as e:
            logger.debug("Failed configuring SQLite WAL mode PRAGMA: %s", e)
        return conn

    def _init_tables(self) -> None:
        """Initializes SQLite schema if not already present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Fills table for manual trade confirmations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fill_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0.0,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Alerts log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    delivered INTEGER DEFAULT 1
                )
            """)

            # Settings key-value store
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Broker account snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS broker_account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    broker TEXT NOT NULL,
                    portfolio_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    buying_power REAL NOT NULL,
                    currency TEXT DEFAULT 'USD'
                )
            """)

            # High-water marks table for trailing stop-losses
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS position_high_water_marks (
                    ticker TEXT PRIMARY KEY,
                    peak_price REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Circuit breaker emergency audit events log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breaker_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    peak_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    hedge_action TEXT DEFAULT 'cash',
                    status TEXT DEFAULT 'EXECUTED',
                    notes TEXT
                )
            """)

            # Safe migration: ensure broker, broker_order_id, status columns exist in fills
            cursor.execute("PRAGMA table_info(fills)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            if "broker" not in existing_cols:
                cursor.execute("ALTER TABLE fills ADD COLUMN broker TEXT DEFAULT 'manual'")
            if "broker_order_id" not in existing_cols:
                cursor.execute("ALTER TABLE fills ADD COLUMN broker_order_id TEXT DEFAULT ''")
            if "status" not in existing_cols:
                cursor.execute("ALTER TABLE fills ADD COLUMN status TEXT DEFAULT 'FILLED'")

            conn.commit()

    def record_fill(
        self,
        fill_date: str,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        commission: float = 0.0,
        notes: str = "",
        broker: str = "manual",
        broker_order_id: str = "",
        status: str = "FILLED",
    ) -> int:
        """Records an executed fill into the persistent ledger."""
        created_at = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO fills (fill_date, ticker, action, shares, price, commission, notes, created_at, broker, broker_order_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (fill_date, ticker.upper(), action.upper(), shares, price, commission, notes, created_at, broker, broker_order_id, status),
            )
            conn.commit()
            return cursor.lastrowid

    def record_broker_account_snapshot(
        self,
        broker: str,
        portfolio_value: float,
        cash: float,
        buying_power: float,
        currency: str = "USD",
    ) -> int:
        """Persists a snapshot of broker account balances."""
        now_iso = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO broker_account_snapshots (timestamp, broker, portfolio_value, cash, buying_power, currency)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now_iso, broker, portfolio_value, cash, buying_power, currency),
            )
            conn.commit()
            return cursor.lastrowid

    def get_latest_broker_snapshot(self, broker: str | None = None) -> dict[str, Any] | None:
        """Retrieves the most recent account snapshot for the broker."""
        with self._get_connection() as conn:
            query = "SELECT * FROM broker_account_snapshots"
            params: tuple = ()
            if broker:
                query += " WHERE broker = ?"
                params = (broker,)
            query += " ORDER BY id DESC LIMIT 1"
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_fills(self, limit: int = 100) -> pd.DataFrame:
        """Retrieves recent executed fills as a pandas DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM fills ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
            return df

    def log_alert(self, level: str, channel: str, title: str, message: str, delivered: bool = True) -> int:
        """Logs an alert dispatch event."""
        now_iso = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts_log (timestamp, level, channel, title, message, delivered)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now_iso, level.upper(), channel.lower(), title, message, 1 if delivered else 0),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_alerts(self, limit: int = 50) -> pd.DataFrame:
        """Retrieves recent alerts dispatched."""
        with self._get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM alerts_log ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )

    def get_alerts(self, limit: int = 50) -> pd.DataFrame:
        """Alias for get_recent_alerts."""
        return self.get_recent_alerts(limit=limit)

    def get_open_positions_from_fills(self) -> dict[str, dict[str, float]]:
        """Aggregates executed fills to calculate current net open positions."""
        df = self.get_fills(limit=10000)
        if df.empty:
            return {}

        positions: dict[str, dict[str, float]] = {}
        for ticker, group in df.groupby("ticker"):
            buys = group[group["action"] == "BUY"]
            sells = group[group["action"] == "SELL"]

            total_bought = float(buys["shares"].sum()) if not buys.empty else 0.0
            total_sold = float(sells["shares"].sum()) if not sells.empty else 0.0
            net_shares = total_bought - total_sold

            avg_buy_price = float((buys["shares"] * buys["price"]).sum() / total_bought) if total_bought > 0 else 0.0

            positions[ticker] = {
                "net_shares": net_shares,
                "shares": net_shares,
                "total_bought": total_bought,
                "total_sold": total_sold,
                "avg_buy_price": round(avg_buy_price, 2),
                "cost_basis": round(avg_buy_price, 2),
            }
        return positions

    def update_high_water_mark(self, ticker: str, current_price: float, cost_basis: float = 0.0) -> float:
        """Updates or registers the peak price (high-water mark) for a ticker. Returns current peak."""
        now_iso = datetime.now(UTC).isoformat()
        initial_peak = max(float(current_price), float(cost_basis))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO position_high_water_marks (ticker, peak_price, cost_basis, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    peak_price = max(position_high_water_marks.peak_price, excluded.peak_price, excluded.cost_basis),
                    cost_basis = CASE WHEN excluded.cost_basis > 0 THEN excluded.cost_basis ELSE position_high_water_marks.cost_basis END,
                    updated_at = excluded.updated_at
                """,
                (ticker.upper(), initial_peak, float(cost_basis), now_iso),
            )
            conn.commit()
            cursor.execute("SELECT peak_price FROM position_high_water_marks WHERE ticker = ?", (ticker.upper(),))
            row = cursor.fetchone()
            return float(row["peak_price"]) if row else initial_peak

    def get_high_water_marks(self) -> dict[str, dict[str, float]]:
        """Retrieves all tracked peak high-water marks and entry costs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ticker, peak_price, cost_basis, updated_at FROM position_high_water_marks")
            rows = cursor.fetchall()
            return {
                r["ticker"]: {
                    "peak_price": float(r["peak_price"]),
                    "cost_basis": float(r["cost_basis"]),
                    "updated_at": r["updated_at"],
                }
                for r in rows
            }

    def reset_high_water_mark(self, ticker: str) -> None:
        """Removes the peak high-water mark tracking for a closed/liquidated ticker."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM position_high_water_marks WHERE ticker = ?", (ticker.upper(),))
            conn.commit()

    def log_circuit_breaker_event(
        self,
        ticker: str,
        trigger_type: str,
        entry_price: float,
        peak_price: float,
        exit_price: float,
        pnl_pct: float,
        hedge_action: str = "cash",
        status: str = "EXECUTED",
        notes: str = "",
    ) -> int:
        """Logs a circuit breaker liquidation event."""
        now_iso = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO circuit_breaker_events (
                    timestamp, ticker, trigger_type, entry_price, peak_price, exit_price, pnl_pct, hedge_action, status, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    ticker.upper(),
                    trigger_type,
                    float(entry_price),
                    float(peak_price),
                    float(exit_price),
                    float(pnl_pct),
                    hedge_action,
                    status,
                    notes,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_circuit_breaker_events(self, limit: int = 50) -> pd.DataFrame:
        """Retrieves recent circuit breaker execution logs."""
        with self._get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM circuit_breaker_events ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )


    def set_setting(self, key: str, value: Any) -> None:
        """Stores a configuration setting atomically."""
        now_iso = datetime.now(UTC).isoformat()
        val_str = json.dumps(value)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (key, val_str, now_iso),
            )
            conn.commit()

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration setting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value_json FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["value_json"])
            return default

    def query_df(self, query: str) -> pd.DataFrame:
        """Executes arbitrary SQL query on the SQLite database and returns a DataFrame."""
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn)


def query_parquet_duckdb(query: str) -> pd.DataFrame:
    """Executes a DuckDB SQL query across data parquet files and returns a DataFrame."""
    con = duckdb.connect(database=":memory:")
    try:
        df = con.execute(query).df()
        return df
    finally:
        con.close()
