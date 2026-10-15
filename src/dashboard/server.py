"""FastAPI backend server for the Argus Quant Trading Terminal.

Serves:
- Modern HTML5 / Tailwind CSS / Vanilla JS single-page application.
- High-performance REST APIs for market data, snapshot loading, broker execution,
  rebalance calculations, and SQLite ledger synchronization.
"""
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.dashboard.alerts import AlertDispatcher
from src.dashboard.database import DashboardDB
from src.dashboard.investor_logic import (
    calculate_share_allocations,
    generate_broker_order_text,
    get_latest_prices,
)
from src.dashboard.snapshot import DashboardSnapshotBuilder, load_today_snapshot
from src.portfolio.broker import (
    AlpacaBrokerClient,
    BaseBrokerClient,
    Trading212BrokerClient,
    get_broker_client,
)
from src.portfolio.risk_monitor import IntradayRiskMonitor
from src.utils.logger import setup_logger

logger = setup_logger("dashboard_server")

# Global active broker choice override (defaults to settings/dry_run)
_ACTIVE_BROKER_OVERRIDE: str | None = None


def get_current_broker() -> BaseBrokerClient:
    """Returns the currently active broker client instance."""
    return get_broker_client(_ACTIVE_BROKER_OVERRIDE)


# Initialize FastAPI app
app = FastAPI(
    title="Argus Quant Trading Terminal API",
    description="Institutional-grade quant trading terminal API and static server",
    version="2.3.0",
)

# CORS middleware for local developer flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Request / Response Models
# -----------------------------------------------------------------------------
class OrderRequest(BaseModel):
    ticker: str
    quantity: float
    side: str = "buy"
    price: float = 0.0


class RebalanceOrderItem(BaseModel):
    ticker: str
    shares: float
    side: str = "buy"
    price: float = 0.0


class BatchRebalanceRequest(BaseModel):
    orders: list[RebalanceOrderItem]


class SizerRequest(BaseModel):
    budget: float = 10000.0
    allow_fractional: bool = False


class BrokerSelectRequest(BaseModel):
    broker: str  # alpaca, trading212, dry_run


class CircuitBreakerRequest(BaseModel):
    tickers: list[str] | None = None
    liquidate_to: str = "cash"  # 'cash' or 'hedged_spy'



# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def api_health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/api/snapshot")
def api_get_snapshot():
    """Returns the pre-computed operations and model snapshot (<5ms latency)."""
    try:
        data = load_today_snapshot()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Failed to load snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/account")
def api_get_account():
    """Returns real-time balance metrics from active broker."""
    broker = get_current_broker()
    try:
        summary = broker.get_account_summary()
        alpaca_configured = AlpacaBrokerClient().is_configured()
        t212_configured = Trading212BrokerClient().is_configured()
        return {
            "broker": broker.name,
            "environment": broker.environment,
            "portfolio_value": float(summary.get("portfolio_value", 100000.0)),
            "cash": float(summary.get("cash", 25000.0)),
            "buying_power": float(summary.get("buying_power", 50000.0)),
            "currency": str(summary.get("currency", "USD")),
            "integrations": {
                "alpaca": alpaca_configured,
                "trading212": t212_configured,
                "dry_run": True,
            },
        }
    except Exception as e:
        logger.error(f"Failed to retrieve account summary: {e}")
        return {
            "broker": broker.name,
            "environment": broker.environment,
            "portfolio_value": 100000.0,
            "cash": 25000.0,
            "buying_power": 50000.0,
            "currency": "USD",
            "integrations": {"alpaca": False, "trading212": False, "dry_run": True},
        }


@app.post("/api/order")
def api_submit_order(req: OrderRequest):
    """Executes a single market order via the active broker and persists fill to SQLite."""
    broker = get_current_broker()
    db = DashboardDB()
    ticker = req.ticker.strip().upper()
    shares = float(req.quantity)
    side = req.side.strip().lower()
    price = float(req.price)

    if not ticker or shares <= 0:
        raise HTTPException(status_code=400, detail="Invalid ticker or quantity.")

    try:
        receipt = broker.submit_order(
            ticker=ticker,
            quantity=shares,
            side=side,
            order_type="market",
            time_in_force="day",
            estimated_price=price,
        )
        fill_status = receipt.get("status", "FILLED")
        fill_px = receipt.get("filled_price") or price or 100.0
        order_id = receipt.get("order_id", f"ord_{int(datetime.now().timestamp())}")

        fill_id = db.record_fill(
            fill_date=datetime.now(UTC).strftime("%Y-%m-%d"),
            ticker=ticker,
            action=side.upper(),
            shares=shares,
            price=fill_px,
            commission=0.0,
            notes=f"Web terminal order via {broker.name} ({broker.environment})",
            broker=f"{broker.name}_{broker.environment}",
            broker_order_id=order_id,
            status=fill_status,
        )

        return {
            "success": True,
            "fill_id": fill_id,
            "order_id": order_id,
            "status": fill_status,
            "ticker": ticker,
            "side": side.upper(),
            "shares": shares,
            "price": fill_px,
        }
    except Exception as e:
        logger.error(f"Order submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rebalance")
def api_submit_rebalance(req: BatchRebalanceRequest):
    """Executes batch rebalance orders and auto-persists fills into SQLite ledger."""
    broker = get_current_broker()
    if not req.orders:
        raise HTTPException(status_code=400, detail="No orders provided.")

    try:
        order_dicts = [ord_item.model_dump() for ord_item in req.orders]
        results = broker.submit_batch_orders(order_dicts, record_to_ledger=True)
        return {"success": True, "executed_count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Rebalance execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/size")
def api_calculate_sizing(req: SizerRequest):
    """Calculates dollar-to-shares allocation and retail copy slips for a given budget."""
    try:
        snapshot = load_today_snapshot()
        actions = snapshot.get("today_cockpit", {})["actions"]
        top_picks = [a for a in actions if a.get("rank", 999) <= 15 and a.get("target_weight", 0.0) > 0]
        tickers = [p["ticker"] for p in top_picks]
        prices = get_latest_prices(tickers)

        summary = calculate_share_allocations(
            budget=req.budget,
            target_picks=top_picks,
            prices=prices,
            allow_fractional=req.allow_fractional,
        )

        date_str = snapshot.get("as_of_date", "")
        slips = {
            "standard": generate_broker_order_text(summary, broker_format="Standard", as_of_date=date_str),
            "fidelity": generate_broker_order_text(summary, broker_format="Fidelity / Schwab", as_of_date=date_str),
            "robinhood": generate_broker_order_text(summary, broker_format="Robinhood / Webull", as_of_date=date_str),
            "ibkr": generate_broker_order_text(summary, broker_format="Interactive Brokers", as_of_date=date_str, clean_csv=False),
            "ibkr_csv": generate_broker_order_text(summary, broker_format="Interactive Brokers", as_of_date=date_str, clean_csv=True),
        }

        return {"summary": summary, "slips": slips}
    except Exception as e:
        logger.error(f"Sizing calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions")
def api_get_positions():
    """Returns open positions from SQLite ledger with real-time mark-to-market P&L."""
    db = DashboardDB()
    open_pos = db.get_open_positions_from_fills()
    tickers = list(open_pos.keys())
    prices = get_latest_prices(tickers) if tickers else {}

    results = []
    total_market_val = 0.0
    total_cost_val = 0.0
    total_unrealized_pnl = 0.0

    for sym, p in sorted(open_pos.items()):
        net_sh = float(p.get("net_shares", 0.0))
        if net_sh <= 0:
            continue
        cost_px = float(p.get("avg_buy_price", 100.0))
        cur_px = float(prices.get(sym, cost_px))
        mv = net_sh * cur_px
        cost_tot = net_sh * cost_px
        pnl = mv - cost_tot
        pnl_pct = (pnl / cost_tot * 100.0) if cost_tot > 0 else 0.0

        total_market_val += mv
        total_cost_val += cost_tot
        total_unrealized_pnl += pnl

        results.append({
            "ticker": sym,
            "shares": round(net_sh, 4),
            "avg_buy_price": round(cost_px, 2),
            "current_price": round(cur_px, 2),
            "market_value": round(mv, 2),
            "cost_value": round(cost_tot, 2),
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 2),
            "fill_count": p.get("fill_count", 1),
        })

    return {
        "positions": results,
        "total_market_value": round(total_market_val, 2),
        "total_cost_value": round(total_cost_val, 2),
        "total_unrealized_pnl": round(total_unrealized_pnl, 2),
        "total_unrealized_pnl_pct": round((total_unrealized_pnl / total_cost_val * 100.0) if total_cost_val > 0 else 0.0, 2),
    }


@app.get("/api/fills")
def api_get_fills(limit: int = 50):
    """Returns executed trade fills from SQLite ledger."""
    db = DashboardDB()
    df = db.get_fills(limit=limit)
    if df.empty:
        return {"fills": []}
    return {"fills": df.to_dict(orient="records")}


@app.post("/api/refresh-snapshot")
def api_refresh_snapshot():
    """Regenerates the pre-computed snapshot file."""
    try:
        builder = DashboardSnapshotBuilder()
        snapshot = builder.generate_and_save_snapshot()
        return {"success": True, "as_of_date": snapshot.get("as_of_date")}
    except Exception as e:
        logger.error(f"Snapshot refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings/broker")
def api_set_broker(req: BrokerSelectRequest):
    """Switches the active broker platform."""
    global _ACTIVE_BROKER_OVERRIDE
    choice = req.broker.strip().lower()
    if choice not in ("alpaca", "trading212", "dry_run"):
        raise HTTPException(status_code=400, detail="Invalid broker choice.")
    _ACTIVE_BROKER_OVERRIDE = choice
    active = get_current_broker()
    return {"success": True, "active_broker": active.name, "environment": active.environment}


@app.post("/api/settings/test-alert")
def api_test_alert():
    """Triggers an operational alert test."""
    db = DashboardDB()
    dispatcher = AlertDispatcher(db=db)
    dispatcher.notify(
        level="INFO",
        title="Operational Alert Test",
        message="Testing Argus Quant notification pipeline from web terminal. All systems operational.",
    )
    return {"success": True, "message": "Test alert dispatched and logged to SQLite audit ledger."}


@app.get("/api/risk/monitor")
def api_get_risk_monitor():
    """Returns intraday risk telemetry, stop-loss thresholds, and breach status for all open positions."""
    try:
        broker = get_current_broker()
        monitor = IntradayRiskMonitor(broker=broker)
        positions = monitor.check_positions()
        events_df = monitor.db.get_circuit_breaker_events(limit=20)
        recent_events = events_df.to_dict(orient="records") if not events_df.empty else []

        breached_count = sum(1 for p in positions if p.get("status") == "BREACHED")
        caution_count = sum(1 for p in positions if p.get("status") == "CAUTION")

        return {
            "positions": positions,
            "total_open": len(positions),
            "breached_count": breached_count,
            "caution_count": caution_count,
            "system_status": "CRITICAL" if breached_count > 0 else ("WARNING" if caution_count > 0 else "NORMAL"),
            "recent_events": recent_events,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"Risk monitor evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/circuit-breaker")
def api_trigger_circuit_breaker(req: CircuitBreakerRequest):
    """Executes emergency liquidation of breached holdings to cash or hedged SPY."""
    try:
        broker = get_current_broker()
        monitor = IntradayRiskMonitor(broker=broker)
        results = monitor.execute_circuit_breaker(
            breached_tickers=req.tickers,
            liquidate_to=req.liquidate_to,
        )
        return {
            "success": True,
            "liquidations_count": len(results),
            "liquidated_to": req.liquidate_to,
            "executions": results,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"Circuit breaker execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# -----------------------------------------------------------------------------
# Static Web Assets Serving
# -----------------------------------------------------------------------------
# Mount static files (css, js, assets)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the single-page HTML5 trading terminal."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(str(index_file))
