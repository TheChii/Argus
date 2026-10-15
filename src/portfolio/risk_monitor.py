"""Intraday Risk Circuit Breakers & Emergency Stop-Loss Monitor."""
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src.dashboard.alerts import AlertDispatcher
from src.dashboard.database import DashboardDB
from src.data.prices import PricePipeline
from src.features.technical import compute_atr
from src.portfolio.broker import BaseBrokerClient, get_broker_client
from src.utils.logger import setup_logger

logger = setup_logger("risk_monitor")


class IntradayRiskMonitor:
    """
    Intraday risk monitor evaluating real-time positions against drawdown thresholds.
    Triggers emergency liquidations when hard stops, trailing stops, or ATR volatility stops are breached.
    """

    def __init__(
        self,
        db: DashboardDB | None = None,
        broker: BaseBrokerClient | None = None,
        hard_stop_pct: float = 0.07,
        trailing_stop_pct: float = 0.07,
        atr_multiplier: float = 2.5,
        alerts: AlertDispatcher | None = None,
        price_pipeline: PricePipeline | None = None,
    ):
        self.db = db or DashboardDB()
        if broker is None:
            try:
                self.broker = get_broker_client()
            except Exception as e:
                logger.warning(f"Default broker client resolution failed: {e}")
                self.broker = None
        else:
            self.broker = broker

        self.hard_stop_pct = float(hard_stop_pct)
        self.trailing_stop_pct = float(trailing_stop_pct)
        self.atr_multiplier = float(atr_multiplier)
        self.alerts = alerts or AlertDispatcher(db=self.db)
        self.price_pipeline = price_pipeline or PricePipeline()

    def get_atr(self, ticker: str, period: int = 14) -> float:
        """Computes current 14-day ATR for ticker, falling back to 0.0 if unavailable."""
        try:
            p_df = self.price_pipeline.load_ticker_prices(ticker)
            if p_df is not None and not p_df.empty and len(p_df) >= 5:
                atr_series = compute_atr(p_df, period=period)
                if not atr_series.empty:
                    val = float(atr_series.dropna().iloc[-1])
                    if val > 0 and not np.isnan(val):
                        return round(val, 4)
        except Exception as e:
            logger.warning(f"Could not compute ATR for {ticker}: {e}")
        return 0.0

    def check_positions(self, live_prices: dict[str, float] | None = None) -> list[dict[str, Any]]:
        """
        Evaluates open positions against hard drawdown, trailing stop, and ATR multiples.
        Returns detailed telemetry including status ('HEALTHY', 'CAUTION', 'BREACHED').
        """
        raw_positions: list[dict[str, Any]] = []
        if self.broker is not None:
            try:
                raw_positions = self.broker.get_open_positions()
            except Exception as e:
                logger.warning(f"Failed to fetch open positions from broker: {e}")

        if not raw_positions:
            pos_map = self.db.get_open_positions_from_fills()
            for tkr, data in pos_map.items():
                net_sh = float(data.get("shares", data.get("net_shares", 0.0)))
                if net_sh > 0:
                    cost = float(data.get("cost_basis", data.get("avg_buy_price", 0.0)))
                    raw_positions.append({
                        "ticker": tkr.upper(),
                        "shares": net_sh,
                        "cost_basis": cost,
                        "avg_entry_price": cost,
                        "current_price": cost,
                    })

        results: list[dict[str, Any]] = []

        for pos in raw_positions:
            ticker = str(pos.get("ticker", "")).upper()
            shares = float(pos.get("shares", pos.get("quantity", pos.get("qty", 0.0))))
            if not ticker or shares <= 0:
                continue

            cost_basis = float(pos.get("cost_basis", pos.get("avg_entry_price", 0.0)))

            # Determine quote price
            current_price = 0.0
            if live_prices and ticker in live_prices:
                current_price = float(live_prices[ticker])
            elif pos.get("current_price") and float(pos.get("current_price")) > 0:
                current_price = float(pos.get("current_price"))
            else:
                try:
                    p_df = self.price_pipeline.load_ticker_prices(ticker)
                    if p_df is not None and not p_df.empty and "close" in p_df.columns:
                        current_price = float(p_df["close"].dropna().iloc[-1])
                except (ValueError, KeyError, OSError) as e:
                    logger.debug("Could not load price history for %s: %s", ticker, e)

            if current_price <= 0:
                current_price = cost_basis if cost_basis > 0 else 100.0

            # Update peak high-water mark
            peak_price = self.db.update_high_water_mark(
                ticker=ticker,
                current_price=current_price,
                cost_basis=cost_basis,
            )
            if cost_basis > 0 and peak_price < cost_basis:
                peak_price = cost_basis

            # Drawdowns
            drawdown_entry_pct = ((current_price - cost_basis) / cost_basis * 100.0) if cost_basis > 0 else 0.0
            drawdown_peak_pct = ((current_price - peak_price) / peak_price * 100.0) if peak_price > 0 else 0.0

            # ATR evaluation
            atr_val = self.get_atr(ticker)
            if atr_val <= 0:
                atr_val = round(current_price * 0.02, 4)

            hard_stop_price = round(cost_basis * (1.0 - self.hard_stop_pct), 2)
            trailing_stop_price = round(peak_price * (1.0 - self.trailing_stop_pct), 2)
            atr_stop_price = round(peak_price - (self.atr_multiplier * atr_val), 2)

            stop_loss_price = max(hard_stop_price, trailing_stop_price, atr_stop_price)

            # Trigger condition checks
            triggers = []
            if cost_basis > 0 and drawdown_entry_pct <= -(self.hard_stop_pct * 100.0 - 1e-4):
                triggers.append("HARD_STOP")
            if peak_price > 0 and drawdown_peak_pct <= -(self.trailing_stop_pct * 100.0 - 1e-4):
                triggers.append("TRAILING_STOP")
            if current_price <= (atr_stop_price + 1e-4):
                triggers.append("ATR_STOP")

            is_breached = len(triggers) > 0
            trigger_type = ", ".join(triggers) if triggers else "NONE"

            if is_breached:
                status = "BREACHED"
            elif drawdown_entry_pct <= -4.0 or drawdown_peak_pct <= -4.0:
                status = "CAUTION"
            else:
                status = "HEALTHY"

            results.append({
                "ticker": ticker,
                "shares": shares,
                "cost_basis": round(cost_basis, 2),
                "current_price": round(current_price, 2),
                "peak_price": round(peak_price, 2),
                "drawdown_from_entry_pct": round(drawdown_entry_pct, 2),
                "drawdown_from_peak_pct": round(drawdown_peak_pct, 2),
                "atr_14": round(atr_val, 2),
                "stop_loss_price": stop_loss_price,
                "hard_stop_price": hard_stop_price,
                "trailing_stop_price": trailing_stop_price,
                "atr_stop_price": atr_stop_price,
                "status": status,
                "trigger_type": trigger_type,
                "unrealized_pl": round((current_price - cost_basis) * shares, 2),
            })

        return results

    def execute_circuit_breaker(
        self,
        breached_tickers: list[str] | None = None,
        liquidate_to: str = "cash",
        live_prices: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Executes immediate market liquidation for breached positions.
        Sells shares, logs to ledger, resets peak price, optionally hedges into SPY, and alerts Telegram.
        """
        checked = self.check_positions(live_prices=live_prices)

        if breached_tickers is not None:
            target_set = {t.upper() for t in breached_tickers}
            to_liquidate = [p for p in checked if p["ticker"] in target_set]
        else:
            to_liquidate = [p for p in checked if p["status"] == "BREACHED"]

        if not to_liquidate:
            logger.info("No breached positions found for emergency liquidation.")
            return []

        executions: list[dict[str, Any]] = []
        fill_date = datetime.now(UTC).strftime("%Y-%m-%d")

        for item in to_liquidate:
            ticker = item["ticker"]
            shares = float(item["shares"])
            exit_price = float(item["current_price"])
            cost_basis = float(item["cost_basis"])
            peak_price = float(item["peak_price"])
            pnl_pct = float(item["drawdown_from_entry_pct"])
            trigger_type = item["trigger_type"] if item["trigger_type"] != "NONE" else "MANUAL_CIRCUIT_BREAKER"

            filled_price = exit_price
            order_id = ""

            # 1. Execute SELL order via broker
            if self.broker is not None:
                try:
                    receipt = self.broker.submit_order(
                        ticker=ticker,
                        quantity=shares,
                        side="sell",
                        order_type="market",
                        time_in_force="day",
                        estimated_price=exit_price,
                    )
                    order_id = receipt.get("order_id", "")
                    if receipt.get("filled_price"):
                        filled_price = float(receipt["filled_price"])
                except Exception as e:
                    logger.error(f"Failed broker sell order for {ticker}: {e}")

            # 2. Record sell fill in SQLite ledger
            broker_label = getattr(self.broker, "name", "dry_run") if self.broker else "dry_run"
            self.db.record_fill(
                fill_date=fill_date,
                ticker=ticker,
                action="SELL",
                shares=shares,
                price=filled_price,
                commission=0.0,
                notes=f"EMERGENCY_CIRCUIT_BREAKER: {trigger_type}",
                broker=broker_label,
                broker_order_id=order_id,
                status="FILLED",
            )

            # 3. Optional SPY hedge reinvestment
            hedge_info: dict[str, Any] = {}
            realized_proceeds = shares * filled_price
            if liquidate_to.lower() == "hedged_spy" and realized_proceeds > 0:
                spy_price = 0.0
                if live_prices and "SPY" in live_prices:
                    spy_price = float(live_prices["SPY"])
                else:
                    try:
                        spy_df = self.price_pipeline.load_ticker_prices("SPY")
                        if spy_df is not None and not spy_df.empty and "close" in spy_df.columns:
                            spy_price = float(spy_df["close"].dropna().iloc[-1])
                    except (ValueError, KeyError, OSError) as e:
                        logger.debug("Could not load SPY benchmark price: %s", e)
                if spy_price <= 0:
                    spy_price = 500.0

                spy_shares = round(realized_proceeds / spy_price, 4)
                if spy_shares > 0:
                    spy_order_id = ""
                    spy_filled = spy_price
                    if self.broker is not None:
                        try:
                            h_receipt = self.broker.submit_order(
                                ticker="SPY",
                                quantity=spy_shares,
                                side="buy",
                                order_type="market",
                                time_in_force="day",
                                estimated_price=spy_price,
                            )
                            spy_order_id = h_receipt.get("order_id", "")
                            if h_receipt.get("filled_price"):
                                spy_filled = float(h_receipt["filled_price"])
                        except Exception as e:
                            logger.error(f"Failed SPY hedge buy order: {e}")

                    self.db.record_fill(
                        fill_date=fill_date,
                        ticker="SPY",
                        action="BUY",
                        shares=spy_shares,
                        price=spy_filled,
                        commission=0.0,
                        notes=f"EMERGENCY_HEDGE_REINVESTMENT: proceeds from {ticker}",
                        broker=broker_label,
                        broker_order_id=spy_order_id,
                        status="FILLED",
                    )
                    hedge_info = {
                        "hedged_ticker": "SPY",
                        "shares": spy_shares,
                        "price": spy_filled,
                        "invested_proceeds": realized_proceeds,
                    }

            # 4. Reset high water mark tracking
            self.db.reset_high_water_mark(ticker)

            # 5. Log circuit breaker event to SQLite
            event_id = self.db.log_circuit_breaker_event(
                ticker=ticker,
                trigger_type=trigger_type,
                entry_price=cost_basis,
                peak_price=peak_price,
                exit_price=filled_price,
                pnl_pct=pnl_pct,
                hedge_action=liquidate_to.lower(),
                status="EXECUTED",
                notes=f"Liquidated {shares} shares at ${filled_price:.2f}",
            )

            # 6. Dispatch high-priority alert to Telegram
            self.alerts.notify_circuit_breaker_triggered(
                ticker=ticker,
                trigger_type=trigger_type,
                entry_price=cost_basis,
                peak_price=peak_price,
                exit_price=filled_price,
                pnl_pct=pnl_pct,
                hedge_action=liquidate_to.lower(),
            )

            executions.append({
                "event_id": event_id,
                "ticker": ticker,
                "shares_liquidated": shares,
                "exit_price": filled_price,
                "realized_proceeds": round(realized_proceeds, 2),
                "cost_basis": cost_basis,
                "peak_price": peak_price,
                "pnl_pct": pnl_pct,
                "trigger_type": trigger_type,
                "liquidate_to": liquidate_to.lower(),
                "hedge": hedge_info,
                "status": "EXECUTED",
            })

        return executions
