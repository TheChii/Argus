import html

import requests

from src.config.settings import get_settings
from src.dashboard.database import DashboardDB
from src.utils.logger import setup_logger

logger = setup_logger("alerts")


class AlertDispatcher:
    """Dispatches operational alerts to Telegram and persists them to SQLite."""

    def __init__(self, db: DashboardDB | None = None):
        self.db = db or DashboardDB()
        settings = get_settings()
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id

    def _send_telegram(self, text: str) -> bool:
        """Sends an HTML formatted message via Telegram Bot API if configured."""
        if not self.bot_token or not self.chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=5.0)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Telegram dispatch failed: {e}")
            return False

    def notify(self, level: str, title: str, message: str) -> None:
        """Dispatches an alert through configured channels and logs to SQLite."""
        logger.info(f"[{level.upper()}] {title}: {message}")

        # Dispatch Telegram (title escaped to prevent malformed HTML entities)
        tg_text = f"<b>[{html.escape(level.upper(), quote=False)}] {html.escape(title, quote=False)}</b>\n\n{message}"
        delivered_tg = self._send_telegram(tg_text)

        # Log to SQLite
        self.db.log_alert(
            level=level,
            channel="telegram" if self.bot_token else "console",
            title=title,
            message=message,
            delivered=delivered_tg if self.bot_token else True,
        )

    def notify_pipeline_failure(self, step_name: str, error_message: str) -> None:
        """Dispatches alert when daily pipeline step encounters an error."""
        safe_step = html.escape(str(step_name), quote=False)
        safe_err = html.escape(str(error_message), quote=False)
        self.notify(
            level="ERROR",
            title="Daily Pipeline Step Failed",
            message=f"Step <b>{safe_step}</b> failed with error:\n<code>{safe_err}</code>",
        )

    def notify_confidence_light_change(self, old_color: str, new_color: str, ic: float, t_stat: float) -> None:
        """Dispatches alert when model confidence light transitions."""
        action_map = {
            "green": "Model edge healthy. Full trading active.",
            "amber": "Model edge degrading. Sizing reduced by 50%.",
            "red": "Confidence lost (IC <= 0). Holding 100% Cash / SPY.",
        }
        self.notify(
            level="WARNING" if new_color in ("amber", "red") else "INFO",
            title=f"Confidence Light Changed: {old_color.upper()} -> {new_color.upper()}",
            message=(
                f"Rolling 60-day live Spearman IC: <b>{ic:+.4f}</b> (t-stat: {t_stat:.2f})\n"
                f"Action: <i>{action_map.get(new_color.lower(), 'Review holdings')}</i>"
            ),
        )

    def notify_weekly_report_ready(self, as_of_date: str, num_orders: int, turnover_pct: float) -> None:
        """Dispatches alert when Friday weekly rebalance report is ready."""
        self.notify(
            level="INFO",
            title=f"Weekly Rebalance Report Ready ({as_of_date})",
            message=(
                f"Weekly portfolio rebalance generated:\n"
                f"- Total Orders: <b>{num_orders}</b>\n"
                f"- 1-Way Turnover: <b>{turnover_pct:.1f}%</b>\n"
                f"Review buy/sell list and execute fills."
            ),
        )

    def notify_circuit_breaker_triggered(
        self,
        ticker: str,
        trigger_type: str,
        entry_price: float,
        peak_price: float,
        exit_price: float,
        pnl_pct: float,
        hedge_action: str = "cash",
    ) -> None:
        """Dispatches an emergency alert when an intraday circuit breaker triggers."""
        safe_ticker = html.escape(str(ticker).upper(), quote=False)
        safe_type = html.escape(str(trigger_type), quote=False)
        safe_hedge = html.escape(str(hedge_action).upper(), quote=False)
        self.notify(
            level="CRITICAL",
            title=f"EMERGENCY CIRCUIT BREAKER: {safe_ticker} LIQUIDATED",
            message=(
                f"Position: <b>{safe_ticker}</b>\n"
                f"Trigger: <b>{safe_type}</b>\n"
                f"Entry Price: <b>${entry_price:.2f}</b>\n"
                f"Peak Price: <b>${peak_price:.2f}</b>\n"
                f"Exit Price: <b>${exit_price:.2f}</b>\n"
                f"Drawdown / PnL: <b>{pnl_pct:+.2f}%</b>\n"
                f"Action: <b>Liquidated to {safe_hedge}</b>"
            ),
        )

