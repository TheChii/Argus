"""Historical News Backfill Engine for Finnhub company news archive."""
from datetime import datetime, timedelta
from typing import Any

from src.config.settings import get_settings
from src.config.universe import get_sp100_constituents
from src.data.news.archive import HeadlineArchive
from src.data.news.finnhub import FinnhubNewsClient
from src.utils.logger import setup_logger

logger = setup_logger("news_backfill")


class HistoricalNewsBackfiller:
    """
    Orchestrates historical company headline backfilling across the universe.

    Guarantees:
    - Splits historical lookback into time windows to bypass Finnhub's 250-article cap per request.
    - Strictly respects Finnhub 60 calls/minute rate limit.
    - Idempotent and deterministic: deduplicates via SHA-256 in HeadlineArchive.
    - Partitions all headlines by year/month.
    """

    def __init__(
        self,
        archive: HeadlineArchive | None = None,
        finnhub_client: FinnhubNewsClient | None = None,
    ):
        self.settings = get_settings()
        self.archive = archive or HeadlineArchive()
        self.finnhub_client = finnhub_client or FinnhubNewsClient()

    def _generate_windows(self, days_back: int, num_windows: int) -> list[tuple[str, str]]:
        """Generates contiguous (from_date, to_date) string pairs."""
        now = datetime.now().date()
        step_days = max(1, days_back // num_windows)
        windows = []

        cur_end = now
        for i in range(num_windows):
            cur_start = now - timedelta(days=min(days_back, (i + 1) * step_days))
            windows.append((cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")))
            cur_end = cur_start

        return windows

    def backfill_tickers(
        self,
        tickers: list[str] | None = None,
        days_back: int = 365,
        num_windows: int = 2,
    ) -> dict[str, Any]:
        """
        Backfills historical company headlines for tickers over `days_back` days.
        """
        if not self.finnhub_client.is_configured():
            logger.error("Finnhub API key is not configured. Cannot perform historical news backfill.")
            return {"status": "error", "message": "Finnhub API key missing"}

        if tickers is None:
            sp100 = get_sp100_constituents()
            tickers = [c["yf_symbol"] for c in sp100]

        windows = self._generate_windows(days_back=days_back, num_windows=num_windows)
        logger.info(
            f"Starting historical news backfill for {len(tickers)} tickers across "
            f"{num_windows} time windows ({days_back} days total lookback)..."
        )
        for w_start, w_end in windows:
            logger.info(f"  Window: {w_start} -> {w_end}")

        total_fetched = 0
        total_inserted = 0
        total_duplicates = 0

        for idx, ticker in enumerate(tickers, 1):
            clean_ticker = ticker.strip().upper()
            ticker_articles: list[dict[str, Any]] = []

            for from_dt, to_dt in windows:
                try:
                    arts = self.finnhub_client.fetch_company_news(clean_ticker, from_date=from_dt, to_date=to_dt)
                    ticker_articles.extend(arts)
                except Exception as e:
                    logger.error(f"Error fetching {clean_ticker} [{from_dt} to {to_dt}]: {e}")

            total_fetched += len(ticker_articles)

            # Ingest into archive (SHA-256 deduplicated & partitioned)
            if ticker_articles:
                res = self.archive.append_articles(ticker_articles)
                total_inserted += res.get("inserted", 0)
                total_duplicates += res.get("duplicates", 0)
                logger.info(
                    f"[{idx}/{len(tickers)}] {clean_ticker}: {len(ticker_articles)} fetched, "
                    f"+{res.get('inserted', 0)} new archived, {res.get('duplicates', 0)} duplicates"
                )
            else:
                logger.info(f"[{idx}/{len(tickers)}] {clean_ticker}: 0 articles returned")

        stats = self.archive.get_archive_stats()
        summary = {
            "tickers_processed": len(tickers),
            "total_articles_fetched": total_fetched,
            "total_inserted": total_inserted,
            "total_duplicates": total_duplicates,
            "archive_total_headlines": stats.get("total_headlines", 0),
            "archive_unique_tickers": stats.get("unique_tickers", 0),
            "earliest_trading_date": stats.get("earliest_trading_date"),
            "latest_trading_date": stats.get("latest_trading_date"),
        }
        logger.info(f"Historical news backfill finished: {summary}")
        return summary
