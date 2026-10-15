"""Finnhub API client for fetching daily company news with rate limiting."""
from datetime import datetime, timedelta
from typing import Any

import requests

from src.config.settings import get_settings
from src.config.universe import denormalize_ticker
from src.utils.logger import setup_logger
from src.utils.rate_limiter import RateLimiter, retry_with_backoff

logger = setup_logger("finnhub")


class FinnhubNewsClient:
    """
    Client for Finnhub company-news endpoint with rate limiting.

    Finnhub free tier allows 60 calls/minute.
    Enforces a minimum interval (~1.1s) between requests.
    """

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str | None = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.finnhub_api_key
        # Delay to stay strictly under 60 calls/min
        self.rate_limiter = RateLimiter(calls_per_second=1.0 / max(1.0, self.settings.finnhub_delay_seconds))

    def is_configured(self) -> bool:
        """Returns True if a Finnhub API key is present."""
        return bool(self.api_key and self.api_key.strip())

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def fetch_company_news(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        timeout: int = 15
    ) -> list[dict[str, Any]]:
        """
        Fetches company news for a single ticker in the specified date range.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            from_date: 'YYYY-MM-DD'
            to_date: 'YYYY-MM-DD'

        Returns:
            List of standardized article dicts.
        """
        if not self.is_configured():
            logger.warning(
                "FINNHUB_API_KEY is not configured in .env. Skipping live Finnhub news fetch. "
                "Get a free API key at https://finnhub.io"
            )
            return []

        # Finnhub expects dot notation for symbols (e.g., BRK.B)
        query_symbol = denormalize_ticker(ticker)
        self.rate_limiter.wait()

        params = {
            "symbol": query_symbol,
            "from": from_date,
            "to": to_date,
            "token": self.api_key,
        }

        resp = requests.get(self.BASE_URL, params=params, timeout=timeout)

        if resp.status_code == 429:
            logger.warning("Finnhub rate limit reached (HTTP 429). Retrying after backoff...")
            resp.raise_for_status()

        if resp.status_code in (401, 403):
            logger.error("Finnhub authentication failed: invalid API key.")
            return []

        resp.raise_for_status()
        raw_items = resp.json()

        if not isinstance(raw_items, list):
            logger.warning(f"Unexpected response format from Finnhub for {ticker}: {type(raw_items)}")
            return []

        articles = []
        for item in raw_items:
            headline = item.get("headline", "").strip()
            if not headline:
                continue

            # Finnhub datetime is unix epoch timestamp in seconds
            pub_epoch = item.get("datetime")
            if not pub_epoch:
                continue

            articles.append({
                "ticker": ticker.upper(),
                "headline": headline,
                "summary": item.get("summary", ""),
                "publish_time": pub_epoch,
                "source": item.get("source", "Finnhub"),
                "url": item.get("url", ""),
            })

        logger.debug(f"Finnhub: fetched {len(articles)} headlines for {ticker} between {from_date} and {to_date}")
        return articles

    def fetch_recent_news_for_tickers(
        self,
        tickers: list[str],
        days_back: int = 1
    ) -> list[dict[str, Any]]:
        """
        Fetches news from the last `days_back` days up to today for a list of tickers.
        """
        if not self.is_configured():
            return []

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        all_articles = []
        logger.info(f"Fetching news for {len(tickers)} tickers from Finnhub ({start_date} to {end_date})...")

        for i, t in enumerate(tickers, 1):
            try:
                arts = self.fetch_company_news(t, start_date, end_date)
                all_articles.extend(arts)
            except Exception as e:
                logger.error(f"Error fetching Finnhub news for {t}: {e}")

        logger.info(f"Finnhub batch sync finished: retrieved {len(all_articles)} total articles.")
        return all_articles
