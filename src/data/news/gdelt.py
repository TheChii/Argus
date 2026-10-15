"""GDELT DOC 2.0 API client for news backfilling."""
from datetime import datetime
from typing import Any

import requests

from src.config.settings import get_settings
from src.utils.logger import setup_logger
from src.utils.rate_limiter import RateLimiter, retry_with_backoff

logger = setup_logger("gdelt")


class GDELTNewsClient:
    """
    Client for querying the GDELT DOC 2.0 API.
    Used primarily for historical news backfilling beyond Finnhub's 1-year window.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self):
        self.settings = get_settings()
        self.rate_limiter = RateLimiter(calls_per_second=1.0 / max(1.0, self.settings.gdelt_delay_seconds))

    @retry_with_backoff(max_retries=3, base_delay=2.5)
    def fetch_articles(
        self,
        ticker: str,
        company_name: str | None = None,
        max_records: int = 50,
        timeout: int = 15
    ) -> list[dict[str, Any]]:
        """
        Queries GDELT for articles matching a ticker / company.
        """
        self.rate_limiter.wait()

        # Build query with exact match quotes to prevent single-letter/common-word false positives
        clean_ticker = ticker.strip().upper()
        if company_name:
            clean_name = company_name.split()[0].strip()
            query = f'"{clean_ticker}" OR "{clean_name}"'
        else:
            query = f'"{clean_ticker}"'

        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_records,
            "sort": "date_desc",
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 429:
            logger.warning("GDELT rate limit reached (HTTP 429). Retrying after backoff...")
            resp.raise_for_status()

        resp.raise_for_status()
        try:
            data = resp.json()
        except requests.JSONDecodeError as e:
            logger.debug("Failed parsing GDELT JSON response: %s", e)
            return []

        articles_data = data.get("articles", [])
        results = []

        for item in articles_data:
            title = item.get("title", "").strip()
            seendate = item.get("seendate", "")
            if not title or not seendate:
                continue

            # GDELT seendate format: YYYYMMDDTHHMMSSZ
            try:
                dt_obj = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
            except ValueError:
                continue

            results.append({
                "ticker": ticker.upper(),
                "headline": title,
                "summary": "",
                "publish_time": dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": item.get("domain", "GDELT"),
                "url": item.get("url", ""),
            })

        return results
