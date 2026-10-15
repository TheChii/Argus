"""Incremental price ingestion and Parquet caching using yfinance."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.config.settings import get_settings
from src.config.universe import get_all_tickers, normalize_ticker_for_yfinance
from src.utils.calendar import get_last_completed_trading_day
from src.utils.logger import setup_logger
from src.utils.rate_limiter import RateLimiter, retry_with_backoff

logger = setup_logger("prices")


class PricePipeline:
    """Manages incremental price updates and caching in Parquet format."""

    REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]

    def __init__(self, prices_dir: Path | None = None):
        self.settings = get_settings()
        self.prices_dir = prices_dir or self.settings.prices_dir
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = RateLimiter(calls_per_second=1.0 / max(0.05, self.settings.yfinance_delay_seconds))

    def get_ticker_parquet_path(self, ticker: str) -> Path:
        """Returns the file path for a given ticker's parquet cache."""
        clean_ticker = normalize_ticker_for_yfinance(ticker)
        return self.prices_dir / f"{clean_ticker}.parquet"

    def load_ticker_prices(self, ticker: str) -> pd.DataFrame | None:
        """Loads cached prices for a single ticker."""
        path = self.get_ticker_parquet_path(ticker)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read parquet cache for {ticker}: {e}")
            return None

    def get_latest_cached_date(self, ticker: str) -> str | None:
        """Returns the maximum date string (YYYY-MM-DD) currently cached for ticker."""
        df = self.load_ticker_prices(ticker)
        if df is not None and not df.empty and "date" in df.columns:
            return str(df["date"].max())
        return None

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _fetch_from_yfinance(self, ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Fetches auto-adjusted OHLCV from yfinance for a single ticker."""
        self.rate_limiter.wait()

        # yfinance download
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            timeout=15,
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        # Handle potential MultiIndex columns from newer yfinance
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        # Normalize column names to lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "date" not in df.columns:
            # Check for alternative index name
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})
            else:
                return pd.DataFrame()

        # Format date as YYYY-MM-DD
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        df["ticker"] = normalize_ticker_for_yfinance(ticker)

        # Ensure standard columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = float("nan")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[self.REQUIRED_COLUMNS].dropna(subset=["date", "close"])
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        return df

    def update_ticker(self, ticker: str, lookback_years: int = 5, force: bool = False) -> dict[str, Any]:
        """
        Incrementally updates the Parquet cache for a single ticker.
        If already up to date, skips API call ("never re-pull in a loop").
        """
        clean_ticker = normalize_ticker_for_yfinance(ticker)
        path = self.get_ticker_parquet_path(clean_ticker)
        existing_df = None if force else self.load_ticker_prices(clean_ticker)

        datetime.now().strftime("%Y-%m-%d")

        if existing_df is not None and not existing_df.empty:
            earliest_date = str(existing_df["date"].min())
            latest_date = str(existing_df["date"].max())
            last_completed_td = get_last_completed_trading_day().strftime("%Y-%m-%d")
            desired_start_dt = (datetime.now().date() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

            combined = existing_df.copy()
            rows_added = 0

            # 1. Backfill earlier history if desired lookback reaches further back
            if desired_start_dt < earliest_date:
                logger.info(f"{clean_ticker}: expanding history backwards from {earliest_date} to {desired_start_dt}...")
                earlier_df = self._fetch_from_yfinance(clean_ticker, start=desired_start_dt, end=earliest_date)
                if not earlier_df.empty:
                    combined = pd.concat([earlier_df, combined], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                    rows_added += (len(combined) - len(existing_df))

            # 2. Forward update if not up to date
            if latest_date < last_completed_td:
                new_df = self._fetch_from_yfinance(clean_ticker, start=latest_date)
                if not new_df.empty:
                    # Check for stock split or retroactive adjustment discontinuity
                    boundary_match_old = combined[combined["date"] == latest_date]
                    boundary_match_new = new_df[new_df["date"] == latest_date]
                    if not boundary_match_old.empty and not boundary_match_new.empty:
                        old_close = float(boundary_match_old["close"].iloc[-1])
                        new_close = float(boundary_match_new["close"].iloc[0])
                        if old_close > 0 and abs(new_close - old_close) / old_close > 0.03:
                            logger.warning(
                                f"Split/adjustment detected for {clean_ticker} on {latest_date}: "
                                f"old close={old_close:.2f}, new close={new_close:.2f}. Triggering full history re-sync."
                            )
                            start_dt = datetime.now().date() - timedelta(days=lookback_years * 365)
                            combined = self._fetch_from_yfinance(clean_ticker, start=start_dt.strftime("%Y-%m-%d"))
                            rows_added = len(combined) - len(existing_df)
                        else:
                            combined = pd.concat([combined, new_df], ignore_index=True)
                            combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                            rows_added = len(combined) - len(existing_df)
                    else:
                        combined = pd.concat([combined, new_df], ignore_index=True)
                        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                        rows_added = len(combined) - len(existing_df)

            if rows_added == 0 and len(combined) == len(existing_df):
                logger.debug(f"{clean_ticker} up to date (earliest: {earliest_date}, latest: {latest_date}). Skipping.")
                return {"ticker": clean_ticker, "status": "skipped", "rows_added": 0, "total_rows": len(existing_df)}
        else:
            # Full initial backfill
            start_dt = datetime.now().date() - timedelta(days=lookback_years * 365)
            start_date = start_dt.strftime("%Y-%m-%d")
            combined = self._fetch_from_yfinance(clean_ticker, start=start_date)
            rows_added = len(combined)

        if combined.empty:
            logger.warning(f"No price data retrieved for {clean_ticker}")
            return {"ticker": clean_ticker, "status": "empty", "rows_added": 0, "total_rows": 0}

        # Atomic Parquet write
        tmp_path = path.with_suffix(".tmp")
        combined.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        tmp_path.replace(path)

        logger.info(f"Updated {clean_ticker}: +{rows_added} new rows (total: {len(combined)} rows, latest: {combined['date'].max()})")
        return {"ticker": clean_ticker, "status": "updated", "rows_added": rows_added, "total_rows": len(combined)}

    def update_all(self, tickers: list[str] | None = None, lookback_years: int = 5, force: bool = False) -> dict[str, Any]:
        """Runs incremental updates for all tickers in the universe."""
        if tickers is None:
            tickers = get_all_tickers()

        logger.info(f"Starting price sync for {len(tickers)} tickers...")
        results = []
        for i, ticker in enumerate(tickers, 1):
            try:
                res = self.update_ticker(ticker, lookback_years=lookback_years, force=force)
                results.append(res)
            except Exception as e:
                logger.error(f"Error updating price for {ticker}: {e}")
                results.append({"ticker": ticker, "status": "error", "error": str(e), "rows_added": 0})

        updated_count = sum(1 for r in results if r.get("status") == "updated")
        skipped_count = sum(1 for r in results if r.get("status") in ("skipped", "up_to_date"))
        error_count = sum(1 for r in results if r.get("status") == "error")

        logger.info(f"Price sync complete: {updated_count} updated, {skipped_count} up-to-date, {error_count} errors.")
        return {
            "total_tickers": len(tickers),
            "updated": updated_count,
            "skipped": skipped_count,
            "errors": error_count,
            "details": results,
        }

    def load_panel(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """Loads and stacks prices for all requested tickers into a single panel DataFrame."""
        if tickers is None:
            tickers = get_all_tickers()

        dfs = []
        for ticker in tickers:
            df = self.load_ticker_prices(ticker)
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        panel = pd.concat(dfs, ignore_index=True)
        return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
