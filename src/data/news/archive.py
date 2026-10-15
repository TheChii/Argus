"""Raw Headline Archive Engine with SHA-256 deduplication and trading day cutoff."""
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import get_settings
from src.utils.calendar import get_current_utc_iso, get_effective_trading_date, parse_to_et
from src.utils.logger import setup_logger

logger = setup_logger("news_archive")


def compute_article_hash(ticker: str, publish_time: str, headline: str) -> str:
    """Computes a deterministic SHA-256 hash for deduplication."""
    norm_text = f"{ticker.strip().upper()}|{publish_time.strip()}|{headline.strip().lower()}"
    return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()


class HeadlineArchive:
    """
    Manages the long-term, raw headline archive.

    Guarantees:
    - Pristine text preservation (headline + summary).
    - Preserves both UTC publish time and fetch time.
    - Deterministic SHA-256 deduplication.
    - Effective trading day tagging based on the 4:00 PM ET cutoff rule.
    - Partitioned Parquet storage by year/month for fast retrieval.
    """

    ARCHIVE_COLUMNS = [
        "article_hash",
        "ticker",
        "headline",
        "summary",
        "publish_time",
        "fetch_time",
        "source",
        "url",
        "effective_trading_date",
    ]

    def __init__(self, archive_dir: Path | None = None):
        self.settings = get_settings()
        self.archive_dir = archive_dir or self.settings.news_archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _get_partition_path(self, year: int, month: int) -> Path:
        """Returns path for year/month partition: archive/year=YYYY/month=MM/headlines.parquet"""
        partition_dir = self.archive_dir / f"year={year:04d}" / f"month={month:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        return partition_dir / "headlines.parquet"

    def load_partition(self, year: int, month: int) -> pd.DataFrame:
        """Loads a single month's headline partition."""
        path = self._get_partition_path(year, month)
        if not path.exists():
            return pd.DataFrame(columns=self.ARCHIVE_COLUMNS)
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.error(f"Error reading partition {path}: {e}")
            return pd.DataFrame(columns=self.ARCHIVE_COLUMNS)

    def save_partition(self, df: pd.DataFrame, year: int, month: int) -> None:
        """Atomically saves a partition DataFrame."""
        path = self._get_partition_path(year, month)
        tmp_path = path.with_suffix(".tmp")
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        tmp_path.replace(path)

    def append_articles(self, raw_articles: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Processes and archives incoming articles.

        Expected raw_article keys:
        - ticker: str
        - headline: str
        - publish_time: datetime, str, or epoch
        - summary: Optional[str]
        - source: Optional[str]
        - url: Optional[str]
        - fetch_time: Optional[str] (generated automatically if omitted)
        """
        if not raw_articles:
            return {"total_received": 0, "inserted": 0, "duplicates": 0}

        fetch_time = get_current_utc_iso()
        processed_rows = []

        for item in raw_articles:
            ticker = str(item.get("ticker", "")).strip().upper()
            headline = str(item.get("headline", "")).strip()
            if not ticker or not headline:
                continue

            pub_raw = item.get("publish_time")
            if pub_raw is None:
                continue

            # Standardize publish time to UTC ISO string
            pub_dt = parse_to_et(pub_raw).astimezone(pd.Timestamp("now", tz="UTC").tz)
            pub_iso = pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Calculate effective trading date
            effective_date = get_effective_trading_date(pub_raw)
            art_hash = compute_article_hash(ticker, pub_iso, headline)

            processed_rows.append({
                "article_hash": art_hash,
                "ticker": ticker,
                "headline": headline,
                "summary": str(item.get("summary", "")).strip(),
                "publish_time": pub_iso,
                "fetch_time": item.get("fetch_time") or fetch_time,
                "source": str(item.get("source", "unknown")).strip(),
                "url": str(item.get("url", "")).strip(),
                "effective_trading_date": effective_date,
            })

        if not processed_rows:
            return {"total_received": len(raw_articles), "inserted": 0, "duplicates": 0}

        incoming_df = pd.DataFrame(processed_rows)
        n_before = len(incoming_df)
        # Deduplicate internally within incoming batch
        incoming_df = incoming_df.drop_duplicates(subset=["article_hash"])
        intra_batch_duplicates = n_before - len(incoming_df)

        # Group by year and month of effective_trading_date for partitioned storage
        incoming_df["_dt"] = pd.to_datetime(incoming_df["effective_trading_date"])
        incoming_df["_year"] = incoming_df["_dt"].dt.year
        incoming_df["_month"] = incoming_df["_dt"].dt.month

        total_inserted = 0
        total_duplicates = intra_batch_duplicates

        for (yr, mo), group in incoming_df.groupby(["_year", "_month"]):
            current_partition = self.load_partition(yr, mo)
            if not current_partition.empty:
                existing_hashes = set(current_partition["article_hash"])
                new_items = group[~group["article_hash"].isin(existing_hashes)]
                dups = len(group) - len(new_items)
                total_duplicates += dups

                if new_items.empty:
                    continue

                clean_new = new_items[self.ARCHIVE_COLUMNS]
                combined = pd.concat([current_partition, clean_new], ignore_index=True)
                combined = combined.sort_values(["effective_trading_date", "publish_time"]).reset_index(drop=True)
                self.save_partition(combined, yr, mo)
                total_inserted += len(clean_new)
            else:
                clean_new = group[self.ARCHIVE_COLUMNS].sort_values(["effective_trading_date", "publish_time"]).reset_index(drop=True)
                self.save_partition(clean_new, yr, mo)
                total_inserted += len(clean_new)

        logger.info(f"Archived headlines: {total_inserted} new headlines stored, {total_duplicates} duplicates skipped.")
        return {
            "total_received": len(raw_articles),
            "inserted": total_inserted,
            "duplicates": total_duplicates,
        }

    def load_headlines(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Loads archived headlines filtered by effective trading date range and optional ticker list.
        """
        # Discover all partition parquet files
        pattern = "year=*/month=*/headlines.parquet"
        files = list(self.archive_dir.glob(pattern))
        if not files:
            return pd.DataFrame(columns=self.ARCHIVE_COLUMNS)

        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {f}: {e}")

        if not dfs:
            return pd.DataFrame(columns=self.ARCHIVE_COLUMNS)

        all_news = pd.concat(dfs, ignore_index=True)

        if start_date:
            all_news = all_news[all_news["effective_trading_date"] >= start_date]
        if end_date:
            all_news = all_news[all_news["effective_trading_date"] <= end_date]
        if tickers:
            norm_tickers = {t.strip().upper() for t in tickers}
            all_news = all_news[all_news["ticker"].isin(norm_tickers)]

        return all_news.sort_values(["effective_trading_date", "publish_time"]).reset_index(drop=True)

    def get_archive_stats(self) -> dict[str, Any]:
        """Returns statistics on the current headline archive."""
        files = list(self.archive_dir.glob("year=*/month=*/headlines.parquet"))
        total_rows = 0
        tickers_seen: set[str] = set()
        min_date = None
        max_date = None

        for f in files:
            try:
                df = pd.read_parquet(f, columns=["ticker", "effective_trading_date"])
                if not df.empty:
                    total_rows += len(df)
                    tickers_seen.update(df["ticker"].unique())
                    d_min = df["effective_trading_date"].min()
                    d_max = df["effective_trading_date"].max()
                    min_date = d_min if min_date is None or d_min < min_date else min_date
                    max_date = d_max if max_date is None or d_max > max_date else max_date
            except (ValueError, KeyError, OSError) as e:
                logger.debug("Failed reading archive partition %s: %s", f, e)

        scores_file = self.archive_dir.parent / "scores.parquet"
        scored_count = 0
        if scores_file.exists():
            try:
                import pyarrow.parquet as pq
                scored_count = pq.read_metadata(scores_file).num_rows
            except (ValueError, OSError) as e:
                logger.debug("Fast Parquet metadata read failed for scores.parquet (%s); falling back to pandas.", e)
                try:
                    scored_df = pd.read_parquet(scores_file, columns=["article_hash"])
                    scored_count = len(scored_df)
                except (ValueError, OSError) as e2:
                    logger.debug("Pandas read failed for scores.parquet (%s); returning 0.", e2)
                    scored_count = 0

        return {
            "total_headlines": total_rows,
            "scored_headlines": scored_count,
            "unique_tickers": len(tickers_seen),
            "partitions_count": len(files),
            "earliest_trading_date": min_date,
            "latest_trading_date": max_date,
        }
