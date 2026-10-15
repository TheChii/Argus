"""FRED Macroeconomic data ingestion and Parquet caching."""
import io
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config.settings import get_settings
from src.utils.calendar import get_last_completed_trading_day
from src.utils.logger import setup_logger
from src.utils.rate_limiter import retry_with_backoff

logger = setup_logger("macro")


class MacroPipeline:
    """Manages ingestion and caching of macroeconomic series from FRED."""

    # Default series required by the model:
    # T10Y2Y: 10Y-2Y Treasury Yield Spread
    # DFF: Federal Funds Effective Rate
    # VIXCLS: CBOE Volatility Index
    # BAMLH0A0HYM2: ICE BofA US High Yield Index OAS
    REQUIRED_SERIES = ["T10Y2Y", "DFF", "VIXCLS", "BAMLH0A0HYM2"]

    def __init__(self, macro_dir: Path | None = None):
        self.settings = get_settings()
        self.macro_dir = macro_dir or self.settings.macro_dir
        self.macro_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.macro_dir / "fred_macro.parquet"

    def load_macro_data(self) -> pd.DataFrame | None:
        """Loads the cached macro Parquet table."""
        if not self.parquet_path.exists():
            return None
        try:
            df = pd.read_parquet(self.parquet_path)
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read macro parquet cache: {e}")
            return None

    def get_latest_cached_date(self) -> str | None:
        """Returns the maximum date currently cached for macro data."""
        df = self.load_macro_data()
        if df is not None and not df.empty and "date" in df.columns:
            return str(df["date"].max())
        return None

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _fetch_series_csv(self, series_id: str, timeout: int = 15) -> pd.DataFrame:
        """Fetches series directly from FRED CSV graph endpoint (no API key required)."""
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text), na_values=[".", " ", "NA", "null"])

        # FRED CSV columns: observation_date, <SERIES_ID>
        date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
        val_col = series_id if series_id in df.columns else df.columns[1]

        df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: series_id})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
        return df.reset_index(drop=True)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _fetch_series_api(self, series_id: str, api_key: str, timeout: int = 15) -> pd.DataFrame:
        """Fetches series from official FRED REST API."""
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "limit": 100000,
        }
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        obs = data.get("observations", [])
        if not obs:
            return pd.DataFrame(columns=["date", series_id])

        rows = []
        for o in obs:
            d = o.get("date")
            val_str = o.get("value")
            val = float(val_str) if val_str not in (".", "", None) else float("nan")
            rows.append({"date": d, series_id: val})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    def fetch_series(self, series_id: str) -> pd.DataFrame:
        """Fetches series data using API key if available, else CSV fallback."""
        if self.settings.fred_api_key:
            try:
                return self._fetch_series_api(series_id, self.settings.fred_api_key)
            except Exception as e:
                logger.warning(f"FRED API fetch failed for {series_id} ({e}). Falling back to CSV endpoint.")

        return self._fetch_series_csv(series_id)

    def update_macro_data(self, series_list: list[str] | None = None, force: bool = False) -> dict[str, Any]:
        """
        Incrementally updates all required macro series and saves to Parquet.
        Merges series on 'date', forward-fills missing dates for non-trading/weekend gaps.
        """
        if series_list is None:
            series_list = self.REQUIRED_SERIES

        existing_df = None if force else self.load_macro_data()

        if existing_df is not None and not existing_df.empty:
            latest_date = str(existing_df["date"].max())
            last_completed_td = get_last_completed_trading_day().strftime("%Y-%m-%d")
            if latest_date >= last_completed_td and not force:
                logger.info(f"Macro data already up to date with last completed trading day ({latest_date}). Skipping.")
                return {
                    "status": "skipped",
                    "latest_date": latest_date,
                    "rows_added": 0,
                    "total_rows": len(existing_df),
                }

        logger.info(f"Fetching {len(series_list)} FRED macro series: {series_list}...")
        merged_df = None
        for s in series_list:
            try:
                s_df = self.fetch_series(s)
                logger.debug(f"Fetched {s}: {len(s_df)} observations (range: {s_df['date'].min()} to {s_df['date'].max()})")
                if merged_df is None:
                    merged_df = s_df
                else:
                    merged_df = pd.merge(merged_df, s_df, on="date", how="outer")
            except Exception as e:
                logger.error(f"Failed to fetch series {s}: {e}")

        if merged_df is None or merged_df.empty:
            logger.warning("No macro data could be fetched.")
            return {"status": "error", "error": "No data retrieved", "rows_added": 0}

        # Sort and clean
        merged_df = merged_df.sort_values("date").reset_index(drop=True)

        # Forward fill series values to propagate latest observed rates across calendar days/holidays
        for s in series_list:
            if s in merged_df.columns:
                merged_df[s] = merged_df[s].ffill()

        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, merged_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            for s in series_list:
                if s in combined.columns:
                    combined[s] = combined[s].ffill()
            rows_added = len(combined) - len(existing_df)
        else:
            combined = merged_df
            rows_added = len(combined)

        # Atomic Parquet write
        tmp_path = self.parquet_path.with_suffix(".tmp")
        combined.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        tmp_path.replace(self.parquet_path)

        latest_saved = combined["date"].max()
        logger.info(f"Macro update complete: +{rows_added} new rows (total: {len(combined)}, latest: {latest_saved})")

        return {
            "status": "updated",
            "latest_date": latest_saved,
            "rows_added": rows_added,
            "total_rows": len(combined),
        }

    def get_macro_features(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Computes standardized macro feature columns:
        - yield_spread: T10Y2Y
        - fed_funds: DFF
        - fed_funds_change_5d: 5-day difference in DFF
        - vix: VIXCLS
        - vix_change_5d: 5-day difference in VIXCLS
        - hy_spread: BAMLH0A0HYM2
        """
        if df is None:
            df = self.load_macro_data()
            if df is None or df.empty:
                return pd.DataFrame()

        feat = df.copy().sort_values("date").reset_index(drop=True)
        if "T10Y2Y" in feat.columns:
            feat["yield_spread"] = feat["T10Y2Y"]
        if "DFF" in feat.columns:
            feat["fed_funds"] = feat["DFF"]
            feat["fed_funds_change_5d"] = feat["DFF"].diff(5)
        if "VIXCLS" in feat.columns:
            feat["vix"] = feat["VIXCLS"]
            feat["vix_change_5d"] = feat["VIXCLS"].diff(5)
        if "BAMLH0A0HYM2" in feat.columns:
            feat["hy_spread"] = feat["BAMLH0A0HYM2"]

        return feat
