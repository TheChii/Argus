"""Relative return features vs SPY and Sector ETFs with defensive error handling."""
import pandas as pd

from src.config.universe import BENCHMARK_TICKER, get_ticker_to_sector_etf_map
from src.utils.logger import setup_logger

logger = setup_logger("relative_features")


class RelativeFeatureExtractor:
    """Computes return differentials relative to benchmark (SPY) and sector ETFs."""

    FEATURE_COLUMNS = [
        "rel_return_spy_5d",
        "rel_return_spy_20d",
        "rel_return_sector_5d",
        "rel_return_sector_20d",
    ]

    def __init__(self, ticker_to_sector_map: dict[str, str] | None = None):
        self.sector_map = ticker_to_sector_map or get_ticker_to_sector_etf_map()

    def extract_relative_features(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds SPY-relative and sector-relative returns to the panel DataFrame.
        Expects panel_df to contain: ['date', 'ticker', 'return_5d', 'return_20d'].
        Guarantees safety on empty inputs, missing SPY benchmark, or missing sector ETFs.
        """
        if panel_df is None or panel_df.empty:
            df = pd.DataFrame() if panel_df is None else panel_df.copy()
            for col in self.FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.Series(dtype=float)
            return df

        df = panel_df.copy()

        for ret_col in ["return_5d", "return_20d"]:
            if ret_col not in df.columns:
                logger.warning(f"Required column '{ret_col}' missing from panel_df; initializing with 0.0.")
                df[ret_col] = 0.0

        has_benchmark = (df["ticker"] == BENCHMARK_TICKER).any()
        if has_benchmark:
            spy_df = df[df["ticker"] == BENCHMARK_TICKER][["date", "return_5d", "return_20d"]]
            spy_5d = dict(zip(spy_df["date"], spy_df["return_5d"]))
            spy_20d = dict(zip(spy_df["date"], spy_df["return_20d"]))
            df["spy_return_5d"] = df["date"].map(spy_5d).fillna(0.0)
            df["spy_return_20d"] = df["date"].map(spy_20d).fillna(0.0)
        else:
            logger.debug(f"Benchmark ticker {BENCHMARK_TICKER} not found in panel; falling back to 0.0 benchmark.")
            df["spy_return_5d"] = 0.0
            df["spy_return_20d"] = 0.0

        sector_etfs = set(self.sector_map.values())
        sector_df = df[df["ticker"].isin(sector_etfs)][["date", "ticker", "return_5d", "return_20d"]]

        if not sector_df.empty:
            sector_etf_col = df["ticker"].map(self.sector_map).fillna(BENCHMARK_TICKER)
            sec_lookup_df = sector_df.rename(
                columns={"ticker": "sector_etf", "return_5d": "sec_return_5d", "return_20d": "sec_return_20d"}
            )
            temp_keys = pd.DataFrame({"date": df["date"], "sector_etf": sector_etf_col}, index=df.index)
            merged_sec = temp_keys.merge(sec_lookup_df, on=["date", "sector_etf"], how="left")
            merged_sec.index = df.index

            sec_return_5d = merged_sec["sec_return_5d"].fillna(df["spy_return_5d"])
            sec_return_20d = merged_sec["sec_return_20d"].fillna(df["spy_return_20d"])
        else:
            sec_return_5d = df["spy_return_5d"]
            sec_return_20d = df["spy_return_20d"]

        df["rel_return_spy_5d"] = df["return_5d"] - df["spy_return_5d"]
        df["rel_return_spy_20d"] = df["return_20d"] - df["spy_return_20d"]
        df["rel_return_sector_5d"] = df["return_5d"] - sec_return_5d
        df["rel_return_sector_20d"] = df["return_20d"] - sec_return_20d

        df = df.drop(columns=["spy_return_5d", "spy_return_20d"])
        return df
