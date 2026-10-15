import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("technical")


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Computes Wilder's Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder exponential moving average
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Neutral 50.0 when both gain and loss are 0 (completely flat price)
    flat_mask = (avg_gain < 1e-7) & (avg_loss < 1e-7)
    rsi[flat_mask] = 50.0
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes Wilder's Average True Range (ATR) from high, low, and close series."""
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)

    high = pd.to_numeric(df.get("high", df["close"]), errors="coerce")
    low = pd.to_numeric(df.get("low", df["close"]), errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1).fillna(close)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1.0 / max(1, period)
    atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=max(1, period // 2)).mean()
    return atr


def compute_ticker_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical features for a single ticker's price history.

    Expects df columns: ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    Returns DataFrame with original columns + technical features.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    df = df.sort_values("date").copy() if "date" in df.columns else df.copy()

    if "close" not in df.columns or "volume" not in df.columns:
        for col in TechnicalFeatureExtractor.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df

    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)

    df["rolling_vol_5d"] = df["return_1d"].rolling(window=5, min_periods=3).std() * np.sqrt(252)
    df["rolling_vol_20d"] = df["return_1d"].rolling(window=20, min_periods=10).std() * np.sqrt(252)

    vol_mean = volume.rolling(window=20, min_periods=10).mean()
    vol_std = volume.rolling(window=20, min_periods=10).std()
    df["volume_zscore_20d"] = (volume - vol_mean) / (vol_std + 1e-8)

    ma50 = close.rolling(window=50, min_periods=25).mean()
    df["dist_from_ma50"] = (close - ma50) / (ma50 + 1e-8)

    df["rsi_14d"] = compute_rsi(close, period=14)

    return df


class TechnicalFeatureExtractor:
    """Extracts technical features across all universe tickers."""

    FEATURE_COLUMNS = [
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "rolling_vol_5d",
        "rolling_vol_20d",
        "volume_zscore_20d",
        "dist_from_ma50",
        "rsi_14d",
    ]

    def extract_panel(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical features for every ticker in the panel DataFrame.
        """
        if panel_df.empty:
            return pd.DataFrame()

        try:
            from src.features.fast_features import fast_compute_panel_technical, is_cpp_available
            if is_cpp_available():
                return fast_compute_panel_technical(panel_df)
        except (ImportError, OSError, ValueError) as e:
            logger.debug("C++ fast panel technical compute unavailable (%s); using Python fallback.", e)

        results = []
        for ticker, grp in panel_df.groupby("ticker"):
            feat_grp = compute_ticker_technical_features(grp)
            results.append(feat_grp)

        if not results:
            return pd.DataFrame()

        out_df = pd.concat(results, ignore_index=True)
        return out_df.sort_values(["date", "ticker"]).reset_index(drop=True)
