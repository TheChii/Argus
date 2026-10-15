"""C++ accelerated feature engineering bindings via ctypes with defensive error handling."""
import ctypes
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("fast_features")

_LIB_PATH = Path(__file__).parent / "libfast_features.so"
_CPP_LIB = None

if _LIB_PATH.exists():
    try:
        _CPP_LIB = ctypes.CDLL(str(_LIB_PATH))

        _CPP_LIB.compute_ticker_technical_cpp.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # close
            ctypes.POINTER(ctypes.c_double),  # volume
            ctypes.c_int,                     # n
            ctypes.POINTER(ctypes.c_double),  # out_ret1
            ctypes.POINTER(ctypes.c_double),  # out_ret5
            ctypes.POINTER(ctypes.c_double),  # out_ret20
            ctypes.POINTER(ctypes.c_double),  # out_ret60
            ctypes.POINTER(ctypes.c_double),  # out_vol5
            ctypes.POINTER(ctypes.c_double),  # out_vol20
            ctypes.POINTER(ctypes.c_double),  # out_vol_zscore20
            ctypes.POINTER(ctypes.c_double),  # out_dist_ma50
            ctypes.POINTER(ctypes.c_double),  # out_rsi14
        ]
        _CPP_LIB.compute_ticker_technical_cpp.restype = ctypes.c_int

        _CPP_LIB.compute_panel_technical_cpp.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # close
            ctypes.POINTER(ctypes.c_double),  # volume
            ctypes.POINTER(ctypes.c_int),     # ticker_offsets
            ctypes.c_int,                     # num_tickers
            ctypes.c_int,                     # total_rows
            ctypes.POINTER(ctypes.c_double),  # out_ret1
            ctypes.POINTER(ctypes.c_double),  # out_ret5
            ctypes.POINTER(ctypes.c_double),  # out_ret20
            ctypes.POINTER(ctypes.c_double),  # out_ret60
            ctypes.POINTER(ctypes.c_double),  # out_vol5
            ctypes.POINTER(ctypes.c_double),  # out_vol20
            ctypes.POINTER(ctypes.c_double),  # out_vol_zscore20
            ctypes.POINTER(ctypes.c_double),  # out_dist_ma50
            ctypes.POINTER(ctypes.c_double),  # out_rsi14
        ]
        _CPP_LIB.compute_panel_technical_cpp.restype = ctypes.c_int

        _CPP_LIB.cross_sectional_rank_cpp.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # in_cols
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # out_cols
            ctypes.c_int,                                     # num_cols
            ctypes.POINTER(ctypes.c_int),                     # date_offsets
            ctypes.c_int,                                     # num_dates
            ctypes.c_int                                      # total_rows
        ]
        _CPP_LIB.cross_sectional_rank_cpp.restype = ctypes.c_int

        logger.debug("Successfully loaded C++ acceleration library: libfast_features.so")
    except Exception as e:
        logger.warning(f"Failed to load libfast_features.so: {e}. Falling back to Python implementation.")
        _CPP_LIB = None


def is_cpp_available() -> bool:
    """Returns True if the C++ acceleration shared library is available."""
    return _CPP_LIB is not None


def fast_compute_ticker_technical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical features for a single ticker's price history using C++.
    Expects columns: ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'].
    Guarantees seamless fallback to pure Python if any exception or error occurs.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    required_cols = {"close", "volume"}
    if not is_cpp_available() or not required_cols.issubset(df.columns):
        from src.features.technical import compute_ticker_technical_features
        return compute_ticker_technical_features(df)

    try:
        df_sorted = df.sort_values("date").copy() if "date" in df.columns else df.copy()
        n = len(df_sorted)
        if n == 0:
            return df_sorted

        close_series = pd.to_numeric(df_sorted["close"], errors="coerce")
        volume_series = pd.to_numeric(df_sorted["volume"], errors="coerce")

        close_arr = np.ascontiguousarray(close_series.values, dtype=np.float64)
        vol_arr = np.ascontiguousarray(volume_series.values, dtype=np.float64)

        ret1 = np.empty(n, dtype=np.float64)
        ret5 = np.empty(n, dtype=np.float64)
        ret20 = np.empty(n, dtype=np.float64)
        ret60 = np.empty(n, dtype=np.float64)
        vol5 = np.empty(n, dtype=np.float64)
        vol20 = np.empty(n, dtype=np.float64)
        vol_z20 = np.empty(n, dtype=np.float64)
        dist_ma50 = np.empty(n, dtype=np.float64)
        rsi14 = np.empty(n, dtype=np.float64)

        c_double_p = ctypes.POINTER(ctypes.c_double)
        res = _CPP_LIB.compute_ticker_technical_cpp(
            close_arr.ctypes.data_as(c_double_p),
            vol_arr.ctypes.data_as(c_double_p),
            ctypes.c_int(n),
            ret1.ctypes.data_as(c_double_p),
            ret5.ctypes.data_as(c_double_p),
            ret20.ctypes.data_as(c_double_p),
            ret60.ctypes.data_as(c_double_p),
            vol5.ctypes.data_as(c_double_p),
            vol20.ctypes.data_as(c_double_p),
            vol_z20.ctypes.data_as(c_double_p),
            dist_ma50.ctypes.data_as(c_double_p),
            rsi14.ctypes.data_as(c_double_p),
        )

        if res != 0:
            logger.warning(f"compute_ticker_technical_cpp returned error code {res}. Falling back to Python.")
            from src.features.technical import compute_ticker_technical_features
            return compute_ticker_technical_features(df)

        df_sorted["return_1d"] = ret1
        df_sorted["return_5d"] = ret5
        df_sorted["return_20d"] = ret20
        df_sorted["return_60d"] = ret60
        df_sorted["rolling_vol_5d"] = vol5
        df_sorted["rolling_vol_20d"] = vol20
        df_sorted["volume_zscore_20d"] = vol_z20
        df_sorted["dist_from_ma50"] = dist_ma50
        df_sorted["rsi_14d"] = rsi14

        return df_sorted
    except Exception as e:
        logger.warning(f"Exception during fast_compute_ticker_technical: {e}. Falling back to Python.")
        from src.features.technical import compute_ticker_technical_features
        return compute_ticker_technical_features(df)


def fast_compute_panel_technical(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical features across all tickers in panel_df using parallel C++ OpenMP kernel.
    Guarantees robust error handling and fallback to Python if any condition is not met.
    """
    if panel_df is None or panel_df.empty:
        return pd.DataFrame() if panel_df is None else panel_df.copy()

    required_cols = {"ticker", "date", "close", "volume"}
    if not is_cpp_available() or not required_cols.issubset(panel_df.columns):
        from src.features.technical import compute_ticker_technical_features
        results = [compute_ticker_technical_features(grp) for _, grp in panel_df.groupby("ticker")]
        if not results:
            return pd.DataFrame()
        out = pd.concat(results, ignore_index=True)
        return out.sort_values(["date", "ticker"]).reset_index(drop=True)

    try:
        df_sorted = panel_df.sort_values(["ticker", "date"]).reset_index(drop=True)
        tickers = df_sorted["ticker"].values
        n = len(df_sorted)

        if n == 0:
            return df_sorted

        split_indices = np.where(tickers[:-1] != tickers[1:])[0] + 1
        offsets = np.empty(len(split_indices) + 2, dtype=np.int32)
        offsets[0] = 0
        offsets[1:-1] = split_indices
        offsets[-1] = n
        num_tickers = len(offsets) - 1

        close_series = pd.to_numeric(df_sorted["close"], errors="coerce")
        vol_series = pd.to_numeric(df_sorted["volume"], errors="coerce")

        close_arr = np.ascontiguousarray(close_series.values, dtype=np.float64)
        vol_arr = np.ascontiguousarray(vol_series.values, dtype=np.float64)
        offsets_arr = np.ascontiguousarray(offsets, dtype=np.int32)

        ret1 = np.empty(n, dtype=np.float64)
        ret5 = np.empty(n, dtype=np.float64)
        ret20 = np.empty(n, dtype=np.float64)
        ret60 = np.empty(n, dtype=np.float64)
        vol5 = np.empty(n, dtype=np.float64)
        vol20 = np.empty(n, dtype=np.float64)
        vol_z20 = np.empty(n, dtype=np.float64)
        dist_ma50 = np.empty(n, dtype=np.float64)
        rsi14 = np.empty(n, dtype=np.float64)

        c_double_p = ctypes.POINTER(ctypes.c_double)
        c_int_p = ctypes.POINTER(ctypes.c_int)

        res = _CPP_LIB.compute_panel_technical_cpp(
            close_arr.ctypes.data_as(c_double_p),
            vol_arr.ctypes.data_as(c_double_p),
            offsets_arr.ctypes.data_as(c_int_p),
            ctypes.c_int(num_tickers),
            ctypes.c_int(n),
            ret1.ctypes.data_as(c_double_p),
            ret5.ctypes.data_as(c_double_p),
            ret20.ctypes.data_as(c_double_p),
            ret60.ctypes.data_as(c_double_p),
            vol5.ctypes.data_as(c_double_p),
            vol20.ctypes.data_as(c_double_p),
            vol_z20.ctypes.data_as(c_double_p),
            dist_ma50.ctypes.data_as(c_double_p),
            rsi14.ctypes.data_as(c_double_p),
        )

        if res != 0:
            logger.warning(f"compute_panel_technical_cpp returned error code {res}. Falling back to Python.")
            from src.features.technical import compute_ticker_technical_features
            results = [compute_ticker_technical_features(grp) for _, grp in panel_df.groupby("ticker")]
            if not results:
                return pd.DataFrame()
            out = pd.concat(results, ignore_index=True)
            return out.sort_values(["date", "ticker"]).reset_index(drop=True)

        df_sorted["return_1d"] = ret1
        df_sorted["return_5d"] = ret5
        df_sorted["return_20d"] = ret20
        df_sorted["return_60d"] = ret60
        df_sorted["rolling_vol_5d"] = vol5
        df_sorted["rolling_vol_20d"] = vol20
        df_sorted["volume_zscore_20d"] = vol_z20
        df_sorted["dist_from_ma50"] = dist_ma50
        df_sorted["rsi_14d"] = rsi14

        return df_sorted.sort_values(["date", "ticker"]).reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Exception during fast_compute_panel_technical: {e}. Falling back to Python.")
        from src.features.technical import compute_ticker_technical_features
        results = [compute_ticker_technical_features(grp) for _, grp in panel_df.groupby("ticker")]
        if not results:
            return pd.DataFrame()
        out = pd.concat(results, ignore_index=True)
        return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def fast_cross_sectional_normalize(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Performs cross-sectional rank normalization across tickers date-by-date using C++ OpenMP kernel.
    Ranks values in uniform percentile range [-0.5, 0.5] with average rank tie-breaking.
    Preserves exact input row order regardless of index type, duplicate index labels, or sorting.
    """
    if df is None or df.empty or not feature_cols:
        return pd.DataFrame() if df is None else df.copy()

    valid_cols = [c for c in feature_cols if c in df.columns]
    if not is_cpp_available() or not valid_cols or "date" not in df.columns:
        from src.features.normalization import rank_normalize_series
        res_df = df.copy()
        for col in valid_cols:
            res_df[col] = res_df.groupby("date")[col].transform(rank_normalize_series)
        return res_df

    try:
        dates_raw = df["date"].values
        sort_idx = np.argsort(dates_raw, kind="stable")
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(len(sort_idx))

        dates_sorted = dates_raw[sort_idx]
        n = len(df)

        split_indices = np.where(dates_sorted[:-1] != dates_sorted[1:])[0] + 1
        offsets = np.empty(len(split_indices) + 2, dtype=np.int32)
        offsets[0] = 0
        offsets[1:-1] = split_indices
        offsets[-1] = n
        num_dates = len(offsets) - 1

        offsets_arr = np.ascontiguousarray(offsets, dtype=np.int32)
        num_cols = len(valid_cols)

        c_double_p = ctypes.POINTER(ctypes.c_double)
        c_int_p = ctypes.POINTER(ctypes.c_int)

        in_arrays = []
        out_arrays = []
        in_ptrs_list = []
        out_ptrs_list = []

        for col in valid_cols:
            col_series = pd.to_numeric(df[col], errors="coerce")
            sorted_col_vals = col_series.values[sort_idx]
            arr_in = np.ascontiguousarray(sorted_col_vals, dtype=np.float64)
            arr_out = np.empty(n, dtype=np.float64)

            in_arrays.append(arr_in)
            out_arrays.append(arr_out)
            in_ptrs_list.append(arr_in.ctypes.data_as(c_double_p))
            out_ptrs_list.append(arr_out.ctypes.data_as(c_double_p))

        in_ptrs_arr = (c_double_p * num_cols)(*in_ptrs_list)
        out_ptrs_arr = (c_double_p * num_cols)(*out_ptrs_list)

        res = _CPP_LIB.cross_sectional_rank_cpp(
            in_ptrs_arr,
            out_ptrs_arr,
            ctypes.c_int(num_cols),
            offsets_arr.ctypes.data_as(c_int_p),
            ctypes.c_int(num_dates),
            ctypes.c_int(n),
        )

        if res != 0:
            logger.warning(f"cross_sectional_rank_cpp returned error code {res}. Falling back to Python.")
            from src.features.normalization import rank_normalize_series
            res_df = df.copy()
            for col in valid_cols:
                res_df[col] = res_df.groupby("date")[col].transform(rank_normalize_series)
            return res_df

        out_df = df.copy()
        for i, col in enumerate(valid_cols):
            out_df[col] = out_arrays[i][inv_idx]

        return out_df
    except Exception as e:
        logger.warning(f"Exception during fast_cross_sectional_normalize: {e}. Falling back to Python.")
        from src.features.normalization import rank_normalize_series
        res_df = df.copy()
        for col in valid_cols:
            res_df[col] = res_df.groupby("date")[col].transform(rank_normalize_series)
        return res_df
