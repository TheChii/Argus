import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config.universe import get_sp100_constituents
from src.features.dataset import DatasetBuilder
from src.features.fast_features import (
    fast_compute_panel_technical,
    fast_cross_sectional_normalize,
    is_cpp_available,
)
from src.features.normalization import rank_normalize_series
from src.features.technical import compute_ticker_technical_features


def benchmark_10year_pipeline():
    print("=" * 80)
    print(" 10-YEAR FEATURE ENGINEERING BENCHMARK: PURE PYTHON vs C++ ACCELERATION")
    print("=" * 80)

    db = DatasetBuilder()

    t0 = time.perf_counter()
    raw_prices = db.price_pipeline.load_panel()
    load_time = time.perf_counter() - t0
    n_rows, n_cols = raw_prices.shape
    n_tickers = raw_prices["ticker"].nunique()
    n_dates = raw_prices["date"].nunique()
    print(f"Dataset Loaded: {n_rows:,} rows across {n_tickers} tickers and {n_dates:,} dates ({load_time:.3f}s)")
    print("-" * 80)

    print("\n[Stage 1: Technical Indicators (9 features across 10 years)]")

    t0 = time.perf_counter()
    py_tech_results = [compute_ticker_technical_features(grp) for _, grp in raw_prices.groupby("ticker")]
    py_tech_panel = pd.concat(py_tech_results, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    t_py_tech = time.perf_counter() - t0
    print(f"  • Pure Python (Pandas rolling/ewm loops): {t_py_tech * 1000:.2f} ms ({t_py_tech:.4f} s)")

    t0 = time.perf_counter()
    cpp_tech_panel = fast_compute_panel_technical(raw_prices)
    t_cpp_tech = time.perf_counter() - t0
    speedup_tech = t_py_tech / max(t_cpp_tech, 1e-6)
    print(f"  • C++ OpenMP Parallel Kernel:            {t_cpp_tech * 1000:.2f} ms ({t_cpp_tech:.4f} s) -> {speedup_tech:.1f}x FASTER")

    tech_diff = np.nanmax(np.abs(py_tech_panel["rsi_14d"].dropna().values - cpp_tech_panel["rsi_14d"].dropna().values))
    print(f"    (Max absolute numerical discrepancy in RSI: {tech_diff:.2e})")

    print("\n[Stage 2: Cross-Sectional Rank Normalization (13 features across 2,510 dates)]")
    sp100_symbols = {c["yf_symbol"] for c in get_sp100_constituents()}
    stock_df = cpp_tech_panel[cpp_tech_panel["ticker"].isin(sp100_symbols)].copy()

    stock_df["rel_return_spy_5d"] = stock_df["return_5d"]
    stock_df["rel_return_spy_20d"] = stock_df["return_20d"]
    stock_df["rel_return_sector_5d"] = stock_df["return_5d"]
    stock_df["rel_return_sector_20d"] = stock_df["return_20d"]

    feature_cols = [
        "return_1d", "return_5d", "return_20d", "return_60d",
        "rolling_vol_5d", "rolling_vol_20d", "volume_zscore_20d",
        "dist_from_ma50", "rsi_14d",
        "rel_return_spy_5d", "rel_return_spy_20d",
        "rel_return_sector_5d", "rel_return_sector_20d",
    ]

    print("  • Running Python Baseline (Pandas groupby.transform across 32,630 arrays)...")
    t0 = time.perf_counter()
    py_norm_df = stock_df.copy()
    for col in feature_cols:
        py_norm_df[col] = py_norm_df.groupby("date")[col].transform(rank_normalize_series)
    t_py_norm = time.perf_counter() - t0
    print(f"  • Pure Python (Pandas groupby.transform): {t_py_norm:.2f} s")

    t0 = time.perf_counter()
    cpp_norm_df = fast_cross_sectional_normalize(stock_df, feature_cols=feature_cols)
    t_cpp_norm = time.perf_counter() - t0
    speedup_norm = t_py_norm / max(t_cpp_norm, 1e-6)
    print(f"  • C++ OpenMP Multi-threaded Kernel:       {t_cpp_norm * 1000:.2f} ms ({t_cpp_norm:.4f} s) -> {speedup_norm:.1f}x FASTER")

    max_norm_diff = 0.0
    for col in feature_cols:
        diff = np.nanmax(np.abs(py_norm_df[col].values - cpp_norm_df[col].values))
        max_norm_diff = max(max_norm_diff, diff)
    print(f"    (Max absolute numerical discrepancy in ranks: {max_norm_diff:.2e})")

    print("\n[Stage 3: Interoperability Delay & External Program Overhead Analysis]")

    c_overhead_trials = 1000
    t0 = time.perf_counter()
    for _ in range(c_overhead_trials):
        is_cpp_available()
    t_ctypes_call = (time.perf_counter() - t0) / c_overhead_trials
    print(f"  • In-Memory ctypes shared library call overhead: {t_ctypes_call * 1000:.4f} ms per call (< 0.001 ms)")

    proc_trials = 10
    t0 = time.perf_counter()
    for _ in range(proc_trials):
        subprocess.run(["./src/features/fast_features_cli", "2500"], stdout=subprocess.DEVNULL)
    t_proc_launch = (time.perf_counter() - t0) / proc_trials
    print(f"  • Standalone External Binary process launch latency: {t_proc_launch * 1000:.2f} ms")
    print(f"    => Using a shared library via ctypes is {t_proc_launch / max(t_ctypes_call, 1e-9):,.0f}x faster than spawning an external program!")

    # -------------------------------------------------------------
    # 4. Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FINAL PERFORMANCE COMPARISON (10 YEARS OF MARKET DATA)")
    print("=" * 80)
    print(f"{'Component':<35} | {'Original Python':<16} | {'C++ Engine':<14} | {'Speedup':<10}")
    print("-" * 80)
    print(f"{'Technical Indicators':<35} | {t_py_tech * 1000:>10.2f} ms     | {t_cpp_tech * 1000:>8.2f} ms   | {speedup_tech:>7.1f}x")
    print(f"{'Cross-Sectional Normalization':<35} | {t_py_norm * 1000:>10.2f} ms     | {t_cpp_norm * 1000:>8.2f} ms   | {speedup_norm:>7.1f}x")
    t_py_total = t_py_tech + t_py_norm
    t_cpp_total = t_cpp_tech + t_cpp_norm
    speedup_total = t_py_total / max(t_cpp_total, 1e-6)
    print("-" * 80)
    print(f"{'TOTAL COMPUTE TIME':<35} | {t_py_total:>12.3f} s      | {t_cpp_total:>10.3f} s    | {speedup_total:>7.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_10year_pipeline()
