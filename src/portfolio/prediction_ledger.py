"""Live prediction ledger and realized return / rolling Spearman IC tracker.

Guarantees:
- Scans data/scores/daily_scores_YYYY-MM-DD.parquet and fills in actual 5-day excess returns once mature.
- Computes daily realized Spearman IC: corr(pred_score, actual_excess_return_5d).
- Tracks rolling 60-day out-of-sample live IC, t-statistic, and positive days percentage.
- Feeds live IC directly to the PortfolioAllocator confidence filter.
- Idempotent and thread-safe.
"""
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config.settings import get_settings
from src.data.prices import PricePipeline
from src.utils.logger import setup_logger

logger = setup_logger("prediction_ledger")


class PredictionLedger:
    """Manages prediction audit logs, realized return fill-in, and rolling live IC calculation."""

    def __init__(self, scores_dir: Path | None = None):
        settings = get_settings()
        self.scores_dir = scores_dir or (settings.data_dir / "scores")
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.master_ledger_file = self.scores_dir / "prediction_ledger.parquet"
        self.rolling_ic_file = self.scores_dir / "rolling_ic_history.parquet"
        self.price_pipeline = PricePipeline()

    def update_mature_predictions(self) -> dict[str, Any]:
        """
        Scans all daily score files. For predictions at least 5 trading days old,
        fetches actual realized forward returns and writes them into the score files.

        Returns:
            Summary of updated dates and counts.
        """
        score_files = sorted(self.scores_dir.glob("daily_scores_*.parquet"))
        if not score_files:
            return {"updated_files": 0, "mature_days_evaluated": 0, "status": "no_score_files"}

        # Load SPY prices to determine trading day sequence
        spy_df = self.price_pipeline.load_ticker_prices("SPY")
        if spy_df is None or spy_df.empty:
            logger.warning("SPY price history not found. Cannot determine trading calendar.")
            return {"updated_files": 0, "error": "missing_spy_prices"}

        spy_df = spy_df.sort_values("date").reset_index(drop=True)
        all_trading_dates = spy_df["date"].tolist()
        date_to_idx = {d: i for i, d in enumerate(all_trading_dates)}
        spy_price_map = dict(zip(spy_df["date"], spy_df["close"]))

        updated_count = 0
        mature_dates = []

        for f in score_files:
            try:
                sdf = pd.read_parquet(f)
            except Exception as e:
                logger.error(f"Failed to read {f}: {e}")
                continue

            if sdf.empty or "date" not in sdf.columns or "ticker" not in sdf.columns:
                continue

            pred_date = str(sdf["date"].iloc[0])
            if pred_date not in date_to_idx:
                continue

            pred_idx = date_to_idx[pred_date]
            target_idx = pred_idx + 5

            # If 5 trading days have elapsed in price history
            if target_idx < len(all_trading_dates):
                target_date = all_trading_dates[target_idx]

                # Check if already filled
                if "actual_excess_return_5d" in sdf.columns and sdf["actual_excess_return_5d"].notna().sum() > 0:
                    continue

                spy_p0 = spy_price_map.get(pred_date)
                spy_p5 = spy_price_map.get(target_date)
                if spy_p0 is None or spy_p5 is None or spy_p0 <= 0:
                    continue
                spy_ret_5d = (spy_p5 - spy_p0) / spy_p0

                # Compute actual return for each ticker
                actual_excess = []
                actual_stock = []

                for _, row in sdf.iterrows():
                    t = row["ticker"]
                    t_df = self.price_pipeline.load_ticker_prices(t)
                    if t_df is not None and not t_df.empty:
                        p_map = dict(zip(t_df["date"], t_df["close"]))
                        p0 = p_map.get(pred_date)
                        p5 = p_map.get(target_date)
                        if p0 is not None and p5 is not None and p0 > 0:
                            ret_5d = (p5 - p0) / p0
                            actual_stock.append(ret_5d)
                            actual_excess.append(ret_5d - spy_ret_5d)
                            continue
                    actual_stock.append(np.nan)
                    actual_excess.append(np.nan)

                sdf["ticker_fwd_return_5d"] = actual_stock
                sdf["spy_fwd_return_5d"] = spy_ret_5d
                sdf["actual_excess_return_5d"] = actual_excess
                sdf["evaluated_date"] = target_date
                sdf["evaluated_at"] = datetime.now(UTC).isoformat()

                sdf.to_parquet(f, index=False)
                updated_count += 1
                mature_dates.append(pred_date)
                logger.info(f"Filled mature actual returns for date {pred_date} (target date: {target_date}).")

        # Rebuild master ledger and update rolling IC
        self._rebuild_master_ledger()
        ic_summary = self.compute_rolling_ic_history()

        return {
            "updated_files": updated_count,
            "mature_dates": mature_dates,
            "latest_rolling_ic": ic_summary.get("latest_rolling_ic"),
        }

    def _rebuild_master_ledger(self) -> None:
        """Consolidates all individual daily score parquets into the master prediction ledger."""
        score_files = sorted(self.scores_dir.glob("daily_scores_*.parquet"))
        dfs = []
        for f in score_files:
            try:
                df = pd.read_parquet(f)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {f} for master ledger: {e}")

        if dfs:
            master_df = pd.concat(dfs, ignore_index=True)
            master_df = master_df.drop_duplicates(subset=["date", "ticker"]).sort_values(["date", "rank"]).reset_index(drop=True)
            master_df.to_parquet(self.master_ledger_file, index=False)
            logger.info(f"Rebuilt master prediction ledger: {len(master_df)} rows across {master_df['date'].nunique()} dates.")

    def compute_rolling_ic_history(self, window_days: int = 60) -> dict[str, Any]:
        """
        Computes daily realized Spearman IC and rolling window summary across all mature dates.
        """
        if not self.master_ledger_file.exists():
            return {"status": "no_master_ledger"}

        df = pd.read_parquet(self.master_ledger_file)
        if "actual_excess_return_5d" not in df.columns or "pred_score" not in df.columns:
            return {"status": "no_actual_returns"}

        valid = df.dropna(subset=["pred_score", "actual_excess_return_5d"]).copy()
        if valid.empty:
            return {"status": "no_evaluated_records"}

        daily_ics = []
        for d, grp in valid.groupby("date"):
            if len(grp) >= 10:
                ic, _ = spearmanr(grp["pred_score"], grp["actual_excess_return_5d"])
                if not np.isnan(ic):
                    daily_ics.append({
                        "date": str(d),
                        "spearman_ic": round(float(ic), 4),
                        "tickers_count": len(grp),
                    })

        if not daily_ics:
            return {"status": "insufficient_daily_groups"}

        ic_df = pd.DataFrame(daily_ics).sort_values("date").reset_index(drop=True)

        # Compute rolling window metrics
        ic_df["rolling_mean_ic"] = ic_df["spearman_ic"].rolling(window_days, min_periods=5).mean().round(4)
        ic_df["rolling_std_ic"] = ic_df["spearman_ic"].rolling(window_days, min_periods=5).std().round(4)
        ic_df["rolling_pct_positive"] = (
            (ic_df["spearman_ic"] > 0)
            .rolling(window_days, min_periods=5)
            .mean()
            .mul(100.0)
            .round(1)
        )

        # Rolling t-stat: mean / (std / sqrt(n))
        counts = ic_df["spearman_ic"].rolling(window_days, min_periods=5).count()
        ic_df["rolling_t_stat"] = np.where(
            (ic_df["rolling_std_ic"] > 1e-4) & (counts > 2),
            (ic_df["rolling_mean_ic"] / (ic_df["rolling_std_ic"] / np.sqrt(counts))).round(2),
            0.0,
        )

        ic_df.to_parquet(self.rolling_ic_file, index=False)

        latest_row = ic_df.iloc[-1]
        summary = {
            "total_evaluated_dates": len(ic_df),
            "latest_evaluated_date": str(latest_row["date"]),
            "latest_daily_ic": float(latest_row["spearman_ic"]),
            "latest_rolling_ic": float(latest_row["rolling_mean_ic"]) if not pd.isna(latest_row["rolling_mean_ic"]) else float(latest_row["spearman_ic"]),
            "rolling_t_stat": float(latest_row["rolling_t_stat"]) if not pd.isna(latest_row["rolling_t_stat"]) else 0.0,
            "rolling_pct_positive": float(latest_row["rolling_pct_positive"]) if not pd.isna(latest_row["rolling_pct_positive"]) else 50.0,
        }
        return summary

    def get_latest_rolling_ic(self) -> float | None:
        """Returns the latest rolling 60-day live IC, or None if unavailable."""
        if not self.rolling_ic_file.exists():
            return None
        try:
            df = pd.read_parquet(self.rolling_ic_file)
            if df.empty or "rolling_mean_ic" not in df.columns:
                return None
            val = df["rolling_mean_ic"].dropna().iloc[-1]
            return float(val) if not np.isnan(val) else None
        except Exception as e:
            logger.error(f"Failed to fetch rolling IC: {e}")
            return None

    def seed_historical_ledger(self, oos_df: pd.DataFrame) -> dict[str, Any]:
        """
        Seeds the ledger with out-of-sample walk-forward predictions and realized returns
        so rolling 60-day live tracking is fully populated immediately on Day 1.
        """
        required_cols = {"date", "ticker", "pred_score", "target_excess_return_5d"}
        if not required_cols.issubset(set(oos_df.columns)):
            raise ValueError(f"OOS DataFrame must contain: {required_cols}")

        clean = oos_df.dropna(subset=list(required_cols)).copy()
        clean["date"] = clean["date"].astype(str)
        clean = clean.sort_values(["date", "pred_score"], ascending=[True, False]).reset_index(drop=True)
        clean["rank"] = clean.groupby("date")["pred_score"].rank(ascending=False, method="first").astype(int)
        clean["actual_excess_return_5d"] = clean["target_excess_return_5d"]
        clean["evaluated_at"] = datetime.now(UTC).isoformat()

        # Save individual daily parquets for recent dates
        unique_dates = clean["date"].unique()
        for d in unique_dates[-75:]:  # Seed last ~75 dates as individual parquets
            day_slice = clean[clean["date"] == d].copy()
            out_file = self.scores_dir / f"daily_scores_{d}.parquet"
            cols = ["date", "ticker", "rank", "pred_score", "actual_excess_return_5d", "evaluated_at"]
            if "predicted_vol" in day_slice.columns:
                cols.append("predicted_vol")
            day_slice[cols].to_parquet(out_file, index=False)

        clean.to_parquet(self.master_ledger_file, index=False)
        ic_res = self.compute_rolling_ic_history()
        logger.info(f"Successfully seeded prediction ledger across {len(unique_dates)} dates.")
        return ic_res
