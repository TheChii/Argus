"""Sanity checks suite: Label shuffling, 1-day extra feature lag, and multiple testing counter."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.models.validation import WalkForwardValidator
from src.utils.logger import setup_logger

logger = setup_logger("sanity_checks")


class ModelSanityChecker:
    """
    Automated sanity checks suite to detect leakage, overfitting, and data snooping.

    1. Label Shuffle Test:
       Shuffles targets within each date and retrains. Mean IC must collapse to chance (~0.0).

    2. 1-Day Extra Feature Lag (Leakage Test):
       Lags all features by 1 extra day. Verifies signal persists with graceful decay rather
       than a catastrophic drop caused by unmasking lookahead leakage.

    3. Multiple Testing Counter:
       Tracks total feature sets tested to penalize data snooping via haircut t-statistics.
    """

    def __init__(
        self,
        validator: WalkForwardValidator | None = None,
        tracker_file: Path | None = None
    ):
        self.validator = validator or WalkForwardValidator(min_train_days=500, test_step_days=60, purge_gap_days=5)
        settings = get_settings()
        self.tracker_file = tracker_file or (settings.logs_dir / "experiment_counter.json")
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)

    def run_label_shuffle_test(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "target_excess_return_5d"
    ) -> dict[str, Any]:
        """
        Shuffles target labels and reruns walk-forward evaluation.
        Performance must collapse to chance.
        """
        logger.info("Executing Sanity Check 1: Label Shuffle Test...")
        shuffled_df = df.copy()

        # Permute target labels strictly within each date
        shuffled_df[target_col] = shuffled_df.groupby("date")[target_col].transform(
            lambda s: np.random.permutation(s.values)
        )

        eval_res = self.validator.evaluate(shuffled_df, feature_cols=feature_cols, target_col=target_col)
        shuffled_ic = eval_res["ic_summary"]["mean_ic"]
        t_stat = eval_res["ic_summary"]["t_stat"]
        passed = abs(shuffled_ic) < 0.008

        logger.info(f"Label Shuffle: Shuffled IC={shuffled_ic:.4f} (t={t_stat:.2f}) -> Passed: {passed}")
        return {
            "test": "label_shuffle",
            "shuffled_mean_ic": shuffled_ic,
            "shuffled_t_stat": t_stat,
            "passed": passed,
            "verdict": "PASSED (collapsed to chance)" if passed else "FAILED (unrealized bias detected)",
        }

    def run_feature_lag_leakage_test(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "target_excess_return_5d",
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> dict[str, Any]:
        """
        Lags every feature one extra trading day per ticker.

        Leakage Check:
        - If features contain subtle same-day lookahead leakage, lagging 1 extra day
          destroys the cheat and causes a catastrophic collapse (>85% drop).
        - Genuine features decay gracefully (e.g. 15% to 50% drop), retaining positive IC.
        """
        logger.info("Executing Sanity Check 2: 1-Day Extra Feature Lag (Leakage Check)...")

        # 1. Baseline evaluation
        base_res = self.validator.evaluate(df, feature_cols=feature_cols, target_col=target_col, date_col=date_col)
        base_ic = base_res["ic_summary"]["mean_ic"]

        # 2. Lag features 1 extra trading day per ticker
        lagged_df = df.sort_values([ticker_col, date_col]).copy()
        for col in feature_cols:
            lagged_df[col] = lagged_df.groupby(ticker_col)[col].shift(1)

        # Drop rows where lag produced NaN
        lagged_df = lagged_df.dropna(subset=feature_cols).reset_index(drop=True)

        lag_res = self.validator.evaluate(lagged_df, feature_cols=feature_cols, target_col=target_col, date_col=date_col)
        lag_ic = lag_res["ic_summary"]["mean_ic"]

        # Calculate decay
        drop_pct = ((base_ic - lag_ic) / base_ic * 100.0) if base_ic > 0 else 0.0
        # Passing criteria: signal remains positive and doesn't suffer unnatural total collapse (>90% drop)
        passed = (lag_ic > 0.003) and (drop_pct < 85.0)

        logger.info(f"Feature Lag Test: Base IC={base_ic:.4f}, Lagged IC={lag_ic:.4f} (Drop: {drop_pct:.1f}%) -> Passed: {passed}")
        return {
            "test": "feature_lag_leakage",
            "base_mean_ic": base_ic,
            "lagged_mean_ic": lag_ic,
            "drop_percentage": round(drop_pct, 1),
            "passed": passed,
            "verdict": "PASSED (signal persists gracefully, no leakage)" if passed else "WARNING (suspicious drop or negative IC)",
        }

    def record_experiment_trial(self, feature_cols: list[str], model_name: str, mean_ic: float) -> dict[str, Any]:
        """
        Logs every feature set and model tested to track multiple-testing inflation.
        Computes haircut significance.
        """
        records = []
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file) as f:
                    records = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Could not read experiment tracker file: %s; resetting tracker records.", e)
                records = []

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records.append({
            "timestamp": now_str,
            "model": model_name,
            "num_features": len(feature_cols),
            "features_hash": str(sorted(feature_cols)),
            "reported_mean_ic": mean_ic,
        })

        with open(self.tracker_file, "w") as f:
            json.dump(records, f, indent=2)

        total_trials = len(records)
        # Haircut penalty: required t-stat scales with sqrt(2 * log(total_trials))
        haircut_hurdle_t = float(np.sqrt(2.0 * np.log(max(2, total_trials))))

        return {
            "total_experiments_recorded": total_trials,
            "multiple_testing_hurdle_t": round(haircut_hurdle_t, 2),
        }
