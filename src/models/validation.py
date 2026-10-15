"""Walk-forward cross-validation with 5-day purge gap and model comparisons."""
from typing import Any

import numpy as np
import pandas as pd

from src.models.metrics import (
    compute_daily_spearman_ic,
    compute_quintile_spread_performance,
    summarize_ic_metrics,
)
from src.models.return_ranker.ensemble import BlendedReturnRanker
from src.models.return_ranker.lgbm import LightGBMReturnRanker
from src.models.return_ranker.ridge import RidgeReturnRanker
from src.utils.logger import setup_logger

logger = setup_logger("walk_forward_val")


class WalkForwardValidator:
    """
    Expanding-window walk-forward cross-validation engine.

    Guarantees:
    - Strict 5-day purge gap between train and test windows to prevent overlapping forward label leakage.
    - Support for Ridge, LightGBM (multi-seed), and 50/50 Blended models.
    - Daily Spearman IC and t-stat evaluation.
    - Top-minus-bottom quintile spread backtest with transaction costs.
    """

    def __init__(
        self,
        min_train_days: int = 500,     # ~2 years of trading history
        test_step_days: int = 60,      # test window step size (~1 quarter)
        purge_gap_days: int = 5,       # 5-day purge gap to prevent forward return overlap
    ):
        self.min_train_days = min_train_days
        self.test_step_days = test_step_days
        self.purge_gap_days = purge_gap_days

    def _instantiate_model(self, model_type: str):
        if model_type == "ridge":
            return RidgeReturnRanker()
        elif model_type == "lgbm":
            return LightGBMReturnRanker()
        elif model_type == "blend":
            return BlendedReturnRanker()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Expected 'ridge', 'lgbm', or 'blend'.")

    def evaluate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        model_type: str = "ridge",
        target_col: str = "target_excess_return_5d",
        date_col: str = "date",
    ) -> dict[str, Any]:
        """
        Executes expanding walk-forward cross-validation for a specific model type.
        """
        unique_dates = sorted(df[date_col].unique())
        total_days = len(unique_dates)

        if total_days < (self.min_train_days + self.purge_gap_days + 20):
            raise ValueError(f"Insufficient historical dates ({total_days}) for walk-forward evaluation.")

        oos_predictions = []
        splits_count = 0

        test_start_idx = self.min_train_days + self.purge_gap_days
        while test_start_idx < total_days:
            test_end_idx = min(test_start_idx + self.test_step_days, total_days)
            train_end_idx = test_start_idx - self.purge_gap_days
            train_dates = set(unique_dates[:train_end_idx])
            test_dates = set(unique_dates[test_start_idx:test_end_idx])

            if not test_dates:
                break

            train_df = df[df[date_col].isin(train_dates)].dropna(subset=[target_col])
            test_df = df[df[date_col].isin(test_dates)].copy()

            if len(train_df) >= 100 and len(test_df) >= 20:
                X_train = train_df[feature_cols].values
                y_train = train_df[target_col].values
                X_test = test_df[feature_cols].values

                model = self._instantiate_model(model_type)
                model.fit(X_train, y_train, feature_names=feature_cols)

                try:
                    test_df["pred_score"] = model.predict(X_test, dates=test_df[date_col].values)
                except TypeError:
                    test_df["pred_score"] = model.predict(X_test)

                oos_predictions.append(test_df[[date_col, "ticker", "pred_score", target_col]])
                splits_count += 1

            # Advance test window contiguously across historical periods
            test_start_idx = test_end_idx

        if not oos_predictions:
            return {"error": "No out-of-sample predictions generated."}

        oos_df = pd.concat(oos_predictions, ignore_index=True)
        daily_ic_df = compute_daily_spearman_ic(oos_df, pred_col="pred_score", target_col=target_col)
        ic_summary = summarize_ic_metrics(daily_ic_df)
        quintile_summary = compute_quintile_spread_performance(oos_df, pred_col="pred_score", fwd_ret_col=target_col)

        logger.info(
            f"Walk-Forward Evaluation Complete [{model_type.upper()}] ({splits_count} splits): "
            f"Mean IC={ic_summary['mean_ic']}, t-stat={ic_summary['t_stat']}, "
            f"Quintile Sharpe={quintile_summary['sharpe_ratio']}"
        )

        return {
            "model_type": model_type,
            "splits_count": splits_count,
            "ic_summary": ic_summary,
            "quintile_summary": quintile_summary,
            "daily_ic_df": daily_ic_df,
            "oos_predictions": oos_df,
        }

    def evaluate_comparison(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "target_excess_return_5d",
    ) -> pd.DataFrame:
        """
        Runs walk-forward evaluation on Ridge, LightGBM, and 50/50 Blend across identical splits.
        """
        results = []
        for m_type in ["ridge", "lgbm", "blend"]:
            res = self.evaluate(df, feature_cols=feature_cols, model_type=m_type, target_col=target_col)
            ic = res["ic_summary"]
            qs = res["quintile_summary"]
            results.append({
                "Model": m_type.upper() if m_type != "blend" else "50/50 BLEND",
                "Mean IC": ic["mean_ic"],
                "IC IR": ic["ic_ir"],
                "t-stat": ic["t_stat"],
                "% Pos Days": ic["pct_positive"],
                "Quintile Return (Ann)": f"{qs['annualized_return'] * 100:.1f}%",
                "Quintile Sharpe": qs["sharpe_ratio"],
            })

        return pd.DataFrame(results)

    def run_label_shuffle_sanity_check(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        model_type: str = "ridge",
        target_col: str = "target_excess_return_5d",
    ) -> dict[str, Any]:
        """
        Sanity Check: Shuffles target labels and reruns walk-forward evaluation.
        Expected result: Mean IC should collapse to approximately 0.0 (chance).
        """
        logger.info(f"Running Label-Shuffle Sanity Check on [{model_type}] (performance must collapse to chance)...")
        shuffled_df = df.copy()

        # Permute target labels strictly within each date using transform
        shuffled_df[target_col] = shuffled_df.groupby("date")[target_col].transform(
            lambda s: np.random.permutation(s.values)
        )
        eval_res = self.evaluate(shuffled_df, feature_cols=feature_cols, model_type=model_type, target_col=target_col)

        mean_ic = eval_res["ic_summary"]["mean_ic"]
        t_stat = eval_res["ic_summary"]["t_stat"]
        collapsed = abs(mean_ic) < 0.008

        logger.info(f"Label Shuffle [{model_type}]: Mean IC={mean_ic:.4f} (t={t_stat:.2f}). Collapsed to chance: {collapsed}")
        return {
            "model_type": model_type,
            "shuffled_mean_ic": mean_ic,
            "shuffled_t_stat": t_stat,
            "collapsed_to_chance": collapsed,
        }
