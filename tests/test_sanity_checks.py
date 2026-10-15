"""Unit tests for Model Sanity Checks Suite."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.sanity_checks import ModelSanityChecker
from src.models.validation import WalkForwardValidator


def test_sanity_checker_label_shuffle():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker_path = Path(tmpdir) / "experiment_counter.json"
        validator = WalkForwardValidator(min_train_days=50, test_step_days=10, purge_gap_days=5)
        checker = ModelSanityChecker(validator=validator, tracker_file=tracker_path)

        n_days = 80
        dates = [f"2026-{i:03d}" for i in range(n_days)]
        tickers = ["A", "B", "C", "D", "E"] * n_days
        rep_dates = sorted(dates * 5)

        df = pd.DataFrame({
            "date": rep_dates,
            "ticker": tickers,
            "f1": np.random.randn(len(rep_dates)),
            "f2": np.random.randn(len(rep_dates)),
            "target_excess_return_5d": np.random.randn(len(rep_dates)),
        })

        res = checker.run_label_shuffle_test(df, feature_cols=["f1", "f2"])
        assert "shuffled_mean_ic" in res
        assert "passed" in res
        assert abs(res["shuffled_mean_ic"]) < 0.05


def test_experiment_tracker_counter():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker_path = Path(tmpdir) / "experiment_counter.json"
        checker = ModelSanityChecker(tracker_file=tracker_path)

        exp1 = checker.record_experiment_trial(["f1", "f2"], "Ridge", 0.022)
        assert exp1["total_experiments_recorded"] == 1

        exp2 = checker.record_experiment_trial(["f1", "f2", "f3"], "LGBM", 0.025)
        assert exp2["total_experiments_recorded"] == 2
        assert exp2["multiple_testing_hurdle_t"] > 1.0
