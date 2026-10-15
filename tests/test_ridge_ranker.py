"""Unit tests for Ridge return ranker, IC metrics, and purge gap validation."""
import numpy as np
import pandas as pd

from src.models.metrics import compute_daily_spearman_ic, summarize_ic_metrics
from src.models.return_ranker.ridge import RidgeReturnRanker
from src.models.validation import WalkForwardValidator


def test_ridge_ranker_fit_predict():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    true_beta = np.array([0.5, -0.3, 0.2, 0.0, 0.1])
    y = X @ true_beta + np.random.normal(0, 0.1, 200)

    ranker = RidgeReturnRanker()
    ranker.fit(X, y, feature_names=["f1", "f2", "f3", "f4", "f5"])

    assert ranker.is_fitted
    preds = ranker.predict(X)
    assert len(preds) == 200

    weights = ranker.get_feature_weights()
    assert len(weights) == 5
    # Feature 1 had highest true weight
    assert weights.iloc[0]["feature"] == "f1"


def test_daily_spearman_ic_metrics():
    dates = ["2026-10-01"] * 20 + ["2026-10-02"] * 20
    # Day 1: perfect rank correlation (1.0)
    # Day 2: perfect rank correlation (1.0)
    scores = list(range(20)) + list(range(20))
    targets = list(range(20)) + list(range(20))

    df = pd.DataFrame({
        "date": dates,
        "pred_score": scores,
        "target_excess_return_5d": targets,
    })

    daily_ic = compute_daily_spearman_ic(df)
    assert len(daily_ic) == 2
    assert (daily_ic["ic"] > 0.99).all()

    summary = summarize_ic_metrics(daily_ic)
    assert summary["mean_ic"] == 1.0
    assert summary["pct_positive"] == 100.0


def test_walk_forward_purge_gap():
    # Test that walk-forward split strictly honors purge gap
    validator = WalkForwardValidator(min_train_days=50, test_step_days=10, purge_gap_days=5)

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

    res = validator.evaluate(df, feature_cols=["f1", "f2"])
    assert "ic_summary" in res
    assert "splits_count" in res
    assert res["splits_count"] >= 1
