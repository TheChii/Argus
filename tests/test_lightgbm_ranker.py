"""Unit tests for LightGBM return ranker and 50/50 Blended ensemble."""
import numpy as np

from src.models.return_ranker.ensemble import BlendedReturnRanker, to_percentile_rank
from src.models.return_ranker.lgbm import LightGBMReturnRanker
from src.models.return_ranker.ridge import RidgeReturnRanker


def test_lightgbm_ranker_constraints():
    np.random.seed(42)
    n = 600
    X = np.random.randn(n, 5)
    # Target correlated with feature 0
    y = 0.5 * X[:, 0] - 0.2 * X[:, 1] + np.random.normal(0, 0.2, n)

    ranker = LightGBMReturnRanker(
        max_depth=3,
        num_leaves=8,
        min_child_samples=50,
        learning_rate=0.03,
        n_estimators=50,
        n_seeds=3,
    )

    ranker.fit(X, y, feature_names=["f0", "f1", "f2", "f3", "f4"])
    assert ranker.is_fitted
    assert len(ranker.models_) == 3

    preds = ranker.predict(X)
    assert len(preds) == n

    # Feature importance
    imp = ranker.get_feature_importances()
    assert len(imp) == 5
    assert imp.iloc[0]["feature"] == "f0"


def test_percentile_ranking():
    arr = np.array([10.0, 50.0, 20.0, 100.0, 5.0])
    p_rank = to_percentile_rank(arr)
    assert abs(p_rank.min() - (-0.5)) < 1e-6
    assert abs(p_rank.max() - 0.5) < 1e-6
    assert abs(p_rank.mean()) < 1e-6


def test_blended_ranker():
    np.random.seed(42)
    n = 600
    X = np.random.randn(n, 4)
    y = 0.4 * X[:, 0] + 0.3 * X[:, 1] + np.random.normal(0, 0.2, n)

    ridge = RidgeReturnRanker()
    lgbm = LightGBMReturnRanker(max_depth=3, min_child_samples=50, n_estimators=40, n_seeds=3)

    blend = BlendedReturnRanker(ridge_ranker=ridge, lgbm_ranker=lgbm, ridge_weight=0.5, lgbm_weight=0.5)
    blend.fit(X, y, feature_names=["f0", "f1", "f2", "f3"])

    assert blend.is_fitted
    blend_preds = blend.predict(X)
    assert len(blend_preds) == n
    assert blend_preds.min() >= -0.55
    assert blend_preds.max() <= 0.55

    # Cross-sectional dates test
    dates = np.array(["2026-01-05"] * 300 + ["2026-01-06"] * 300)
    date_preds = blend.predict(X, dates=dates)
    assert len(date_preds) == n
    # Each date sub-array should be centered and span [-0.5, 0.5]
    d1_preds = date_preds[:300]
    d2_preds = date_preds[300:]
    assert abs(d1_preds.mean()) < 0.05
    assert abs(d2_preds.mean()) < 0.05
