"""Unit tests for SHAP Explainability, Prediction Ledger, and Weekly Report Generator."""
import json

import numpy as np
import pandas as pd

from src.models.explainability.shap_explainer import ModelExplainer
from src.models.return_ranker.ensemble import BlendedReturnRanker
from src.models.return_ranker.lgbm import LightGBMReturnRanker
from src.models.return_ranker.ridge import RidgeReturnRanker
from src.portfolio.prediction_ledger import PredictionLedger
from src.reports.weekly_report import WeeklyReportGenerator


def test_model_explainer_ridge_and_lgbm():
    np.random.seed(42)
    N = 100
    P = 5
    feature_names = ["return_20d", "rel_return_spy_20d", "rsi_14d", "dist_ma50", "rolling_vol_5d"]

    X = np.random.randn(N, P)
    # Target heavily influenced by feature 1 (rel_return_spy_20d) and feature 0 (return_20d)
    y = 0.5 * X[:, 1] + 0.3 * X[:, 0] + 0.1 * np.random.randn(N)

    # 1. Ridge attribution test
    ridge = RidgeReturnRanker(alphas=[1.0, 10.0])
    ridge.fit(X, y, feature_names=feature_names)

    explainer = ModelExplainer(feature_names=feature_names)
    attr_ridge = explainer.compute_attributions(ridge, X, feature_names=feature_names)
    assert attr_ridge.shape == (N, P)

    # 2. LightGBM TreeSHAP test
    lgbm = LightGBMReturnRanker(n_seeds=2, n_estimators=20, max_depth=3, min_child_samples=10)
    lgbm.fit(X, y, feature_names=feature_names)

    attr_lgbm = explainer.compute_attributions(lgbm, X, feature_names=feature_names)
    assert attr_lgbm.shape == (N, P)

    # 3. Blended model attribution
    blend = BlendedReturnRanker(ridge_ranker=ridge, lgbm_ranker=lgbm)
    blend.fit(X, y, feature_names=feature_names)

    attr_blend = explainer.compute_attributions(blend, X, feature_names=feature_names)
    assert attr_blend.shape == (N, P)

    # 4. Top reasons & explain_dataframe test
    sample_df = pd.DataFrame(X, columns=feature_names)
    sample_df["ticker"] = [f"SYM_{i}" for i in range(N)]
    sample_df["pred_score"] = blend.predict(X)

    explained_df = explainer.explain_dataframe(sample_df, blend, feature_names, top_k=3)
    assert "shap_top_1" in explained_df.columns
    assert "shap_top_2" in explained_df.columns
    assert "shap_top_3" in explained_df.columns
    assert "shap_top_reasons" in explained_df.columns

    reasons = json.loads(explained_df["shap_top_reasons"].iloc[0])
    assert len(reasons) == 3
    assert "feature" in reasons[0]
    assert "attribution" in reasons[0]
    assert "description" in reasons[0]

    # Verify all actual dataset feature names are mapped in FEATURE_DESCRIPTIONS
    from src.models.explainability.shap_explainer import FEATURE_DESCRIPTIONS
    required_features = ["dist_from_ma50", "yield_spread", "vix", "hy_spread", "regime_prob_0", "fed_funds_change_5d"]
    for rf in required_features:
        assert rf in FEATURE_DESCRIPTIONS, f"Feature '{rf}' missing from FEATURE_DESCRIPTIONS"
        assert not FEATURE_DESCRIPTIONS[rf].startswith("Feature"), f"Feature '{rf}' has default fallback description"


def test_prediction_ledger_and_rolling_ic(tmp_path):
    ledger = PredictionLedger(scores_dir=tmp_path)

    # Create 15 synthetic daily prediction parquet files with realized returns
    dates = pd.date_range("2026-01-01", periods=15, freq="B").strftime("%Y-%m-%d").tolist()
    records = []

    for d in dates:
        tickers = [f"SYM_{i}" for i in range(15)]
        # Simulate positive correlation (IC > 0)
        pred_scores = np.linspace(1.0, 0.0, 15)
        actual_returns = pred_scores * 0.05 + np.random.randn(15) * 0.01

        day_df = pd.DataFrame({
            "date": d,
            "ticker": tickers,
            "rank": np.arange(1, 16),
            "pred_score": pred_scores,
            "actual_excess_return_5d": actual_returns,
            "predicted_vol": 0.20,
        })
        day_file = tmp_path / f"daily_scores_{d}.parquet"
        day_df.to_parquet(day_file, index=False)
        records.append(day_df)

    # Rebuild master ledger
    master_df = pd.concat(records, ignore_index=True)
    master_df.to_parquet(ledger.master_ledger_file, index=False)

    ic_summary = ledger.compute_rolling_ic_history(window_days=10)
    assert ic_summary["total_evaluated_dates"] == 15
    assert ic_summary["latest_daily_ic"] > 0.5  # Synthetic positive correlation
    assert ic_summary["latest_rolling_ic"] > 0.5
    assert ic_summary["rolling_pct_positive"] > 80.0

    latest_ic = ledger.get_latest_rolling_ic()
    assert latest_ic is not None
    assert latest_ic > 0.0


def test_weekly_report_generation(tmp_path):
    report_gen = WeeklyReportGenerator(reports_dir=tmp_path)

    target_df = pd.DataFrame([
        {
            "ticker": "AAPL",
            "rank": 1,
            "pred_score": 0.45,
            "target_weight": 0.075,
            "predicted_vol": 0.18,
            "selection_reason": "new_buy_top_10pct",
            "shap_top_1": "20-day return outperformance vs SPY (+0.025)",
            "shap_top_2": "14-day RSI momentum / oversold recovery (+0.015)",
            "shap_top_3": "5-day realized price volatility (-0.005)",
        },
        {
            "ticker": "MSFT",
            "rank": 2,
            "pred_score": 0.40,
            "target_weight": 0.072,
            "predicted_vol": 0.19,
            "selection_reason": "retained_buffer",
            "shap_top_1": "1-day return outperformance vs Sector ETF (+0.018)",
            "shap_top_2": "Low-volatility bull regime probability (+0.012)",
            "shap_top_3": "Price position relative to 50-day MA (+0.008)",
        },
    ])

    portfolio_meta = {
        "confidence_filter_triggered": False,
        "exposure_scale": 1.0,
        "cash_weight": 0.05,
        "vix": 18.5,
        "retained_tickers": ["MSFT"],
        "new_buy_tickers": ["AAPL"],
        "dropped_tickers": ["TSLA"],
    }

    orders = [
        {
            "ticker": "TSLA",
            "action": "SELL",
            "current_weight": 0.06,
            "target_weight": 0.0,
            "delta_weight": -0.06,
            "pct_change": -6.0,
        },
        {
            "ticker": "AAPL",
            "action": "BUY",
            "current_weight": 0.0,
            "target_weight": 0.075,
            "delta_weight": 0.075,
            "pct_change": 7.5,
        },
    ]

    trade_summary = {
        "total_orders": 2,
        "sells_count": 1,
        "buys_count": 1,
        "one_way_turnover": 0.0675,
        "target_positions_count": 2,
    }

    ic_summary = {
        "latest_rolling_ic": 0.0346,
        "rolling_t_stat": 3.99,
        "rolling_pct_positive": 53.0,
    }

    report_content = report_gen.generate_report(
        as_of_date="2026-10-09",
        target_df=target_df,
        portfolio_meta=portfolio_meta,
        orders=orders,
        trade_summary=trade_summary,
        ic_summary=ic_summary,
        cost_bps=7.5,
    )

    # Verify Markdown file created
    md_file = tmp_path / "weekly_report_2026-10-09.md"
    json_file = tmp_path / "weekly_report_2026-10-09.json"
    assert md_file.exists()
    assert json_file.exists()

    # Verify content mentions key elements
    assert "AAPL" in report_content
    assert "TSLA" in report_content
    assert "MSFT" in report_content
    assert "SHAP" in report_content
    assert "0.0346" in report_content
    assert "18.5" in report_content
