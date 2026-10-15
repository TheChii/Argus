"""SHAP and feature attribution explainer for Ridge, LightGBM, and Blended Return Rankers.

Guarantees:
- Exact TreeSHAP for LightGBM models via native C++ pred_contrib=True averaged across bagged seeds.
- Exact LinearSHAP for Ridge models: phi_j(x) = beta_j * (x_j - x_bar_j).
- Normalized ensemble attribution for Blended models.
- Formats top 3 rationale drivers per portfolio pick with human-readable explanations.
- Anomaly detector: Flags suspicious or contradictory drivers ('Odd reasons usually mean a bug').
"""
import json
from typing import Any

import numpy as np
import pandas as pd

from src.models.return_ranker.ensemble import BlendedReturnRanker
from src.models.return_ranker.lgbm import LightGBMReturnRanker
from src.models.return_ranker.ridge import RidgeReturnRanker
from src.utils.logger import setup_logger

logger = setup_logger("shap_explainer")

# Human-readable feature interpretations
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "return_1d": "1-day short-term price momentum",
    "return_5d": "5-day weekly price momentum",
    "return_20d": "20-day monthly trend momentum",
    "return_60d": "60-day quarterly trend momentum",
    "rel_return_spy_1d": "1-day return outperformance vs SPY",
    "rel_return_spy_5d": "5-day weekly return outperformance vs SPY",
    "rel_return_spy_20d": "20-day monthly return outperformance vs SPY",
    "rel_return_spy_60d": "60-day quarterly return outperformance vs SPY",
    "rel_return_sector_1d": "1-day return outperformance vs Sector ETF",
    "rel_return_sector_5d": "5-day weekly return outperformance vs Sector ETF",
    "rel_return_sector_20d": "20-day monthly return outperformance vs Sector ETF",
    "rel_return_sector_60d": "60-day quarterly return outperformance vs Sector ETF",
    "rsi_14d": "14-day RSI momentum / oversold recovery",
    "dist_ma50": "Price position relative to 50-day moving average",
    "dist_from_ma50": "Price position relative to 50-day moving average",
    "rolling_vol_5d": "5-day realized price volatility",
    "rolling_vol_20d": "20-day monthly realized price volatility",
    "volume_zscore_20d": "20-day trading volume expansion / surge",
    "yield_spread": "10Y-2Y Treasury yield curve spread",
    "yield_spread_10y2y": "10Y-2Y Treasury yield curve spread",
    "vix": "Macro market fear gauge (VIX level)",
    "vix_level": "Macro market fear gauge (VIX level)",
    "vix_change_5d": "5-day short-term VIX compression / expansion",
    "hy_spread": "High Yield credit risk spread (ICE BofA)",
    "credit_spread_change_20d": "20-day High Yield credit risk spread shift",
    "fed_funds_change_5d": "5-day Fed Funds interest rate policy shift",
    "fed_funds_change_20d": "20-day Fed Funds interest rate policy shift",
    "regime_prob_0": "Low-volatility bull regime probability",
    "regime_prob_1": "Normal-volatility neutral regime probability",
    "regime_prob_2": "High-volatility bear/crisis regime probability",
    "hmm_prob_low_vol": "Low-volatility bull regime probability",
    "hmm_prob_high_vol": "High-volatility bear regime probability",
    "sentiment_mean": "Daily FinBERT news sentiment tone",
    "sentiment_count": "Daily news headline publication volume",
    "sentiment_dispersion": "News sentiment conviction / disagreement",
    "sentiment_3d_mean": "3-day decaying news sentiment momentum",
    "earnings_yield": "Earnings yield (E/P) corporate valuation attractiveness",
    "book_to_price": "Book-to-price (B/P) asset-backed valuation discount",
    "sales_to_price": "Sales-to-price (S/P) top-line revenue yield",
    "fcf_yield": "Free cash flow yield (FCF/P) liquidity return",
    "dividend_yield": "Dividend yield cash distribution return",
    "ev_to_ebitda": "Enterprise multiple (EV/EBITDA) valuation discount",
    "market_cap_log": "Size factor (log market capitalization)",
    "style_size": "Barra size factor exposure",
    "style_volatility": "Barra volatility factor exposure",
    "style_momentum": "Barra momentum factor exposure",
    "style_value": "Barra value composite factor exposure",
}


class ModelExplainer:
    """Computes feature attributions and formats human-readable rationale for top stock picks."""

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names or []

    def compute_attributions(
        self,
        model: RidgeReturnRanker | LightGBMReturnRanker | BlendedReturnRanker | Any,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Computes sample-level feature attribution matrix (N, P).
        Positive values indicate features that push expected return higher.
        """
        p = X.shape[1]

        # 1. Blended Return Ranker
        if isinstance(model, BlendedReturnRanker):
            attr_ridge = self._compute_ridge_attributions(model.ridge_ranker, X)
            attr_lgbm = self._compute_lgbm_attributions(model.lgbm_ranker, X)

            std_ridge = np.std(attr_ridge) if np.std(attr_ridge) > 1e-6 else 1.0
            std_lgbm = np.std(attr_lgbm) if np.std(attr_lgbm) > 1e-6 else 1.0

            z_ridge = attr_ridge / std_ridge
            z_lgbm = attr_lgbm / std_lgbm
            return model.ridge_weight * z_ridge + model.lgbm_weight * z_lgbm

        # 2. Ridge Return Ranker (LinearSHAP)
        if isinstance(model, RidgeReturnRanker):
            return self._compute_ridge_attributions(model, X)

        # 3. LightGBM Return Ranker (TreeSHAP)
        if isinstance(model, LightGBMReturnRanker):
            return self._compute_lgbm_attributions(model, X)

        # Fallback for standard scikit-learn linear model
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).ravel()
            mean_x = np.mean(X, axis=0) if len(X) > 1 else np.zeros(p)
            return (X - mean_x) * coef

        return np.zeros_like(X)

    def _compute_ridge_attributions(self, ridge: RidgeReturnRanker, X: np.ndarray) -> np.ndarray:
        """Exact LinearSHAP for Ridge: phi_j(x) = beta_j * (x_j - x_bar_j)."""
        if not ridge.is_fitted or ridge.model is None:
            return np.zeros_like(X)
        coef = np.asarray(ridge.model.coef_).ravel()
        # Features are rank-normalized cross-sectionally (mean ~0), so attribution is X * coef
        mean_x = np.mean(X, axis=0) if len(X) > 1 else np.zeros(X.shape[1])
        return (X - mean_x) * coef

    def _compute_lgbm_attributions(self, lgbm_ranker: LightGBMReturnRanker, X: np.ndarray) -> np.ndarray:
        """Exact TreeSHAP for multi-seed LightGBM using native pred_contrib=True."""
        models = getattr(lgbm_ranker, "models_", None) or getattr(lgbm_ranker, "models", [])
        if not lgbm_ranker.is_fitted or not models:
            return np.zeros_like(X)

        len(X)
        n_features = X.shape[1]
        seed_attributions = []

        for bst in models:
            # pred_contrib returns (N, P + 1) where column P is the base expected value
            contribs = bst.predict(X, pred_contrib=True)
            # Take only the feature attribution columns (0 to P-1)
            feat_contribs = contribs[:, :n_features]
            seed_attributions.append(feat_contribs)

        return np.mean(seed_attributions, axis=0)

    def get_top_reasons(
        self,
        attribution_row: np.ndarray,
        feature_values_row: np.ndarray,
        feature_names: list[str],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Extracts top K positive feature drivers for a single stock prediction.
        """
        reasons = []
        # Sort descending by attribution
        ranked_indices = np.argsort(attribution_row)[::-1]

        for idx in ranked_indices[:top_k]:
            attr_val = float(attribution_row[idx])
            feat_val = float(feature_values_row[idx])
            feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            desc = FEATURE_DESCRIPTIONS.get(feat_name, feat_name.replace("_", " ").title())

            reasons.append({
                "feature": feat_name,
                "attribution": round(attr_val, 4),
                "feature_value": round(feat_val, 4),
                "description": desc,
                "summary": f"{desc} ({feat_name}: attribution {attr_val:+.4f})",
            })

        return reasons

    def check_for_anomalies(self, reasons: list[dict[str, Any]]) -> str | None:
        """
        Sanity check: Detects unusual drivers ('Odd reasons usually mean a bug').
        Returns a warning string if anomalous patterns are detected.
        """
        warnings = []
        for r in reasons:
            feat = r["feature"]
            val = r["feature_value"]
            attr = r["attribution"]

            # Contradiction: Negative momentum strongly driving a top buy pick
            if "rel_return" in feat and val < -0.20 and attr > 0.05:
                warnings.append(f"Odd signal: Severe negative {feat} ({val:.2f}) driving high buy score.")

            # Contradiction: Extreme realized volatility driving high buy score
            if "rolling_vol" in feat and val > 1.0 and attr > 0.05:
                warnings.append(f"Odd signal: Extreme volatility {feat} ({val:.2f}) driving high buy score.")

        return "; ".join(warnings) if warnings else None

    def explain_dataframe(
        self,
        df: pd.DataFrame,
        model: Any,
        feature_cols: list[str],
        top_k: int = 3,
    ) -> pd.DataFrame:
        """
        Appends SHAP attribution explanations and top 3 reasons to prediction DataFrame.
        """
        result_df = df.copy()
        X = result_df[feature_cols].values
        attributions = self.compute_attributions(model, X, feature_names=feature_cols)

        reasons_list = []
        top_1_list = []
        top_2_list = []
        top_3_list = []
        warning_list = []

        for i in range(len(result_df)):
            row_reasons = self.get_top_reasons(attributions[i], X[i], feature_cols, top_k=top_k)
            reasons_list.append(json.dumps(row_reasons))

            top_1_list.append(row_reasons[0]["summary"] if len(row_reasons) > 0 else "")
            top_2_list.append(row_reasons[1]["summary"] if len(row_reasons) > 1 else "")
            top_3_list.append(row_reasons[2]["summary"] if len(row_reasons) > 2 else "")

            warn = self.check_for_anomalies(row_reasons)
            warning_list.append(warn or "")

        result_df["shap_top_1"] = top_1_list
        result_df["shap_top_2"] = top_2_list
        result_df["shap_top_3"] = top_3_list
        result_df["shap_top_reasons"] = reasons_list
        result_df["shap_warning"] = warning_list

        return result_df
