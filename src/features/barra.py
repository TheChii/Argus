"""Barra Multi-Factor Model: Factor exposure matrix construction and cross-sectional alpha neutralization.

Guarantees & Properties:
1. 11 GICS Sector dummy variables + 4 Standardized Style factors (Size, Volatility, Momentum, Value).
2. Regularized linear projection residualization:
   alpha_neutral = alpha - X (X^T X + lambda I)^(-1) X^T alpha
3. Zero lookahead; purely cross-sectional per time-step.
4. Numerical stability via L2 ridge regularization (lambda = 1e-4).
5. Comprehensive diagnostic telemetry (pre- and post-neutralization factor correlations and sector spreads).
"""
from typing import Any

import numpy as np
import pandas as pd

from src.config.universe import SECTOR_ETF_MAP, get_sector_for_ticker
from src.data.fundamentals import FundamentalPipeline
from src.utils.logger import setup_logger

logger = setup_logger("barra_factor_model")

# Standard 11 GICS Sectors
GICS_SECTORS: list[str] = sorted(SECTOR_ETF_MAP.keys())

# Core Barra style factor names
STYLE_FACTORS: list[str] = ["size", "volatility", "momentum", "value"]


class BarraFactorModel:
    """
    Constructs cross-sectional factor exposure matrices and neutralizes return forecasts (alphas)
    against sector and style factor biases.
    """

    def __init__(
        self,
        sectors: list[str] | None = None,
        fundamental_pipeline: FundamentalPipeline | None = None,
        l2_reg: float = 1e-4,
    ):
        self.sectors = sectors or GICS_SECTORS
        self.fundamental_pipeline = fundamental_pipeline or FundamentalPipeline()
        self.l2_reg = l2_reg

    def build_factor_exposure_matrix(
        self,
        panel_slice: pd.DataFrame,
        fundamentals_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Builds cross-sectional factor exposure matrix X in R^{N x K}.

        Parameters:
            panel_slice: DataFrame containing cross-section of stocks.
                         Must have 'ticker' either as column or index.
            fundamentals_df: Optional DataFrame with pre-loaded fundamentals.

        Returns:
            (exposure_matrix_df, factor_names_list)
        """
        df = panel_slice.copy()
        if "ticker" in df.columns:
            df = df.set_index("ticker")

        tickers = list(df.index)
        n = len(tickers)
        if n == 0:
            return pd.DataFrame(), []

        # Load fundamentals if not provided
        if fundamentals_df is None or fundamentals_df.empty:
            try:
                fundamentals_df = self.fundamental_pipeline.load_fundamentals(tickers=tickers)
            except Exception as e:
                logger.debug(f"Could not load fundamentals: {e}; using fallback records.")
                fundamentals_df = pd.DataFrame([
                    self.fundamental_pipeline.get_fallback_fundamentals(t) for t in tickers
                ])

        fund_map: dict[str, dict[str, Any]] = {}
        if not fundamentals_df.empty:
            if "ticker" in fundamentals_df.columns:
                fund_map = fundamentals_df.set_index("ticker").to_dict(orient="index")
            else:
                fund_map = fundamentals_df.to_dict(orient="index")

        # -------------------------------------------------------------
        # 1. Sector Dummy Matrix (11 GICS Sectors + optional Other)
        # -------------------------------------------------------------
        sector_series = []
        for t in tickers:
            sec = None
            if "sector" in df.columns:
                sec = df.loc[t, "sector"]
            if not sec and t in fund_map:
                sec = fund_map[t].get("sector")
            if not sec:
                sec = get_sector_for_ticker(t) or "Other"
            sector_series.append(sec)

        sector_col_data: dict[str, np.ndarray] = {}
        for s in self.sectors:
            col_name = f"sector_{s.lower().replace(' ', '_')}"
            sector_col_data[col_name] = np.array([1.0 if sec == s else 0.0 for sec in sector_series], dtype=float)

        # Check for unmapped or other sectors
        other_mask = np.array([1.0 if sec not in self.sectors else 0.0 for sec in sector_series], dtype=float)
        if np.any(other_mask > 0):
            sector_col_data["sector_other"] = other_mask

        # -------------------------------------------------------------
        # 2. Standardized Style Factors
        # -------------------------------------------------------------
        # A. Size Factor (log market cap)
        size_raw = np.zeros(n, dtype=float)
        for i, t in enumerate(tickers):
            val = None
            if "market_cap_log" in df.columns:
                val = df.loc[t, "market_cap_log"]
            elif t in fund_map:
                val = fund_map[t].get("market_cap_log")
            if val is None or np.isnan(val) or val <= 0:
                val = self.fundamental_pipeline.get_fallback_fundamentals(t).get("market_cap_log", 25.0)
            size_raw[i] = float(val)

        # B. Volatility Factor (20-day rolling vol or predicted vol)
        vol_raw = np.zeros(n, dtype=float)
        for i, t in enumerate(tickers):
            val = None
            if "rolling_vol_20d" in df.columns:
                val = df.loc[t, "rolling_vol_20d"]
            elif "predicted_vol" in df.columns:
                val = df.loc[t, "predicted_vol"]
            elif "rolling_vol_5d" in df.columns:
                val = df.loc[t, "rolling_vol_5d"]
            if val is None or np.isnan(val):
                val = 0.20
            vol_raw[i] = float(val)

        # C. Momentum Factor (20-day or 60-day return)
        mom_raw = np.zeros(n, dtype=float)
        for i, t in enumerate(tickers):
            val = None
            if "return_20d" in df.columns:
                val = df.loc[t, "return_20d"]
            elif "return_60d" in df.columns:
                val = df.loc[t, "return_60d"]
            elif "return_5d" in df.columns:
                val = df.loc[t, "return_5d"]
            if val is None or np.isnan(val):
                val = 0.0
            mom_raw[i] = float(val)

        # D. Value Composite Factor: composite of B/P, E/P, S/P
        bp_raw = np.zeros(n, dtype=float)
        ey_raw = np.zeros(n, dtype=float)
        sp_raw = np.zeros(n, dtype=float)
        for i, t in enumerate(tickers):
            rec = fund_map.get(t) or self.fundamental_pipeline.get_fallback_fundamentals(t)
            bp_raw[i] = float(rec.get("book_to_price", 0.25))
            ey_raw[i] = float(rec.get("earnings_yield", 0.04))
            sp_raw[i] = float(rec.get("sales_to_price", 0.35))

        def standardize(arr: np.ndarray) -> np.ndarray:
            if len(arr) <= 1:
                return np.zeros_like(arr)
            mu = np.nanmean(arr)
            std = np.nanstd(arr)
            if std < 1e-7:
                return np.zeros_like(arr)
            z = (arr - mu) / std
            return np.clip(np.nan_to_num(z, nan=0.0), -3.0, 3.0)

        size_std = standardize(size_raw)
        vol_std = standardize(vol_raw)
        mom_std = standardize(mom_raw)

        value_composite = (standardize(bp_raw) + standardize(ey_raw) + standardize(sp_raw)) / 3.0
        val_std = standardize(value_composite)

        # Assemble full factor matrix
        factor_dict: dict[str, np.ndarray] = dict(sector_col_data)
        factor_dict["style_size"] = size_std
        factor_dict["style_volatility"] = vol_std
        factor_dict["style_momentum"] = mom_std
        factor_dict["style_value"] = val_std

        factor_df = pd.DataFrame(factor_dict, index=tickers)
        factor_names = list(factor_df.columns)
        return factor_df, factor_names

    def neutralize_alpha(
        self,
        scores: pd.Series,
        factor_matrix: pd.DataFrame,
        method: str = "full_barra",
        l2_reg: float | None = None,
        preserve_scale: bool = True,
    ) -> pd.Series:
        """
        Neutralizes raw alpha / score vector against factor exposures via regularized projection:
        alpha_neutral = alpha - X (X^T X + lambda I)^(-1) X^T alpha

        Parameters:
            scores: pd.Series of expected returns or ranking scores indexed by ticker.
            factor_matrix: DataFrame of shape (N, K) indexed by ticker.
            method: 'full_barra' (sectors + styles), 'sector_only', 'style_only', or 'none'.
            l2_reg: Ridge penalty parameter. Defaults to self.l2_reg (1e-4).
            preserve_scale: If True, adds back original mean so downstream score levels remain aligned.

        Returns:
            pd.Series of neutralized alpha scores indexed by ticker.
        """
        if method == "none" or factor_matrix.empty or scores.empty:
            return scores.copy()

        # Align indices
        common_idx = [t for t in scores.index if t in factor_matrix.index]
        if len(common_idx) < 3:
            return scores.copy()

        y = scores.loc[common_idx].values.astype(float)
        X_sub = factor_matrix.loc[common_idx]

        if method == "sector_only":
            cols = [c for c in X_sub.columns if c.startswith("sector_")]
        elif method == "style_only":
            cols = [c for c in X_sub.columns if c.startswith("style_")]
        else:  # 'full_barra'
            cols = list(X_sub.columns)

        if not cols:
            return scores.copy()

        X = X_sub[cols].values.astype(float)
        n, p = X.shape

        # If style_only without sector dummies, add intercept column so scores are centered
        if method == "style_only":
            X = np.hstack([np.ones((n, 1), dtype=float), X])
            p += 1

        reg = self.l2_reg if l2_reg is None else l2_reg
        XtX = X.T @ X
        lambda_I = reg * np.eye(p)
        Xty = X.T @ y

        try:
            beta = np.linalg.solve(XtX + lambda_I, Xty)
            y_pred = X @ beta
            residual = y - y_pred
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if solve encounters singularity
            beta = np.linalg.pinv(XtX + lambda_I) @ Xty
            y_pred = X @ beta
            residual = y - y_pred

        if preserve_scale:
            neutral = residual + np.nanmean(y)
        else:
            neutral = residual

        neutral_series = pd.Series(neutral, index=common_idx, name=f"{scores.name or 'score'}_neutral")

        # For any tickers in scores but not factor_matrix, retain raw
        for t in scores.index:
            if t not in neutral_series.index:
                neutral_series[t] = scores[t]

        return neutral_series.reindex(scores.index)

    def compute_factor_diagnostics(
        self,
        raw_scores: pd.Series,
        neutral_scores: pd.Series,
        factor_matrix: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Computes comprehensive diagnostic metrics comparing raw and neutralized scores:
        - Factor correlations (before vs after)
        - Max style factor loading
        - R^2 proportion explained
        - Sector score spreads
        """
        common = [t for t in raw_scores.index if t in neutral_scores.index and t in factor_matrix.index]
        if len(common) < 3:
            return {
                "raw_correlations": {},
                "neutral_correlations": {},
                "max_raw_correlation": 0.0,
                "max_neutral_correlation": 0.0,
                "r2_explained": 0.0,
                "sector_biases_raw": {},
                "sector_biases_neutral": {},
            }

        y_raw = raw_scores.loc[common].values.astype(float)
        y_neu = neutral_scores.loc[common].values.astype(float)
        X_df = factor_matrix.loc[common]

        raw_corr: dict[str, float] = {}
        neu_corr: dict[str, float] = {}
        sector_raw: dict[str, float] = {}
        sector_neu: dict[str, float] = {}

        for col in X_df.columns:
            feat_vals = X_df[col].values.astype(float)
            if np.std(feat_vals) > 1e-6:
                if np.std(y_raw) > 1e-6:
                    c_raw = float(np.corrcoef(feat_vals, y_raw)[0, 1])
                else:
                    c_raw = 0.0

                if np.std(y_neu) > 1e-6:
                    c_neu = float(np.corrcoef(feat_vals, y_neu)[0, 1])
                else:
                    c_neu = 0.0

                raw_corr[col] = round(0.0 if np.isnan(c_raw) else c_raw, 4)
                neu_corr[col] = round(0.0 if np.isnan(c_neu) else c_neu, 4)

            if col.startswith("sector_"):
                mask = feat_vals > 0.5
                sec_name = col.replace("sector_", "")
                if np.any(mask):
                    sector_raw[sec_name] = round(float(np.mean(y_raw[mask])), 4)
                    sector_neu[sec_name] = round(float(np.mean(y_neu[mask])), 4)

        style_raw_corrs = [abs(v) for k, v in raw_corr.items() if k.startswith("style_")]
        style_neu_corrs = [abs(v) for k, v in neu_corr.items() if k.startswith("style_")]

        var_raw = float(np.var(y_raw))
        var_neu = float(np.var(y_neu - np.mean(y_raw)))
        r2 = max(0.0, 1.0 - (var_neu / var_raw)) if var_raw > 1e-8 else 0.0

        return {
            "raw_correlations": raw_corr,
            "neutral_correlations": neu_corr,
            "max_raw_correlation": round(float(max(style_raw_corrs)) if style_raw_corrs else 0.0, 4),
            "max_neutral_correlation": round(float(max(style_neu_corrs)) if style_neu_corrs else 0.0, 4),
            "r2_explained": round(float(r2), 4),
            "sector_biases_raw": sector_raw,
            "sector_biases_neutral": sector_neu,
        }
