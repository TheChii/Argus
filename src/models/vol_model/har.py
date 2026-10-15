"""Heterogeneous Autoregressive (HAR) model for forecasting 5-day forward realized volatility."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import setup_logger

logger = setup_logger("har_vol_model")


class HARVolatilityModel:
    """
    Log-linear Heterogeneous Autoregressive (HAR) Realized Volatility Model.

    Predicts next-5-day realized volatility RV_{t->t+5} from:
    - RV_daily: 1-day realized vol component
    - RV_weekly: 5-day historical realized vol component
    - RV_monthly: 20-day historical realized vol component
    - VIX: Market CBOE Volatility Index

    Log-transformation ensures strictly positive volatility predictions:
    log(RV_{t->t+5}) = b0 + bd*log(RV_d) + bw*log(RV_w) + bm*log(RV_m) + bv*log(VIX)
    """

    HAR_FEATURES = ["log_rv_daily", "log_rv_weekly", "log_rv_monthly", "log_vix"]

    def __init__(self, alphas: list[float] | None = None):
        self.alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0]
        self.model = RidgeCV(alphas=self.alphas)
        self.is_fitted = False
        self.feature_names_ = self.HAR_FEATURES

    @staticmethod
    def extract_har_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts HAR input features from DataFrame.
        Expects df columns: ['return_1d', 'rolling_vol_20d', 'vix']
        """
        feat = pd.DataFrame(index=df.index)

        # 1-day realized vol
        ret1 = df["return_1d"].fillna(0.0)
        rv_d = (np.abs(ret1) * np.sqrt(252)).clip(lower=0.01, upper=5.0)

        # 5-day realized vol (strictly per-ticker)
        if "rolling_vol_5d" in df.columns:
            rv_w = df["rolling_vol_5d"].fillna(df.get("rolling_vol_20d", 0.20)).clip(lower=0.01, upper=5.0)
        elif "ticker" in df.columns and df["ticker"].nunique() > 1:
            rv_w = df.groupby("ticker")["return_1d"].transform(
                lambda s: np.sqrt((s ** 2).rolling(window=5, min_periods=3).mean() * 252)
            ).fillna(0.20).clip(lower=0.01, upper=5.0)
        else:
            ret_sq = (ret1 ** 2)
            rv_w = np.sqrt(ret_sq.rolling(window=5, min_periods=3).mean() * 252).fillna(0.20).clip(lower=0.01, upper=5.0)

        # 20-day realized vol
        rv_m = df["rolling_vol_20d"].fillna(0.20).clip(lower=0.01, upper=5.0)

        # VIX
        vix = (df["vix"].fillna(15.0) / 100.0).clip(lower=0.05, upper=1.5)

        feat["log_rv_daily"] = np.log(rv_d)
        feat["log_rv_weekly"] = np.log(rv_w.fillna(rv_m))
        feat["log_rv_monthly"] = np.log(rv_m)
        feat["log_vix"] = np.log(vix)

        return feat

    def fit(self, df: pd.DataFrame, target_col: str = "target_realized_vol_5d") -> dict[str, Any]:
        """
        Fits log-linear Ridge model on historical realized volatility targets.
        """
        clean_mask = (df[target_col].notna()) & (df[target_col] > 0.001) & (df["return_60d"].notna())
        clean_df = df[clean_mask].copy()

        if len(clean_df) < 50:
            raise ValueError(f"Insufficient training observations for HAR model ({len(clean_df)} rows).")

        X_har = self.extract_har_features(clean_df)
        y_log = np.log(clean_df[target_col].clip(lower=0.01, upper=5.0).values)

        self.model.fit(X_har.values, y_log)
        self.is_fitted = True

        y_pred_log = self.model.predict(X_har.values)
        y_pred = np.exp(y_pred_log)
        y_true = clean_df[target_col].values

        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        logger.info(f"HAR Vol Model fitted: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, alpha={self.model.alpha_:.2f}")
        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "best_alpha": float(self.model.alpha_),
            "coefficients": dict(zip(self.HAR_FEATURES, [float(c) for c in self.model.coef_])),
            "intercept": float(self.model.intercept_),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts 5-day forward annualized volatility (sigma).
        Outputs are clipped within realistic boundaries [0.05, 1.50] (5% to 150%).
        """
        if not self.is_fitted:
            # Fallback to 20d rolling vol if not fitted
            return df["rolling_vol_20d"].fillna(0.20).clip(lower=0.05, upper=1.50).values

        X_har = self.extract_har_features(df)
        pred_log = self.model.predict(X_har.values)
        pred_vol = np.exp(pred_log)
        # Financial sanity clamp: 5% to 150% annualized volatility
        return np.clip(pred_vol, 0.05, 1.50)
