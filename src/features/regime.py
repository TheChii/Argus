"""Market regime detection using a 3-state Gaussian Hidden Markov Model (HMM)."""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.utils.logger import setup_logger

logger = setup_logger("market_regime")


class MarketRegimeDetector:
    """
    Fits a 3-state Gaussian HMM on SPY daily returns and VIX levels.

    Produces 3 posterior state probabilities:
    - regime_prob_0: Low Volatility / Bull state
    - regime_prob_1: Medium Volatility / Normal state
    - regime_prob_2: High Volatility / Bear & Crisis state

    States are sorted by mean VIX to ensure consistent semantic ordering.
    """

    FEATURE_COLUMNS = ["regime_prob_0", "regime_prob_1", "regime_prob_2"]

    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=150,
            random_state=random_state,
        )
        self.state_order_ = None
        self.is_fitted = False

    def fit(self, spy_returns: np.ndarray, vix_levels: np.ndarray) -> "MarketRegimeDetector":
        """Fits Gaussian HMM parameters strictly on provided returns and VIX."""
        mask = (~np.isnan(spy_returns)) & (~np.isnan(vix_levels))
        if np.sum(mask) < self.n_states * 10:
            logger.warning("Insufficient data points to fit Gaussian HMM.")
            return self

        clean_X = np.column_stack([spy_returns[mask], vix_levels[mask]])
        self.model.fit(clean_X)
        vix_means = self.model.means_[:, 1]
        self.state_order_ = np.argsort(vix_means)
        self.is_fitted = True
        return self

    def predict_filtered_proba(self, spy_returns: np.ndarray, vix_levels: np.ndarray) -> np.ndarray:
        """
        Computes forward-only filtered state probabilities: P(S_t | x_{1:t}).
        Strictly zero future lookahead: does NOT use backward smoothing.
        """
        n_samples = len(spy_returns)
        if not self.is_fitted or n_samples == 0:
            return np.full((n_samples, self.n_states), 1.0 / self.n_states)

        clean_X = np.column_stack([
            np.nan_to_num(spy_returns, nan=0.0),
            np.nan_to_num(vix_levels, nan=15.0)
        ])

        # Emission log likelihoods
        framelogprob = self.model._compute_log_likelihood(clean_X)
        frame_prob = np.exp(np.clip(framelogprob, -50.0, 50.0))

        filtered = np.zeros((n_samples, self.n_states), dtype=float)
        # t = 0 initialization: alpha_0 = pi * B(x_0)
        alpha = self.model.startprob_ * frame_prob[0]
        s0 = np.sum(alpha)
        alpha = alpha / s0 if s0 > 0 else np.full(self.n_states, 1.0 / self.n_states)
        filtered[0] = alpha

        # Forward pass: alpha_t = (alpha_{t-1} @ A) * B(x_t)
        for t in range(1, n_samples):
            alpha = (alpha @ self.model.transmat_) * frame_prob[t]
            st = np.sum(alpha)
            if st > 1e-12:
                alpha = alpha / st
            else:
                alpha = np.full(self.n_states, 1.0 / self.n_states)
            filtered[t] = alpha

        # Sort columns according to semantic order (State 0 = low VIX, State 2 = high VIX)
        if self.state_order_ is not None:
            filtered = filtered[:, self.state_order_]

        return filtered

    def fit_predict_regimes(
        self,
        dates: list[str],
        spy_returns: np.ndarray,
        vix_levels: np.ndarray
    ) -> pd.DataFrame:
        """
        Fits HMM on historical series and returns strictly forward-filtered regime probabilities.
        """
        try:
            self.fit(spy_returns, vix_levels)
            probs = self.predict_filtered_proba(spy_returns, vix_levels)
            res_df = pd.DataFrame(probs, columns=self.FEATURE_COLUMNS)
            res_df["date"] = dates
            # Clean forward fill without backward fill
            for col in self.FEATURE_COLUMNS:
                res_df[col] = res_df[col].ffill().fillna(1.0 / self.n_states)
            return res_df[["date"] + self.FEATURE_COLUMNS]

        except Exception as e:
            logger.error(f"HMM fitting/filtering failed ({e}). Returning uniform regime probabilities.")
            out = pd.DataFrame({"date": dates})
            for col in self.FEATURE_COLUMNS:
                out[col] = 1.0 / self.n_states
            return out
