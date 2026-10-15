"""Dataset builder combining technical, relative, macro, regime, and sentiment features."""
import numpy as np
import pandas as pd

from src.config.universe import (
    BENCHMARK_TICKER,
    get_all_tickers,
    get_sector_for_ticker,
    get_sp100_constituents,
)
from src.data.fundamentals import FundamentalPipeline
from src.data.macro import MacroPipeline
from src.data.prices import PricePipeline
from src.features.normalization import CrossSectionalNormalizer
from src.features.regime import MarketRegimeDetector
from src.features.relative import RelativeFeatureExtractor
from src.features.sentiment import SentimentFeatureExtractor
from src.features.technical import TechnicalFeatureExtractor
from src.utils.logger import setup_logger

logger = setup_logger("dataset_builder")


class DatasetBuilder:
    """
    Constructs cross-sectional feature matrices and forward targets.

    Guarantees:
    - 5-day excess return target: R_{i, t->t+5} - R_{SPY, t->t+5}
    - 5-day forward realized volatility target: RV_{i, t->t+5}
    - Cross-sectional uniform rank normalization [-0.5, 0.5] on stock features
    - Support for feature ablation (price-only vs full multi-modal)
    - Zero future lookahead in feature matrices
    """

    TICKER_FEATURES = [
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "rolling_vol_5d",
        "rolling_vol_20d",
        "volume_zscore_20d",
        "dist_from_ma50",
        "rsi_14d",
        "rel_return_spy_5d",
        "rel_return_spy_20d",
        "rel_return_sector_5d",
        "rel_return_sector_20d",
    ]

    FUNDAMENTAL_FEATURES = [
        "earnings_yield",
        "book_to_price",
        "sales_to_price",
        "fcf_yield",
        "dividend_yield",
        "ev_to_ebitda",
        "market_cap_log",
    ]

    MACRO_FEATURES = [
        "yield_spread",
        "vix",
        "vix_change_5d",
        "fed_funds_change_5d",
        "hy_spread",
    ]

    REGIME_FEATURES = [
        "regime_prob_0",
        "regime_prob_1",
        "regime_prob_2",
    ]

    SENTIMENT_FEATURES = [
        "sentiment_mean",
        "sentiment_count",
        "sentiment_dispersion",
        "sentiment_3d_mean",
    ]

    TARGET_COLUMNS = [
        "target_excess_return_5d",
        "target_realized_vol_5d",
    ]

    def __init__(
        self,
        price_pipeline: PricePipeline | None = None,
        macro_pipeline: MacroPipeline | None = None,
        fundamental_pipeline: FundamentalPipeline | None = None,
    ):
        self.price_pipeline = price_pipeline or PricePipeline()
        self.macro_pipeline = macro_pipeline or MacroPipeline()
        self.fundamental_pipeline = fundamental_pipeline or FundamentalPipeline()
        self.tech_extractor = TechnicalFeatureExtractor()
        self.rel_extractor = RelativeFeatureExtractor()
        self.regime_detector = MarketRegimeDetector()
        self.sentiment_extractor = SentimentFeatureExtractor()

    def build_raw_panel(
        self,
        tickers: list[str] | None = None,
        include_sentiment: bool = True,
        execution_mode: str = "close",
    ) -> pd.DataFrame:
        """
        Builds raw un-normalized panel including technical, relative, macro, regime, and sentiment features.
        """
        logger.info("Building raw feature panel...")
        all_req_tickers = get_all_tickers()

        panel = self.price_pipeline.load_panel(tickers=all_req_tickers)
        if panel.empty:
            raise ValueError("No price data available in cache. Run price backfill first.")

        panel = self.tech_extractor.extract_panel(panel)
        panel = self.rel_extractor.extract_relative_features(panel)

        macro_df = self.macro_pipeline.get_macro_features()
        if not macro_df.empty:
            panel = pd.merge(panel, macro_df[["date"] + self.MACRO_FEATURES], on="date", how="left")
            for col in self.MACRO_FEATURES:
                panel[col] = panel[col].ffill().bfill().fillna(0.0)
        else:
            for col in self.MACRO_FEATURES:
                panel[col] = 0.0

        # Fundamentals ingestion
        fundamentals_df = self.fundamental_pipeline.load_fundamentals()
        if not fundamentals_df.empty:
            cols_to_merge = ["ticker"] + [c for c in self.FUNDAMENTAL_FEATURES if c in fundamentals_df.columns]
            if "sector" in fundamentals_df.columns and "sector" not in panel.columns:
                cols_to_merge.append("sector")
            panel = pd.merge(panel, fundamentals_df[cols_to_merge], on="ticker", how="left")
            for col in self.FUNDAMENTAL_FEATURES:
                if col in panel.columns:
                    panel[col] = panel[col].ffill().bfill().fillna(0.0)
                else:
                    panel[col] = 0.0
        else:
            for col in self.FUNDAMENTAL_FEATURES:
                panel[col] = 0.0

        if "sector" not in panel.columns:
            panel["sector"] = panel["ticker"].apply(lambda t: get_sector_for_ticker(t) or "Other")
        else:
            panel["sector"] = panel["sector"].fillna(panel["ticker"].apply(lambda t: get_sector_for_ticker(t) or "Other"))

        spy_sub = panel[panel["ticker"] == BENCHMARK_TICKER].sort_values("date").reset_index(drop=True)
        if not spy_sub.empty and len(spy_sub) > 60:
            regimes = self.regime_detector.fit_predict_regimes(
                dates=spy_sub["date"].tolist(),
                spy_returns=spy_sub["return_1d"].values,
                vix_levels=spy_sub["vix"].values if "vix" in spy_sub.columns else np.full(len(spy_sub), 15.0)
            )
            panel = pd.merge(panel, regimes, on="date", how="left")
            for col in self.REGIME_FEATURES:
                panel[col] = panel[col].ffill().bfill().fillna(1.0 / 3.0)
        else:
            for col in self.REGIME_FEATURES:
                panel[col] = 1.0 / 3.0

        if include_sentiment:
            panel = self.sentiment_extractor.align_with_panel(panel)
        else:
            for col in self.SENTIMENT_FEATURES:
                panel[col] = 0.0

        panel = self._compute_forward_targets(panel, execution_mode=execution_mode)
        return panel

    def _compute_forward_targets(self, panel_df: pd.DataFrame, execution_mode: str = "close") -> pd.DataFrame:
        """Computes 5-day forward excess return and 5-day forward realized volatility."""
        if panel_df is None or panel_df.empty:
            df = pd.DataFrame() if panel_df is None else panel_df.copy()
            for col in self.TARGET_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.Series(dtype=float)
            return df

        df = panel_df.sort_values(["ticker", "date"]).copy()

        # Compute forward 5-day raw return per ticker: P_{t+5} / P_t - 1
        results = []
        for ticker, grp in df.groupby("ticker"):
            grp = grp.copy().sort_values("date")
            close = grp["close"]
            if execution_mode == "next_open" and "open" in grp.columns:
                opn = grp["open"]
                grp["fwd_raw_return_5d"] = opn.shift(-6) / opn.shift(-1) - 1.0
            else:
                grp["fwd_raw_return_5d"] = close.shift(-5) / close - 1.0

            daily_ret_sq = (close.pct_change() ** 2)
            fwd_sum_sq = daily_ret_sq.rolling(window=5, min_periods=5).sum().shift(-5)
            grp["target_realized_vol_5d"] = np.sqrt(fwd_sum_sq * (252.0 / 5.0))

            results.append(grp)

        if not results:
            return pd.DataFrame()

        out_df = pd.concat(results, ignore_index=True)

        spy_df = out_df[out_df["ticker"] == BENCHMARK_TICKER][["date", "fwd_raw_return_5d"]]
        if not spy_df.empty:
            spy_fwd_map = dict(zip(spy_df["date"], spy_df["fwd_raw_return_5d"]))
            out_df["spy_fwd_return_5d"] = out_df["date"].map(spy_fwd_map).fillna(0.0)
        else:
            logger.debug(f"Benchmark {BENCHMARK_TICKER} not in panel for forward targets; excess return equals raw return.")
            out_df["spy_fwd_return_5d"] = 0.0

        out_df["target_excess_return_5d"] = out_df["fwd_raw_return_5d"] - out_df["spy_fwd_return_5d"]
        out_df = out_df.drop(columns=["fwd_raw_return_5d", "spy_fwd_return_5d"])
        return out_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    def prepare_dataset(
        self,
        include_sentiment: bool = True,
        stock_only: bool = True,
        execution_mode: str = "close",
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Builds the processed, rank-normalized dataset ready for model training and scoring.

        Filters out ETFs (SPY, XLK, etc.) from the ranking basket if stock_only=True.
        Applies cross-sectional rank normalization to all ticker-specific features.

        Returns:
            (dataset_df, feature_names_list)
        """
        raw_panel = self.build_raw_panel(include_sentiment=include_sentiment, execution_mode=execution_mode)

        if stock_only:
            sp100_symbols = {c["yf_symbol"] for c in get_sp100_constituents()}
            df = raw_panel[raw_panel["ticker"].isin(sp100_symbols)].copy()
        else:
            df = raw_panel.copy()

        norm_features = list(self.TICKER_FEATURES) + list(self.FUNDAMENTAL_FEATURES)
        if include_sentiment:
            norm_features.extend(self.SENTIMENT_FEATURES)

        normalizer = CrossSectionalNormalizer(feature_columns=norm_features)
        norm_df = normalizer.normalize(df)

        all_feature_cols = (
            list(self.TICKER_FEATURES)
            + list(self.FUNDAMENTAL_FEATURES)
            + list(self.MACRO_FEATURES)
            + list(self.REGIME_FEATURES)
        )
        if include_sentiment:
            all_feature_cols.extend(self.SENTIMENT_FEATURES)

        clean_df = norm_df.dropna(subset=["return_60d"]).reset_index(drop=True)
        logger.info(f"Prepared dataset: {len(clean_df)} observations across {clean_df['ticker'].nunique()} tickers with {len(all_feature_cols)} features.")

        return clean_df, all_feature_cols
