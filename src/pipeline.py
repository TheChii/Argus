"""Daily Pipeline Orchestrator for Prices, Macro, and News Archive."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.config.universe import get_all_tickers, get_sp100_constituents
from src.dashboard.alerts import AlertDispatcher
from src.dashboard.snapshot import DashboardSnapshotBuilder
from src.data.fundamentals import FundamentalPipeline
from src.data.macro import MacroPipeline
from src.data.news.archive import HeadlineArchive
from src.data.news.finnhub import FinnhubNewsClient
from src.data.prices import PricePipeline
from src.features.dataset import DatasetBuilder
from src.models.explainability.shap_explainer import ModelExplainer
from src.models.return_ranker.ensemble import BlendedReturnRanker
from src.models.sentiment.scorer import FinBERTScorer
from src.models.vol_model.har import HARVolatilityModel
from src.portfolio.allocator import PortfolioAllocator
from src.portfolio.prediction_ledger import PredictionLedger
from src.reports.weekly_report import WeeklyReportGenerator
from src.utils.calendar import get_current_utc_iso
from src.utils.logger import setup_logger

logger = setup_logger("pipeline")


class DailyPipelineOrchestrator:
    """
    Orchestrates the daily end-of-day data ingestion pipeline.

    Guarantees:
    - Idempotent execution (safe to run multiple times per day).
    - Catches up on any missed trading days.
    - Rate-limited and cached: avoids repeated fetches of existing data.
    - Logs structured status reports to JSON.
    """

    def __init__(self, data_dir: Path | None = None):
        self.settings = get_settings()
        if data_dir:
            self.settings.data_dir = data_dir
        self.settings.ensure_directories()

        self.price_pipeline = PricePipeline(prices_dir=self.settings.prices_dir)
        self.macro_pipeline = MacroPipeline(macro_dir=self.settings.macro_dir)
        self.fundamental_pipeline = FundamentalPipeline(fundamentals_dir=self.settings.fundamentals_dir)
        self.headline_archive = HeadlineArchive(archive_dir=self.settings.news_archive_dir)
        self.finnhub_client = FinnhubNewsClient()
        self.finbert_scorer = FinBERTScorer()
        self.snapshot_builder = DashboardSnapshotBuilder()
        self.alerts = AlertDispatcher()

    def run(
        self,
        tickers: list[str] | None = None,
        lookback_years: int = 5,
        news_days_back: int = 2,
        force_prices: bool = False,
        force_macro: bool = False,
        rebalance: bool = False,
        auto_rebalance_weekly: bool = False,
        sizing_method: str = "inv_vol",
        commit_holdings: bool = False,
        cost_bps: float = 7.5,
        portfolio_value: float = 100_000.0,
    ) -> dict[str, Any]:
        """
        Executes the daily ingestion steps:
        1. Prices sync for all universe tickers.
        2. FRED Macro series sync.
        3. News archive sync.
        4. FinBERT scoring of new headlines.
        5. Daily model inference, audit logging & portfolio trade list generation.
        """
        run_start = datetime.now()
        run_date = run_start.strftime("%Y-%m-%d")

        # Concurrency Protection: Acquire non-blocking lock to prevent overlapping runs
        lock_file = self.settings.data_dir / "pipeline.lock"
        lock_fd = None
        try:
            import fcntl
            lock_fd = open(lock_file, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            logger.warning("Another pipeline run is currently in progress. Exiting to prevent concurrency collision.")
            if lock_fd:
                try:
                    lock_fd.close()
                except OSError as e:
                    logger.debug("Failed closing lock file descriptor: %s", e)
            return {
                "status": "busy",
                "message": "Another daily pipeline instance is currently running.",
                "run_date": run_date,
            }
        except (ImportError, AttributeError, OSError) as e:
            logger.debug("fcntl locking not supported or failed (%s); proceeding without flock.", e)

        logger.info(f"========== Starting Daily Pipeline Run for {run_date} ==========")

        if tickers is None:
            tickers = get_all_tickers()

        # Step 1: Prices update
        logger.info("--- Step 1: Syncing Equity & ETF Prices ---")
        price_results = self.price_pipeline.update_all(
            tickers=tickers,
            lookback_years=lookback_years,
            force=force_prices,
        )

        # Step 2: Macro update
        logger.info("--- Step 2: Syncing FRED Macroeconomic Data ---")
        macro_results = self.macro_pipeline.update_macro_data(force=force_macro)

        # Step 2b: Corporate Valuation Fundamentals update
        logger.info("--- Step 2b: Syncing Corporate Valuation Fundamentals ---")
        try:
            sp100_symbols = [c["yf_symbol"] for c in get_sp100_constituents()]
            target_fund_tickers = [t for t in tickers if t in sp100_symbols] if tickers else sp100_symbols
            fund_df = self.fundamental_pipeline.sync_universe(tickers=target_fund_tickers, force=force_prices)
            fund_results = {"total_synced": len(fund_df)}
        except Exception as e:
            logger.warning(f"Fundamental sync encountered error: {e}; proceeding with existing cache.")
            fund_results = {"error": str(e)}

        # Step 3: News Archive update
        logger.info(f"--- Step 3: Syncing News Headlines (lookback: {news_days_back} days) ---")
        news_results = {"total_received": 0, "inserted": 0, "duplicates": 0}

        if self.finnhub_client.is_configured():
            # Finnhub query: only for company constituents (exclude raw ETF tickers)
            sp100_symbols = [c["yf_symbol"] for c in get_sp100_constituents()]
            target_news_tickers = [t for t in tickers if t in sp100_symbols]
            raw_articles = self.finnhub_client.fetch_recent_news_for_tickers(
                tickers=target_news_tickers,
                days_back=news_days_back,
            )
            news_results = self.headline_archive.append_articles(raw_articles)
        else:
            logger.info("FINNHUB_API_KEY not configured. Skipping live news fetch.")

        archive_stats = self.headline_archive.get_archive_stats()

        # Step 4: Score newly archived headlines with FinBERT
        logger.info("--- Step 4: Scoring Unscored Headlines with FinBERT ---")
        all_headlines = self.headline_archive.load_headlines()
        if not all_headlines.empty:
            scores_df = self.finbert_scorer.score_articles(all_headlines.to_dict(orient="records"))
            total_scored = len(scores_df)
        else:
            total_scored = 0

        # Step 5: Update mature predictions in ledger & compute rolling live IC
        logger.info("--- Step 5: Updating Prediction Ledger & Rolling Live IC ---")
        ledger = PredictionLedger()
        ledger.update_mature_predictions()
        latest_rolling_ic = ledger.get_latest_rolling_ic()
        ic_history = ledger.compute_rolling_ic_history()

        # Step 6: Daily model inference, SHAP explanations & score logging
        logger.info("--- Step 6: Daily Model Inference & SHAP Explainability ---")
        scoring_results: dict[str, Any] = {}
        trade_orders: list[dict[str, Any]] = []
        weekly_report_path: str | None = None

        try:
            scores_dir = self.settings.data_dir / "scores"
            scores_dir.mkdir(parents=True, exist_ok=True)
            daily_scores_file = scores_dir / f"daily_scores_{run_date}.parquet"

            dataset_builder = DatasetBuilder(
                price_pipeline=self.price_pipeline,
                macro_pipeline=self.macro_pipeline,
            )
            panel_df, feature_cols = dataset_builder.prepare_dataset(include_sentiment=True)

            if not panel_df.empty:
                max_date = str(panel_df["date"].max())
                today_slice = panel_df[panel_df["date"] == max_date].copy()

                # Train on history up to 5 days prior to max_date (purge gap)
                train_dates = sorted(panel_df[panel_df["date"] < max_date]["date"].unique())
                if len(train_dates) > 5:
                    train_dates = train_dates[:-5]

                train_df = panel_df[panel_df["date"].isin(set(train_dates))].dropna(subset=["target_excess_return_5d"])

                if len(train_df) >= 100 and not today_slice.empty:
                    X_train = train_df[feature_cols].values
                    y_train = train_df["target_excess_return_5d"].values
                    X_today = today_slice[feature_cols].values

                    # Fit blended return ranker
                    ranker = BlendedReturnRanker()
                    ranker.fit(X_train, y_train, feature_names=feature_cols)
                    today_slice["pred_score"] = ranker.predict(X_today)

                    # Fit HAR vol model for position sizing
                    har_model = HARVolatilityModel()
                    har_model.fit(train_df)
                    today_slice["predicted_vol"] = har_model.predict(today_slice)

                    # SHAP Feature Attributions & Top 3 Reasons
                    explainer = ModelExplainer(feature_names=feature_cols)
                    today_slice = explainer.explain_dataframe(today_slice, ranker, feature_cols, top_k=3)

                    # Sort and save audit trail
                    today_slice = today_slice.sort_values("pred_score", ascending=False).reset_index(drop=True)
                    today_slice["rank"] = np.arange(1, len(today_slice) + 1)

                    audit_cols = [
                        "date", "ticker", "rank", "pred_score", "predicted_vol",
                        "shap_top_1", "shap_top_2", "shap_top_3", "shap_top_reasons",
                    ]
                    available_audit = [c for c in audit_cols if c in today_slice.columns]
                    today_slice[available_audit].to_parquet(daily_scores_file, index=False)
                    logger.info(f"Saved {len(today_slice)} daily scores with SHAP attributions to: {daily_scores_file}")

                    top_names = today_slice["ticker"].head(5).tolist()
                    bottom_names = today_slice["ticker"].tail(5).tolist()

                    # Step 7: Rules-based portfolio allocation & rebalancing
                    allocator = PortfolioAllocator()
                    current_holdings = allocator.load_holdings()

                    # Extract current VIX from macro
                    macro_df = self.macro_pipeline.load_macro_data()
                    current_vix = None
                    if macro_df is not None and not macro_df.empty and "vix" in macro_df.columns:
                        vix_s = macro_df["vix"].dropna()
                        if not vix_s.empty:
                            current_vix = float(vix_s.iloc[-1])

                    target_weights_df, portfolio_meta = allocator.compute_target_weights(
                        today_slice,
                        vol_df=today_slice[["ticker", "predicted_vol"]],
                        current_holdings=current_holdings,
                        current_vix=current_vix,
                        rolling_60d_ic=latest_rolling_ic,
                        method=sizing_method,
                        top_n=15,
                        use_turnover_buffer=True,
                        use_exposure_scaling=True,
                        use_confidence_filter=True,
                        neutralization_mode="full_barra",
                        max_sector_weight=0.25,
                        max_names_per_sector=3,
                        return_metadata=True,
                    )

                    # Propagate SHAP descriptions into target_weights_df for reporting
                    for sc in ["shap_top_1", "shap_top_2", "shap_top_3"]:
                        if sc in today_slice.columns and sc not in target_weights_df.columns:
                            s_map = dict(zip(today_slice["ticker"], today_slice[sc]))
                            target_weights_df[sc] = target_weights_df["ticker"].map(s_map)

                    orders, trade_summary = allocator.generate_trade_list(target_weights_df, current_holdings=current_holdings)
                    trade_orders = orders

                    # Determine if weekly rebalancing should execute (Friday close in America/New_York)
                    import zoneinfo
                    et_tz = zoneinfo.ZoneInfo("America/New_York")
                    is_friday = (datetime.now(tz=et_tz).weekday() == 4) or (pd.to_datetime(max_date).weekday() == 4)
                    is_rebalance = rebalance or (auto_rebalance_weekly and is_friday)

                    if is_rebalance and commit_holdings:
                        new_holdings = {
                            row["ticker"]: float(row["target_weight"])
                            for _, row in target_weights_df.iterrows()
                            if row["ticker"] not in ("CASH",)
                        }
                        allocator.save_holdings(new_holdings)

                    if is_rebalance:
                        report_gen = WeeklyReportGenerator()
                        report_gen.generate_report(
                            as_of_date=max_date,
                            target_df=target_weights_df,
                            portfolio_meta=portfolio_meta,
                            orders=orders,
                            trade_summary=trade_summary,
                            ic_summary=ic_history,
                        )
                        weekly_report_path = str(self.settings.data_dir / "reports" / f"weekly_report_{max_date}.md")

                    scoring_results = {
                        "date_scored": max_date,
                        "tickers_scored": len(today_slice),
                        "top_5": top_names,
                        "bottom_5": bottom_names,
                        "rebalance_executed": is_rebalance,
                        "trade_orders_count": len(orders),
                        "one_way_turnover": trade_summary.get("one_way_turnover", 0.0),
                        "exposure_scale": portfolio_meta.get("exposure_scale", 1.0),
                        "confidence_filter_triggered": portfolio_meta.get("confidence_filter_triggered", False),
                        "live_rolling_60d_ic": latest_rolling_ic,
                        "weekly_report_path": weekly_report_path,
                    }
                else:
                    scoring_results = {"status": "insufficient_history", "train_rows": len(train_df)}
            else:
                scoring_results = {"status": "empty_panel"}

        except Exception as e:
            logger.error(f"Daily inference / portfolio step encountered error: {e}")
            scoring_results = {"status": "error", "error": str(e)}

        run_end = datetime.now()
        duration_seconds = (run_end - run_start).total_seconds()

        report = {
            "run_date": run_date,
            "timestamp": get_current_utc_iso(),
            "duration_seconds": round(duration_seconds, 2),
            "prices": {
                "total_tickers": price_results.get("total_tickers", 0),
                "updated": price_results.get("updated", 0),
                "skipped": price_results.get("skipped", 0),
                "errors": price_results.get("errors", 0),
            },
            "macro": {
                "status": macro_results.get("status"),
                "latest_date": macro_results.get("latest_date"),
                "rows_added": macro_results.get("rows_added", 0),
                "total_rows": macro_results.get("total_rows", 0),
            },
            "fundamentals": fund_results,
            "news": {
                "fetched_batch": news_results.get("total_received", 0),
                "new_archived": news_results.get("inserted", 0),
                "duplicates_skipped": news_results.get("duplicates", 0),
                "total_archive_size": archive_stats.get("total_headlines", 0),
                "total_scored_headlines": total_scored,
            },
            "scoring": scoring_results,
            "trade_orders": trade_orders if scoring_results.get("rebalance_executed") else [],
        }

        # Step 8: Precompute operations dashboard snapshot & dispatch alerts
        logger.info("--- Step 8: Exporting Dashboard Snapshot & Alert Checks ---")
        try:
            old_light = None
            snap_file = self.settings.data_dir / "dashboard" / "today_snapshot.json"
            if snap_file.exists():
                try:
                    with open(snap_file, encoding="utf-8") as f:
                        prev_snap = json.load(f)
                    old_light = prev_snap.get("meta", {}).get("status_strip", {}).get("confidence_light")
                except Exception:
                    pass

            new_snap = self.snapshot_builder.generate_and_save_snapshot()
            status_strip = new_snap.get("meta", {}).get("status_strip", {})
            new_light = status_strip.get("confidence_light", "green")
            new_ic = status_strip.get("rolling_60d_ic", 0.0)
            new_t = status_strip.get("rolling_t_stat", 0.0)

            # 1. Check confidence light transition
            if old_light and old_light != new_light:
                self.alerts.notify_confidence_light_change(
                    old_color=old_light,
                    new_color=new_light,
                    ic=new_ic,
                    t_stat=new_t,
                )

            # 2. Check for pipeline step failures
            if price_results.get("errors", 0) > 0:
                self.alerts.notify_pipeline_failure("Price Pipeline", f"{price_results['errors']} ticker download errors encountered")
            if scoring_results.get("status") == "error":
                self.alerts.notify_pipeline_failure("Scoring & Allocation", str(scoring_results.get("error")))

            # 3. Check for weekly report readiness
            if weekly_report_path:
                turnover_pct = scoring_results.get("one_way_turnover", 0.0) * 100
                self.alerts.notify_weekly_report_ready(
                    as_of_date=scoring_results.get("date_scored", run_date),
                    num_orders=len(trade_orders),
                    turnover_pct=turnover_pct,
                )

            report["dashboard_snapshot_saved"] = True
        except Exception as e:
            logger.error(f"Failed to generate dashboard snapshot or dispatch alerts: {e}")
            report["dashboard_snapshot_saved"] = False
            report["dashboard_error"] = str(e)

        # Write execution report to logs directory
        log_file = self.settings.logs_dir / f"pipeline_run_{run_date}.json"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Execution report saved to {log_file}")
        except Exception as e:
            logger.error(f"Failed to write execution log: {e}")

        # Release concurrency lock
        if lock_fd:
            try:
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass

        logger.info(f"========== Pipeline Run Finished in {duration_seconds:.1f}s ==========")
        return report

    def get_status_summary(self) -> dict[str, Any]:
        """Provides a comprehensive health check and coverage summary."""
        tickers = get_all_tickers()
        cached_prices = 0
        latest_price_dates = []

        for t in tickers:
            d = self.price_pipeline.get_latest_cached_date(t)
            if d:
                cached_prices += 1
                latest_price_dates.append(d)

        macro_latest = self.macro_pipeline.get_latest_cached_date()
        macro_df = self.macro_pipeline.load_macro_data()
        macro_rows = len(macro_df) if macro_df is not None else 0

        archive_stats = self.headline_archive.get_archive_stats()
        scores_df = self.finbert_scorer.load_scores()
        archive_stats["scored_headlines"] = len(scores_df)

        earliest_price_date = None
        spy_p = self.price_pipeline.load_ticker_prices("SPY")
        if spy_p is not None and not spy_p.empty and "date" in spy_p.columns:
            earliest_price_date = str(spy_p["date"].min())

        return {
            "universe": {
                "total_universe_tickers": len(tickers),
                "tickers_with_price_cache": cached_prices,
                "coverage_pct": round((cached_prices / len(tickers)) * 100, 1) if tickers else 0.0,
                "max_price_date": max(latest_price_dates) if latest_price_dates else None,
                "min_price_date": earliest_price_date or (min(latest_price_dates) if latest_price_dates else None),
            },
            "macro": {
                "cached": macro_latest is not None,
                "latest_date": macro_latest,
                "total_rows": macro_rows,
                "series_present": self.macro_pipeline.REQUIRED_SERIES,
            },
            "news_archive": archive_stats,
        }
