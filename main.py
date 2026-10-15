"""CLI interface for Argus Quant: data pipeline, FinBERT scoring, models & backtesting."""
import os
import sys
from pathlib import Path

# Auto-switch to project virtual environment if invoked via system Python
_venv_dir = Path(__file__).resolve().parent / ".venv"
_venv_python = _venv_dir / "bin" / "python"
if _venv_python.exists() and Path(sys.prefix).resolve() != _venv_dir.resolve():
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

import argparse
import json

import pandas as pd

from src.backtest.cost_adjusted import CostAdjustedBacktest
from src.config.settings import get_settings
from src.data.macro import MacroPipeline
from src.data.news.archive import HeadlineArchive
from src.data.news.backfill import HistoricalNewsBackfiller
from src.data.prices import PricePipeline
from src.features.dataset import DatasetBuilder
from src.models.sanity_checks import ModelSanityChecker
from src.models.sentiment.scorer import FinBERTScorer
from src.models.validation import WalkForwardValidator
from src.models.vol_model.har import HARVolatilityModel
from src.pipeline import DailyPipelineOrchestrator
from src.portfolio.allocator import PortfolioAllocator
from src.portfolio.prediction_ledger import PredictionLedger
from src.reports.weekly_report import WeeklyReportGenerator
from src.utils.logger import setup_logger

logger = setup_logger("cli")


def cmd_run_daily(args):
    orchestrator = DailyPipelineOrchestrator()
    tickers = args.tickers.split(",") if args.tickers else None
    report = orchestrator.run(
        tickers=tickers,
        lookback_years=args.lookback_years,
        news_days_back=args.news_days_back,
        force_prices=args.force,
        force_macro=args.force,
        rebalance=args.rebalance,
        auto_rebalance_weekly=getattr(args, "auto_rebalance", False),
        sizing_method=args.sizing_method,
        commit_holdings=args.commit_holdings,
    )
    print("\n--- Pipeline Run Summary ---")
    print(json.dumps(report, indent=2))


def cmd_status(args):
    orchestrator = DailyPipelineOrchestrator()
    summary = orchestrator.get_status_summary()
    print("\n================ DATA PIPELINE STATUS ================")
    print(f"Universe Coverage: {summary['universe']['tickers_with_price_cache']} / {summary['universe']['total_universe_tickers']} tickers ({summary['universe']['coverage_pct']}%)")
    print(f"Price Dates:       {summary['universe']['min_price_date']} -> {summary['universe']['max_price_date']}")
    print(f"Macro Status:      {'Cached' if summary['macro']['cached'] else 'Missing'} (Latest: {summary['macro']['latest_date']}, Rows: {summary['macro']['total_rows']})")
    print(f"News Archive:      {summary['news_archive']['total_headlines']} headlines across {summary['news_archive']['unique_tickers']} tickers")
    print(f"Scored Headlines:  {summary['news_archive'].get('scored_headlines', 0)}")
    if summary['news_archive']['latest_trading_date']:
        print(f"News Date Range:   {summary['news_archive']['earliest_trading_date']} -> {summary['news_archive']['latest_trading_date']}")
    print("======================================================\n")


def cmd_backfill_prices(args):
    pipeline = PricePipeline()
    tickers = args.tickers.split(",") if args.tickers else None
    results = pipeline.update_all(tickers=tickers, lookback_years=args.years, force=args.force)
    print("\n--- Price Backfill Summary ---")
    print(f"Updated: {results['updated']}, Skipped: {results['skipped']}, Errors: {results['errors']}")


def cmd_backfill_macro(args):
    pipeline = MacroPipeline()
    res = pipeline.update_macro_data(force=args.force)
    print("\n--- Macro Sync Summary ---")
    print(json.dumps(res, indent=2))


def cmd_backfill_news(args):
    """Backfills historical company headlines across the universe via Finnhub."""
    backfiller = HistoricalNewsBackfiller()
    tickers = args.tickers.split(",") if args.tickers else None
    res = backfiller.backfill_tickers(
        tickers=tickers,
        days_back=args.days_back,
        num_windows=args.windows,
    )
    print("\n--- News Backfill Summary ---")
    print(json.dumps(res, indent=2))

    if args.score:
        print("\nScoring newly ingested headlines with FinBERT...")
        archive = HeadlineArchive()
        scorer = FinBERTScorer()
        headlines_df = archive.load_headlines()
        if not headlines_df.empty:
            scored_df = scorer.score_articles(headlines_df.to_dict(orient="records"), batch_size=args.batch_size)
            print(f"Scoring complete. Total cached scores: {len(scored_df)}")


def cmd_sample_test(args):
    """Runs a quick end-to-end test on a small subset of tickers."""
    sample_tickers = ["SPY", "AAPL", "MSFT", "XLK"]
    print(f"Running sample pipeline test on: {sample_tickers}")
    orchestrator = DailyPipelineOrchestrator()
    report = orchestrator.run(tickers=sample_tickers, lookback_years=5, news_days_back=1)
    print("\n--- Sample Test Result ---")
    print(json.dumps(report, indent=2))


def cmd_score_headlines(args):
    """Batch scores all unscored headlines with FinBERT."""
    archive = HeadlineArchive()
    scorer = FinBERTScorer()
    headlines_df = archive.load_headlines()
    if headlines_df.empty:
        print("No headlines found in archive.")
        return
    print(f"Found {len(headlines_df)} total headlines in archive. Scoring unscored headlines...")
    scored_df = scorer.score_articles(headlines_df.to_dict(orient="records"), batch_size=args.batch_size)
    print(f"\nScoring complete. Total cached scores: {len(scored_df)}")
    print(scored_df[["article_hash", "sentiment_score", "scored_at"]].tail(5).to_string())


def cmd_train_baseline(args):
    """Trains HAR volatility model and Ridge baseline with walk-forward validation."""
    print("Building analysis dataset...")
    builder = DatasetBuilder()
    df, feature_cols = builder.prepare_dataset(include_sentiment=args.with_sentiment)

    print(f"\n1. Fitting HAR Volatility Model on {len(df)} observations...")
    har_model = HARVolatilityModel()
    har_stats = har_model.fit(df)
    print(f"   HAR Vol Model: R2 = {har_stats['r2']:.3f}, RMSE = {har_stats['rmse']:.3f}, MAE = {har_stats['mae']:.3f}")

    print("\n2. Running Walk-Forward Cross-Validation on Ridge Return Ranker...")
    print(f"   Features ({len(feature_cols)}): {feature_cols}")
    print(f"   Purge gap: 5 trading days | Min train window: {args.min_train_days} days")
    validator = WalkForwardValidator(min_train_days=args.min_train_days, test_step_days=60, purge_gap_days=5)
    results = validator.evaluate(df, feature_cols=feature_cols, model_type="ridge")

    ic = results["ic_summary"]
    qs = results["quintile_summary"]
    print("\n================ RIDGE BASELINE EVALUATION ================")
    print(f"Out-of-Sample Splits:    {results['splits_count']}")
    print(f"Mean Daily Spearman IC:  {ic['mean_ic']} (IR: {ic['ic_ir']}, Positive Days: {ic['pct_positive']}%)")
    print(f"IC t-statistic:          {ic['t_stat']}")
    print(f"Quintile Spread Return:  {qs['annualized_return'] * 100:.1f}% ann. (mean/period: {qs['mean_period_spread'] * 100:.2f}%)")
    print(f"Quintile Sharpe Ratio:   {qs['sharpe_ratio']}")
    print("===========================================================\n")

    if args.run_sanity_check:
        print("3. Running Label Shuffle Sanity Check (collapsed performance check)...")
        sanity = validator.run_label_shuffle_sanity_check(df, feature_cols=feature_cols, model_type="ridge")
        print(f"   Shuffled Mean IC: {sanity['shuffled_mean_ic']} (t-stat: {sanity['shuffled_t_stat']})")
        print(f"   Collapsed to Chance: {sanity['collapsed_to_chance']}\n")


def cmd_train_lightgbm(args):
    """Walk-forward evaluation comparing Ridge, LightGBM (multi-seed), and 50/50 Blend."""
    print("Building analysis dataset...")
    builder = DatasetBuilder()
    df, feature_cols = builder.prepare_dataset(include_sentiment=args.with_sentiment)

    print("Running Walk-Forward Comparison across Ridge, Multi-Seed LightGBM, and 50/50 Blend...")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Purge gap: 5 trading days | Min train window: {args.min_train_days} days | Retrain step: 60 days\n")

    validator = WalkForwardValidator(min_train_days=args.min_train_days, test_step_days=60, purge_gap_days=5)
    comp_df = validator.evaluate_comparison(df, feature_cols=feature_cols)

    print("================ WALK-FORWARD MODEL COMPARISON ================")
    print(comp_df.to_string(index=False))
    print("================================================================\n")


def cmd_run_sanity_checks(args):
    """Runs automated sanity checks: Label Shuffle & 1-Day Extra Feature Lag (Leakage Check)."""
    print("Building analysis dataset for Sanity Checks...")
    builder = DatasetBuilder()
    df, feature_cols = builder.prepare_dataset(include_sentiment=False)

    checker = ModelSanityChecker(validator=WalkForwardValidator(min_train_days=args.min_train_days, test_step_days=60, purge_gap_days=5))

    s1 = checker.run_label_shuffle_test(df, feature_cols=feature_cols)
    s2 = checker.run_feature_lag_leakage_test(df, feature_cols=feature_cols)
    exp = checker.record_experiment_trial(feature_cols, "Ridge+LGBM", s2.get("base_mean_ic", 0.022))

    print("\n================ SANITY CHECKS REPORT ================")
    print(f"1. Label Shuffle Test:        {s1['verdict']} (Shuffled IC: {s1['shuffled_mean_ic']:.4f}, t={s1['shuffled_t_stat']:.2f})")
    print(f"2. 1-Day Extra Lag Test:      {s2['verdict']} (Base IC: {s2['base_mean_ic']:.4f} -> Lagged IC: {s2['lagged_mean_ic']:.4f}, Decay: {s2['drop_percentage']}%)")
    print(f"3. Multiple Testing Counter:  {exp['total_experiments_recorded']} trials recorded (Hurdle t-stat: {exp['multiple_testing_hurdle_t']})")
    print("======================================================\n")


def cmd_backtest(args):
    """Runs cost-adjusted weekly quintile backtest on out-of-sample predictions."""
    print("Building analysis dataset...")
    builder = DatasetBuilder()
    df, feature_cols = builder.prepare_dataset(include_sentiment=args.with_sentiment)

    validator = WalkForwardValidator(min_train_days=args.min_train_days, test_step_days=60, purge_gap_days=5)
    print(f"Generating out-of-sample walk-forward predictions for [{args.model.upper()}]...")
    res = validator.evaluate(df, feature_cols=feature_cols, model_type=args.model)
    oos_df = res["oos_predictions"]

    print(f"Running Cost-Adjusted Weekly Backtest (cost: {args.cost_bps} bps per trade)...")
    bt = CostAdjustedBacktest(cost_bps=args.cost_bps, rebalance_interval=5)
    summary = bt.run(oos_df, pred_col="pred_score", target_col="target_excess_return_5d")

    print("\n================ COST-ADJUSTED BACKTEST RESULTS ================")
    print("Strategy:                  Top vs Bottom Quintile (Q5 - Q1) Spread")
    print(f"Model Evaluated:           {args.model.upper()}")
    print("Rebalance Frequency:       Weekly (every 5 trading days)")
    print(f"Transaction Cost:          {args.cost_bps} bps per trade")
    print(f"Total Rebalance Periods:   {summary['rebalance_periods']}")
    print(f"Mean Weekly Turnover:      {summary['mean_weekly_turnover']}%")
    print(f"Gross Annualized Return:   {summary['annualized_return_gross']}%")
    print(f"Net Annualized Return:     {summary['annualized_return_net']}%")
    print(f"Annualized Volatility:     {summary['annualized_volatility']}%")
    print(f"Gross Sharpe Ratio:        {summary['sharpe_ratio_gross']}")
    print(f"Net Sharpe Ratio:          {summary['sharpe_ratio_net']}")
    print(f"Maximum Drawdown:          {summary['max_drawdown']}%")
    print(f"Calmar Ratio:              {summary['calmar_ratio']}")
    print(f"Win Rate (% positive):     {summary['win_rate_pct']}%")
    print(f"Total Costs Deducted:      {summary['total_transaction_cost_pct']}%")
    print("================================================================\n")


def cmd_run_ablation(args):
    """Compares performance of Price-Only vs Price+Macro vs Full."""
    print("Running Feature Ablation Study...")
    builder = DatasetBuilder()
    df, all_features = builder.prepare_dataset(include_sentiment=True)

    feature_subsets = {
        "Price-Only": builder.TICKER_FEATURES,
        "Price + Macro": builder.TICKER_FEATURES + builder.MACRO_FEATURES + builder.REGIME_FEATURES,
        "Full (+ Sentiment)": all_features,
    }

    validator = WalkForwardValidator(min_train_days=args.min_train_days, test_step_days=60, purge_gap_days=5)
    records = []

    for name, f_cols in feature_subsets.items():
        print(f"\nEvaluating subset: {name} ({len(f_cols)} features)...")
        res = validator.evaluate(df, feature_cols=f_cols, model_type="ridge")
        ic = res["ic_summary"]
        qs = res["quintile_summary"]
        records.append({
            "Feature Set": name,
            "Num Features": len(f_cols),
            "Mean IC": ic["mean_ic"],
            "IC t-stat": ic["t_stat"],
            "% Pos IC": ic["pct_positive"],
            "Quintile Sharpe": qs["sharpe_ratio"],
        })

    summary_df = pd.DataFrame(records)
    print("\n================ FEATURE ABLATION COMPARISON ================")
    print(summary_df.to_string(index=False))
    print("=============================================================\n")


def cmd_report(args):
    """Displays or regenerates the weekly portfolio rebalance report."""
    settings = get_settings()
    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.date:
        report_file = reports_dir / f"weekly_report_{args.date}.md"
    else:
        existing = sorted(reports_dir.glob("weekly_report_*.md"))
        report_file = existing[-1] if existing else None

    if report_file and report_file.exists():
        print(f"\n================ WEEKLY PORTFOLIO REPORT ({report_file.name}) ================\n")
        with open(report_file, encoding="utf-8") as f:
            print(f.read())
        print("=================================================================================\n")
    else:
        print("\nNo weekly report found. Generating report from latest daily scores...")
        scores_dir = settings.data_dir / "scores"
        score_files = sorted(scores_dir.glob("daily_scores_*.parquet"))
        if not score_files:
            print("Error: No daily scores found in data/scores. Run 'main.py run-daily' first.")
            return

        latest_file = score_files[-1]
        scores_df = pd.read_parquet(latest_file)
        as_of_date = str(scores_df["date"].iloc[0])

        allocator = PortfolioAllocator()
        ledger = PredictionLedger()
        target_df, meta = allocator.compute_target_weights(
            scores_df,
            vol_df=scores_df[["ticker", "predicted_vol"]] if "predicted_vol" in scores_df.columns else None,
            return_metadata=True,
        )
        orders, trade_summary = allocator.generate_trade_list(target_df)
        ic_summary = ledger.compute_rolling_ic_history()

        report_gen = WeeklyReportGenerator()
        content = report_gen.generate_report(
            as_of_date=as_of_date,
            target_df=target_df,
            portfolio_meta=meta,
            orders=orders,
            trade_summary=trade_summary,
            ic_summary=ic_summary,
        )
        print(f"\n================ WEEKLY PORTFOLIO REPORT ({as_of_date}) ================\n")
        print(content)
        print("=========================================================================\n")


def cmd_update_ledger(args):
    """Updates prediction ledger with realized returns and recalculates rolling IC."""
    ledger = PredictionLedger()

    if args.seed:
        print("Seeding prediction ledger from out-of-sample walk-forward predictions...")
        builder = DatasetBuilder()
        df, feature_cols = builder.prepare_dataset(include_sentiment=False)
        validator = WalkForwardValidator(min_train_days=500, test_step_days=60, purge_gap_days=5)
        res = validator.evaluate(df, feature_cols=feature_cols, model_type="ridge")
        oos_df = res["oos_predictions"]
        ledger.seed_historical_ledger(oos_df)
        print("\nPrediction Ledger seeded successfully!")
    else:
        print("Checking for mature predictions (5 trading days old) to fill realized returns...")
        ledger.update_mature_predictions()

    ic_info = ledger.compute_rolling_ic_history()
    print("\n================ PREDICTION LEDGER & ROLLING IC STATUS ================")
    print(f"Total Evaluated Dates:   {ic_info.get('total_evaluated_dates', 0)}")
    print(f"Latest Evaluated Date:   {ic_info.get('latest_evaluated_date', 'None')}")
    print(f"Latest Daily IC:         {ic_info.get('latest_daily_ic', 'None')}")
    print(f"Rolling 60-Day Mean IC:  {ic_info.get('latest_rolling_ic', 'None')}")
    print(f"Rolling 60-Day t-stat:   {ic_info.get('rolling_t_stat', 'None')}")
    print(f"Rolling % Positive Days: {ic_info.get('rolling_pct_positive', 'None')}%")
    print("=======================================================================\n")


def cmd_setup_automation(args):
    """Displays setup instructions and verified configuration for automated scheduling."""
    settings = get_settings()
    script_path = settings.project_root / "scripts" / "cron_daily.sh"
    workflow_path = settings.project_root / ".github" / "workflows" / "daily_pipeline.yml"

    print("\n================ PRODUCTION AUTOMATION SETUP ================")
    print("1. Cron Schedule (Linux / macOS):")
    print("   Run 'crontab -e' and add the following entry:")
    print(f"   15 16 * * 1-5 {script_path} >> {settings.data_dir}/logs/cron_daily.log 2>&1\n")
    print("   Schedule: Monday-Friday at 16:15 ET (after US market close at 16:00 ET).")
    print(f"   Runner Script: {script_path} ({'Executable' if script_path.exists() and os.access(script_path, os.X_OK) else 'Created'})\n")

    print("2. GitHub Actions Scheduled Workflow:")
    print(f"   Workflow File: {workflow_path} ({'Ready' if workflow_path.exists() else 'Missing'})")
    print("   Cron: '15 21 * * 1-5' UTC (Monday through Friday)")
    print("   Repository Secrets Required:")
    print("   - FINNHUB_API_KEY (for live headlines)")
    print("   - FRED_API_KEY (for macro indicators)\n")
    print("3. Manual Execution Command:")
    print(f"   bash {script_path}")
    print("=============================================================\n")


def cmd_dashboard(args):
    """Launches the Unified Argus Quant Trading Terminal (HTML5/Tailwind/FastAPI)."""
    import subprocess
    settings = get_settings()
    app_path = settings.project_root / "src" / "dashboard" / "app.py"
    port = getattr(args, "port", 8501)
    host = getattr(args, "host", "0.0.0.0")

    cmd = [
        sys.executable,
        str(app_path),
        "--host",
        str(host),
        "--port",
        str(port),
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTerminal stopped.")


def cmd_investor_dashboard(args):
    """Launches the Unified Argus Quant Trading Terminal (Alias)."""
    return cmd_dashboard(args)


def cmd_export_snapshot(args):
    """Generates and exports the dashboard precomputed snapshot."""
    from src.dashboard.snapshot import DashboardSnapshotBuilder
    print("Building precomputed operations dashboard snapshot...")
    builder = DashboardSnapshotBuilder()
    snap = builder.generate_and_save_snapshot()
    header = snap.get("meta", {}).get("status_strip", {})
    cockpit = snap.get("today_cockpit", {}).get("summary", {})
    print("\n================ DASHBOARD SNAPSHOT EXPORTED ================")
    print(f"As-of Date:       {snap.get('as_of_date')}")
    print(f"Confidence Light: {header.get('confidence_light', 'N/A').upper()} ({header.get('confidence_status', 'N/A')})")
    print(f"Rolling 60d IC:   {header.get('rolling_60d_ic')} (t-stat: {header.get('rolling_t_stat')})")
    print(f"Regime:           {header.get('regime_name')} | VIX: {header.get('vix')}")
    print(f"Total Orders:     {cockpit.get('total_orders')} (Buys: {cockpit.get('buys_count')}, Sells: {cockpit.get('sells_count')})")
    print(f"1-Way Turnover:   {cockpit.get('one_way_turnover', 0.0) * 100:.1f}%")
    print(f"Snapshot File:    {builder.snapshot_file}")
    print("=============================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Argus Quant Data Pipeline, Features, Models & Backtesting CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run-daily
    sub_daily = subparsers.add_parser("run-daily", help="Run the daily end-of-day data sync")
    sub_daily.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: all)")
    sub_daily.add_argument("--lookback-years", type=int, default=5, help="Years of history for initial pulls (default: 5)")
    sub_daily.add_argument("--news-days-back", type=int, default=2, help="Days of news to fetch (default: 2)")
    sub_daily.add_argument("--force", action="store_true", help="Force refresh existing caches")
    sub_daily.add_argument("--rebalance", action="store_true", help="Execute weekly portfolio rebalance & output trade list")
    sub_daily.add_argument("--auto-rebalance", action="store_true", help="Automatically rebalance on weekly schedule (Friday close)")
    sub_daily.add_argument("--sizing-method", type=str, default="inv_vol", choices=["inv_vol", "equal"], help="Position sizing method")
    sub_daily.add_argument("--commit-holdings", action="store_true", help="Persist new target weights to current_holdings.json")
    sub_daily.set_defaults(func=cmd_run_daily)

    # status
    sub_status = subparsers.add_parser("status", help="Print pipeline cache status and health summary")
    sub_status.set_defaults(func=cmd_status)

    # backfill-prices
    sub_prices = subparsers.add_parser("backfill-prices", help="Initialize or update price cache")
    sub_prices.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    sub_prices.add_argument("--years", type=int, default=10, help="Lookback years (default: 10)")
    sub_prices.add_argument("--force", action="store_true", help="Force refetch")
    sub_prices.set_defaults(func=cmd_backfill_prices)

    # backfill-macro
    sub_macro = subparsers.add_parser("backfill-macro", help="Initialize or update FRED macro cache")
    sub_macro.add_argument("--force", action="store_true", help="Force refetch")
    sub_macro.set_defaults(func=cmd_backfill_macro)

    # backfill-news
    sub_news = subparsers.add_parser("backfill-news", help="Backfill historical company headlines from Finnhub")
    sub_news.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: all S&P 100)")
    sub_news.add_argument("--days-back", type=int, default=365, help="Total days of history to backfill (default: 365)")
    sub_news.add_argument("--windows", type=int, default=2, help="Number of time windows to split query into (default: 2)")
    sub_news.add_argument("--score", action="store_true", help="Automatically run FinBERT scoring after backfilling")
    sub_news.add_argument("--batch-size", type=int, default=32, help="Inference batch size if scoring")
    sub_news.set_defaults(func=cmd_backfill_news)

    # sample-test
    sub_sample = subparsers.add_parser("sample-test", help="Run quick test on a 4-ticker subset")
    sub_sample.set_defaults(func=cmd_sample_test)

    # score-headlines
    sub_score = subparsers.add_parser("score-headlines", help="Run FinBERT sentiment scoring on archived headlines")
    sub_score.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    sub_score.set_defaults(func=cmd_score_headlines)

    # train-baseline
    sub_train = subparsers.add_parser("train-baseline", help="Train HAR vol model and walk-forward Ridge return ranker")
    sub_train.add_argument("--min-train-days", type=int, default=500, help="Minimum days for initial train window (default: 500)")
    sub_train.add_argument("--with-sentiment", action="store_true", help="Include sentiment features")
    sub_train.add_argument("--run-sanity-check", action="store_true", help="Run label shuffle sanity check")
    sub_train.set_defaults(func=cmd_train_baseline)

    # train-lightgbm
    sub_lgbm = subparsers.add_parser("train-lightgbm", help="Train and compare Ridge, LightGBM, and 50/50 Blend")
    sub_lgbm.add_argument("--min-train-days", type=int, default=500, help="Minimum days for initial train window (default: 500)")
    sub_lgbm.add_argument("--with-sentiment", action="store_true", help="Include sentiment features")
    sub_lgbm.set_defaults(func=cmd_train_lightgbm)

    # run-sanity-checks
    sub_sanity = subparsers.add_parser("run-sanity-checks", help="Run label shuffle and 1-day feature lag tests")
    sub_sanity.add_argument("--min-train-days", type=int, default=500, help="Minimum days for initial train window (default: 500)")
    sub_sanity.set_defaults(func=cmd_run_sanity_checks)

    # backtest
    sub_bt = subparsers.add_parser("backtest", help="Run weekly cost-adjusted quintile backtest")
    sub_bt.add_argument("--model", type=str, default="blend", choices=["ridge", "lgbm", "blend"], help="Model to backtest (default: blend)")
    sub_bt.add_argument("--cost-bps", type=float, default=7.5, help="Transaction cost in basis points per trade (default: 7.5)")
    sub_bt.add_argument("--min-train-days", type=int, default=500, help="Minimum days for initial train window (default: 500)")
    sub_bt.add_argument("--with-sentiment", action="store_true", help="Include sentiment features")
    sub_bt.set_defaults(func=cmd_backtest)

    # run-ablation
    sub_ablation = subparsers.add_parser("run-ablation", help="Compare Price-Only vs Price+Macro vs Full model")
    sub_ablation.add_argument("--min-train-days", type=int, default=500, help="Minimum days for initial train window (default: 500)")
    sub_ablation.set_defaults(func=cmd_run_ablation)

    # report
    sub_rep = subparsers.add_parser("report", help="Display or regenerate weekly portfolio rebalance report")
    sub_rep.add_argument("--date", type=str, default=None, help="As-of date (YYYY-MM-DD) for the report (default: latest)")
    sub_rep.set_defaults(func=cmd_report)

    # update-ledger
    sub_led = subparsers.add_parser("update-ledger", help="Fill in mature realized returns and compute rolling live IC")
    sub_led.add_argument("--seed", action="store_true", help="Seed ledger using historical walk-forward predictions")
    sub_led.set_defaults(func=cmd_update_ledger)

    # setup-automation
    sub_auto = subparsers.add_parser("setup-automation", help="Display setup guide for cron and GitHub Actions automation")
    sub_auto.set_defaults(func=cmd_setup_automation)

    # dashboard
    sub_dash = subparsers.add_parser("dashboard", help="Launch the Unified Argus Quant Trading Terminal (XTB Design)")
    sub_dash.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind terminal server (default: 0.0.0.0)")
    sub_dash.add_argument("--port", type=int, default=8501, help="Port to bind terminal server (default: 8501)")
    sub_dash.add_argument("--headless", action="store_true", help="Deprecated legacy flag (retained for backward compatibility)")
    sub_dash.set_defaults(func=cmd_dashboard)

    # investor-dashboard (alias)
    sub_inv = subparsers.add_parser("investor-dashboard", help="Launch the Unified Argus Quant Trading Terminal (Alias)")
    sub_inv.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind terminal server (default: 0.0.0.0)")
    sub_inv.add_argument("--port", type=int, default=8501, help="Port to bind terminal server (default: 8501)")
    sub_inv.add_argument("--headless", action="store_true", help="Deprecated legacy flag (retained for backward compatibility)")
    sub_inv.set_defaults(func=cmd_investor_dashboard)

    # export-snapshot
    sub_snap = subparsers.add_parser("export-snapshot", help="Generate pre-computed dashboard snapshot JSON")
    sub_snap.set_defaults(func=cmd_export_snapshot)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
