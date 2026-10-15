"""Comprehensive unit tests for Operations Dashboard snapshot, database, and alerts."""
import time

import pandas as pd

from src.dashboard.alerts import AlertDispatcher
from src.dashboard.database import DashboardDB
from src.dashboard.snapshot import DashboardSnapshotBuilder, load_today_snapshot


def test_dashboard_db_fills_and_positions(tmp_path):
    """Verifies SQLite ledger for trade fills and position aggregation."""
    db_file = tmp_path / "test_ledger.db"
    db = DashboardDB(db_path=db_file)

    # 1. Record BUY fill
    db.record_fill(
        fill_date="2026-03-27",
        ticker="AAPL",
        action="BUY",
        shares=100.0,
        price=180.50,
        commission=1.50,
        notes="Target rebalance entry",
    )

    # 2. Record another BUY fill
    db.record_fill(
        fill_date="2026-03-27",
        ticker="MSFT",
        action="BUY",
        shares=50.0,
        price=400.00,
        commission=1.50,
        notes="Target rebalance entry",
    )

    # 3. Partial SELL
    db.record_fill(
        fill_date="2026-03-27",
        ticker="AAPL",
        action="SELL",
        shares=40.0,
        price=185.00,
        commission=1.00,
        notes="Rebalance trim",
    )

    fills = db.get_fills()
    assert len(fills) == 3

    # Check positions aggregation
    positions = db.get_open_positions_from_fills()
    assert "AAPL" in positions
    assert "MSFT" in positions

    # AAPL: 100 bought - 40 sold = 60 shares
    assert positions["AAPL"]["net_shares"] == 60.0
    assert positions["MSFT"]["net_shares"] == 50.0
    assert positions["AAPL"]["total_bought"] == 100.0
    assert positions["AAPL"]["total_sold"] == 40.0

    # Test SQLite analytical query
    df = db.query_df("SELECT action, sum(shares) as total_shares FROM fills GROUP BY action ORDER BY action")
    assert not df.empty
    assert "total_shares" in df.columns


def test_alert_dispatcher_logging(tmp_path):
    """Verifies alert dispatching records audit trail into SQLite."""
    db_file = tmp_path / "alerts_ledger.db"
    db = DashboardDB(db_path=db_file)
    alerts = AlertDispatcher(db=db)

    # Test confidence light change alert
    alerts.notify_confidence_light_change(
        old_color="green",
        new_color="amber",
        ic=0.015,
        t_stat=0.85,
    )

    # Test pipeline failure alert
    alerts.notify_pipeline_failure("Price Pipeline", "404 Not Found for mock ticker")

    # Test weekly report alert
    alerts.notify_weekly_report_ready("2026-03-27", num_orders=4, turnover_pct=14.5)

    logged = db.get_alerts()
    assert len(logged) == 3
    assert any("Confidence Light Changed" in t for t in logged["title"])
    assert any("Daily Pipeline Step Failed" in t for t in logged["title"])
    assert any("Weekly Rebalance Report Ready" in t for t in logged["title"])


def test_dashboard_snapshot_generation(tmp_path, monkeypatch):
    """Verifies end-to-end dashboard snapshot compilation and serialization."""
    # Setup isolated test directory structure
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True)
    dashboard_dir = tmp_path / "dashboard"
    dashboard_dir.mkdir(parents=True)

    # Create mock daily scores parquet
    today = "2026-03-27"
    yesterday = "2026-03-26"

    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    scores_df_today = pd.DataFrame({
        "date": [today] * 5,
        "ticker": tickers,
        "rank": [1, 2, 3, 4, 5],
        "pred_score": [0.08, 0.06, 0.04, 0.02, 0.01],
        "predicted_vol": [0.015, 0.018, 0.025, 0.020, 0.019],
        "shap_top_1": ["momentum_20d (+0.035)", "momentum_20d (+0.025)", "volatility_60d (-0.010)", "news_sentiment (+0.015)", "pe_ratio (-0.005)"],
        "shap_top_2": ["news_sentiment (+0.020)", "rsi_14d (+0.012)", "momentum_5d (+0.015)", "rsi_14d (+0.008)", "momentum_20d (+0.005)"],
        "shap_top_3": ["macro_term_spread (+0.010)", "har_vol (-0.005)", "rsi_14d (+0.005)", "har_vol (-0.004)", "har_vol (-0.002)"],
    })
    scores_df_today.to_parquet(scores_dir / f"daily_scores_{today}.parquet", index=False)

    scores_df_prev = pd.DataFrame({
        "date": [yesterday] * 5,
        "ticker": tickers,
        "rank": [3, 1, 2, 5, 4],
        "pred_score": [0.05, 0.07, 0.06, 0.01, 0.02],
    })
    scores_df_prev.to_parquet(scores_dir / f"daily_scores_{yesterday}.parquet", index=False)

    # Monkeypatch settings
    from src.config import settings as settings_module
    s = settings_module.get_settings()
    monkeypatch.setattr(s, "data_dir", tmp_path)

    builder = DashboardSnapshotBuilder()
    snapshot = builder.generate_and_save_snapshot()

    # Validate high-level schema
    assert "as_of_date" in snapshot
    assert snapshot["as_of_date"] == today
    assert "meta" in snapshot
    assert snapshot["meta"]["is_paper"] is True

    # Validate status strip
    strip = snapshot["meta"]["status_strip"]
    assert strip["confidence_light"] in ("green", "amber", "red")
    assert "rolling_60d_ic" in strip
    assert "regime_name" in strip
    assert "vix" in strip
    assert "exposure_scale" in strip

    # Validate today cockpit
    cockpit = snapshot["today_cockpit"]
    assert "actions" in cockpit
    assert len(cockpit["actions"]) == 5

    # Check rank change calculation for AAPL (was 3, now 1 -> change = +2)
    aapl_action = next(a for a in cockpit["actions"] if a["ticker"] == "AAPL")
    assert aapl_action["rank"] == 1
    assert aapl_action["rank_change_1d"] == 2
    assert aapl_action["action"] in ("BUY", "HOLD", "SELL")

    # Validate persistence file and load latency
    assert builder.snapshot_file.exists()

    t0 = time.perf_counter()
    loaded = load_today_snapshot()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert loaded["as_of_date"] == today
    assert latency_ms < 50.0  # Ultra-fast <50ms precomputed load requirement

    # Validate pipeline status step 4 schema
    ps_steps = snapshot["pipeline_status"]["steps"]
    step_4 = next(s for s in ps_steps if s["step"] == 4)
    assert "records" in step_4
    assert isinstance(step_4["records"], int)

    # Validate regime_probs dictionary
    assert "low_vol" in strip["regime_probs"]

    # Validate deterministic equity curve generation
    snapshot_2 = builder.generate_and_save_snapshot()
    eq1 = [p["strategy"] for p in snapshot["live_record"]["equity_curve"]]
    eq2 = [p["strategy"] for p in snapshot_2["live_record"]["equity_curve"]]
    assert eq1 == eq2, "Equity curve points must be deterministic across runs"


def test_dashboard_empty_scores_fallback_generation(tmp_path, monkeypatch):
    """Verifies that when scores parquets are absent, snapshot builds valid baseline cockpit actions."""
    from src.config import settings as settings_module

    s = settings_module.get_settings()
    monkeypatch.setattr(s, "data_dir", tmp_path)

    builder = DashboardSnapshotBuilder()
    snapshot = builder.generate_and_save_snapshot()

    assert "today_cockpit" in snapshot
    assert "actions" in snapshot["today_cockpit"]
    assert len(snapshot["today_cockpit"]["actions"]) > 0

    first_act = snapshot["today_cockpit"]["actions"][0]
    assert "ticker" in first_act
    assert "action" in first_act
    assert "target_weight" in first_act
    assert "predicted_vol" in first_act
    assert first_act["action"] == "HOLD"
    assert first_act["target_weight"] > 0

