"""Unit and Integration Tests for Barra Factor Model and Sector Neutralization."""
import numpy as np
import pandas as pd

from src.features.barra import GICS_SECTORS, STYLE_FACTORS, BarraFactorModel
from src.portfolio.allocator import PortfolioAllocator


def test_barra_exposure_matrix_construction():
    barra = BarraFactorModel()
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "UNH", "PG", "NEE"]
    panel_df = pd.DataFrame({
        "ticker": tickers,
        "return_20d": np.random.uniform(-0.05, 0.05, len(tickers)),
        "rolling_vol_20d": np.random.uniform(0.15, 0.35, len(tickers)),
        "market_cap_log": np.random.uniform(24.0, 28.0, len(tickers)),
    })

    X, factor_names = barra.build_factor_exposure_matrix(panel_df)

    assert len(X) == len(tickers)
    assert set(X.index) == set(tickers)

    # 1. Verify sector dummy sum
    sector_cols = [c for c in factor_names if c.startswith("sector_")]
    assert len(sector_cols) >= len(GICS_SECTORS)
    row_sums = X[sector_cols].sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)

    # 2. Verify standardized style factors
    for sf in STYLE_FACTORS:
        col = f"style_{sf}"
        assert col in factor_names
        vals = X[col].values
        assert abs(np.mean(vals)) < 1e-4
        assert np.all(vals >= -3.0) and np.all(vals <= 3.0)


def test_barra_alpha_neutralization_orthogonality():
    barra = BarraFactorModel(l2_reg=1e-5)
    tickers = [f"TICKER_{i}" for i in range(50)]

    # Distribute across 5 sectors
    sectors = ["Information Technology", "Health Care", "Financials", "Energy", "Consumer Discretionary"]
    stock_sectors = [sectors[i % len(sectors)] for i in range(50)]

    # Create synthetic style loadings
    np.random.seed(42)
    size_raw = np.linspace(-2.0, 2.0, 50)
    vol_raw = np.random.normal(0, 1, 50)
    mom_raw = np.random.normal(0, 1, 50)
    val_raw = np.random.normal(0, 1, 50)

    panel_df = pd.DataFrame({
        "ticker": tickers,
        "sector": stock_sectors,
        "market_cap_log": size_raw + 25.0,
        "rolling_vol_20d": vol_raw * 0.05 + 0.20,
        "return_20d": mom_raw * 0.03,
        "book_to_price": val_raw * 0.10 + 0.30,
        "earnings_yield": 0.05,
        "sales_to_price": 0.50,
    })

    X, _ = barra.build_factor_exposure_matrix(panel_df)

    # Raw alpha heavily biased towards Information Technology (+0.10) and Size (+0.05 * size)
    raw_alpha = pd.Series(
        [0.10 if s == "Information Technology" else 0.0 for s in stock_sectors],
        index=tickers,
    ) + 0.05 * X["style_size"] + np.random.normal(0, 0.02, 50)

    # Compute neutralized alpha
    neutral_alpha = barra.neutralize_alpha(raw_alpha, X, method="full_barra", preserve_scale=True)

    # Verify style factor correlations are eliminated (near zero)
    diag = barra.compute_factor_diagnostics(raw_alpha, neutral_alpha, X)
    assert diag["max_neutral_correlation"] < 0.05
    assert diag["r2_explained"] > 0.50

    # Verify sector bias is flattened
    neu_tech = diag["sector_biases_neutral"].get("information_technology", 0.0)
    neu_fin = diag["sector_biases_neutral"].get("financials", 0.0)
    assert abs(neu_tech - neu_fin) < 0.02


def test_allocator_sector_constraints():
    allocator = PortfolioAllocator(
        target_holdings=15,
        max_weight=0.08,
        max_sector_weight=0.25,
        max_names_per_sector=3,
        neutralization_mode="full_barra",
    )

    # Universe of 55 stocks across 5 real GICS sectors
    # Stocks 0-14 in Information Technology with HIGHEST raw scores
    tickers = []
    sectors = []
    scores = []

    for i in range(15):
        tickers.append(f"TECH_{i}")
        sectors.append("Information Technology")
        scores.append(1.0 - i * 0.01)  # 1.00 down to 0.86

    for i in range(10):
        tickers.append(f"HLTH_{i}")
        sectors.append("Health Care")
        scores.append(0.70 - i * 0.01)

    for i in range(10):
        tickers.append(f"FIN_{i}")
        sectors.append("Financials")
        scores.append(0.50 - i * 0.01)

    for i in range(10):
        tickers.append(f"ENGY_{i}")
        sectors.append("Energy")
        scores.append(0.30 - i * 0.01)

    for i in range(10):
        tickers.append(f"IND_{i}")
        sectors.append("Industrials")
        scores.append(0.20 - i * 0.01)

    scores_df = pd.DataFrame({
        "ticker": tickers,
        "sector": sectors,
        "pred_score": scores,
        "predicted_vol": [0.20] * len(tickers),
        "return_20d": [0.02] * len(tickers),
        "rolling_vol_20d": [0.20] * len(tickers),
        "market_cap_log": [25.0] * len(tickers),
    })

    target_df, meta = allocator.compute_target_weights(
        scores_df,
        top_n=15,
        use_turnover_buffer=False,
        use_exposure_scaling=False,
        use_confidence_filter=False,
        return_metadata=True,
    )

    assert len(target_df) == 15

    # 1. Max names per sector: Information Technology must have at most 3 names despite having 15 highest scores
    tech_picks = target_df[target_df["sector"] == "Information Technology"]
    assert len(tech_picks) <= 3, f"Expected <= 3 tech picks, got {len(tech_picks)}"

    # 2. Strict 8% single-stock cap
    for w in target_df["target_weight"]:
        assert w <= 0.0801, f"Stock weight {w} exceeded 8% cap"

    # 3. Strict 25% single-sector cap across all sectors
    for sec_info in meta["sector_breakdown"]:
        sec_w = sec_info["weight"]
        assert sec_w <= 0.2501, f"Sector {sec_info['sector']} weight {sec_w} exceeded 25% cap"

    # 4. Total weights sum to 1.0 (within rounding tolerance)
    assert abs(target_df["target_weight"].sum() - 1.0) < 0.01
