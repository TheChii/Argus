"""Portfolio allocation and trade list generation: rules-based portfolio construction.

Guarantees & Rules:
1. Target names: ~15 names (target_holdings=15).
2. Position cap: Capped at ~8% each (max_weight=0.08) with iterative redistribution of excess weight.
3. Turnover buffer (hysteresis):
   - Buy only when entering top 10% (percentile >= 0.90).
   - Sell only when falling below top 40% (percentile < 0.60).
   - Retain existing holdings in buffer zone [0.60, 0.90) to minimize unnecessary turnover.
4. Exposure scaling: Cut equity exposure when VIX or predicted vol spikes (S = min(1.0, vix_base / vix)).
5. Confidence filter: If live rolling 60-day Spearman IC <= 0, hold 100% Cash or SPY.
6. Execution order: SELL orders generated first to release capital, then BUY orders.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config.settings import get_settings
from src.config.universe import get_sector_for_ticker
from src.features.barra import GICS_SECTORS, BarraFactorModel
from src.utils.logger import setup_logger

logger = setup_logger("portfolio_allocator")


class PortfolioAllocator:
    """Manages rules-based portfolio construction, position sizing, and trade order generation."""

    def __init__(
        self,
        portfolio_dir: Path | None = None,
        target_holdings: int = 15,
        max_weight: float = 0.08,
        buy_percentile: float = 0.90,
        sell_percentile: float = 0.60,
        vix_baseline: float = 20.0,
        rolling_ic_threshold: float = 0.0,
        confidence_fallback: str = "cash",
        sector_neutral: bool = True,
        max_sector_weight: float = 0.25,
        max_names_per_sector: int = 3,
        neutralization_mode: str = "full_barra",
    ):
        settings = get_settings()
        self.portfolio_dir = portfolio_dir or (settings.data_dir / "portfolio")
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)
        self.holdings_file = self.portfolio_dir / "current_holdings.json"

        self.target_holdings = target_holdings
        self.max_weight = max_weight
        self.buy_percentile = buy_percentile
        self.sell_percentile = sell_percentile
        self.vix_baseline = vix_baseline
        self.rolling_ic_threshold = rolling_ic_threshold
        self.confidence_fallback = confidence_fallback.lower()
        self.sector_neutral = sector_neutral
        self.max_sector_weight = max_sector_weight
        self.max_names_per_sector = max_names_per_sector
        self.neutralization_mode = neutralization_mode
        self.barra_model = BarraFactorModel()

    def compute_target_weights(
        self,
        scores_df: pd.DataFrame,
        vol_df: pd.DataFrame | None = None,
        current_holdings: dict[str, float] | None = None,
        current_vix: float | None = None,
        rolling_60d_ic: float | None = None,
        method: str = "inv_vol",
        top_n: int | None = None,
        use_turnover_buffer: bool = True,
        use_exposure_scaling: bool = True,
        use_confidence_filter: bool = True,
        score_col: str = "pred_score",
        ticker_col: str = "ticker",
        return_metadata: bool = False,
        neutralization_mode: str | None = None,
        max_sector_weight: float | None = None,
        max_names_per_sector: int | None = None,
        factor_matrix: pd.DataFrame | None = None,
    ):
        """
        Computes target portfolio weights adhering to all quantitative portfolio rules.

        Returns:
            target_df if return_metadata=False (with target_df.attrs['metadata']),
            or (target_df, metadata_dict) if return_metadata=True.
        """
        n_target = top_n if top_n is not None else self.target_holdings
        neut_mode = neutralization_mode if neutralization_mode is not None else self.neutralization_mode
        eff_max_sector_weight = max_sector_weight if max_sector_weight is not None else self.max_sector_weight
        eff_max_names_per_sector = max_names_per_sector if max_names_per_sector is not None else self.max_names_per_sector

        metadata: dict[str, Any] = {
            "confidence_filter_triggered": False,
            "exposure_scale": 1.0,
            "cash_weight": 0.0,
            "vix": current_vix,
            "rolling_60d_ic": rolling_60d_ic,
            "retained_count": 0,
            "new_buys_count": 0,
            "dropped_count": 0,
            "retained_tickers": [],
            "new_buy_tickers": [],
            "dropped_tickers": [],
            "sector_breakdown": [],
            "max_sector_weight": eff_max_sector_weight,
            "max_names_per_sector": eff_max_names_per_sector,
            "neutralization_mode": neut_mode,
        }

        if current_holdings is None:
            current_holdings = self.load_holdings()

        # Clean non-equity cash keys from current holdings
        held_equity_keys = {t: w for t, w in current_holdings.items() if t not in ("CASH", "SPY") and w > 0.001}

        # ---------------------------------------------------------
        # Rule 1: Confidence Filter
        # ---------------------------------------------------------
        if use_confidence_filter and rolling_60d_ic is not None:
            if rolling_60d_ic <= self.rolling_ic_threshold:
                logger.warning(
                    f"Confidence filter TRIGGERED: live rolling 60d IC={rolling_60d_ic:.4f} "
                    f"<= {self.rolling_ic_threshold}. Allocating 100% to {self.confidence_fallback.upper()}."
                )
                metadata["confidence_filter_triggered"] = True
                metadata["exposure_scale"] = 0.0
                metadata["dropped_tickers"] = list(held_equity_keys.keys())
                metadata["dropped_count"] = len(held_equity_keys)

                if self.confidence_fallback == "spy":
                    target_df = pd.DataFrame([{
                        ticker_col: "SPY",
                        "sector": "Broad Market ETF",
                        "rank": 1,
                        score_col: 0.0,
                        "target_weight": 1.0,
                        "predicted_vol": 0.16,
                        "selection_reason": "confidence_fallback_spy",
                    }])
                    metadata["cash_weight"] = 0.0
                else:
                    target_df = pd.DataFrame([{
                        ticker_col: "CASH",
                        "sector": "Cash & Short-Term Bills",
                        "rank": 1,
                        score_col: 0.0,
                        "target_weight": 1.0,
                        "predicted_vol": 0.0,
                        "selection_reason": "confidence_fallback_cash",
                    }])
                    metadata["cash_weight"] = 1.0

                target_df.attrs["metadata"] = metadata
                return (target_df, metadata) if return_metadata else target_df

        clean = scores_df.dropna(subset=[score_col]).copy()
        if len(clean) == 0:
            empty_df = pd.DataFrame(columns=[ticker_col, "sector", "rank", score_col, "target_weight", "predicted_vol", "selection_reason"])
            metadata["cash_weight"] = 1.0
            empty_df.attrs["metadata"] = metadata
            return (empty_df, metadata) if return_metadata else empty_df

        # Resolve sector mappings
        ticker_sectors: dict[str, str] = {}
        for _, row in clean.iterrows():
            t = str(row[ticker_col])
            sec = None
            if "sector" in row and pd.notna(row["sector"]):
                sec = str(row["sector"])
            if not sec or sec == "None":
                sec = get_sector_for_ticker(t)
            ticker_sectors[t] = sec or "Other"

        has_real_sectors = any(s in GICS_SECTORS for s in ticker_sectors.values())

        # ---------------------------------------------------------
        # Barra Factor Neutralization (if applicable)
        # ---------------------------------------------------------
        if neut_mode != "none" and (has_real_sectors or factor_matrix is not None):
            try:
                if factor_matrix is None:
                    factor_matrix, _ = self.barra_model.build_factor_exposure_matrix(clean)

                raw_s = clean.set_index(ticker_col)[score_col]
                neutral_s = self.barra_model.neutralize_alpha(
                    raw_s,
                    factor_matrix,
                    method=neut_mode,
                    preserve_scale=True,
                )
                clean["raw_score"] = clean[score_col]
                clean[score_col] = clean[ticker_col].map(neutral_s).fillna(clean[score_col])

                diagnostics = self.barra_model.compute_factor_diagnostics(
                    raw_s,
                    neutral_s,
                    factor_matrix,
                )
                metadata["barra_diagnostics"] = diagnostics
            except Exception as e:
                logger.warning(f"Barra factor neutralization skipped: {e}")

        # Sort descending by score
        clean = clean.sort_values(score_col, ascending=False).reset_index(drop=True)
        n_total = len(clean)

        # Cross-sectional percentile rank in [0, 1] where 1.0 is highest expected return
        ranks = rankdata(clean[score_col].values, method="average")
        clean["percentile"] = (ranks - 1.0) / max(n_total - 1.0, 1.0)
        clean["rank"] = np.arange(1, n_total + 1)

        # Merge predicted vol
        if vol_df is not None and not vol_df.empty:
            vol_map = dict(zip(vol_df[ticker_col], vol_df["predicted_vol"]))
            clean["predicted_vol"] = clean[ticker_col].map(vol_map).fillna(0.20)
        elif "predicted_vol" in clean.columns:
            clean["predicted_vol"] = clean["predicted_vol"].fillna(0.20)
        else:
            clean["predicted_vol"] = 0.20

        # ---------------------------------------------------------
        # Rule 2: Turnover Buffer (Hysteresis) & Sector Quotas
        # ---------------------------------------------------------
        selected_tickers: list[str] = []
        ticker_meta: dict[str, str] = {}
        sector_counts: dict[str, int] = dict.fromkeys(set(ticker_sectors.values()), 0)

        def can_add_sector(t: str) -> bool:
            if not has_real_sectors or eff_max_names_per_sector is None:
                return True
            s = ticker_sectors.get(t, "Other")
            if s not in GICS_SECTORS:
                return True
            return sector_counts.get(s, 0) < eff_max_names_per_sector

        def record_pick(t: str, reason: str):
            selected_tickers.append(t)
            ticker_meta[t] = reason
            s = ticker_sectors.get(t, "Other")
            sector_counts[s] = sector_counts.get(s, 0) + 1

        if use_turnover_buffer and held_equity_keys:
            # Check currently held names against sell_percentile (top 40%)
            scores_by_ticker = clean.set_index(ticker_col)
            retained = []
            dropped = []

            for t in held_equity_keys:
                if t in scores_by_ticker.index:
                    pct = float(scores_by_ticker.loc[t, "percentile"])
                    if pct >= self.sell_percentile:
                        retained.append((t, float(scores_by_ticker.loc[t, score_col])))
                    else:
                        dropped.append(t)
                else:
                    dropped.append(t)

            metadata["dropped_tickers"] = dropped
            metadata["dropped_count"] = len(dropped)

            # Sort retained by score descending
            retained.sort(key=lambda x: x[1], reverse=True)
            retained_candidates = [x[0] for x in retained]
            retained_final = []

            for t in retained_candidates:
                if len(selected_tickers) < n_target and can_add_sector(t):
                    record_pick(t, "retained_buffer")
                    retained_final.append(t)

            metadata["retained_tickers"] = retained_final
            metadata["retained_count"] = len(retained_final)

            # New entrants must be in top 10% (percentile >= buy_percentile)
            slots_needed = n_target - len(selected_tickers)
            new_buys_final = []
            if slots_needed > 0:
                buy_candidates = clean[
                    (~clean[ticker_col].isin(set(held_equity_keys.keys()))) &
                    (clean["percentile"] >= self.buy_percentile)
                ]
                for _, row in buy_candidates.iterrows():
                    nb = str(row[ticker_col])
                    if len(selected_tickers) < n_target and can_add_sector(nb):
                        record_pick(nb, "new_buy_top_10pct")
                        new_buys_final.append(nb)

            metadata["new_buy_tickers"] = new_buys_final
            metadata["new_buys_count"] = len(new_buys_final)

            # If still short, backfill: pass 1 respecting sector quota
            if len(selected_tickers) < n_target:
                for _, row in clean.iterrows():
                    bf = str(row[ticker_col])
                    if bf not in selected_tickers and can_add_sector(bf):
                        record_pick(bf, "backfill_top_ranked")
                        if len(selected_tickers) >= n_target:
                            break

            # If still short, backfill: pass 2 relaxing sector quota
            if len(selected_tickers) < n_target:
                for _, row in clean.iterrows():
                    bf = str(row[ticker_col])
                    if bf not in selected_tickers:
                        record_pick(bf, "backfill_sector_relaxed")
                        if len(selected_tickers) >= n_target:
                            break
        else:
            # Day 1 / Cold Start: Pass 1 respecting sector quota
            new_buys_final = []
            for _, row in clean.iterrows():
                t = str(row[ticker_col])
                if can_add_sector(t):
                    record_pick(t, "top_ranked_entry")
                    new_buys_final.append(t)
                    if len(selected_tickers) >= n_target:
                        break

            # Pass 2 relaxing sector quota if still short
            if len(selected_tickers) < n_target:
                for _, row in clean.iterrows():
                    t = str(row[ticker_col])
                    if t not in selected_tickers:
                        record_pick(t, "top_ranked_entry_relaxed")
                        new_buys_final.append(t)
                        if len(selected_tickers) >= n_target:
                            break

            metadata["new_buy_tickers"] = new_buys_final
            metadata["new_buys_count"] = len(new_buys_final)

        portfolio_slice = clean[clean[ticker_col].isin(selected_tickers)].copy()
        portfolio_slice["selection_reason"] = portfolio_slice[ticker_col].map(ticker_meta).fillna("selected")
        portfolio_slice = portfolio_slice.sort_values(score_col, ascending=False).reset_index(drop=True)

        k = len(portfolio_slice)
        if k == 0:
            empty_df = pd.DataFrame(columns=[ticker_col, "sector", "rank", score_col, "target_weight", "predicted_vol", "selection_reason"])
            metadata["cash_weight"] = 1.0
            empty_df.attrs["metadata"] = metadata
            return (empty_df, metadata) if return_metadata else empty_df

        # ---------------------------------------------------------
        # Rule 3: Position Sizing (Inverse-Vol with Dual Cap Redistribution)
        # ---------------------------------------------------------
        if method == "inv_vol":
            vols = np.clip(portfolio_slice["predicted_vol"].values, 0.05, 1.50)
            inv_vols = 1.0 / vols
            raw_weights = inv_vols / np.sum(inv_vols)
        else:
            raw_weights = np.full(k, 1.0 / k)

        weights = raw_weights.copy()
        slice_tickers = portfolio_slice[ticker_col].tolist()
        slice_sectors = [ticker_sectors.get(t, "Other") for t in slice_tickers]
        enforce_sector_cap = has_real_sectors and (eff_max_sector_weight is not None)

        min_positions_for_cap = int(np.ceil(1.0 / self.max_weight))
        if k >= min_positions_for_cap:
            for _ in range(100):
                changed = False
                # 1. Individual weight cap
                over_indiv = weights > self.max_weight
                if np.any(over_indiv):
                    excess = np.sum(weights[over_indiv] - self.max_weight)
                    weights[over_indiv] = self.max_weight
                    changed = True

                    # Find uncapped positions
                    uncapped = weights < (self.max_weight - 1e-5)
                    if enforce_sector_cap:
                        sec_sums = {}
                        for s_val, w_val in zip(slice_sectors, weights):
                            sec_sums[s_val] = sec_sums.get(s_val, 0.0) + w_val
                        sec_eligible = np.array([sec_sums.get(s, 0.0) < (eff_max_sector_weight - 1e-5) for s in slice_sectors])
                        uncapped = uncapped & sec_eligible

                    if np.any(uncapped) and excess > 1e-7:
                        u_sum = np.sum(weights[uncapped])
                        if u_sum > 0:
                            weights[uncapped] += excess * (weights[uncapped] / u_sum)
                        else:
                            weights[uncapped] += excess / np.sum(uncapped)

                # 2. Sector cap
                if enforce_sector_cap:
                    sec_sums = {}
                    for s_val, w_val in zip(slice_sectors, weights):
                        sec_sums[s_val] = sec_sums.get(s_val, 0.0) + w_val

                    for s_name, s_total in sec_sums.items():
                        if s_name in GICS_SECTORS and s_total > eff_max_sector_weight + 1e-5:
                            excess_sec = s_total - eff_max_sector_weight
                            s_indices = [idx for idx, s in enumerate(slice_sectors) if s == s_name]
                            s_w_sum = np.sum(weights[s_indices])
                            if s_w_sum > 0:
                                weights[s_indices] *= (eff_max_sector_weight / s_w_sum)
                                changed = True

                            other_mask = (np.array(slice_sectors) != s_name) & (weights < (self.max_weight - 1e-5))
                            other_sec_eligible = np.array([sec_sums.get(s, 0.0) < (eff_max_sector_weight - 1e-5) for s in slice_sectors])
                            other_mask = other_mask & other_sec_eligible

                            if np.any(other_mask) and excess_sec > 1e-7:
                                o_sum = np.sum(weights[other_mask])
                                if o_sum > 0:
                                    weights[other_mask] += excess_sec * (weights[other_mask] / o_sum)
                                else:
                                    weights[other_mask] += excess_sec / np.sum(other_mask)

                if not changed:
                    break
        elif top_n is None:
            weights = np.minimum(weights, self.max_weight)

        # ---------------------------------------------------------
        # Rule 4: Exposure Scaling (VIX / Volatility Spike)
        # ---------------------------------------------------------
        exposure_scale = 1.0
        if use_exposure_scaling and current_vix is not None and current_vix > self.vix_baseline:
            exposure_scale = min(1.0, self.vix_baseline / float(current_vix))
            logger.info(
                f"High VIX detected ({current_vix:.2f} > {self.vix_baseline:.2f}). "
                f"Scaling equity exposure to {exposure_scale:.1%}"
            )

        weights = weights * exposure_scale
        portfolio_slice["target_weight"] = np.round(weights, 4)
        portfolio_slice["sector"] = slice_sectors

        metadata["exposure_scale"] = round(exposure_scale, 4)
        metadata["cash_weight"] = round(max(0.0, 1.0 - float(np.sum(portfolio_slice["target_weight"]))), 4)

        # Compute aggregate sector exposures
        sec_agg: dict[str, dict[str, Any]] = {}
        for t, w, s in zip(portfolio_slice[ticker_col], portfolio_slice["target_weight"], slice_sectors):
            if s not in sec_agg:
                sec_agg[s] = {"sector": s, "weight": 0.0, "count": 0, "tickers": []}
            sec_agg[s]["weight"] = round(sec_agg[s]["weight"] + float(w), 4)
            sec_agg[s]["count"] += 1
            sec_agg[s]["tickers"].append(t)
        metadata["sector_breakdown"] = sorted(sec_agg.values(), key=lambda x: x["weight"], reverse=True)

        out_cols = [ticker_col, "sector", "rank", score_col, "target_weight", "predicted_vol", "selection_reason"]
        res_df = portfolio_slice[out_cols].reset_index(drop=True)
        res_df.attrs["metadata"] = metadata
        return (res_df, metadata) if return_metadata else res_df

    def generate_trade_list(
        self,
        target_df: pd.DataFrame,
        current_holdings: dict[str, float] | None = None,
        min_trade_threshold: float = 0.002,  # 20 bps minimum rebalance threshold
        ticker_col: str = "ticker",
        weight_col: str = "target_weight",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Diffs target weights against current holdings and produces ordered execution trade list.

        SELL orders are placed first to raise liquidity before BUY orders.
        """
        if current_holdings is None:
            current_holdings = self.load_holdings()

        # Build target map excluding CASH virtual tracker
        target_map = {
            row[ticker_col]: float(row[weight_col])
            for _, row in target_df.iterrows()
            if row[ticker_col] not in ("CASH",)
        }
        held_map = {k: float(v) for k, v in current_holdings.items() if k not in ("CASH",)}

        all_tickers = sorted(set(target_map.keys()) | set(held_map.keys()))

        orders = []
        turnover_sum = 0.0

        for t in all_tickers:
            tgt_w = float(target_map.get(t, 0.0))
            cur_w = float(held_map.get(t, 0.0))
            delta_w = tgt_w - cur_w

            turnover_sum += abs(delta_w)

            if abs(delta_w) < min_trade_threshold:
                continue

            action = "BUY" if delta_w > 0 else "SELL"
            orders.append({
                "ticker": t,
                "action": action,
                "current_weight": round(cur_w, 4),
                "target_weight": round(tgt_w, 4),
                "delta_weight": round(delta_w, 4),
                "pct_change": round(delta_w * 100.0, 2),
            })

        # Orders sorted: SELLs first to release cash, then BUYs
        orders.sort(key=lambda x: (0 if x["action"] == "SELL" else 1, -abs(x["delta_weight"])))

        one_way_turnover = round(turnover_sum * 0.5, 4)
        summary = {
            "total_orders": len(orders),
            "sells_count": sum(1 for o in orders if o["action"] == "SELL"),
            "buys_count": sum(1 for o in orders if o["action"] == "BUY"),
            "one_way_turnover": one_way_turnover,
            "target_positions_count": len(target_map),
        }

        return orders, summary

    def load_holdings(self) -> dict[str, float]:
        """Loads current portfolio holdings from JSON file."""
        if not self.holdings_file.exists():
            return {}
        try:
            with open(self.holdings_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read holdings file {self.holdings_file}: {e}")
            return {}

    def save_holdings(self, holdings: dict[str, float]) -> None:
        """Saves current portfolio holdings atomically to JSON file."""
        tmp_path = self.holdings_file.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(holdings, f, indent=2)
        tmp_path.replace(self.holdings_file)
        logger.info(f"Saved {len(holdings)} portfolio holdings to {self.holdings_file}")
