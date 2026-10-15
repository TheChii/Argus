"""Universe definitions for S&P 100, Sector ETFs, and Benchmark."""
import json
from pathlib import Path

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("universe")

# Mapping of GICS Sectors to SPDR Sector ETFs
SECTOR_ETF_MAP: dict[str, str] = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

BENCHMARK_TICKER = "SPY"


def normalize_ticker_for_yfinance(symbol: str) -> str:
    """
    Normalizes ticker symbols for Yahoo Finance.
    e.g., BRK.B -> BRK-B, BF.B -> BF-B
    """
    symbol = symbol.strip().upper()
    return symbol.replace(".", "-")


def denormalize_ticker(symbol: str) -> str:
    """
    Converts Yahoo Finance ticker format back to standard dot notation.
    e.g., BRK-B -> BRK.B
    """
    symbol = symbol.strip().upper()
    return symbol.replace("-", ".")


def get_sector_etfs() -> list[str]:
    """Returns the list of 11 SPDR Sector ETFs."""
    return sorted(set(SECTOR_ETF_MAP.values()))


def get_sp100_constituents(refresh_from_wiki: bool = False) -> list[dict[str, str]]:
    """
    Returns list of S&P 100 constituents with Symbol, Name, Sector, and Sector_ETF.
    Uses cached JSON file by default, with option to re-fetch from Wikipedia.
    """
    json_path = Path(__file__).resolve().parent / "sp100_default.json"

    if refresh_from_wiki:
        try:
            url = "https://en.wikipedia.org/wiki/S%26P_100"
            tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
            for table in tables:
                if "Symbol" in table.columns and "Sector" in table.columns:
                    records = table[["Symbol", "Name", "Sector"]].to_dict(orient="records")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(records, f, indent=2)
                    break
        except (ValueError, OSError, TimeoutError) as e:
            # Fall back to existing cached file if Wikipedia fetch fails
            logger.debug("Wikipedia S&P 100 constituent update failed (%s); using cached registry.", e)

    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
    else:
        # Minimal emergency fallback if json file missing
        records = [
            {"Symbol": "AAPL", "Name": "Apple Inc.", "Sector": "Information Technology"},
            {"Symbol": "MSFT", "Name": "Microsoft Corporation", "Sector": "Information Technology"},
            {"Symbol": "NVDA", "Name": "NVIDIA Corporation", "Sector": "Information Technology"},
            {"Symbol": "AMZN", "Name": "Amazon.com Inc.", "Sector": "Consumer Discretionary"},
            {"Symbol": "GOOGL", "Name": "Alphabet Inc. (Class A)", "Sector": "Communication Services"},
        ]

    # Enrich with Sector ETF and normalized symbol
    enriched = []
    for item in records:
        sym = item["Symbol"]
        sector = item.get("Sector", "")
        sector_etf = SECTOR_ETF_MAP.get(sector, "SPY")
        enriched.append({
            "symbol": sym,
            "yf_symbol": normalize_ticker_for_yfinance(sym),
            "name": item.get("Name", ""),
            "sector": sector,
            "sector_etf": sector_etf,
        })
    return enriched


def get_all_tickers() -> list[str]:
    """
    Returns a unified, sorted list of all Yahoo Finance tickers required by the model:
    - S&P 100 constituents
    - Benchmark (SPY)
    - 11 Sector ETFs
    """
    constituents = get_sp100_constituents()
    tickers = {c["yf_symbol"] for c in constituents}
    tickers.add(BENCHMARK_TICKER)
    tickers.update(get_sector_etfs())
    return sorted(tickers)


def get_ticker_to_sector_etf_map() -> dict[str, str]:
    """Returns mapping from normalized ticker to its sector ETF."""
    mapping = {}
    for c in get_sp100_constituents():
        mapping[c["yf_symbol"]] = c["sector_etf"]
    # Sector ETFs map to themselves
    for etf in get_sector_etfs():
        mapping[etf] = etf
    mapping[BENCHMARK_TICKER] = BENCHMARK_TICKER
    return mapping


def get_sector_for_ticker(symbol: str) -> str:
    """Returns the GICS sector name for a ticker, or empty string if not found."""
    norm = normalize_ticker_for_yfinance(symbol)
    for c in get_sp100_constituents():
        if c["yf_symbol"] == norm or c["symbol"] == symbol:
            return c.get("sector", "")
    return ""
