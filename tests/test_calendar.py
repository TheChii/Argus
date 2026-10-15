"""Tests for Calendar and 4:00 PM ET Cutoff Rules."""
import zoneinfo
from datetime import datetime

from src.utils.calendar import (
    get_effective_trading_date,
    get_next_trading_day,
    is_trading_day,
)

ET = zoneinfo.ZoneInfo("America/New_York")


def test_weekend_not_trading_day():
    # 2026-10-10 is Saturday, 2026-10-11 is Sunday
    assert not is_trading_day("2026-10-10")
    assert not is_trading_day("2026-10-11")


def test_regular_weekday_is_trading_day():
    # 2026-10-07 is Wednesday
    assert is_trading_day("2026-10-07")


def test_market_holidays():
    # New Year's Day 2024
    assert not is_trading_day("2024-01-01")
    # Christmas Day 2024
    assert not is_trading_day("2024-12-25")
    # Good Friday 2024 (2024-03-29)
    assert not is_trading_day("2024-03-29")


def test_news_before_4pm_cutoff():
    # News published at 2:30 PM ET on Wednesday 2026-10-07
    dt = datetime(2026, 10, 7, 14, 30, tzinfo=ET)
    effective = get_effective_trading_date(dt)
    assert effective == "2026-10-07"


def test_news_after_4pm_cutoff():
    # News published at 4:15 PM ET on Wednesday 2026-10-07 -> goes to next trading day (Thursday 2026-10-08)
    dt = datetime(2026, 10, 7, 16, 15, tzinfo=ET)
    effective = get_effective_trading_date(dt)
    assert effective == "2026-10-08"


def test_news_on_friday_after_close():
    # News published at 5:00 PM ET on Friday 2026-10-09 -> goes to Monday 2026-10-12 (or next trading day)
    dt = datetime(2026, 10, 9, 17, 0, tzinfo=ET)
    effective = get_effective_trading_date(dt)
    # 2026-10-12 is Monday
    next_td = get_next_trading_day("2026-10-09").strftime("%Y-%m-%d")
    assert effective == next_td


def test_news_on_weekend():
    # News published on Saturday 2026-10-10 -> goes to next trading day
    dt = datetime(2026, 10, 10, 10, 0, tzinfo=ET)
    effective = get_effective_trading_date(dt)
    next_td = get_next_trading_day("2026-10-09").strftime("%Y-%m-%d")
    assert effective == next_td
