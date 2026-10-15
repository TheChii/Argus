"""Market trading calendar and cutoff utilities."""
import zoneinfo
from datetime import date, datetime, time

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

from src.config.settings import get_settings


class USTradingHolidayCalendar(AbstractHolidayCalendar):
    """NYSE / NASDAQ standard holiday calendar."""
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2021-06-18", observance=nearest_workday),
        Holiday("July4th", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_CALENDAR = USTradingHolidayCalendar()
_HOLIDAYS_CACHE = set(_CALENDAR.holidays("2000-01-01", "2040-12-31").date)
_ET_TZ = zoneinfo.ZoneInfo("America/New_York")
_UTC_TZ = zoneinfo.ZoneInfo("UTC")


def is_trading_day(d: date | datetime | str) -> bool:
    """Checks whether a given calendar date is an open US equity trading day."""
    if isinstance(d, str):
        d = pd.to_datetime(d).date()
    elif isinstance(d, datetime):
        d = d.date()

    # Monday=0, Sunday=6
    if d.weekday() >= 5:
        return False
    return d not in _HOLIDAYS_CACHE


def get_next_trading_day(d: date | datetime | str) -> date:
    """Returns the next trading day strictly after the given date."""
    if isinstance(d, str):
        curr = pd.to_datetime(d).date()
    elif isinstance(d, datetime):
        curr = d.date()
    else:
        curr = d

    curr += pd.Timedelta(days=1).to_pytimedelta()
    while not is_trading_day(curr):
        curr += pd.Timedelta(days=1).to_pytimedelta()
    return curr


def get_last_completed_trading_day() -> date:
    """
    Returns the most recent finalized trading day based on 4:00 PM ET close.
    If run after 4:00 PM ET on an open trading day, returns today.
    Otherwise returns the preceding open trading day.
    """
    settings = get_settings()
    now_et = datetime.now(tz=_ET_TZ)
    today = now_et.date()
    cutoff = time(settings.market_close_hour, settings.market_close_minute)

    if is_trading_day(today) and now_et.time() >= cutoff:
        return today

    candidate = today - pd.Timedelta(days=1).to_pytimedelta()
    while not is_trading_day(candidate):
        candidate -= pd.Timedelta(days=1).to_pytimedelta()
    return candidate


def parse_to_et(dt_val: datetime | str | int | float) -> datetime:
    """Converts various datetime representations (epoch, ISO, naive) to America/New_York."""
    if isinstance(dt_val, (int, float)):
        # If timestamp is in milliseconds
        if dt_val > 1e11:
            dt_val = dt_val / 1000.0
        dt_obj = datetime.fromtimestamp(dt_val, tz=_UTC_TZ)
    elif isinstance(dt_val, str):
        dt_obj = pd.to_datetime(dt_val)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.tz_localize("UTC")
    elif isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            dt_obj = dt_val.replace(tzinfo=_UTC_TZ)
        else:
            dt_obj = dt_val
    else:
        raise ValueError(f"Unsupported datetime format: {type(dt_val)}")

    return dt_obj.astimezone(_ET_TZ)


def get_effective_trading_date(publish_time: datetime | str | int | float) -> str:
    """
    Determines which trading date a headline belongs to.

    Rule:
    - Converted to America/New_York time.
    - If published at or after 4:00 PM ET (16:00), it belongs to the next trading day.
    - If published before 4:00 PM ET on a trading day, it belongs to today's trading day.
    - If published on a weekend or market holiday, it belongs to the next trading day.

    Returns:
        ISO date string 'YYYY-MM-DD'
    """
    et_time = parse_to_et(publish_time)
    settings = get_settings()
    cutoff_time = time(settings.market_close_hour, settings.market_close_minute)

    curr_date = et_time.date()

    if et_time.time() >= cutoff_time:
        # After 4:00 PM ET: belongs strictly to the next trading day
        effective_date = get_next_trading_day(curr_date)
    else:
        # Before 4:00 PM ET: belongs to today if today is a trading day, else next trading day
        if is_trading_day(curr_date):
            effective_date = curr_date
        else:
            effective_date = get_next_trading_day(curr_date)

    return effective_date.strftime("%Y-%m-%d")


def get_current_utc_iso() -> str:
    """Returns the current UTC time as an ISO string."""
    return datetime.now(tz=_UTC_TZ).strftime("%Y-%m-%dT%H:%M:%SZ")
