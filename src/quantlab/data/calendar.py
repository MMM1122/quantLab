"""Exchange calendar (XNYS) helpers. Missing-bar checks use real sessions, never calendar days."""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

CALENDAR_NAME = "XNYS"
_CAL_START = "1995-01-01"


@lru_cache(maxsize=1)
def _calendar():
    return xcals.get_calendar(CALENDAR_NAME, start=_CAL_START)


def sessions(start: str | date | pd.Timestamp, end: str | date | pd.Timestamp) -> pd.DatetimeIndex:
    """Trading sessions in [start, end] (inclusive), tz-naive midnight timestamps."""
    cal = _calendar()
    s = max(pd.Timestamp(start).normalize(), cal.first_session)
    e = min(pd.Timestamp(end).normalize(), cal.last_session)
    if s > e:
        return pd.DatetimeIndex([])
    out = cal.sessions_in_range(s, e)
    return pd.DatetimeIndex(out).tz_localize(None) if out.tz is not None else pd.DatetimeIndex(out)


def is_session(d: str | date | pd.Timestamp) -> bool:
    return bool(_calendar().is_session(pd.Timestamp(d).normalize()))


def last_completed_session(asof: pd.Timestamp | None = None) -> pd.Timestamp:
    """Most recent session whose close is at or before ``asof`` (default: now, NY time)."""
    now = asof if asof is not None else pd.Timestamp.now(tz="America/New_York")
    if now.tzinfo is None:
        now = now.tz_localize("America/New_York")
    cal = _calendar()
    day = now.tz_convert("America/New_York").normalize().tz_localize(None)
    sess = sessions(day - pd.Timedelta(days=10), day)
    if len(sess) and sess[-1] == day:
        close = cal.session_close(day).tz_convert("America/New_York")
        if now < close:
            sess = sess[:-1]
    return sess[-1]
