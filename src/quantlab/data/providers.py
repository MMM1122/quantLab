"""Market-data providers.

* ``YFinanceProvider`` — research data via the unofficial ``yfinance`` package. Yahoo data is
  for personal research use only; check Yahoo's terms. It may be rate-limited, revised or
  unavailable, which is why snapshots are immutable and hashed.
* ``SyntheticProvider`` — seeded geometric-Brownian-motion fixtures for offline demos and
  tests. Every snapshot from it is flagged ``is_synthetic`` and labelled
  ``SYNTHETIC — NOT REAL MARKET PERFORMANCE``. Real and synthetic data are never mixed in one
  snapshot.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quantlab.core.errors import DataUnavailableError
from quantlab.data import calendar
from quantlab.data.adjustments import back_adjusted_close
from quantlab.domain.models import BAR_COLUMNS, PriceBasis

YF_TERMS_WARNING = (
    "yfinance retrieves Yahoo Finance data through an unofficial interface intended for "
    "personal research use. Review Yahoo's terms; data may be delayed, revised or unavailable."
)


@dataclass
class ProviderResult:
    bars: pd.DataFrame
    provider: str
    price_basis: str
    is_synthetic: bool
    parameters: dict[str, Any] = field(default_factory=dict)
    retrieved_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    warnings: list[str] = field(default_factory=list)


class MarketDataProvider(Protocol):
    name: str

    def fetch(self, symbols: list[str], start: str, end: str) -> ProviderResult: ...


def canonicalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column set, dtypes and (symbol, session_date) ordering."""
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    out = df[BAR_COLUMNS].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.tz_localize(None).dt.normalize()
    out["session_date"] = out["session_date"].astype("datetime64[ns]")
    for c in ("open", "high", "low", "close", "adj_close", "volume", "dividend", "split_ratio"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["currency"] = out["currency"].astype(str)
    out["source"] = out["source"].astype(str)
    out = out.sort_values(["symbol", "session_date"], kind="mergesort").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- yfinance


def _default_yf_download(symbol: str, start: str, end_exclusive: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so offline use never needs network

    return yf.Ticker(symbol).history(
        start=start, end=end_exclusive, auto_adjust=False, actions=True, back_adjust=False
    )


class YFinanceProvider:
    """Daily OHLCV + dividends + splits via yfinance (auto_adjust=False).

    Yahoo's Open/High/Low/Close are split-adjusted but not dividend-adjusted, so snapshots
    from this provider have ``price_basis = split_adjusted``; ``Adj Close`` is the
    split+dividend back-adjusted close.
    """

    name = "yfinance"

    def __init__(self, downloader: Callable[[str, str, str], pd.DataFrame] | None = None,
                 now: Callable[[], pd.Timestamp] | None = None):
        self._download = downloader or _default_yf_download
        self._now = now or (lambda: pd.Timestamp.now(tz="America/New_York"))

    def fetch(self, symbols: list[str], start: str, end: str) -> ProviderResult:
        warnings.warn(YF_TERMS_WARNING, UserWarning, stacklevel=2)
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frames = []
        errors: dict[str, str] = {}
        notes: list[str] = []
        last_done = calendar.last_completed_session(self._now())
        for sym in symbols:
            try:
                raw = self._download(sym, start, end_excl)
            except Exception as exc:  # network/provider errors are converted, not swallowed
                errors[sym] = f"{type(exc).__name__}: {exc}"
                continue
            if raw is None or len(raw) == 0:
                errors[sym] = "provider returned no rows"
                continue
            df, sym_notes = clean_provider_rows(sym, self._normalize(sym, raw), last_done)
            notes += sym_notes
            if df.empty:
                errors[sym] = "no completed-session rows after cleanup"
                continue
            frames.append(df)
        if errors:
            raise DataUnavailableError(f"yfinance fetch failed for {sorted(errors)}: {errors}")
        bars = canonicalize_bars(pd.concat(frames, ignore_index=True))
        return ProviderResult(
            bars=bars,
            provider=self.name,
            price_basis=PriceBasis.SPLIT_ADJUSTED.value,
            is_synthetic=False,
            parameters={"symbols": symbols, "start": start, "end": end, "auto_adjust": False, "actions": True,
                        "last_completed_session": last_done.strftime("%Y-%m-%d"), "provider_cleanup": notes},
            warnings=[YF_TERMS_WARNING, *notes],
        )

    @staticmethod
    def _normalize(symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        cols = {c.lower().replace(" ", "_"): c for c in df.columns}
        required = ["open", "high", "low", "close", "volume"]
        for r in required:
            if r not in cols:
                raise DataUnavailableError(f"{symbol}: provider frame missing column {r}")
        adj = df[cols["adj_close"]] if "adj_close" in cols else df[cols["close"]]
        div = df[cols["dividends"]] if "dividends" in cols else 0.0
        spl = df[cols["stock_splits"]] if "stock_splits" in cols else 0.0
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "session_date": idx.normalize(),
                "open": df[cols["open"]].to_numpy(dtype=float),
                "high": df[cols["high"]].to_numpy(dtype=float),
                "low": df[cols["low"]].to_numpy(dtype=float),
                "close": df[cols["close"]].to_numpy(dtype=float),
                "adj_close": np.asarray(adj, dtype=float),
                "volume": df[cols["volume"]].to_numpy(dtype=float),
                "dividend": np.asarray(div, dtype=float) if not np.isscalar(div) else 0.0,
                "split_ratio": np.asarray(spl, dtype=float) if not np.isscalar(spl) else 0.0,
                "currency": "USD",
                "source": "yfinance",
            }
        )
        # yfinance reports "Stock Splits" = 0 on non-split days
        out["split_ratio"] = out["split_ratio"].where(out["split_ratio"] > 0, 1.0)
        out["dividend"] = out["dividend"].fillna(0.0)
        return out


def clean_provider_rows(symbol: str, df: pd.DataFrame, last_completed: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Handle known Yahoo quirks explicitly; every change is reported (stored in snapshot provenance).

    1. Bars for sessions that have not closed yet (intraday download) are dropped — a partial bar
       would otherwise be treated as a final close.
    2. Action-only rows (no prices, but a dividend/split) dated on a non-session day are moved to the
       next XNYS session (the effective ex-date); on a session day they are merged into that
       session's price row.
    3. Rows with no prices and no actions are dropped.
    4. Exact duplicate rows are collapsed. Conflicting duplicates are KEPT so validation flags them.
    """
    notes: list[str] = []
    df = df.copy()
    prices = df[["open", "high", "low", "close"]]
    no_price = prices.isna().all(axis=1)
    has_action = (df["dividend"].fillna(0) > 0) | (df["split_ratio"].fillna(1) != 1.0)

    future = df["session_date"] > last_completed
    if future.any():
        notes.append(f"{symbol}: dropped {int(future.sum())} bar(s) for session(s) not yet closed "
                     f"(after {last_completed:%Y-%m-%d})")
        df, no_price, has_action = df[~future], no_price[~future], has_action[~future]

    empty = no_price & ~has_action
    if empty.any():
        notes.append(f"{symbol}: dropped {int(empty.sum())} empty provider row(s) without prices or actions")
        df, no_price, has_action = df[~empty], no_price[~empty], has_action[~empty]

    actions_only = df[no_price & has_action]
    if len(actions_only):
        df = df[~(no_price & has_action)].copy()
        for _, a in actions_only.iterrows():
            d = pd.Timestamp(a["session_date"])
            target = d if calendar.is_session(d) else calendar.sessions(d, d + pd.Timedelta(days=10))[0]
            m = df["session_date"] == target
            if not m.any():
                notes.append(f"{symbol}: dropped corporate action on {d:%Y-%m-%d} — no price row for session "
                             f"{target:%Y-%m-%d} (dividend {a['dividend']}, split {a['split_ratio']})")
                continue
            i = df.index[m][0]
            if a["dividend"] and a["dividend"] > 0:
                df.loc[i, "dividend"] = max(float(df.loc[i, "dividend"] or 0), float(a["dividend"]))
            if a["split_ratio"] and a["split_ratio"] != 1.0:
                df.loc[i, "split_ratio"] = float(a["split_ratio"])
            moved = "" if target == d else f" moved from non-session day {d:%Y-%m-%d} to {target:%Y-%m-%d}"
            notes.append(f"{symbol}: merged action-only row into session {target:%Y-%m-%d}{moved}")

    dup_exact = df.duplicated(keep="first")
    if dup_exact.any():
        notes.append(f"{symbol}: collapsed {int(dup_exact.sum())} exact duplicate row(s)")
        df = df[~dup_exact]
    return df.reset_index(drop=True), notes


# --------------------------------------------------------------------------- synthetic

_SYNTH_PARAMS = {
    # symbol: (start price, annual drift, annual vol, annual dividend yield, dividends/yr)
    "SPY": (200.0, 0.08, 0.17, 0.016, 4),
    "QQQ": (100.0, 0.11, 0.22, 0.006, 4),
    "IWM": (120.0, 0.06, 0.22, 0.013, 4),
    "TLT": (125.0, 0.01, 0.14, 0.030, 12),
    "GLD": (115.0, 0.05, 0.15, 0.0, 0),
}


class SyntheticProvider:
    """Seeded GBM bars on real XNYS sessions. Clearly labelled synthetic.

    Prices are *unadjusted* (as-traded). Dividends reduce the price on the ex-date. One
    optional 2-for-1 split (``split_symbol`` on ``split_date``) exercises split handling.
    """

    name = "synthetic"

    def __init__(self, seed: int = 7, split_symbol: str | None = "QQQ", split_date: str = "2021-06-01"):
        self.seed = seed
        self.split_symbol = split_symbol
        self.split_date = split_date

    def fetch(self, symbols: list[str], start: str, end: str) -> ProviderResult:
        sess = calendar.sessions(start, end)
        if len(sess) < 2:
            raise DataUnavailableError("synthetic provider: fewer than 2 sessions in range")
        frames = []
        for k, sym in enumerate(symbols):
            frames.append(self._one(sym, sess, k))
        bars = canonicalize_bars(pd.concat(frames, ignore_index=True))
        return ProviderResult(
            bars=bars,
            provider=self.name,
            price_basis=PriceBasis.UNADJUSTED.value,
            is_synthetic=True,
            parameters={
                "symbols": symbols,
                "start": start,
                "end": end,
                "seed": self.seed,
                "split_symbol": self.split_symbol,
                "split_date": self.split_date,
                "model": "GBM daily log-returns; quarterly/monthly dividends; seeded",
            },
        )

    def _one(self, sym: str, sess: pd.DatetimeIndex, k: int) -> pd.DataFrame:
        # derive a per-symbol stream from the seed and the symbol name (order-independent)
        sym_key = sum((i + 1) * ord(ch) for i, ch in enumerate(sym))
        rng = np.random.default_rng([self.seed, sym_key])
        p0, mu, vol, dy, nd = _SYNTH_PARAMS.get(sym, (50.0 + 10 * k, 0.06, 0.20, 0.01, 4))
        n = len(sess)
        dt = 1 / 252
        rets = rng.normal((mu - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n)
        # occasional fat-tail days so drawdowns are non-trivial
        shocks = rng.random(n) < 0.004
        rets[shocks] += rng.normal(-0.03, 0.03, shocks.sum())
        dividend = np.zeros(n)
        split = np.ones(n)
        if nd > 0 and dy > 0:
            months = {4: (3, 6, 9, 12), 12: tuple(range(1, 13))}[nd]
            s = pd.Series(np.arange(n), index=sess)
            for (_y, m), grp in s.groupby([sess.year, sess.month]):
                if m in months and len(grp) > 15:
                    dividend[int(grp.iloc[15])] = -1.0  # placeholder, amount set below
        if self.split_symbol == sym:
            sd = pd.Timestamp(self.split_date)
            pos = sess.searchsorted(sd)
            if 0 < pos < n:
                split[pos] = 2.0
        close = np.empty(n)
        open_ = np.empty(n)
        prev = p0
        for t in range(n):
            ref = prev
            if split[t] != 1.0:
                ref = ref / split[t]
            if dividend[t] < 0:
                dividend[t] = round(ref * dy / nd, 4)
                ref = ref - dividend[t]
            gap = rng.normal(0, vol * np.sqrt(dt) * 0.3)
            open_[t] = ref * np.exp(gap)
            close[t] = ref * np.exp(rets[t])
            prev = close[t]
        span = np.abs(rng.normal(0, vol * np.sqrt(dt) * 0.6, n))
        high = np.maximum(open_, close) * (1 + span)
        low = np.minimum(open_, close) * (1 - span * 0.9)
        volume = np.round(rng.lognormal(16.5, 0.35, n))
        close_r, open_r = np.round(close, 4), np.round(open_, 4)
        high_r = np.maximum(np.round(high, 4), np.maximum(open_r, close_r))
        low_r = np.minimum(np.round(low, 4), np.minimum(open_r, close_r))
        adj = back_adjusted_close(close_r, dividend, split, PriceBasis.UNADJUSTED)
        return pd.DataFrame(
            {
                "symbol": sym,
                "session_date": sess,
                "open": open_r,
                "high": high_r,
                "low": low_r,
                "close": close_r,
                "adj_close": np.round(adj, 6),
                "volume": volume,
                "dividend": dividend,
                "split_ratio": split,
                "currency": "USD",
                "source": "synthetic",
            }
        )


def get_provider(name: str, seed: int = 7) -> MarketDataProvider:
    if name == "yfinance":
        return YFinanceProvider()
    if name == "synthetic":
        return SyntheticProvider(seed=seed)
    raise ValueError(f"unknown provider {name!r} (expected 'yfinance' or 'synthetic')")
