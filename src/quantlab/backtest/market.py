"""Session-grid market panel used by the engine (truth for fills and valuation)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.data import calendar
from quantlab.data.adjustments import adjusted_ohlc, total_return_index
from quantlab.domain.models import PriceBasis, PriceMode


@dataclass
class MarketPanel:
    sessions: pd.DatetimeIndex
    symbols: list[str]
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    dividend: pd.DataFrame
    split: pd.DataFrame
    signal_price: pd.DataFrame  # causal total-return index (NaN where no bar)
    price_basis: str
    price_mode: str

    @property
    def apply_splits_to_holdings(self) -> bool:
        return self.price_mode == PriceMode.RAW_WITH_DIVIDENDS and self.price_basis == PriceBasis.UNADJUSTED

    @property
    def credit_dividends(self) -> bool:
        return self.price_mode == PriceMode.RAW_WITH_DIVIDENDS


def build_panel(
    bars: pd.DataFrame,
    price_basis: str,
    price_mode: str = PriceMode.RAW_WITH_DIVIDENDS,
    symbols: list[str] | None = None,
) -> MarketPanel:
    """Pivot long bars onto the XNYS session grid spanning the data.

    Missing bars become NaN prices (never forward-filled here); corporate actions on missing
    rows are lost, which the data-quality report flags as missing sessions.
    """
    price_mode = PriceMode(price_mode)
    df = bars if symbols is None else bars[bars["symbol"].isin(symbols)]
    if df.empty:
        raise ValueError("no bars for requested symbols")
    if price_mode == PriceMode.ADJUSTED_TOTAL_RETURN:
        df = adjusted_ohlc(df)
    syms = sorted(df["symbol"].unique().tolist()) if symbols is None else [s for s in symbols if s in set(df["symbol"])]
    grid = calendar.sessions(df["session_date"].min(), df["session_date"].max())

    def piv(col: str, fill=None) -> pd.DataFrame:
        p = df.pivot(index="session_date", columns="symbol", values=col).reindex(index=grid, columns=syms)
        return p.fillna(fill) if fill is not None else p

    close = piv("close")
    dividend = piv("dividend", 0.0)
    split = piv("split_ratio", 1.0)
    basis_for_tr = PriceBasis.SPLIT_ADJUSTED if price_mode == PriceMode.ADJUSTED_TOTAL_RETURN else price_basis
    sig = pd.DataFrame(
        {
            s: total_return_index(close[s].to_numpy(), dividend[s].to_numpy(), split[s].to_numpy(), basis_for_tr)
            for s in syms
        },
        index=grid,
    )
    return MarketPanel(
        sessions=grid,
        symbols=syms,
        open=piv("open"),
        high=piv("high"),
        low=piv("low"),
        close=close,
        volume=piv("volume"),
        dividend=dividend,
        split=split,
        signal_price=sig.astype(float),
        price_basis=price_basis,
        price_mode=str(price_mode),
    )
