"""Plotly figures shared by the dashboard and HTML exports. Titles always carry data labels."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantlab.backtest.metrics import drawdown_series
from quantlab.core.labels import BACKTEST_LABEL, SYNTHETIC_LABEL

PALETTE = ["#2F5D8A", "#B5651D", "#4F7F52", "#8C4A6B", "#6B6B6B", "#A08C3B"]


def _title(text: str, is_synthetic: bool, simulated: bool = True) -> str:
    parts = [text]
    if simulated:
        parts.append(BACKTEST_LABEL)
    if is_synthetic:
        parts.append(SYNTHETIC_LABEL)
    return "<br><sup>" .join([parts[0], " · ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def _layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 50, "r": 20, "t": 70, "b": 40},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        hovermode="x unified",
    )
    return fig


def equity_and_drawdown(curves: dict[str, pd.Series], is_synthetic: bool, title: str = "Equity curve") -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05)
    for k, (name, nav) in enumerate(curves.items()):
        color = PALETTE[k % len(PALETTE)]
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name=name, line={"color": color, "width": 1.6}),
                      row=1, col=1)
        dd = drawdown_series(nav)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, name=f"{name} DD", showlegend=False,
                                 line={"color": color, "width": 1}, fill="tozeroy" if k == 0 else None),
                      row=2, col=1)
    fig.update_yaxes(title_text="NAV (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_layout(title=_title(title, is_synthetic))
    return _layout(fig, 560)


def price_chart(bars: pd.DataFrame, symbol: str, is_synthetic: bool, kind: str = "candlestick") -> go.Figure:
    g = bars[bars["symbol"] == symbol].sort_values("session_date")
    fig = go.Figure()
    if kind == "candlestick":
        fig.add_trace(go.Candlestick(x=g["session_date"], open=g["open"], high=g["high"], low=g["low"],
                                     close=g["close"], name=symbol))
        fig.update_layout(xaxis_rangeslider_visible=False)
    else:
        fig.add_trace(go.Scatter(x=g["session_date"], y=g["close"], name="close", line={"color": PALETTE[0]}))
        fig.add_trace(go.Scatter(x=g["session_date"], y=g["adj_close"], name="adj_close (back-adjusted)",
                                 line={"color": PALETTE[1], "dash": "dot"}))
    fig.update_layout(title=_title(f"{symbol} daily bars", is_synthetic, simulated=False))
    return _layout(fig)


def signal_chart(price: pd.Series, weight: pd.Series, symbol: str, is_synthetic: bool, title: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=price.index, y=price.values, name=f"{symbol} total-return index",
                             line={"color": PALETTE[0]}), row=1, col=1)
    fig.add_trace(go.Scatter(x=weight.index, y=weight.values, name="target weight", line={"shape": "hv",
                             "color": PALETTE[1]}), row=2, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
    fig.update_layout(title=_title(title, is_synthetic, simulated=False))
    return _layout(fig, 480)


def bar_compare(df: pd.DataFrame, x: str, y: str, color: str | None, title: str, is_synthetic: bool,
                error_y: str | None = None) -> go.Figure:
    fig = go.Figure()
    groups = [(None, df)] if color is None else list(df.groupby(color, sort=False))
    for k, (name, g) in enumerate(groups):
        fig.add_trace(go.Bar(x=g[x], y=g[y], name=str(name) if name is not None else y,
                             marker_color=PALETTE[k % len(PALETTE)],
                             error_y={"type": "data", "array": g[error_y]} if error_y else None))
    fig.update_layout(barmode="group", title=_title(title, is_synthetic))
    return _layout(fig)
