"""Streamlit helpers shared by dashboard pages (UI layer only — no domain logic here)."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from quantlab.context import AppContext, build_context
from quantlab.core.labels import BACKTEST_LABEL, NOT_ADVICE, SYNTHETIC_LABEL
from quantlab.domain.models import DataSnapshot


@st.cache_resource(show_spinner=False)
def _context(data_dir: str | None) -> AppContext:
    return build_context(data_dir)


def get_ctx() -> AppContext:
    return _context(os.environ.get("QUANTLAB_DATA_DIR"))


def safety_banner() -> None:
    st.caption(f"🔒 Live trading disabled · local research & simulation only · {NOT_ADVICE}")


def synthetic_banner(is_synthetic: bool) -> None:
    if is_synthetic:
        st.warning(f"**{SYNTHETIC_LABEL}** — seeded fixture data for offline demos and tests.", icon="⚠️")


def backtest_banner(is_synthetic: bool) -> None:
    st.info(f"**{BACKTEST_LABEL}**", icon="🧪")
    synthetic_banner(is_synthetic)


def not_implemented(feature: str, detail: str = "") -> None:
    st.error(f"**Not implemented yet:** {feature}. {detail}  \nThis page shows no simulated or placeholder results.")


def snapshot_label(s: DataSnapshot, quality: str | None = None) -> str:
    tag = "SYNTHETIC" if s.is_synthetic else "REAL"
    derived = " · derived" if s.parent_id else ""
    q = f" · {quality}" if quality else ""
    return f"[{tag}] {', '.join(s.symbols)} {s.start}→{s.end} · {s.provider}{derived}{q} · {s.id[:8]}"


def snapshot_selector(ctx: AppContext, key: str, include_derived: bool = False,
                      require_tradable: bool = False) -> DataSnapshot | None:
    snaps = ctx.snapshots.list(include_derived=include_derived)
    if require_tradable:
        snaps = [s for s in snaps if ctx.snapshots.quality_status(s.id) != "FAILED"]
    if not snaps:
        st.info("No data snapshots yet. Ingest data on **Market Explorer** (real via yfinance, or the labelled "
                "synthetic fixture for offline use).")
        return None
    labels = {snapshot_label(s, ctx.snapshots.quality_status(s.id)): s for s in snaps}
    choice = st.selectbox("Data snapshot", list(labels), key=key)
    return labels[choice]


def fmt_pct(v: Any) -> str:
    return "n/a" if v is None else f"{v * 100:.2f}%"


def fmt_num(v: Any, d: int = 2) -> str:
    return "n/a" if v is None else f"{v:,.{d}f}"


def df_download(df: pd.DataFrame, label: str, filename: str, key: str, header: str | None = None) -> None:
    data = (f"# {header}\n" if header else "") + df.to_csv(index=False)
    st.download_button(label, data=data.encode(), file_name=filename, mime="text/csv", key=key)
