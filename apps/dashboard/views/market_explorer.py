from datetime import date

import pandas as pd
import streamlit as st

from quantlab.core.errors import QuantLabError
from quantlab.core.labels import data_label
from quantlab.data.ingestion import ingest
from quantlab.data.providers import YF_TERMS_WARNING, YFinanceProvider, get_provider
from quantlab.data.refresh import lineage, refresh_snapshot
from quantlab.reports import charts
from quantlab.storage import duckdb_store
from quantlab.ui.common import df_download, get_ctx, safety_banner, snapshot_selector, synthetic_banner

ctx = get_ctx()
st.title("Market Explorer")
safety_banner()

with st.expander("Ingest data", expanded=not ctx.snapshots.list()):
    with st.form("ingest"):
        provider = st.radio("Provider", ["yfinance", "synthetic"], horizontal=True,
                            help="yfinance = real Yahoo data (needs internet). synthetic = seeded offline fixture.")
        symbols = st.multiselect("Symbols", ctx.settings.base["universe"], default=ctx.settings.base["universe"])
        c1, c2, c3 = st.columns(3)
        start = c1.date_input("Start", date(2015, 1, 1), min_value=date(1995, 1, 1))
        end = c2.date_input("End", date.today() if provider == "yfinance" else date(2025, 12, 31))
        seed = c3.number_input("Synthetic seed", value=7, step=1)
        st.caption(YF_TERMS_WARNING)
        submitted = st.form_submit_button("Ingest")
    if submitted:
        try:
            with st.spinner("Fetching and validating…"):
                res = ingest(symbols, str(start), str(end), get_provider(provider, int(seed)), ctx.snapshots, ctx.db)
            msg = "created" if res.created else "already existed (identical content; deduplicated)"
            st.success(f"Snapshot {res.snapshot.id} {msg}. Quality: {res.quality.status} {res.quality.counts}")
            synthetic_banner(res.snapshot.is_synthetic)
        except QuantLabError as exc:
            st.error(f"Ingest failed — no snapshot created and no trading action taken: {exc}")

snap = snapshot_selector(ctx, "mx_snap", include_derived=True)
if snap is None:
    st.stop()
if not snap.is_synthetic and snap.parent_id is None:
    if st.button("Refresh from provider (new snapshot + revision check)"):
        try:
            with st.spinner("Re-downloading full history…"):
                rr = refresh_snapshot(ctx, snap.id, YFinanceProvider())
            t = rr.diff["totals"]
            msg = (f"{rr.diff['status']}: new snapshot {rr.new_snapshot_id[:8]} · appended {t['appended']} · "
                   f"raw revisions {t['raw_revised']} · removed {t['removed']} · adj-only {t['adj_only_revised']}")
            (st.warning if rr.diff["status"] == "REVISIONS_DETECTED" else st.success)(msg)
        except QuantLabError as exc:
            st.error(f"Refresh failed — existing snapshot untouched: {exc}")
synthetic_banner(snap.is_synthetic)
st.caption(f"{data_label(snap.is_synthetic)} · provider `{snap.provider}` · price basis `{snap.price_basis}` · "
           f"hash `{snap.dataset_hash[:16]}…` · retrieved {snap.retrieved_at[:19]}")

tab_q, tab_p, tab_a, tab_s, tab_l = st.tabs(["Data quality", "Prices", "Corporate actions", "Summary", "Lineage"])
with tab_l:
    rows = lineage(ctx, snap.id)
    if not rows:
        st.write("No refresh history for this snapshot.")
    for r in rows:
        st.markdown(f"**{r['diff']['status']}** · {r['base_snapshot_id'][:8]} → {r['new_snapshot_id'][:8]} · "
                    f"{r['created_at'][:19]}")
        st.json(r["diff"]["totals"])
with tab_q:
    status = ctx.snapshots.quality_status(snap.id)
    issues = ctx.snapshots.quality_issues(snap.id)
    if status is None:
        st.warning("No quality report stored for this snapshot.")
    elif status == "FAILED":
        st.error(f"Quality status: {status} — backtests and paper trading are blocked on this snapshot.")
    else:
        st.success(f"Quality status: {status}")
    if issues:
        df = pd.DataFrame(issues)
        df["details"] = df["details"].astype(str)
        st.dataframe(df, hide_index=True, width="stretch")
    elif status is not None:
        st.write("No issues found.")

with tab_p:
    c1, c2, c3 = st.columns(3)
    sym = c1.selectbox("Symbol", snap.symbols, key="mx_sym")
    d0 = c2.date_input("From", pd.Timestamp(snap.start).date(), key="mx_d0")
    d1 = c3.date_input("To", pd.Timestamp(snap.end).date(), key="mx_d1")
    kind = st.radio("Chart", ["line", "candlestick"], horizontal=True)
    bars = duckdb_store.query_bars(snap.storage_uri, [sym], str(d0), str(d1))
    if bars.empty:
        st.info("No bars in this range.")
    else:
        st.plotly_chart(charts.price_chart(bars, sym, snap.is_synthetic, kind), width="stretch")
        df_download(bars, "Download bars CSV", f"{sym}_{snap.id[:8]}.csv", "mx_dl", header=data_label(snap.is_synthetic))

with tab_a:
    bars = duckdb_store.query_bars(snap.storage_uri)
    acts = bars[(bars["dividend"] > 0) | (bars["split_ratio"] != 1.0)][["symbol", "session_date", "dividend", "split_ratio", "close"]]
    st.dataframe(acts, hide_index=True, width="stretch")

with tab_s:
    st.dataframe(duckdb_store.symbol_summary(snap.storage_uri), hide_index=True, width="stretch")
    if snap.parent_id:
        st.write("Derived from parent snapshot", snap.parent_id)
        st.json(snap.manifest)
