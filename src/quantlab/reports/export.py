"""Report exports (CSV / JSON / Markdown / HTML). Every number comes from stored artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.backtest.service import get_run, load_artifact
from quantlab.context import AppContext
from quantlab.core.errors import InputValidationError
from quantlab.core.labels import BACKTEST_LABEL, NOT_ADVICE, SYNTHETIC_LABEL
from quantlab.reports import charts

FORMATS = ("json", "csv", "md", "html")

METRIC_ROWS = [
    ("total_return", "Total return", "pct"),
    ("cagr", "CAGR", "pct"),
    ("annual_volatility", "Annual volatility", "pct"),
    ("sharpe", "Sharpe", "num"),
    ("sharpe_ci95_low", "Sharpe 95% CI low", "num"),
    ("sharpe_ci95_high", "Sharpe 95% CI high", "num"),
    ("sortino", "Sortino", "num"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("max_drawdown_duration_sessions", "Max DD duration (sessions)", "int"),
    ("turnover_annual", "Turnover (x/yr)", "num"),
    ("time_in_market", "Time in market", "pct"),
    ("n_fills", "Fills", "int"),
    ("total_commission", "Commission (USD)", "usd"),
    ("total_slippage_cost", "Slippage cost (USD)", "usd"),
    ("total_dividends", "Dividends received (USD)", "usd"),
    ("final_nav", "Final NAV (USD)", "usd"),
]


def fmt(v: Any, kind: str) -> str:
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "usd":
        return f"{v:,.2f}"
    return f"{v:.3f}"


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    cols = {k: metrics.get(k) for k in ("net", "gross", "benchmark") if metrics.get(k)}
    rows = []
    for key, label, kind in METRIC_ROWS:
        rows.append({"metric": label, **{c: fmt(m.get(key), kind) for c, m in cols.items()}})
    return pd.DataFrame(rows)


def backtest_markdown(run: dict[str, Any]) -> str:
    r = run["result"]
    lines = [f"# Backtest report — {r['strategy']['name']} ({', '.join(r['symbols'])})", ""]
    lines.append(f"**{BACKTEST_LABEL}**  ")
    if r["is_synthetic"]:
        lines.append(f"**{SYNTHETIC_LABEL}**  ")
    lines.append(f"_{NOT_ADVICE}_")
    lines += ["", "## Provenance", ""]
    s = r["snapshot"]
    lines += [
        f"- Run ID: `{r['run_id']}`",
        f"- Snapshot: `{s['id']}` · provider `{s['provider']}` · price basis `{s['price_basis']}`",
        f"- Dataset hash: `{s['dataset_hash']}`",
        f"- Config hash: `{r['config_hash']}` · code version `{r.get('code_version')}` · seed {r['seed']}",
        f"- Window: {r['metrics']['net']['start']} → {r['metrics']['net']['end']} "
        f"({r['metrics']['net']['n_sessions']} sessions)",
        f"- Strategy parameters: `{json.dumps(r['strategy']['parameters'])}`",
        f"- Costs: commission {r['config']['commission_bps']} bps (min {r['config']['min_commission']} USD), "
        f"slippage {r['config']['slippage_bps']} bps · price mode `{r['config']['price_mode']}` · "
        f"risk-free {r['config']['risk_free_rate']:.2%}",
        "",
        "## Metrics (net of costs, gross, benchmark buy-and-hold)",
        "",
    ]
    lines.append(df_to_markdown(metrics_table(r["metrics"])))
    rel = r["metrics"].get("relative") or {}
    if rel:
        lines += ["", "## Relative to benchmark", ""]
        for k, v in rel.items():
            lines.append(f"- {k}: {fmt(v, 'pct' if 'cagr' in k or 'return' in k or 'tracking' in k else 'num')}")
    lines += ["", "## Assumptions", ""] + [f"- {a}" for a in r["assumptions"]]
    if r["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in r["warnings"]]
    return "\n".join(lines) + "\n"


def export_backtest(ctx: AppContext, run_id: str, fmt_: str, out_dir: str | Path | None = None) -> list[Path]:
    if fmt_ not in FORMATS:
        raise InputValidationError(f"format must be one of {FORMATS}")
    run = get_run(ctx, run_id)
    if "result" not in run:
        raise InputValidationError(f"run {run_id} has no completed result to export")
    r = run["result"]
    out = Path(out_dir or ctx.settings.exports_dir) / f"backtest_{run['id']}"
    out.mkdir(parents=True, exist_ok=True)
    tag = "SYNTHETIC_" if r["is_synthetic"] else ""
    paths: list[Path] = []
    if fmt_ == "json":
        p = out / f"{tag}result.json"
        p.write_text(json.dumps(r, indent=2, sort_keys=True))
        paths.append(p)
    elif fmt_ == "csv":
        header = f"# {BACKTEST_LABEL}" + (f" | {SYNTHETIC_LABEL}" if r["is_synthetic"] else "") + f" | run {run['id']}\n"
        for name in ("equity", "trades", "orders"):
            df = load_artifact(ctx, run_id, name)
            p = out / f"{tag}{name}.csv"
            p.write_text(header + df.to_csv(index=False))
            paths.append(p)
    elif fmt_ == "md":
        p = out / f"{tag}report.md"
        p.write_text(backtest_markdown(run))
        paths.append(p)
    else:
        p = out / f"{tag}report.html"
        p.write_text(backtest_html(ctx, run))
        paths.append(p)
    return paths


def backtest_html(ctx: AppContext, run: dict[str, Any]) -> str:
    r = run["result"]
    eq = load_artifact(ctx, run["id"], "equity").set_index("session")["nav"]
    curves = {r["strategy"]["name"]: eq}
    try:
        b = load_artifact(ctx, run["id"], "benchmark_equity").set_index("session")["nav"]
        curves[f"{r['benchmark_symbol']} buy & hold"] = b
    except Exception:  # benchmark is optional; absence is reported in warnings
        pass
    fig = charts.equity_and_drawdown(curves, r["is_synthetic"], f"{r['strategy']['name']} — equity & drawdown")
    md = backtest_markdown(run)
    table = metrics_table(r["metrics"]).to_html(index=False, border=0)
    banner = f"<div class='banner'>{html.escape(BACKTEST_LABEL)}"
    if r["is_synthetic"]:
        banner += f" · <b>{html.escape(SYNTHETIC_LABEL)}</b>"
    banner += "</div>"
    body_md = "<pre>" + html.escape(md) + "</pre>"
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>QuantLab backtest {run['id']}</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;
color:#222;background:#FAF8F4}} .banner{{background:#7A2E2E;color:#fff;padding:8px 12px;border-radius:4px}}
table{{border-collapse:collapse}} td,th{{padding:4px 10px;border-bottom:1px solid #ddd;text-align:right}}
td:first-child,th:first-child{{text-align:left}} pre{{white-space:pre-wrap;background:#fff;padding:12px}}</style>
</head><body>{banner}<h1>{html.escape(r['strategy']['name'])} — {html.escape(', '.join(r['symbols']))}</h1>
{fig.to_html(full_html=False, include_plotlyjs=True)}
<h2>Metrics</h2>{table}<h2>Full report</h2>{body_md}</body></html>"""
