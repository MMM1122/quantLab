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
    cols: dict[str, dict[str, Any]] = {k: metrics[k] for k in ("net", "gross", "benchmark") if metrics.get(k)}
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


# ------------------------------------------------------------------ experiments (DQ-001)
AGG_COLS = [
    ("strategy", "strategy", None), ("scenario", "scenario", None), ("policy", "policy", None),
    ("n_seeds", "seeds", "int"), ("delta_total_return_mean", "Δ total return", "pct"),
    ("delta_total_return_ci_low", "CI low", "pct"), ("delta_total_return_ci_high", "CI high", "pct"),
    ("delta_sharpe_mean", "Δ Sharpe", "num"), ("delta_max_drawdown_mean", "Δ max DD", "pct"),
    ("n_trades_mean", "trades", "num"), ("missed_signals_mean", "missed", "num"),
    ("spurious_signals_mean", "spurious", "num"), ("signal_agreement_mean", "agreement", "pct"),
    ("refused_decisions_mean", "refused", "num"),
]


def experiment_table(aggregate: list[dict]) -> pd.DataFrame:
    rows = []
    for r in aggregate:
        row = {}
        for key, label, kind in AGG_COLS:
            v = r.get(key)
            row[label] = v if kind is None else fmt(v, kind)
        if r.get("research_only"):
            row["policy"] = f"{row['policy']} (research-only)"
        rows.append(row)
    return pd.DataFrame(rows)


def experiment_markdown(exp: dict[str, Any]) -> str:
    r = exp["results"]
    cfg = r["design"]["config"]
    lines = [f"# {r['code']} — data-quality experiment report", "", f"**{r['label']}**  "]
    if r["is_synthetic"]:
        lines.append(f"**{SYNTHETIC_LABEL}**  ")
    lines += [f"_{NOT_ADVICE}_", "", "## Hypothesis", "", r["hypothesis"], "", "## Design", ""]
    lines += [
        f"- Experiment ID: `{r['experiment_id']}` · code version `{r['design'].get('code_version')}`",
        f"- Source snapshot: `{r['design']['snapshot']['id']}` (hash `{r['design']['snapshot']['dataset_hash']}`), "
        f"verified unchanged after the run: {r['source_snapshot_verified_unchanged']}",
        f"- Symbol: {cfg['symbol']} · strategies: {', '.join(cfg['strategies'])}",
        f"- Evaluation window: {r['evaluation_window'][0]} → {r['evaluation_window'][1]}",
        f"- Seeds: {cfg['seeds']} · policies: {', '.join(cfg['policies'])}",
        f"- Costs: commission {cfg['commission_bps']} bps, slippage {cfg['slippage_bps']} bps, "
        f"capital {cfg['initial_cash']:,.0f} USD",
        "- Scenarios: " + "; ".join(
            f"{s['name']} (missing {s['missing_rate']:.0%}, delayed {s['delay_rate']:.0%} lag≤{s['delay_max_lag']}, "
            f"corrupted {s['corrupt_rate']:.0%})" for s in cfg["scenarios"]),
        f"- Backtests executed: {r['n_runs']} (+{len(r['baseline'])} clean baselines)",
        "", "## Missing-data rules", "",
    ]
    lines += [f"- {x}" for x in r["design"]["missing_data_rules"]]
    lines += ["", "## Clean baselines", ""]
    for sid, b in r["baseline"].items():
        m = b["metrics"]
        ci = b.get("sharpe_bootstrap_ci95") or [None, None]
        lines.append(f"- {sid}: total return {fmt(m['total_return'], 'pct')}, CAGR {fmt(m['cagr'], 'pct')}, "
                     f"Sharpe {fmt(m['sharpe'], 'num')} (block-bootstrap 95% CI {fmt(ci[0], 'num')} … "
                     f"{fmt(ci[1], 'num')}), max DD {fmt(m['max_drawdown'], 'pct')}, trades {b['n_trades']}")
    if r.get("detection"):
        det = pd.DataFrame(r["detection"]).groupby("scenario").sum(numeric_only=True).reset_index()
        lines += ["", "## Did validation detect the injected problems?", "", df_to_markdown(det.drop(columns=["seed"]))]
    lines += ["", "## Results (mean across seeds; Δ vs clean baseline; 95% Student-t CI)", "",
              df_to_markdown(experiment_table(r["aggregate"]))]
    lines += ["", "## Findings (generated from the numbers above)", ""] + [f"- {f}" for f in r["findings"]]
    lines += ["", "## Limitations", "",
              "- Results are specific to this snapshot, symbol, window and parameter set; they are not general claims.",
              "- With few seeds, confidence intervals are wide; 'not distinguishable from zero' is the honest reading.",
              "- Delays are modelled as a stale bar delivered for the session (value of t-k, as-of stamp t-k).",
              "- Fills use pristine prices; real outages may also affect the ability to trade."]
    if r["is_synthetic"]:
        lines.append(f"- {SYNTHETIC_LABEL}: numbers describe the synthetic fixture, not real markets.")
    return "\n".join(lines) + "\n"


def experiment_html(exp: dict[str, Any]) -> str:
    r = exp["results"]
    agg = pd.DataFrame(r["aggregate"])
    agg = agg[agg.scenario != "clean"].copy()
    figs: list[str] = []
    for sid, g in agg.groupby("strategy", sort=False):
        g = g.assign(err=(g["delta_total_return_ci_high"] - g["delta_total_return_mean"]).fillna(0) * 100,
                     y=g["delta_total_return_mean"] * 100)
        figs.append(charts.bar_compare(g, "scenario", "y", "policy", f"{sid}: Δ total return vs clean (pp, mean ± 95% CI)",
                                       r["is_synthetic"], error_y="err").to_html(full_html=False, include_plotlyjs=not figs))
    md = experiment_markdown(exp)
    banner = f"<div class='banner'>{html.escape(r['label'])}"
    if r["is_synthetic"]:
        banner += f" · <b>{html.escape(SYNTHETIC_LABEL)}</b>"
    banner += "</div>"
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{r['code']} {r['experiment_id']}</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;
color:#222;background:#FAF8F4}} .banner{{background:#7A2E2E;color:#fff;padding:8px 12px;border-radius:4px}}
pre{{white-space:pre-wrap;background:#fff;padding:12px}}</style></head><body>{banner}
<h1>{r['code']} data-quality experiment</h1>{''.join(figs)}<pre>{html.escape(md)}</pre></body></html>"""


def walkforward_markdown(exp: dict[str, Any]) -> str:
    r = exp["results"]
    cfg = r["design"]["config"]
    lines = [f"# Walk-forward validation — {r['strategy']} ({', '.join(r['symbols'])})", "", f"**{r['label']}**  "]
    if r["is_synthetic"]:
        lines.append(f"**{SYNTHETIC_LABEL}**  ")
    lines += [f"_{NOT_ADVICE}_", "", "## Design", "",
              f"- Experiment ID: `{r['experiment_id']}` · code version `{r['design'].get('code_version')}`",
              f"- Snapshot `{r['design']['snapshot']['id']}` (hash `{r['design']['snapshot']['dataset_hash']}`)",
              f"- Train {cfg['train_sessions']} sessions ({'expanding' if cfg['expanding'] else 'rolling'}), test "
              f"{cfg['test_sessions']} sessions, embargo {cfg['embargo_sessions']}",
              f"- Objective: {r['objective']} · selection rule: {r['design']['selection_rule']}",
              f"- Parameter grid ({r['n_configs']} valid sets): `{json.dumps(cfg['param_grid'])}`",
              f"- Costs: commission {cfg['commission_bps']} bps, slippage {cfg['slippage_bps']} bps",
              "", "## Out-of-sample metrics (stitched test windows)", ""]
    m, b = r["oos_metrics"], r.get("benchmark_metrics") or {}
    rows = [{"metric": label, "out-of-sample": fmt(m.get(k), kind),
             "benchmark": fmt(b.get(k), kind) if b else "n/a"} for k, label, kind in METRIC_ROWS]
    lines.append(df_to_markdown(pd.DataFrame(rows)))
    lines += ["", "## Folds", ""]
    fr = pd.DataFrame([{"fold": f["fold"], "train": f"{f['train_start']}→{f['train_end']}",
                        "test": f"{f['test_start']}→{f['test_end']}", "chosen": json.dumps(f["chosen_params"]),
                        f"train {r['objective']}": fmt(f["train_objective"], "num"),
                        f"test {r['objective']}": fmt(f["test_objective"], "num"),
                        "test return": fmt(f["test_total_return"], "pct"),
                        "OOS rank": fmt(f["oos_percentile_rank"], "num")} for f in r["folds"]])
    lines.append(df_to_markdown(fr))
    lines += ["", "## Findings (generated from the numbers above)", ""] + [f"- {x}" for x in r["findings"]]
    lines += ["", "## Assumptions", ""] + [f"- {x}" for x in r["assumptions"]]
    return "\n".join(lines) + "\n"


def walkforward_html(exp: dict[str, Any], oos: pd.Series, bench: pd.Series | None) -> str:
    r = exp["results"]
    curves = {"out-of-sample (chosen params)": oos}
    if bench is not None:
        curves["benchmark buy & hold"] = bench
    fig = charts.equity_and_drawdown(curves, r["is_synthetic"], f"Walk-forward {r['strategy']} — out-of-sample equity")
    banner = f"<div class='banner'>{html.escape(r['label'])}"
    if r["is_synthetic"]:
        banner += f" · <b>{html.escape(SYNTHETIC_LABEL)}</b>"
    banner += "</div>"
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Walk-forward {r['experiment_id']}</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;
color:#222;background:#FAF8F4}} .banner{{background:#7A2E2E;color:#fff;padding:8px 12px;border-radius:4px}}
pre{{white-space:pre-wrap;background:#fff;padding:12px}}</style></head><body>{banner}
<h1>Walk-forward validation</h1>{fig.to_html(full_html=False, include_plotlyjs=True)}
<pre>{html.escape(walkforward_markdown(exp))}</pre></body></html>"""


def export_experiment(ctx: AppContext, experiment_id: str, fmt_: str, out_dir: str | Path | None = None) -> list[Path]:
    from quantlab.research.dq_001 import experiment_runs, get_experiment

    if fmt_ not in FORMATS:
        raise InputValidationError(f"format must be one of {FORMATS}")
    exp = get_experiment(ctx, experiment_id)
    if exp["status"] != "COMPLETED":
        raise InputValidationError(f"experiment {experiment_id} is {exp['status']}; nothing to export")
    r = exp["results"]
    out = Path(out_dir or ctx.settings.exports_dir) / f"experiment_{exp['id']}"
    out.mkdir(parents=True, exist_ok=True)
    tag = "SYNTHETIC_" if r["is_synthetic"] else ""
    header = f"# {r['label']}" + (f" | {SYNTHETIC_LABEL}" if r["is_synthetic"] else "") + f" | {r['code']} {exp['id']}\n"
    paths = []
    if fmt_ == "json":
        p = out / f"{tag}result.json"
        p.write_text(json.dumps(r, indent=2, sort_keys=True))
        paths.append(p)
    elif fmt_ == "csv" and exp["code"] == "WALK-FORWARD":
        folder = Path(exp["artifact_uri"])
        for name, df in (("folds", pd.read_csv(folder / "folds.csv")),
                         ("grid_results", pd.read_csv(folder / "grid_results.csv")),
                         ("oos_equity", pd.read_parquet(folder / "oos_equity.parquet"))):
            p = out / f"{tag}{name}.csv"
            p.write_text(header + df.to_csv(index=False))
            paths.append(p)
    elif fmt_ == "csv":
        for name, df in (("runs", experiment_runs(ctx, exp["id"])), ("aggregate", pd.DataFrame(r["aggregate"])),
                         ("policy_comparison", pd.DataFrame(r["policy_comparison"]))):
            p = out / f"{tag}{name}.csv"
            p.write_text(header + df.to_csv(index=False))
            paths.append(p)
    elif fmt_ == "md":
        p = out / f"{tag}report.md"
        p.write_text(walkforward_markdown(exp) if exp["code"] == "WALK-FORWARD" else experiment_markdown(exp))
        paths.append(p)
    else:
        p = out / f"{tag}report.html"
        if exp["code"] == "WALK-FORWARD":
            oos, bench = load_walkforward_curves(exp)
            p.write_text(walkforward_html(exp, oos, bench))
        else:
            p.write_text(experiment_html(exp))
        paths.append(p)
    return paths


def load_walkforward_curves(exp: dict[str, Any]) -> tuple[pd.Series, pd.Series | None]:
    folder = Path(exp["artifact_uri"])
    oos = pd.read_parquet(folder / "oos_equity.parquet").set_index("session")["nav"]
    bp = folder / "benchmark_equity.parquet"
    bench = pd.read_parquet(bp).set_index("session")["nav"] if bp.exists() else None
    return oos, bench
