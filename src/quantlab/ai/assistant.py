"""AI Research Assistant: explains and compares *stored* results.

Pipeline: read-only tools -> facts (with sources) -> deterministic draft (always produced)
-> optional LLM rewrite -> numeric verification -> answer. If an LLM answer contains any number
that cannot be traced to a stored fact, it is rejected and the deterministic draft is returned.
The assistant has no order tool and never imports the paper broker.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from quantlab.ai.provider import LLMProvider, provider_from_env
from quantlab.ai.tools import ReadOnlyResearchTools
from quantlab.core.errors import QuantLabError
from quantlab.core.labels import NOT_ADVICE

ORDER_INTENT = re.compile(r"\b(buy|sell|place|submit|execute|send)\b.{0,40}\b(order|orders|trade|trades|shares|position)\b",
                          re.I)
SYSTEM_PROMPT = (
    "You are QuantLab's research assistant. You explain stored backtest and experiment results.\n"
    "Rules:\n"
    "1. Use ONLY numbers that appear in FACTS (you may reformat a fraction as a percentage). Never invent, "
    "estimate or compute new numbers.\n"
    "2. Results are simulations; always state the label and data label. If data is synthetic, say so plainly.\n"
    "3. Never give investment advice and never suggest placing orders. You have no ability to trade.\n"
    "4. Mention statistical uncertainty (confidence intervals) when present.\n"
    "5. Be concise (under 250 words), use short bullet points."
)


@dataclass
class AssistantResponse:
    text: str
    mode: str  # deterministic | llm | refused
    task: str
    sources: list[str]
    facts: dict[str, Any]
    verification: dict[str, Any] = field(default_factory=dict)
    provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(x: float | None, d: int = 2) -> str:
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _pp(x: float | None, d: int = 2) -> str:
    return "n/a" if x is None else f"{x * 100:+.{d}f} pp"


def _num(x: float | None, d: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{d}f}"


# ------------------------------------------------------------------ verification
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|\b(?=[0-9]*[a-f])[0-9a-f]{8}\b", re.I)
_UUID_FULL = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_NUM = re.compile(r"(?<![A-Za-z0-9_])[-+−]?\d[\d,]*(?:\.\d+)?\s*(%|pp)?")


def _collect_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        out.append(float(obj))
    elif isinstance(obj, str) and _UUID_FULL.match(obj) and obj[:8].isdigit():
        out.append(float(obj[:8]))  # short run ids shown in tables
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _collect_numbers(v, out)


def verify_numbers(text: str, context: dict[str, Any]) -> dict[str, Any]:
    """Every number in ``text`` must match a stored value (as-is or x100 for %), within the
    rounding implied by how many decimals the text shows."""
    allowed: list[float] = []
    _collect_numbers(context, allowed)
    allowed += [float(i) for i in range(0, 11)] + [95.0, 252.0, 100.0]
    clean = _UUID.sub(" ", _DATE.sub(" ", text))
    clean = re.sub(r"\b[A-Z]{2,5}\b", " ", clean)  # tickers
    unverified, checked = [], 0
    for m in _NUM.finditer(clean):
        tok = m.group(0).strip()
        raw = tok.replace("%", "").replace("pp", "").replace(",", "").replace("−", "-").replace("+", "").strip()
        try:
            v = abs(float(raw))
        except ValueError:
            continue
        checked += 1
        dec = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
        ok = any(abs(v - abs(a)) <= tol or abs(v - abs(a) * 100) <= tol for a in allowed)
        if not ok:
            unverified.append(tok)
    return {"verified": not unverified, "numbers_checked": checked, "unverified_numbers": unverified}


# ------------------------------------------------------------------ deterministic writers
def explain_backtest(f: dict[str, Any]) -> str:
    x = f["facts"]
    lines = [f"**{f['strategy']}** on {', '.join(f['symbols'])}, {f['window'][0]} → {f['window'][1]}  ",
             f"_{f['label']} · {f['data_label']}_", ""]
    lines.append(f"- Net total return {_pct(x.get('net.total_return'))}, CAGR {_pct(x.get('net.cagr'))}; "
                 f"without costs the total return would be {_pct(x.get('gross.total_return'))} "
                 f"(cost drag {_pp(x.get('derived.cost_drag_total_return'))}; commission "
                 f"{_num(x.get('net.total_commission'))} USD, slippage {_num(x.get('net.total_slippage_cost'))} USD).")
    lines.append(f"- Risk: annual volatility {_pct(x.get('net.annual_volatility'))}, maximum drawdown "
                 f"{_pct(x.get('net.max_drawdown'))} lasting {_num(x.get('net.max_drawdown_duration_sessions'), 0)} "
                 f"sessions; Sharpe {_num(x.get('net.sharpe'))} (95% CI {_num(x.get('net.sharpe_ci95_low'))} to "
                 f"{_num(x.get('net.sharpe_ci95_high'))}, risk-free {_pct(f['risk_free_rate'])}).")
    if "benchmark.cagr" in x:
        lines.append(f"- Versus {f['benchmark_symbol']} buy & hold: benchmark CAGR {_pct(x.get('benchmark.cagr'))}, "
                     f"excess CAGR {_pp(x.get('relative.excess_cagr'))}, beta {_num(x.get('relative.beta'))}, "
                     f"benchmark max drawdown {_pct(x.get('benchmark.max_drawdown'))}.")
    lines.append(f"- Activity: {_num(x.get('net.n_fills'), 0)} fills, turnover {_num(x.get('net.turnover_annual'))}x "
                 f"per year, time in market {_pct(x.get('net.time_in_market'), 1)}.")
    lo, hi = x.get("net.sharpe_ci95_low"), x.get("net.sharpe_ci95_high")
    if lo is not None and hi is not None and lo <= 0 <= hi:
        lines.append("- The Sharpe confidence interval includes zero: risk-adjusted performance is not "
                     "statistically distinguishable from zero over this sample.")
    if f["is_synthetic"]:
        lines.append("- These numbers come from SYNTHETIC data and say nothing about real markets.")
    for w in f["warnings"]:
        lines.append(f"- Warning: {w}")
    lines.append(f"\n_{NOT_ADVICE} Backtests are simulations; past results do not predict future returns._")
    return "\n".join(lines)


def stats_lines(st: dict[str, Any]) -> list[str]:
    out = [f"- Statistical tests on {st['n_obs']} common sessions ({st['window'][0]} → {st['window'][1]}), "
           f"reference {st['reference']}:"]
    for p in st["pairwise"]:
        verdict = "not significant" if (p["jk_p_value"] or 1) >= 0.05 else "significant at 5%"
        out.append(f"  - {p['strategy']}: Sharpe difference {_num(p['sharpe_diff'])}, Jobson-Korkie p = "
                   f"{_num(p['jk_p_value'], 3)} ({verdict}); bootstrap 95% CI {_num(p['bootstrap_ci_low'])} to "
                   f"{_num(p['bootstrap_ci_high'])}.")
    for k, d in st["deflated_sharpe"].items():
        out.append(f"  - {k}: probability true Sharpe > 0 is {_num(d['psr_vs_zero'], 3)}; deflated for "
                   f"{d['n_trials']} trials it is {_num(d['deflated_sharpe'], 3)}.")
    return out


def compare_backtests(fs: list[dict[str, Any]]) -> str:
    rows = ["| run | strategy | symbols | window | CAGR | Sharpe (95% CI) | max DD | turnover |", "|---|---|---|---|---|---|---|---|"]
    for f in fs:
        x = f["facts"]
        rows.append(f"| {f['run_id'][:8]} | {f['strategy']} | {', '.join(f['symbols'])} | {f['window'][0]}→{f['window'][1]} | "
                    f"{_pct(x.get('net.cagr'))} | {_num(x.get('net.sharpe'))} ({_num(x.get('net.sharpe_ci95_low'))} to "
                    f"{_num(x.get('net.sharpe_ci95_high'))}) | {_pct(x.get('net.max_drawdown'))} | "
                    f"{_num(x.get('net.turnover_annual'))} |")
    best = max(fs, key=lambda f: f["facts"].get("net.sharpe") if f["facts"].get("net.sharpe") is not None else -1e9)
    lines = ["\n".join(rows), ""]
    lines.append(f"- Highest point-estimate Sharpe: {best['strategy']} ({best['run_id'][:8]}), "
                 f"{_num(best['facts'].get('net.sharpe'))}.")
    overlaps = []
    for f in fs:
        if f is best:
            continue
        if (f["facts"].get("net.sharpe_ci95_high") or -1e9) >= (best["facts"].get("net.sharpe_ci95_low") or 1e9):
            overlaps.append(f"{f['strategy']} ({f['run_id'][:8]})")
    if overlaps:
        lines.append(f"- Its Sharpe confidence interval overlaps with {', '.join(overlaps)}: the ranking is not "
                     "statistically reliable.")
    if len({f["is_synthetic"] for f in fs}) > 1:
        lines.append("- WARNING: mixing synthetic and real-data runs; they are not comparable.")
    if len({tuple(f["window"]) for f in fs}) > 1:
        lines.append("- WARNING: runs cover different windows; differences partly reflect the period.")
    if any(f["is_synthetic"] for f in fs):
        lines.append("- Synthetic data involved: results describe the fixture, not real markets.")
    lines.append(f"\n_{NOT_ADVICE} Higher backtest Sharpe does not prove exploitable alpha._")
    return "\n".join(lines)


def summarize_experiment(f: dict[str, Any]) -> str:
    lines = [f"**{f['code']}** on {f['symbol']}, window {f['window'][0]} → {f['window'][1]}, "
             f"{len(f['seeds'])} seeds  ", f"_{f['label']} · {f['data_label']}_", "", f"Hypothesis: {f['hypothesis']}", ""]
    by_strat: dict[str, list[dict]] = {}
    for a in f["aggregate"]:
        if a["scenario"] != "clean":
            by_strat.setdefault(a["strategy"], []).append(a)
    for sid, rows in by_strat.items():
        rows = [r for r in rows if r.get("delta_total_return_mean") is not None]
        if not rows:
            continue
        worst = min(rows, key=lambda r: r["delta_total_return_mean"])
        tradable = [r for r in rows if not r["research_only"]]
        best_t = max(tradable, key=lambda r: r["delta_total_return_mean"]) if tradable else None
        lines.append(f"- {sid}: largest mean loss vs clean in {worst['scenario']} / {worst['policy']} "
                     f"({_pct(worst['delta_total_return_mean'])}, 95% CI {_pct(worst['delta_total_return_ci_low'])} to "
                     f"{_pct(worst['delta_total_return_ci_high'])}).")
        if best_t:
            lines.append(f"  Smallest mean impact among tradable policies: {best_t['scenario']} / {best_t['policy']} "
                         f"({_pct(best_t['delta_total_return_mean'])}).")
    if f["research_only_policies"]:
        lines.append(f"- {', '.join(f['research_only_policies'])}: research-only counterfactual, not a tradable policy.")
    lines.append("- Wide confidence intervals with few seeds mean many differences are not statistically reliable.")
    if f["is_synthetic"]:
        lines.append("- SYNTHETIC data: findings describe the fixture, not real markets.")
    lines.append(f"\n_{NOT_ADVICE}_")
    return "\n".join(lines)


def summarize_walkforward(f: dict[str, Any]) -> str:
    x = f["facts"]
    lines = [f"**Walk-forward validation of {f['strategy']}** on {', '.join(f['symbols'])}, out-of-sample "
             f"{f['window'][0]} → {f['window'][1]}  ", f"_{f['label']} · {f['data_label']}_", ""]
    lines.append(f"- {f['n_folds']} folds, {f['n_configs']} parameter sets per fold, selection by train-window "
                 f"{f['objective']} only.")
    lines.append(f"- Out-of-sample: total return {_pct(x.get('oos.total_return'))}, CAGR {_pct(x.get('oos.cagr'))}, "
                 f"Sharpe {_num(x.get('oos.sharpe'))} (95% CI {_num(x.get('oos.sharpe_ci95_low'))} to "
                 f"{_num(x.get('oos.sharpe_ci95_high'))}), max drawdown {_pct(x.get('oos.max_drawdown'))}.")
    if "benchmark.total_return" in x:
        lines.append(f"- Benchmark buy & hold over the same span: total return {_pct(x.get('benchmark.total_return'))}, "
                     f"Sharpe {_num(x.get('benchmark.sharpe'))}.")
    if x.get("mean_oos_percentile_rank") is not None:
        lines.append(f"- Average out-of-sample percentile rank of the chosen parameters: "
                     f"{_num(x.get('mean_oos_percentile_rank'))} (a value near one half means selection added no information).")
    distinct = {json.dumps(p, sort_keys=True) for p in f["chosen_params"]}
    lines.append(f"- {len(distinct)} distinct parameter set(s) were chosen across folds.")
    if f["is_synthetic"]:
        lines.append("- SYNTHETIC data: findings describe the fixture, not real markets.")
    lines.append(f"\n_{NOT_ADVICE}_")
    return "\n".join(lines)


# ------------------------------------------------------------------ assistant
class ResearchAssistant:
    def __init__(self, db_path, provider: LLMProvider | None | str = "env"):
        self.tools = ReadOnlyResearchTools(db_path)
        self.provider = provider_from_env() if provider == "env" else provider

    def _finish(self, task: str, draft: str, context: dict[str, Any], sources: list[str],
                question: str | None) -> AssistantResponse:
        facts = context if isinstance(context, dict) else {"items": context}
        det_check = verify_numbers(draft, context)
        if self.provider is None:
            return AssistantResponse(draft, "deterministic", task, sources, facts, det_check)
        user = (f"TASK: {task}\nQUESTION: {question or 'Explain the results.'}\n\nFACTS (JSON):\n"
                f"{json.dumps(context, default=str)[:60000]}\n\nDRAFT (all numbers verified):\n{draft}")
        try:
            out = self.provider.complete(SYSTEM_PROMPT, user)
        except Exception as exc:  # provider failure -> deterministic answer, failure reported
            return AssistantResponse(draft, "deterministic", task, sources, facts,
                                     {**det_check, "llm_error": f"{type(exc).__name__}"}, self.provider.name)
        check = verify_numbers(out, context)
        if not check["verified"] or (ORDER_INTENT.search(out or "") and "cannot" not in out.lower()):
            note = (f"\n\n_(LLM answer discarded: it contained numbers not found in stored data "
                    f"{check['unverified_numbers']} or order language. Showing the deterministic analysis.)_")
            return AssistantResponse(draft + note, "deterministic", task, sources, facts,
                                     {**check, "llm_rejected": True}, self.provider.name)
        return AssistantResponse(out, "llm", task, sources, facts, check, self.provider.name)

    def _refuse_orders(self, task: str, question: str | None) -> AssistantResponse | None:
        if question and ORDER_INTENT.search(question):
            return AssistantResponse(
                "I can't place, submit or recommend orders. I'm a read-only research assistant: I can explain "
                "stored backtests and experiments. Paper orders are entered manually on the Paper Trading page.",
                "refused", task, [], {}, {"verified": True, "numbers_checked": 0, "unverified_numbers": []})
        return None

    def explain(self, run_id: str, question: str | None = None) -> AssistantResponse:
        if r := self._refuse_orders("explain", question):
            return r
        f = self.tools.get_backtest_facts(run_id)
        x = f["facts"]
        if "gross.total_return" in x and "net.total_return" in x:
            x["derived.cost_drag_total_return"] = x["gross.total_return"] - x["net.total_return"]
        return self._finish("explain_backtest", explain_backtest(f), f, [f["source"]], question)

    def compare(self, run_ids: list[str], question: str | None = None) -> AssistantResponse:
        if r := self._refuse_orders("compare", question):
            return r
        fs = self.tools.compare_backtests(run_ids)
        text = compare_backtests(fs)
        context: dict[str, Any] = {"runs": fs}
        try:
            st = self.tools.compare_statistics(run_ids)
            context["statistics"] = st
            body, _, tail = text.rpartition("\n\n_")
            text = body + "\n" + "\n".join(stats_lines(st)) + "\n\n_" + tail
        except QuantLabError as exc:  # e.g. runs on non-overlapping windows
            text += f"\n\n(Statistical tests not available: {exc})"
        return self._finish("compare_backtests", text, context, [f["source"] for f in fs], question)

    def summarize_experiment(self, experiment_id: str, question: str | None = None) -> AssistantResponse:
        if r := self._refuse_orders("summarize", question):
            return r
        f = self.tools.get_experiment_facts(experiment_id)
        draft = summarize_walkforward(f) if f["kind"] == "walkforward" else summarize_experiment(f)
        return self._finish("summarize_experiment", draft, f, [f["source"]], question)
