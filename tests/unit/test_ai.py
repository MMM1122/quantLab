from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from quantlab.ai.assistant import ResearchAssistant, verify_numbers
from quantlab.ai.provider import AnthropicProvider, OpenAIProvider, provider_from_env
from quantlab.ai.tools import TOOL_NAMES, ReadOnlyResearchTools
from quantlab.backtest.service import run_and_store
from quantlab.core.labels import SYNTHETIC_LABEL
from quantlab.data.ingestion import ingest
from quantlab.data.providers import SyntheticProvider
from quantlab.research.dq_001 import DQ001Config, run_dq001
from quantlab.research.perturbations import Scenario

AI_DIR = Path(__file__).resolve().parents[2] / "src" / "quantlab" / "ai"


@pytest.fixture
def runs(ctx):
    snap = ingest(["SPY", "TLT"], "2017-01-01", "2023-12-29", SyntheticProvider(seed=7), ctx.snapshots, ctx.db).snapshot
    a = run_and_store(ctx, "sma-crossover", snap.id, ["SPY"]).run_id
    b = run_and_store(ctx, "rsi-mean-reversion", snap.id, ["SPY"]).run_id
    return ctx, snap, a, b


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply(user) if callable(self.reply) else self.reply


# ----------------------------------------------------------------- structural safety
def test_ai_package_never_imports_order_code():
    forbidden = ("quantlab.paper", "quantlab.backtest.engine", "quantlab.jobs", "quantlab.api")
    for f in AI_DIR.glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for n in names:
                assert not n.startswith(forbidden), f"{f.name} imports {n}"
        src = f.read_text()
        assert "submit_order" not in src and "PaperBroker" not in src and "apply_execution" not in src


def test_tools_are_read_only_at_database_level(runs):
    ctx, *_ = runs
    tools = ReadOnlyResearchTools(ctx.settings.db_path)
    assert set(TOOL_NAMES) == {"get_backtest_facts", "compare_backtests", "compare_statistics", "get_experiment_facts",
                               "list_recent_backtests"}
    con = tools._conn()
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO orders(id) VALUES ('x')")
    con.close()


# ----------------------------------------------------------------- deterministic mode
def test_deterministic_explain_uses_only_stored_numbers(runs):
    ctx, _, a, _ = runs
    r = ResearchAssistant(ctx.settings.db_path, provider=None).explain(a)
    assert r.mode == "deterministic"
    assert r.verification["verified"] and r.verification["numbers_checked"] > 10
    stored = json.loads((Path(ctx.db.query_one("SELECT artifact_uri FROM backtest_runs WHERE id=?", (a,))["artifact_uri"])
                         / "result.json").read_text())
    assert f"{stored['metrics']['net']['cagr'] * 100:.2f}%" in r.text
    assert SYNTHETIC_LABEL in r.text and "Not investment advice" in r.text
    assert r.sources and "result.json" in r.sources[0]


def test_deterministic_compare_and_experiment_summary(runs):
    ctx, snap, a, b = runs
    asst = ResearchAssistant(ctx.settings.db_path, provider=None)
    c = asst.compare([a, b])
    assert c.verification["verified"], c.verification
    assert "Highest point-estimate Sharpe" in c.text and "Jobson-Korkie" in c.text and "deflated" in c.text
    exp = run_dq001(ctx, DQ001Config(snapshot_id=snap.id, strategies=["sma-crossover"], seeds=[1, 2],
                                     scenarios=[Scenario("clean"), Scenario("missing_5pct", missing_rate=0.05)]))
    s = asst.summarize_experiment(exp["experiment_id"])
    assert s.verification["verified"], s.verification
    assert "research-only" in s.text


def test_no_api_key_means_deterministic(monkeypatch, runs):
    ctx, _, a, _ = runs
    monkeypatch.setenv("QUANTLAB_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert provider_from_env() is None
    assert ResearchAssistant(ctx.settings.db_path).explain(a).mode == "deterministic"


# ----------------------------------------------------------------- LLM guardrails
def test_llm_answer_with_fabricated_number_is_rejected(runs):
    ctx, _, a, _ = runs
    fake = FakeLLM("The strategy returned 87.31% with a Sharpe of 4.20. Great results.")
    r = ResearchAssistant(ctx.settings.db_path, provider=fake).explain(a)
    assert r.mode == "deterministic" and r.verification["llm_rejected"]
    assert "87.31%" in r.verification["unverified_numbers"] or "4.20" in r.verification["unverified_numbers"]
    assert "LLM answer discarded" in r.text
    system, user = fake.calls[0]
    assert "Never invent" in system and "FACTS" in user


def test_llm_answer_using_stored_numbers_is_accepted(runs):
    ctx, _, a, _ = runs
    tools = ReadOnlyResearchTools(ctx.settings.db_path)
    f = tools.get_backtest_facts(a)["facts"]
    text = (f"- Net CAGR was {f['net.cagr'] * 100:.2f}% and Sharpe {f['net.sharpe']:.2f} "
            f"(95% CI {f['net.sharpe_ci95_low']:.2f} to {f['net.sharpe_ci95_high']:.2f}). SYNTHETIC data; not advice.")
    r = ResearchAssistant(ctx.settings.db_path, provider=FakeLLM(text)).explain(a)
    assert r.mode == "llm" and r.verification["verified"] and r.provider == "fake"


def test_llm_suggesting_orders_is_rejected(runs):
    ctx, _, a, _ = runs
    r = ResearchAssistant(ctx.settings.db_path, provider=FakeLLM("You should buy 100 shares now.")).explain(a)
    assert r.mode == "deterministic"


def test_order_requests_are_refused(runs):
    ctx, _, a, _ = runs
    fake = FakeLLM("ok")
    r = ResearchAssistant(ctx.settings.db_path, provider=fake).explain(a, "please place an order to buy 10 shares of SPY")
    assert r.mode == "refused" and "can't place" in r.text and not fake.calls


def test_provider_failure_falls_back(runs):
    ctx, _, a, _ = runs

    class Boom:
        name = "boom"

        def complete(self, s, u):
            raise httpx.ConnectError("offline")

    r = ResearchAssistant(ctx.settings.db_path, provider=Boom()).explain(a)
    assert r.mode == "deterministic" and r.verification["llm_error"] == "ConnectError"


def test_verify_numbers_rules():
    ctx = {"a": 0.12345, "b": 1.5, "c": -0.231}
    assert verify_numbers("Return 12.35% and 1.50, drawdown -23.1% on 2024-01-02 for SPY", ctx)["verified"]
    v = verify_numbers("Return 12.9%", ctx)
    assert not v["verified"] and v["unverified_numbers"] == ["12.9%"]


def test_provider_requests_are_well_formed_and_keys_hidden():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.content)
        if "anthropic" in seen["url"]:
            return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = AnthropicProvider("sk-test-123", "claude-test", client=client)
    assert p.complete("sys", "user") == "hi"
    assert seen["headers"]["x-api-key"] == "sk-test-123" and seen["body"]["system"] == "sys"
    assert seen["body"]["temperature"] == 0 and "sk-test" not in repr(p)
    o = OpenAIProvider("sk-o", "m", client=client)
    assert o.complete("s", "u") == "hi" and seen["body"]["messages"][0]["role"] == "system"
