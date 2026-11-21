"""`quantlab ai ...` sub-commands (read-only research assistant)."""

from __future__ import annotations

from quantlab.cli.main import _ctx, _print


def _assistant(args):
    from quantlab.ai.assistant import ResearchAssistant

    return ResearchAssistant(_ctx(args).settings.db_path)


def _out(r, as_json: bool) -> int:
    if as_json:
        d = r.to_dict()
        d.pop("facts")
        _print(d)
    else:
        print(r.text)
        print(f"\n[mode={r.mode} · numbers verified={r.verification.get('verified')} · sources={r.sources}]")
    return 0


def register(sub) -> None:
    ai = sub.add_parser("ai", help="read-only research assistant").add_subparsers(dest="cmd", required=True)
    e = ai.add_parser("explain")
    e.add_argument("--run", required=True)
    e.add_argument("--question", default=None)
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=lambda a: _out(_assistant(a).explain(a.run, a.question), a.json))
    c = ai.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--question", default=None)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=lambda a: _out(_assistant(a).compare(a.runs, a.question), a.json))
    s = ai.add_parser("summarize")
    s.add_argument("--experiment", required=True)
    s.add_argument("--question", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=lambda a: _out(_assistant(a).summarize_experiment(a.experiment, a.question), a.json))
