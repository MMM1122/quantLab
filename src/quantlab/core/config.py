"""Settings: YAML defaults + environment overrides.

Safety boundary: ``LIVE_TRADING_ENABLED`` must be false. V1 has no live broker; if the
variable is set to a truthy value QuantLab refuses to start rather than silently ignoring it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quantlab.core.errors import ConfigError, LiveTradingDisabledError

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
_TRUTHY = {"1", "true", "yes", "on"}


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_yaml(name: str) -> dict[str, Any]:
    p = CONFIG_DIR / name
    if not p.exists():
        raise ConfigError(f"missing config file {p}")
    with p.open() as f:
        return yaml.safe_load(f) or {}


def assert_live_trading_disabled() -> None:
    if os.environ.get("LIVE_TRADING_ENABLED", "false").strip().lower() in _TRUTHY:
        raise LiveTradingDisabledError(
            "LIVE_TRADING_ENABLED is set, but QuantLab V1 has no live broker integration. "
            "Unset it (default false). Live trading requires a separate, human-approved release."
        )


@dataclass
class Settings:
    data_dir: Path
    base: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    llm_provider: str = "none"

    live_trading_enabled: bool = False  # always False in V1; kept for explicit reporting

    @property
    def db_path(self) -> Path:
        return self.data_dir / "quantlab.db"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.snapshots_dir, self.runs_dir, self.experiments_dir, self.exports_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_settings(data_dir: str | Path | None = None) -> Settings:
    _load_dotenv(REPO_ROOT / ".env")
    assert_live_trading_disabled()
    dd = Path(data_dir or os.environ.get("QUANTLAB_DATA_DIR", "data"))
    if not dd.is_absolute():
        dd = (REPO_ROOT / dd).resolve()
    s = Settings(
        data_dir=dd,
        base=_load_yaml("base.yaml"),
        risk=_load_yaml("risk.yaml"),
        llm_provider=os.environ.get("QUANTLAB_LLM_PROVIDER", "none").strip().lower() or "none",
    )
    if s.risk.get("allow_short") or s.risk.get("allow_margin"):
        raise ConfigError("Shorting and margin cannot be enabled in QuantLab V1.")
    s.ensure_dirs()
    return s
