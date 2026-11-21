"""Optional LLM provider adapters. Used only to *phrase* explanations of stored facts.

Keys come from the environment and are never logged. Without a key the assistant runs in
deterministic mode. Providers receive facts as data and have no tools — they cannot read the
database or reach any order system.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx

from quantlab.core.errors import ConfigError


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str | None = None, client: httpx.Client | None = None):
        self._key = api_key
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self._client = client or httpx.Client(timeout=60)

    def complete(self, system: str, user: str) -> str:
        r = self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 1200, "temperature": 0, "system": system,
                  "messages": [{"role": "user", "content": user}]},
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")

    def __repr__(self) -> str:  # never expose the key
        return f"AnthropicProvider(model={self.model!r})"


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str | None = None, client: httpx.Client | None = None):
        self._key = api_key
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = client or httpx.Client(timeout=60)

    def complete(self, system: str, user: str) -> str:
        r = self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json={"model": self.model, "temperature": 0,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r})"


def provider_from_env(name: str | None = None) -> LLMProvider | None:
    name = (name or os.environ.get("QUANTLAB_LLM_PROVIDER", "none")).strip().lower()
    if name in ("", "none"):
        return None
    if name == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        return AnthropicProvider(key) if key else None
    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        return OpenAIProvider(key) if key else None
    raise ConfigError(f"unknown QUANTLAB_LLM_PROVIDER {name!r}")
