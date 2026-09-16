"""Bounded, provider-agnostic optional LLM clients.

The core analyzer never calls this module.  A provider is constructed only
after the user explicitly enables an LLM feature and supplies both provider and
model identifiers.
"""

from __future__ import annotations

import json
import time
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Any


class LLMProvider(ABC):
    def __init__(self, *, timeout_seconds: float = 20.0, retries: int = 2, max_context_chars: int = 12_000):
        self.timeout_seconds = float(timeout_seconds)
        self.retries = int(retries)
        self.max_context_chars = int(max_context_chars)

    def complete(self, system: str, prompt: str) -> str:
        bounded_system = system[: self.max_context_chars]
        bounded_prompt = prompt[: self.max_context_chars]
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._complete_once(bounded_system, bounded_prompt)
            except Exception as exc:  # provider-specific clients expose different errors
                last_error = exc
                if attempt >= self.retries:
                    break
                # Keep retries bounded and avoid a long blocking backoff in a linter.
                time.sleep(min(0.25 * (attempt + 1), 0.75))
        assert last_error is not None
        raise last_error

    @abstractmethod
    def _complete_once(self, system: str, prompt: str) -> str: ...


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        timeout_seconds: float = 20.0,
        retries: int = 2,
        max_context_chars: int = 12_000,
        client: Any | None = None,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            retries=retries,
            max_context_chars=max_context_chars,
        )
        if not model:
            raise ValueError("an explicit Anthropic model is required")
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout_seconds)
        self.client: Any = client
        self.model = model

    def _complete_once(self, system: str, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        timeout_seconds: float = 20.0,
        retries: int = 2,
        max_context_chars: int = 12_000,
        client: Any | None = None,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            retries=retries,
            max_context_chars=max_context_chars,
        )
        if not model:
            raise ValueError("an explicit OpenAI model is required")
        if client is None:
            import openai

            client = openai.OpenAI(timeout=timeout_seconds)
        self.client: Any = client
        self.model = model

    def _complete_once(self, system: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            timeout=self.timeout_seconds,
        )
        return response.choices[0].message.content or ""


class OllamaProvider(LLMProvider):
    """Local Ollama backend over its HTTP API."""

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        timeout_seconds: float = 20.0,
        retries: int = 2,
        max_context_chars: int = 12_000,
        opener: Callable[..., Any] | None = None,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            retries=retries,
            max_context_chars=max_context_chars,
        )
        if not model:
            raise ValueError("an explicit Ollama model is required")
        self.model = model
        self.host = host.rstrip("/")
        self._opener = opener or urllib.request.urlopen

    def _complete_once(self, system: str, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with self._opener(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", ""))


@lru_cache(maxsize=16)
def _cached_provider(
    name: str,
    model: str,
    timeout_seconds: float,
    retries: int,
    max_context_chars: int,
) -> LLMProvider:
    if name == "anthropic":
        return AnthropicProvider(
            model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            max_context_chars=max_context_chars,
        )
    if name == "openai":
        return OpenAIProvider(
            model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            max_context_chars=max_context_chars,
        )
    if name == "ollama":
        return OllamaProvider(
            model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            max_context_chars=max_context_chars,
        )
    raise ValueError(f"unknown LLM provider: {name}")


def get_provider(
    name: str | None = None,
    model: str | None = None,
    *,
    timeout_seconds: float = 20.0,
    retries: int = 2,
    max_context_chars: int = 12_000,
) -> LLMProvider:
    """Return a reused client; never infer provider/model from the environment."""

    if not name or not model:
        raise ValueError("LLM assistance requires explicit provider and model configuration")
    normalized = name.strip().lower()
    if normalized == "local":
        normalized = "ollama"
    return _cached_provider(
        normalized,
        model,
        float(timeout_seconds),
        int(retries),
        int(max_context_chars),
    )
