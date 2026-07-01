"""Provider-agnostic LLM interface. No keys are ever required for core use."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str) -> str: ...


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        import anthropic  # type: ignore

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("LEAKPROOF_LLM_MODEL", "claude-opus-4-8")

    def complete(self, system: str, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        import openai  # type: ignore

        self.client = openai.OpenAI()
        self.model = model or os.environ.get("LEAKPROOF_LLM_MODEL", "gpt-4o")

    def complete(self, system: str, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


class OllamaProvider(LLMProvider):
    """Local/Ollama backend over the HTTP API (no SDK dependency)."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("LEAKPROOF_LLM_MODEL", "llama3")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def complete(self, system: str, prompt: str) -> str:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    name = (name or os.environ.get("LEAKPROOF_LLM_PROVIDER", "anthropic")).lower()
    if name == "anthropic":
        return AnthropicProvider(model)
    if name == "openai":
        return OpenAIProvider(model)
    if name in ("ollama", "local"):
        return OllamaProvider(model)
    raise ValueError(f"unknown LLM provider: {name}")
