"""Optional provider contracts use injected clients; no network or credentials."""

import io
import json
from types import SimpleNamespace

import pytest

from ml_leakproof.core.config import Config, LLMConfig
from ml_leakproof.core.models import Category, Finding, Layer, Location, Severity
from ml_leakproof.llm import explain_all, explain_finding, get_provider, triage_file
from ml_leakproof.llm import provider as providers
from ml_leakproof.llm.explain import code_context


class StubProvider(providers.LLMProvider):
    def __init__(self, response="helpful explanation", fail=0, **kwargs):
        super().__init__(**kwargs)
        self.response, self.fail, self.calls = response, fail, []

    def _complete_once(self, system, prompt):
        self.calls.append((system, prompt))
        if len(self.calls) <= self.fail:
            raise RuntimeError("provider unavailable")
        return self.response


@pytest.fixture
def config():
    return Config(llm=LLMConfig(enabled=True, provider="ollama", model="test-model", retries=0))


def finding(path=None):
    return Finding(
        rule_id="P001",
        category=Category.PREPROCESSING,
        severity=Severity.HIGH,
        layer=Layer.STATIC,
        message="learned from test",
        location=Location(file=path, line=1, snippet="x"),
        confidence=0.9,
    )


def test_bounded_context_and_retry_budget(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda _: None)
    provider = StubProvider(fail=1, retries=1, max_context_chars=4)
    assert provider.complete("system!", "prompt!") == "helpful explanation"
    assert provider.calls == [("syst", "prom")] * 2
    provider = StubProvider(fail=3, retries=1)
    with pytest.raises(RuntimeError):
        provider.complete("s", "p")
    assert len(provider.calls) == 2


def test_injected_provider_clients():
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking"),
                SimpleNamespace(type="text", text="explanation"),
            ]
        )

    anthropic = providers.AnthropicProvider(
        "model", client=SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    assert anthropic.complete("system", "prompt") == "explanation"
    assert requests[0]["model"] == "model"

    def completion(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    openai = providers.OpenAIProvider(
        "model",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
        ),
    )
    assert openai.complete("system", "prompt") == ""
    assert requests[-1]["timeout"] == 20

    def opener(request, timeout):
        requests.append(json.loads(request.data))
        assert timeout == 20
        return io.BytesIO(b'{"response":"local response"}')

    ollama = providers.OllamaProvider("model", opener=opener)
    assert ollama.complete("system", "prompt") == "local response"
    assert requests[-1]["stream"] is False


@pytest.mark.parametrize(
    "cls", [providers.OpenAIProvider, providers.AnthropicProvider, providers.OllamaProvider]
)
def test_explicit_model_required(cls):
    with pytest.raises(ValueError):
        cls("")


def test_provider_selection_and_cache(monkeypatch):
    providers._cached_provider.cache_clear()
    with pytest.raises(ValueError):
        get_provider()
    with pytest.raises(ValueError):
        get_provider("unknown", "model")
    assert get_provider("local", "model") is get_provider("ollama", "model")
    for name, attr in [("openai", "OpenAIProvider"), ("anthropic", "AnthropicProvider")]:
        monkeypatch.setattr(providers, attr, lambda model, **kwargs: StubProvider(**kwargs))
        assert isinstance(get_provider(name, "model"), StubProvider)
    providers._cached_provider.cache_clear()


def test_explanations_preserve_deterministic_result(tmp_path, monkeypatch, config):
    path = tmp_path / "code.py"
    path.write_text("x = 1\ny = 2\n")
    original = finding(path)
    fake = StubProvider("Review this.\n```diff\n- old\n+ new\n```")
    monkeypatch.setattr(providers, "get_provider", lambda *a, **kw: fake)
    enriched = explain_finding(original, config)
    assert original.evidence == {}
    assert enriched.rule_id == original.rule_id and enriched.fix == original.fix
    assert (
        enriched.confidence == original.confidence
        and enriched.advisory_only == original.advisory_only
    )
    assert enriched.evidence["llm_suggested_diff"] == "- old\n+ new"
    assert code_context(original, max_chars=3) == "x ="
    fake.fail = 100
    result, diagnostics = explain_all([original], config)
    assert result == [original] and diagnostics[0].code == "LP251"
    assert code_context(finding(tmp_path / "missing")) == "x"
    assert code_context(finding()) == "x"


@pytest.mark.parametrize(
    "response,code",
    [("not json", "LP250"), ("{}", "LP253"), ('[7, {}, {"line":99,"reason":"oops"}]', "LP253")],
)
def test_invalid_triage_responses(tmp_path, monkeypatch, config, response, code):
    path = tmp_path / "code.py"
    path.write_text("if broken")
    monkeypatch.setattr(providers, "get_provider", lambda *a, **kw: StubProvider(response))
    findings, diagnostics = triage_file(path, config)
    assert findings == [] and diagnostics[0].code == code


def test_triage_is_bounded_advisory_and_read_failure_visible(tmp_path, monkeypatch, config):
    path = tmp_path / "code.py"
    path.write_text("if broken")
    response = "```json\n" + json.dumps([{"line": 1, "reason": "review"}] * 20) + "\n```"
    monkeypatch.setattr(providers, "get_provider", lambda *a, **kw: StubProvider(response))
    findings, diagnostics = triage_file(path, config)
    assert len(findings) == 10 and not diagnostics
    assert all(f.advisory_only and f.rule_id == "LP-TRIAGE" for f in findings)
    assert triage_file(tmp_path / "missing", config)[1][0].code == "LP252"
    import ml_leakproof.llm as llm

    with pytest.raises(AttributeError):
        _ = llm.unknown
