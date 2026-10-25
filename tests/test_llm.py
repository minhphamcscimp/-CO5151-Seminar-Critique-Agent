"""Tests for the LLM layer, cost accounting and the reflection loop. Owner: Van.

No API key and no network: the Gemini client is exercised through a stub that
mimics the OpenAI SDK response object.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from seminar_critique.agents.base import UNTRUSTED_RULE
from seminar_critique.agents.critic import RUBRIC_AXES, SYSTEM as CRITIC_SYSTEM
from seminar_critique.agents.reflection import NO_CHANGE
from seminar_critique.agents.summarizer import SYSTEM as SUMMARIZER_SYSTEM
from seminar_critique.configs import CONFIGS, resolve_models
from seminar_critique.llm import FakeLLM, GeminiClient
from seminar_critique.llm.pricing import estimate_cost_usd, price_per_million
from seminar_critique.orchestrator import run_pipeline
from seminar_critique.schemas import Task

TASK = Task(
    id="t-llm",
    topic="self-improvement-reflection",
    input="Critique this claim.",
    rubric_axes=["assumptions", "reproducibility"],
    key_points=["fake"],
)


# --- stubs that look like the OpenAI SDK --------------------------------------
@dataclass
class _Usage:
    prompt_tokens: int = 120
    completion_tokens: int = 30


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list
    usage: _Usage


class _StubCompletions:
    def __init__(self, text: str = "stub answer"):
        self.text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response([_Choice(_Message(self.text))], _Usage())


def _client_with_stub(text: str = "stub answer") -> tuple[GeminiClient, _StubCompletions]:
    client = GeminiClient(api_key="test-key-not-real")
    stub = _StubCompletions(text)
    client._client.chat.completions = stub  # type: ignore[attr-defined]
    return client, stub


# --- tests --------------------------------------------------------------------
def test_missing_key_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here, so nothing can be loaded
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_temperature_and_model_reach_the_api():
    """The two gaps of the MAD client: temperature was dropped, model was fixed."""
    client, stub = _client_with_stub()
    client.complete([{"role": "user", "content": "hi"}], model="gemini-2.5-flash", temperature=0.7)
    assert stub.calls[0]["temperature"] == 0.7
    assert stub.calls[0]["model"] == "gemini-2.5-flash"

    client.complete([{"role": "user", "content": "hi"}], model="gemini-2.5-pro")
    assert stub.calls[1]["model"] == "gemini-2.5-pro"  # no single-model allowlist


def test_usage_latency_and_cost_are_filled():
    client, _ = _client_with_stub()
    resp = client.complete([{"role": "user", "content": "hi"}], model="gemini-2.5-flash")
    assert resp.text == "stub answer"
    assert resp.usage.prompt_tokens == 120
    assert resp.usage.completion_tokens == 30
    assert resp.usage.latency_s >= 0
    assert resp.usage.cost_usd == pytest.approx((120 * 0.30 + 30 * 2.50) / 1_000_000)


def test_empty_response_raises():
    client, stub = _client_with_stub()
    stub.create = lambda **kw: _Response([_Choice(_Message(None))], _Usage())  # type: ignore
    with pytest.raises(RuntimeError, match="empty response"):
        client.complete([{"role": "user", "content": "hi"}], model="gemini-2.5-flash")


def test_pricing_uses_longest_prefix_and_env_override(monkeypatch):
    monkeypatch.delenv("GEMINI_PRICE_IN_PER_M", raising=False)
    monkeypatch.delenv("GEMINI_PRICE_OUT_PER_M", raising=False)
    assert price_per_million("gemini-2.5-flash-lite-preview") == (0.10, 0.40)
    assert price_per_million("fake-model") == (0.0, 0.0)
    assert estimate_cost_usd("totally-unknown-model", 1000, 1000) == 0.0

    monkeypatch.setenv("GEMINI_PRICE_IN_PER_M", "1.0")
    monkeypatch.setenv("GEMINI_PRICE_OUT_PER_M", "2.0")
    assert estimate_cost_usd("gemini-2.5-flash", 1_000_000, 0) == 1.0


def test_resolve_models_keeps_one_model_for_every_config(monkeypatch):
    for name in ("A", "B", "C"):
        cfg = resolve_models(CONFIGS[name], "gemini")
        assert set(cfg.models.values()) == {"gemini-3.1-flash-lite"}
        assert CONFIGS[name].models["critic"] == "fake-model"  # original untouched
    assert resolve_models(CONFIGS["A"], "fake").models["baseline"] == "fake-model"


def test_prompts_carry_the_untrusted_rule_and_rubric():
    assert UNTRUSTED_RULE in SUMMARIZER_SYSTEM
    assert UNTRUSTED_RULE in CRITIC_SYSTEM
    assert len(RUBRIC_AXES) == 5  # the five axes of the instructor's rubric
    for axis in ("assumptions", "baseline fairness", "reproducibility"):
        assert axis in RUBRIC_AXES


class _NoChangeLLM(FakeLLM):
    """Answers the reflection agent with the agreed no-op marker."""

    def complete(self, messages, *, model, temperature=0.0, max_tokens=2048):
        resp = super().complete(messages, model=model, temperature=temperature)
        if "Draft:" in messages[-1]["content"]:
            resp.text = NO_CHANGE
        return resp


def test_reflection_stops_early_and_keeps_the_draft():
    result = run_pipeline(TASK, CONFIGS["C"], _NoChangeLLM())
    reflections = [s for s in result.trace if s.agent == "reflection"]
    assert len(reflections) == 1  # stopped after the first no-op instead of 2
    assert result.answer != NO_CHANGE  # the synthesizer draft is kept
    assert result.answer == [s for s in result.trace if s.agent == "synthesizer"][0].output


def test_config_c_still_runs_both_reflections_when_the_draft_changes():
    result = run_pipeline(TASK, CONFIGS["C"], FakeLLM())
    assert len([s for s in result.trace if s.agent == "reflection"]) == 2
