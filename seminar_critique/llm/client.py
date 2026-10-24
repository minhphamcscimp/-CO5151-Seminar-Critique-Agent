"""LLM access layer. Owner: Van.

Everyone else codes against `LLMClient` and tests with `FakeLLM`,
so nobody needs an API key to work on their own module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..schemas import Usage


@dataclass
class LLMResponse:
    text: str
    usage: Usage


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Return the model text and the usage of this single call."""
        ...


class FakeLLM:
    """Deterministic offline client for tests and for developing other modules."""

    def complete(self, messages, *, model, temperature=0.0, max_tokens=2048):
        last = messages[-1]["content"]
        words = len(last.split())
        text = f"[fake:{model}] reply to {words} words"
        return LLMResponse(
            text=text,
            usage=Usage(prompt_tokens=words, completion_tokens=len(text.split())),
        )


class GeminiClient:
    """TODO (Van): real client.

    Reuse ideas from third_party/MAD/code/utils/agent.py:
    retry with backoff on 429/503 and the OutOfQuota handling.
    Must fix these gaps of the MAD code:
      1. actually send `temperature` to the API
      2. accept any model name per call (no single-model allowlist)
      3. fill Usage from response.usage and measure latency_s
    Read the key from the GEMINI_API_KEY environment variable only.
    """

    def complete(self, messages, *, model, temperature=0.0, max_tokens=2048):
        raise NotImplementedError("GeminiClient is not implemented yet")


def get_client(name: str = "fake") -> LLMClient:
    if name == "fake":
        return FakeLLM()
    if name == "gemini":
        return GeminiClient()
    raise ValueError(f"unknown llm client: {name}")
