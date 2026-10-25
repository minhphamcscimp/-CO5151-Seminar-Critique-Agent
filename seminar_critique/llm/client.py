"""LLM access layer. Owner: Van.

Everyone else codes against `LLMClient` and tests with `FakeLLM`,
so nobody needs an API key to work on their own module.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import backoff
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from ..schemas import Usage
from .pricing import estimate_cost_usd

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
# 429 = rate limit, 5xx = transient server side; anything else is our own bug.
RETRY_STATUS = (429, 500, 502, 503, 504)


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


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader so nobody has to export the key by hand.

    Only fills variables that are not already set, and never prints the value.
    .env is in .gitignore, so the key cannot be committed by accident.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class OutOfQuotaError(RuntimeError):
    """The API key has no quota left; retrying would not help."""


def _log_retry(details: dict) -> None:
    error = details.get("exception")
    name = type(error).__name__ if error else "unknown error"
    print(
        f"[llm] call failed ({name}); retry {details['tries']} in {details['wait']:.1f}s",
        flush=True,
    )


def _give_up(error: Exception) -> bool:
    """Retry only on rate limits and transient server errors."""
    if isinstance(error, APIStatusError):
        return error.status_code not in RETRY_STATUS
    return False


class GeminiClient:
    """Real client for the Gemini OpenAI-compatible endpoint.

    Built on third_party/MAD/code/utils/agent.py, with the three gaps of that
    code fixed:
      1. `temperature` is really sent to the API (MAD stored it and never used it);
      2. the model name comes from each call, so a config can pick its own model
         (MAD asserted against a one-item `support_models` list);
      3. `Usage` is filled from `response.usage`, `latency_s` is measured and
         `cost_usd` is computed, so the report can show real numbers.

    The key is read from GEMINI_API_KEY only; it is never passed through a Config
    and never written into a trace.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = GEMINI_BASE_URL,
        timeout: float = 120.0,
        sleep_time: float = 0.0,
    ) -> None:
        _load_dotenv()
        key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and put the key there, "
                "or run with --llm fake."
            )
        # max_retries=0: the backoff decorator below is the only retry policy.
        self._client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        self._sleep_time = sleep_time

    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        model = model or DEFAULT_GEMINI_MODEL
        started = time.perf_counter()
        response = self._create(
            messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        latency_s = time.perf_counter() - started

        text = response.choices[0].message.content
        if text is None:
            raise RuntimeError(f"{model} returned an empty response")

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return LLMResponse(
            text=text,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
                latency_s=round(latency_s, 3),
            ),
        )

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError),
        max_tries=4,
        max_time=300,
        on_backoff=_log_retry,
        giveup=_give_up,
    )
    def _create(self, *, messages: list[dict], model: str, temperature: float, max_tokens: int):
        if self._sleep_time:
            time.sleep(self._sleep_time)  # client-side rate limiting between calls
        try:
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as e:
            message = str(e).casefold()
            if any(w in message for w in ("quota", "resource_exhausted", "billing")):
                raise OutOfQuotaError(f"Gemini quota exhausted for {model}") from e
            raise


def get_client(name: str = "fake") -> LLMClient:
    if name == "fake":
        return FakeLLM()
    if name == "gemini":
        return GeminiClient()
    raise ValueError(f"unknown llm client: {name}")
