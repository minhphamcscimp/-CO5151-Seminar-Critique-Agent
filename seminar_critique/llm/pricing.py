"""Token prices used to turn Usage into cost_usd. Owner: Van.

Numbers are USD per 1,000,000 tokens, taken from the Gemini API pricing page
(https://ai.google.dev/gemini-api/docs/pricing), paid tier, read on 2026-10-24.

If the group runs on the free tier the real bill is 0 USD. We still report the
converted cost so configs A / B / C can be compared on the same scale; the report
must say "free tier, cost converted at the paid-tier price of 2026-10-24".

Prices can be overridden without touching code:
    GEMINI_PRICE_IN_PER_M=0.10  GEMINI_PRICE_OUT_PER_M=0.40
"""
from __future__ import annotations

import os

PRICE_DATE = "2026-10-24"
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"

# model prefix -> (input USD / 1M tokens, output USD / 1M tokens)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # offline clients cost nothing
    "fake-model": (0.0, 0.0),
}

DEFAULT_PRICE = (0.0, 0.0)


def price_per_million(model: str) -> tuple[float, float]:
    """Return (input, output) USD per 1M tokens for `model`.

    Longest matching prefix wins, so "gemini-2.5-flash-002" uses the
    "gemini-2.5-flash" row. Unknown models cost 0 and that is visible in the CSV.
    """
    env_in = os.getenv("GEMINI_PRICE_IN_PER_M")
    env_out = os.getenv("GEMINI_PRICE_OUT_PER_M")
    if env_in and env_out:
        return float(env_in), float(env_out)

    name = (model or "").lower()
    best = ""
    for prefix in PRICE_TABLE:
        if name.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return PRICE_TABLE[best] if best else DEFAULT_PRICE


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost of one call. Rounded to 8 decimals so the CSV stays readable."""
    price_in, price_out = price_per_million(model)
    cost = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
    return round(cost, 8)
