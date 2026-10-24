"""Reflection agent. Owner: Van.

Goal: improve a draft using EXTERNAL feedback (benchmark lookups, retrieved evidence),
not the model's own opinion alone. This is the point of the Huang et al. (ICLR 2024)
discussion: intrinsic self-correction does not work, feedback-grounded correction can.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Config, TraceStep
from .base import call_agent

SYSTEM = (
    "Revise the draft critique using ONLY the external evidence given. "
    "If the evidence does not change anything, return the draft unchanged."
)  # TODO (Van): refine the prompt


def reflect(
    draft: str, evidence: list[dict], llm: LLMClient, cfg: Config
) -> tuple[str, TraceStep]:
    user = f"Draft:\n{draft}\n\nExternal evidence: {evidence}"
    return call_agent(llm, cfg, "reflection", SYSTEM, user, tool_calls=evidence)
