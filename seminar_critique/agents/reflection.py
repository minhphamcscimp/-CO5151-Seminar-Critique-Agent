"""Reflection agent. Owner: Van.

Goal: improve a draft using EXTERNAL feedback (benchmark lookups, retrieved evidence),
not the model's own opinion alone. This is the point of the Huang et al. (ICLR 2024)
discussion: intrinsic self-correction does not work, feedback-grounded correction can.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Config, TraceStep
from .base import UNTRUSTED_RULE, call_agent

NO_CHANGE = "NO CHANGE"

SYSTEM = (
    "You revise a draft critique using ONLY the external evidence supplied below "
    "(benchmark records and retrieved paper chunks). Your own opinion is not evidence.\n"
    "Procedure: for each piece of evidence decide whether it contradicts, confirms or "
    "does not touch the draft. Change only the sentences a contradiction forces you to "
    "change, and cite the evidence in that sentence. Keep the template and the headings "
    "of the draft.\n"
    f"If no piece of evidence changes anything, reply with exactly: {NO_CHANGE}\n"
    "Never add a new weakness that the evidence does not support, and never delete a "
    "weakness just because the evidence is silent about it.\n"
    + UNTRUSTED_RULE
)


def reflect(
    draft: str, evidence: list[dict], llm: LLMClient, cfg: Config
) -> tuple[str, TraceStep]:
    user = f"Draft:\n{draft}\n\nExternal evidence: {evidence or '(none)'}"
    return call_agent(llm, cfg, "reflection", SYSTEM, user, tool_calls=evidence)
