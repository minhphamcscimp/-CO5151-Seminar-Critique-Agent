"""Critic agent. Owner: Van (tool results come from Tin's tools/).

Goal: critique along the instructor's rubric axes:
assumptions, baseline fairness, data reliability, generalizability, reproducibility.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Config, TraceStep
from .base import UNTRUSTED_RULE, call_agent

RUBRIC_AXES = [
    "assumptions",
    "baseline fairness",
    "data reliability",
    "generalizability",
    "reproducibility",
]

SYSTEM = (
    "You are a strict seminar reviewer. Cover every rubric axis you are given, "
    "in the order given, one block per axis:\n"
    "AXIS: <name>\n"
    "WEAKNESS: one concrete problem, not a generic remark.\n"
    "EVIDENCE: the section, table or benchmark record you rely on, quoted or cited; "
    "write 'no evidence in the material' when you have none.\n"
    "SEVERITY: high / medium / low.\n"
    "Rules: at most 3 sentences per block; never invent a number or a citation; "
    "when a benchmark record contradicts the material, say so and give both numbers; "
    "judge the work, not the writing style.\n"
    + UNTRUSTED_RULE
)


def critique(
    summary: str,
    rubric_axes: list[str],
    tool_results: list[dict],
    llm: LLMClient,
    cfg: Config,
) -> tuple[str, TraceStep]:
    axes = ", ".join(rubric_axes or RUBRIC_AXES)
    evidence = tool_results or "(no benchmark lookup in this config)"
    user = f"Summary:\n{summary}\n\nRubric axes: {axes}\n\nBenchmark evidence: {evidence}"
    return call_agent(llm, cfg, "critic", SYSTEM, user, tool_calls=tool_results)
