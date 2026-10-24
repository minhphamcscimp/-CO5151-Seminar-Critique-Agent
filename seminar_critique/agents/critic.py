"""Critic agent. Owner: Van (tool results come from Tin's tools/).

Goal: critique along the instructor's rubric axes:
assumptions, baseline fairness, data reliability, generalizability, reproducibility.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Config, TraceStep
from .base import call_agent

SYSTEM = (
    "You are a critical reviewer. For each rubric axis give concrete, evidence-based "
    "weaknesses. Cite the section or table you rely on."
)  # TODO (Van): refine the prompt


def critique(
    summary: str,
    rubric_axes: list[str],
    tool_results: list[dict],
    llm: LLMClient,
    cfg: Config,
) -> tuple[str, TraceStep]:
    axes = ", ".join(rubric_axes) or "all axes"
    user = f"Summary:\n{summary}\n\nRubric axes: {axes}\n\nBenchmark evidence: {tool_results}"
    return call_agent(llm, cfg, "critic", SYSTEM, user, tool_calls=tool_results)
