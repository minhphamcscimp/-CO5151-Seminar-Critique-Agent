"""Critic agent. Owner: Van (tool results come from Tin's tools/).

Goal: critique along the instructor's rubric axes:
assumptions, baseline fairness, data reliability, generalizability, reproducibility.
"""
from __future__ import annotations

import json

from ..llm import LLMClient
from ..schemas import Config, TraceStep
from ..security import sanitize_untrusted
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
    summary = sanitize_untrusted(summary, "summarizer_output")
    evidence = sanitize_untrusted(json.dumps(tool_results, ensure_ascii=False), "benchmark_tool")
    user = f"Summary:\n{summary}\n\nRubric axes: {axes}\n\nBenchmark evidence: {evidence}"
    return call_agent(llm, cfg, "critic", SYSTEM, user, tool_calls=tool_results)
