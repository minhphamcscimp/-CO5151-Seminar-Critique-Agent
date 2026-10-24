"""Synthesizer agent. Owner: Van.

Goal: merge summary and critique into a structured critique the student can use.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Config, TraceStep
from .base import call_agent

SYSTEM = (
    "You write the final structured critique: strengths, weaknesses per rubric axis, "
    "and 3 questions a reader could ask in Q&A."
)  # TODO (Van): refine the prompt


def synthesize(
    summary: str, critique_text: str, llm: LLMClient, cfg: Config
) -> tuple[str, TraceStep]:
    user = f"Summary:\n{summary}\n\nCritique:\n{critique_text}"
    return call_agent(llm, cfg, "synthesizer", SYSTEM, user)
