"""Synthesizer agent. Owner: Van.

Goal: merge summary and critique into a structured critique the student can use.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Config, TraceStep
from ..security import sanitize_untrusted
from .base import UNTRUSTED_RULE, call_agent

SYSTEM = (
    "You write the final critique a student brings to a seminar. "
    "Use exactly this template and no other headings:\n"
    "## Summary\n(3 sentences: problem, method, headline result.)\n"
    "## Strengths\n(2-3 bullets, each with the evidence it rests on.)\n"
    "## Weaknesses by rubric axis\n"
    "(one bullet per axis, format: - <axis> [severity]: <weakness> (evidence))\n"
    "## Questions for the authors\n(exactly 3 questions, each answerable with a number "
    "or an experiment, not yes/no.)\n"
    "Rules: keep only claims supported by the summary or the critique; drop a point "
    "rather than invent support; do not repeat the same weakness in two axes; "
    "maximum 450 words.\n"
    + UNTRUSTED_RULE
)


def synthesize(
    summary: str, critique_text: str, llm: LLMClient, cfg: Config
) -> tuple[str, TraceStep]:
    summary = sanitize_untrusted(summary, "summarizer_output")
    critique_text = sanitize_untrusted(critique_text, "critic_output")
    user = f"Summary:\n{summary}\n\nCritique:\n{critique_text}"
    return call_agent(llm, cfg, "synthesizer", SYSTEM, user)