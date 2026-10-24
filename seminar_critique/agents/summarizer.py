"""Summarizer agent. Owner: Van.

Goal: turn the seminar material into a short, faithful summary
(problem, method, setup, results) that the Critic can work from.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Chunk, Config, TraceStep
from .base import call_agent

SYSTEM = (
    "You summarize a research paper for a critique. Keep problem, method, "
    "experimental setup and results. Do not add claims that are not in the text."
)  # TODO (Van): refine the prompt


def summarize(
    material: str, chunks: list[Chunk], llm: LLMClient, cfg: Config
) -> tuple[str, TraceStep]:
    context = "\n".join(f"[{c.paper} {c.section}] {c.text}" for c in chunks)
    user = f"Material:\n{material}\n\nRetrieved context:\n{context}"
    return call_agent(llm, cfg, "summarizer", SYSTEM, user)
