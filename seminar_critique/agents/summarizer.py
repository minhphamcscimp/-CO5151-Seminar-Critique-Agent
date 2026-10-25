"""Summarizer agent. Owner: Van.

Goal: turn the seminar material into a short, faithful summary
(problem, method, setup, results) that the Critic can work from.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..schemas import Chunk, Config, TraceStep
from ..security import sanitize_untrusted
from .base import UNTRUSTED_RULE, call_agent

SYSTEM = (
    "You summarize research material so that a reviewer can critique it.\n"
    "Write exactly these four sections, in this order, each 2-4 sentences:\n"
    "1. Problem - what question the work attacks and why it matters.\n"
    "2. Method - the proposed approach and what is new in it.\n"
    "3. Experimental setup - datasets, baselines, models, metrics, budget.\n"
    "4. Results - the main numbers, with the table or section they come from.\n"
    "Rules: use only the material and the retrieved context; never invent a number; "
    "write 'not stated' when the material does not say; keep the authors' own claims "
    "as claims, do not endorse them. Maximum 300 words.\n"
    + UNTRUSTED_RULE
)


def summarize(
    material: str, chunks: list[Chunk], llm: LLMClient, cfg: Config
) -> tuple[str, TraceStep]:
    # Chunk text and its paper/section label are untrusted (threat model entry point 2).
    context = "\n".join(
        sanitize_untrusted(c.text, f"{c.paper} {c.section}") for c in chunks
    ) or "(none)"
    user = f"Material:\n{material}\n\nRetrieved context:\n{context}"
    return call_agent(llm, cfg, "summarizer", SYSTEM, user)