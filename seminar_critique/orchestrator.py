"""Pipeline and reflection loop. Owner: Van.

CONTRACT (do not change without a "[contract]" PR):
    run_pipeline(task, config, llm=None) -> Result

Flow for config B and C:
    retrieve (optional) -> Summarizer -> tool lookup (optional) -> Critic
    -> Synthesizer -> Reflection loop (config C, max N iterations)
Config A is a single LLM call, used as the simple baseline.
"""
from __future__ import annotations

import time

from .agents import critique, reflect, summarize, synthesize
from .agents.base import call_agent
from .configs import resolve_models
from .llm import GeminiClient, LLMClient, get_client
from .retrieval import retrieve
from .schemas import Chunk, Config, Result, Task, TraceStep, Usage
from .security import sanitize_untrusted
from .tools import call_tool

BASELINE_SYSTEM = "Write a structured critique of the following seminar material."


def _normalize(text: str) -> str:
    """Ignore whitespace and case when comparing two drafts."""
    return " ".join(text.split()).casefold()


def _is_unchanged(previous: str, current: str) -> bool:
    """True when a reflection pass produced no real change.

    Also true when the model answers with the agreed no-op marker, so the
    reflection prompt can say "reply NO CHANGE if the evidence adds nothing".
    """
    current_norm = _normalize(current)
    return current_norm == _normalize(previous) or current_norm.startswith("no change")


def run_pipeline(task: Task, cfg: Config, llm: LLMClient | None = None) -> Result:
    llm = llm or get_client("fake")
    # Placeholder model names become the real model when running on Gemini,
    # so eval/runner.py does not have to know anything about model names.
    cfg = resolve_models(cfg, "gemini" if isinstance(llm, GeminiClient) else "fake")
    started = time.perf_counter()
    trace: list[TraceStep] = []

    # Entry point 1 of the threat model: the seminar material is untrusted.
    material = sanitize_untrusted(task.input, "task_input")

    if not cfg.use_pipeline:
        answer, step = call_agent(llm, cfg, "baseline", BASELINE_SYSTEM, material)
        trace.append(step)
    else:
        chunks: list[Chunk] = []
        if cfg.use_retrieval:
            chunks = retrieve(task.input[:200], task.topic)

        summary, step = summarize(material, chunks, llm, cfg)
        trace.append(step)

        tool_results: list[dict] = []
        if cfg.use_tools:
            tool_results.append(call_tool("lookup_benchmark", name=task.topic, metric="accuracy"))

        critique_text, step = critique(summary, task.rubric_axes, tool_results, llm, cfg)
        trace.append(step)

        answer, step = synthesize(summary, critique_text, llm, cfg)
        trace.append(step)

        for _ in range(cfg.max_reflection_iters):
            if not tool_results and not chunks:
                # Reflection may only use external evidence. With no evidence
                # there is nothing to revise against, so another call would just
                # be intrinsic self-correction (Huang et al., ICLR 2024).
                break
            previous = answer
            answer, step = reflect(answer, tool_results, llm, cfg)
            trace.append(step)
            if _is_unchanged(previous, answer):
                # The evidence changed nothing; keep the previous draft (the
                # model may have answered only with the NO CHANGE marker) and
                # stop instead of paying for another identical call.
                answer = previous
                break

    total = Usage()
    for s in trace:
        total = total + s.usage
    total.latency_s = time.perf_counter() - started  # wall-clock time of the whole run
    return Result(task.id, cfg.name, answer, trace, total)
