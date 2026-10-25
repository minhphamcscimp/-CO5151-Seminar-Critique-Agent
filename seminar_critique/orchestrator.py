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
from .llm import LLMClient, get_client
from .retrieval import retrieve
from .schemas import Chunk, Config, Result, Task, TraceStep, Usage
from .security import sanitize_untrusted
from .tools import call_tool

BASELINE_SYSTEM = "Write a structured critique of the following seminar material."


def run_pipeline(task: Task, cfg: Config, llm: LLMClient | None = None) -> Result:
    llm = llm or get_client("fake")
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
            answer, step = reflect(answer, tool_results, llm, cfg)
            trace.append(step)
            # TODO (Van): stop early when the evidence changes nothing.

    total = Usage()
    for s in trace:
        total = total + s.usage
    total.latency_s = time.perf_counter() - started  # wall-clock time of the whole run
    return Result(task.id, cfg.name, answer, trace, total)
