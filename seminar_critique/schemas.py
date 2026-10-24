"""Shared data contracts for the whole team.

RULE: changing anything in this file affects all four modules.
Open a PR titled "[contract] ..." and get a review from every module owner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Usage:
    """Cost accounting for one LLM call or one whole run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost_usd + other.cost_usd,
            self.latency_s + other.latency_s,
        )


@dataclass
class Chunk:
    """One retrieved piece of a paper (owner: retrieval)."""

    id: str
    paper: str
    section: str
    text: str
    score: float = 0.0


@dataclass
class Task:
    """One evaluation task (owner: eval). Loaded from eval/tasks/*.json."""

    id: str
    topic: str
    input: str  # the seminar material the agents must critique
    rubric_axes: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)  # expected findings

    @classmethod
    def from_json(cls, path: str) -> "Task":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


@dataclass
class Config:
    """One experiment configuration A / B / C (owner: eval + orchestrator)."""

    name: str
    use_pipeline: bool = True  # False = single LLM call (baseline)
    use_retrieval: bool = False
    use_tools: bool = False
    max_reflection_iters: int = 0
    temperature: float = 0.0
    # model per agent, so strong and cheap models can be mixed
    models: dict[str, str] = field(
        default_factory=lambda: {
            "baseline": "fake-model",
            "summarizer": "fake-model",
            "critic": "fake-model",
            "synthesizer": "fake-model",
            "reflection": "fake-model",
        }
    )


@dataclass
class TraceStep:
    """One step of a run. The list of steps is the trace shown in the report."""

    agent: str
    input: str
    output: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass
class Result:
    """Output of run_pipeline (owner: orchestrator)."""

    task_id: str
    config_name: str
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
