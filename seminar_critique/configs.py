"""Experiment configurations A / B / C. Owners: Tam (definition) and Van (code).

Agreed meaning (Van + Tam, 24/10):
    A = one single LLM call (baseline)
    B = Summarizer -> Critic -> Synthesizer, with retrieval and the benchmark tool
    C = B plus a reflection loop of at most 2 iterations
Rule from the proposal: keep the SAME model across configs so only one thing
changes at a time. `resolve_models` enforces that at run time.
"""
from __future__ import annotations

from dataclasses import replace

from .schemas import Config

CONFIGS: dict[str, Config] = {
    # A: simplest baseline, one LLM call, no pipeline
    "A": Config(name="A", use_pipeline=False),
    # B: three agents with retrieval and the benchmark tool, no reflection
    "B": Config(name="B", use_pipeline=True, use_retrieval=True, use_tools=True),
    # C: B plus the reflection loop (max 2 iterations)
    "C": Config(
        name="C",
        use_pipeline=True,
        use_retrieval=True,
        use_tools=True,
        max_reflection_iters=2,
    ),
}


def resolve_models(cfg: Config, llm_name: str) -> Config:
    """Return a copy of `cfg` whose placeholder models fit the client in use.

    The configs ship with "fake-model" so the whole team can run offline.
    When the pipeline runs on Gemini, every agent switches to the one real
    model (GEMINI_MODEL, default gemini-3.1-flash-lite): same model for A, B
    and C, which is what the comparison in Part 5 needs. A model that was set
    on purpose (anything other than "fake-model") is left alone.
    """
    if llm_name != "gemini":
        return cfg
    from .llm import DEFAULT_GEMINI_MODEL

    models = {
        agent: (DEFAULT_GEMINI_MODEL if name == "fake-model" else name)
        for agent, name in cfg.models.items()
    }
    return replace(cfg, models=models)
