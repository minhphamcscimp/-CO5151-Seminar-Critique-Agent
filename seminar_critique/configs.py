"""Experiment configurations A / B / C. Owners: Tam (definition) and Van (code).

TODO (Tam + Van): agree on the final meaning of A / B / C and write it in the report (Part 5).
Rule from the proposal: keep the SAME model across configs so only one thing changes at a time.
"""
from __future__ import annotations

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
