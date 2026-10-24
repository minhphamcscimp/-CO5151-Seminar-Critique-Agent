"""Benchmark lookup tool. Owner: Tin.

CONTRACT: lookup_benchmark(name, metric) -> dict   (READ ONLY)

Gives the Critic and the Reflection agent real external feedback,
for example "what is the published accuracy of method X on benchmark Y".
Read the answers from a local file (for example data/benchmarks.json),
never from a live write-capable service.
"""
from __future__ import annotations


def lookup_benchmark(name: str, metric: str) -> dict:
    """STUB: returns a fixed record.

    TODO (Tin): load a local JSON of benchmark numbers with their source
    (paper, table), and return {"found": False} when nothing matches.
    """
    return {
        "found": True,
        "benchmark": name,
        "metric": metric,
        "value": None,
        "source": "(stub) replace with paper and table reference",
    }
