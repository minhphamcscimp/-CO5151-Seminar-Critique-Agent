"""Benchmark lookup tool. Owner: Tin.

CONTRACT: lookup_benchmark(name, metric) -> dict   (READ ONLY)

Gives the Critic and the Reflection agent real external feedback,
for example "what is the published accuracy of method X on benchmark Y".
Read the answers from the fixed local file data/benchmarks.json,
never from a live write-capable service.
"""
from __future__ import annotations

import json
from pathlib import Path


BENCHMARKS_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmarks.json"


def lookup_benchmark(name: str, metric: str) -> dict:
    """Look up a benchmark/metric pair, ignoring case and surrounding whitespace.

    Values use the paper's 0–100 scale. Missing or malformed files raise an
    error; they must not be confused with a valid lookup that found no match.
    """
    if not isinstance(name, str) or not isinstance(metric, str):
        raise TypeError("name and metric must be strings")
    with BENCHMARKS_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("benchmarks.json must contain a list of records")

    matches = []
    for record in records:
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key].strip()
            for key in ("benchmark", "metric", "method", "source")
        ):
            raise ValueError("benchmark records need benchmark, metric, method and source")
        value = record.get("value")
        if type(value) not in (int, float) or not 0 <= value <= 100:
            raise ValueError("benchmark value must be a finite number between 0 and 100")
        if (record["benchmark"].casefold(), record["metric"].casefold()) == (
            name.strip().casefold(), metric.strip().casefold()
        ):
            matches.append(record)

    # shortcut: one published method per benchmark/metric, extend the contract if comparisons are needed.
    if len(matches) > 1:
        raise ValueError("ambiguous benchmark/metric pair in benchmarks.json")
    if matches:
        return {**matches[0], "found": True}
    return {"found": False, "source": None}
