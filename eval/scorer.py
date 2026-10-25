"""Deterministic lexical coverage metrics for evaluation tasks.

These metrics measure whether expected phrases are mentioned, not whether the
answer is semantically correct. Human validation is still needed for report claims.
"""
from __future__ import annotations

import re
import unicodedata

from seminar_critique.schemas import Result, Task


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+", normalized)


def _coverage(expected: list[str], answer_tokens: list[str]) -> float:
    if not expected:
        return 0.0

    hits = 0
    for phrase in expected:
        phrase_tokens = _tokens(phrase)
        if not phrase_tokens:
            continue
        width = len(phrase_tokens)
        if any(
            answer_tokens[index:index + width] == phrase_tokens
            for index in range(len(answer_tokens) - width + 1)
        ):
            hits += 1
    return hits / len(expected)


def score(task: Task, result: Result) -> dict[str, float]:
    """Return phrase recall and rubric-axis phrase coverage in the answer."""
    answer_tokens = _tokens(result.answer)
    return {
        "key_point_recall": _coverage(task.key_points, answer_tokens),
        "rubric_axis_coverage": _coverage(task.rubric_axes, answer_tokens),
    }
