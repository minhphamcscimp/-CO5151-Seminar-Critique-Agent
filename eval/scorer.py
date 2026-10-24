"""Scoring. Owner: Tam.

CONTRACT: score(task, result) -> dict with at least {"key_point_recall": float}

Decide in the report (Part 5.1) what counts as correct:
the key_points come from the instructor's "Key defense points" and "Required" lists.
If you add an LLM-as-judge, check it on a sample labeled by humans (Part 5.5)
and report the agreement.
"""
from __future__ import annotations

from seminar_critique.schemas import Result, Task


def score(task: Task, result: Result) -> dict:
    """STUB: share of key points that appear as plain text in the answer.

    TODO (Tam): replace with a better rule or an LLM judge, keep the return keys.
    """
    if not task.key_points:
        return {"key_point_recall": 0.0}
    answer = result.answer.lower()
    hit = sum(1 for k in task.key_points if k.lower() in answer)
    return {"key_point_recall": hit / len(task.key_points)}
