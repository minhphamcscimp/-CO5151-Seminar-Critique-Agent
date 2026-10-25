"""Tests for the evaluation scorer, runner, task set, and summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eval import runner, summarize
from eval.scorer import score
from seminar_critique.schemas import Result, Task


def test_score_matches_case_and_punctuation_and_reports_axis_coverage():
    task = Task(
        id="score",
        topic="test",
        input="",
        rubric_axes=["baseline fairness", "reproducibility"],
        key_points=["external feedback", "self-correction"],
    )
    result = Result(
        task_id=task.id,
        config_name="A",
        answer="External-feedback matters; SELF correction is discussed. "
        "Baseline fairness was considered.",
    )

    assert score(task, result) == {
        "key_point_recall": 1.0,
        "rubric_axis_coverage": 0.5,
    }


def test_score_returns_zero_for_empty_lists_and_does_not_match_empty_phrases():
    task = Task(
        id="empty",
        topic="test",
        input="",
        rubric_axes=[""],
        key_points=[],
    )
    result = Result(task_id=task.id, config_name="A", answer="anything")

    assert score(task, result) == {
        "key_point_recall": 0.0,
        "rubric_axis_coverage": 0.0,
    }


def test_task_set_has_valid_contracts_and_covers_two_topics():
    task_paths = sorted(Path("eval/tasks").glob("*.json"))
    tasks = [Task.from_json(str(path)) for path in task_paths]

    assert len(tasks) == 9
    assert len({task.id for task in tasks}) == len(tasks)
    assert {task.topic for task in tasks} >= {
        "self-improvement-reflection",
        "tool-use-mcp",
    }
    for task in tasks:
        assert task.id and task.input
        assert len(task.key_points) in range(3, 6)
        assert len(task.rubric_axes) == 5


def test_runner_and_summary_with_fake_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    task = {
        "id": "smoke",
        "topic": "self-improvement-reflection",
        "input": "Critique this claim.",
        "rubric_axes": ["assumptions"],
        "key_points": ["external feedback"],
    }
    (tasks_dir / "smoke.json").write_text(json.dumps(task), encoding="utf-8")
    results_path = tmp_path / "results.csv"
    traces_dir = tmp_path / "traces"

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval.runner",
            "--tasks", str(tasks_dir),
            "--repeats", "1",
            "--out", str(results_path),
            "--traces", str(traces_dir),
        ],
    )
    runner.main()

    with results_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 3
    assert {row["config"] for row in rows} == {"A", "B", "C"}
    assert {"key_point_recall", "rubric_axis_coverage"} <= set(rows[0])
    assert all(float(row["prompt_tokens"]) > 0 for row in rows)

    summary_path = tmp_path / "summary.md"
    monkeypatch.setattr(
        "sys.argv",
        ["eval.summarize", "--input", str(results_path), "--output", str(summary_path)],
    )
    summarize.main()
    summary = summary_path.read_text(encoding="utf-8")
    assert "| A | 1 | 1 |" in summary
    assert "| B | 1 | 1 |" in summary
    assert "| C | 1 | 1 |" in summary
    assert "phrase-match proxies" in summary


def test_summary_rejects_empty_results(tmp_path: Path):
    results_path = tmp_path / "empty.csv"
    results_path.write_text(
        "task_id,config,run,key_point_recall,rubric_axis_coverage,"
        "prompt_tokens,completion_tokens,cost_usd,latency_s\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no evaluation rows"):
        summarize._read_results(results_path)
