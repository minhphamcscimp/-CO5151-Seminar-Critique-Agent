"""Tests for the evaluation scorer, runner, task set, and summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from eval import runner, summarize
from eval.scorer import score
from seminar_critique.llm import LLMResponse, OutOfQuotaError
from seminar_critique.schemas import Result, Task, Usage


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
        [
            "eval.summarize",
            "--input", str(results_path),
            "--output", str(summary_path),
            "--repeats", "1",
        ],
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


def test_summary_excludes_partial_tasks_and_warns_below_five(tmp_path: Path):
    complete_rows = []
    for config in ("A", "B", "C"):
        complete_rows.append({
            "task_id": "complete",
            "config": config,
            "run": "0",
            "key_point_recall": "0.5",
            "rubric_axis_coverage": "0.4",
            "prompt_tokens": "10",
            "completion_tokens": "5",
            "cost_usd": "0.01",
            "latency_s": "1.0",
        })
    partial_row = {
        **complete_rows[0],
        "task_id": "interrupted",
        "config": "A",
    }
    rows_by_config = {
        "A": [complete_rows[0], partial_row],
        "B": [complete_rows[1]],
        "C": [complete_rows[2]],
    }

    summary = summarize._render(rows_by_config, expected_repeats=1)

    assert "**Warning:** only 1 complete task(s)" in summary
    assert "Incomplete tasks excluded: interrupted." in summary
    assert "| A | 1 | 1 |" in summary
    assert "| B | 1 | 1 |" in summary
    assert "| C | 1 | 1 |" in summary


def test_summary_with_no_complete_tasks_withholds_metrics():
    partial_rows = {
        "A": [{
            "task_id": "interrupted",
            "config": "A",
            "run": "0",
            "key_point_recall": 0.5,
            "rubric_axis_coverage": 0.4,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost_usd": 0.01,
            "latency_s": 1.0,
        }],
    }

    summary = summarize._render(partial_rows, expected_repeats=1)

    assert "**No complete task available; metrics are withheld.**" in summary
    assert "Incomplete tasks excluded: interrupted." in summary


def test_runner_keeps_only_fully_completed_tasks_when_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for task_id in ("01-complete", "02-interrupted"):
        task = {
            "id": task_id,
            "topic": "self-improvement-reflection",
            "input": "Critique this claim.",
            "rubric_axes": ["assumptions"],
            "key_points": ["external feedback"],
        }
        (tasks_dir / f"{task_id}.json").write_text(json.dumps(task), encoding="utf-8")

    results_path = tmp_path / "results.csv"
    traces_dir = tmp_path / "traces"
    calls = []

    def run_until_interrupted(task, config, _llm):
        if task.id == "02-interrupted":
            raise KeyboardInterrupt
        calls.append((task.id, config.name))
        return Result(
            task_id=task.id,
            config_name=config.name,
            answer="external feedback",
        )

    monkeypatch.setattr(runner, "run_pipeline", run_until_interrupted)
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
    assert len(calls) == 3
    assert len(rows) == 3
    assert {row["task_id"] for row in rows} == {"01-complete"}
    assert {row["config"] for row in rows} == {"A", "B", "C"}


def test_paced_client_enforces_minimum_between_request_starts(monkeypatch: pytest.MonkeyPatch):
    now = [10.0]
    sleeps = []
    client = Mock()

    def complete(*_args, **_kwargs):
        now[0] += 1.0
        return LLMResponse(text="ok", usage=Usage())

    def sleep(duration):
        sleeps.append(duration)
        now[0] += duration

    client.complete.side_effect = complete
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runner.time, "sleep", sleep)
    paced = runner._PacedClient(client, min_interval_s=4.1)

    paced.complete([], model="test-model")
    paced.complete([], model="test-model")

    assert sleeps == pytest.approx([3.1])
    assert client.complete.call_count == 2


def test_runner_resolves_gemini_model_before_wrapping_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    task = {
        "id": "gemini-model",
        "topic": "self-improvement-reflection",
        "input": "Critique this claim.",
        "rubric_axes": ["assumptions"],
        "key_points": ["external feedback"],
    }
    (tasks_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    configs_seen = []

    def run_pipeline(task, config, llm):
        configs_seen.append(config)
        return Result(task_id=task.id, config_name=config.name, answer="answer")

    monkeypatch.setattr(runner, "get_client", lambda _: Mock())
    monkeypatch.setattr(runner, "run_pipeline", run_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval.runner",
            "--tasks", str(tasks_dir),
            "--configs", "A",
            "--repeats", "1",
            "--llm", "gemini",
            "--out", str(tmp_path / "results.csv"),
            "--traces", str(tmp_path / "traces"),
        ],
    )

    runner.main()

    assert configs_seen[0].models["baseline"] != "fake-model"


def test_runner_resume_skips_completed_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    task = {
        "id": "resume",
        "topic": "self-improvement-reflection",
        "input": "Critique this claim.",
        "rubric_axes": ["assumptions"],
        "key_points": ["external feedback"],
    }
    (tasks_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    results_path = tmp_path / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=runner.FIELDS)
        writer.writeheader()
        writer.writerow({
            "task_id": "resume",
            "config": "A",
            "run": 0,
            "key_point_recall": 0.0,
            "rubric_axis_coverage": 0.0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cost_usd": 0.0,
            "latency_s": 0.1,
        })

    configs_seen = []

    def run_pipeline(task, config, _llm):
        configs_seen.append(config.name)
        return Result(task_id=task.id, config_name=config.name, answer="answer")

    monkeypatch.setattr(runner, "run_pipeline", run_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval.runner",
            "--tasks", str(tasks_dir),
            "--repeats", "1",
            "--resume",
            "--out", str(results_path),
            "--traces", str(tmp_path / "traces"),
        ],
    )

    runner.main()

    with results_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert configs_seen == ["B", "C"]
    assert len(rows) == 3
    assert [row["config"] for row in rows] == ["A", "B", "C"]


def test_runner_preserves_completed_rows_when_quota_is_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    task = {
        "id": "quota",
        "topic": "self-improvement-reflection",
        "input": "Critique this claim.",
        "rubric_axes": ["assumptions"],
        "key_points": ["external feedback"],
    }
    (tasks_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    results_path = tmp_path / "results.csv"

    def run_pipeline(task, config, _llm):
        if config.name == "B":
            raise OutOfQuotaError("Gemini quota exhausted for test-model")
        return Result(task_id=task.id, config_name=config.name, answer="answer")

    monkeypatch.setattr(runner, "run_pipeline", run_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval.runner",
            "--tasks", str(tasks_dir),
            "--configs", "A", "B",
            "--repeats", "1",
            "--out", str(results_path),
            "--traces", str(tmp_path / "traces"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.main()

    assert exc_info.value.code == 1
    with results_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [(row["config"], row["run"]) for row in rows] == [("A", "0")]
    error_output = capsys.readouterr().err
    assert "current run was not saved" in error_output
    assert "with --resume" in error_output
