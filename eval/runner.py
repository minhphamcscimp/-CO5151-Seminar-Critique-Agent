"""Experiment runner. Owner: Tam.

    python -m eval.runner --tasks eval/tasks --configs A B C --repeats 3

Runs every task with every config several times, saves each trace under traces/,
and writes one CSV row per run into eval/results/. Commit the CSV, not the traces,
so the numbers in the report can be reproduced from the repository. The scorer's
coverage values are deterministic phrase matches, not semantic correctness scores.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from pathlib import Path
from typing import Any

from seminar_critique.configs import CONFIGS, resolve_models
from seminar_critique.llm import LLMClient, LLMResponse, OutOfQuotaError, get_client
from seminar_critique.orchestrator import run_pipeline
from seminar_critique.run import save_trace
from seminar_critique.schemas import Task

from .scorer import score

FIELDS = [
    "task_id", "config", "run", "key_point_recall", "rubric_axis_coverage",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_s",
]


class _PacedClient:
    def __init__(self, client: LLMClient, min_interval_s: float) -> None:
        self._client = client
        self._min_interval_s = min_interval_s
        self._last_call_started: float | None = None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        if self._last_call_started is not None:
            elapsed = time.monotonic() - self._last_call_started
            delay = self._min_interval_s - elapsed
            if delay > 0:
                time.sleep(delay)
        self._last_call_started = time.monotonic()
        return self._client.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="eval/tasks")
    p.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=sorted(CONFIGS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--llm", choices=["fake", "gemini"], default="fake")
    p.add_argument("--out", default="eval/results/results.csv")
    p.add_argument("--traces", default="traces")
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from completed task/config/repeat rows already in --out",
    )
    p.add_argument(
        "--gemini-call-interval",
        type=float,
        default=6.0,
        help="minimum seconds between Gemini API request starts (default: 6, about 10/min)",
    )
    args = p.parse_args()

    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    if args.gemini_call_interval < 0:
        p.error("--gemini-call-interval must be non-negative")
    task_files = sorted(glob.glob(os.path.join(args.tasks, "*.json")))
    if not task_files:
        p.error(f"no task JSON files found in {args.tasks!r}")
    tasks = [Task.from_json(path) for path in task_files]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_runs: set[tuple[str, str, int]] = set()
    if args.resume and output_path.exists() and output_path.stat().st_size:
        with output_path.open(newline="", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            if reader.fieldnames != FIELDS:
                p.error(f"cannot resume: {output_path} does not have the expected CSV columns")
            task_ids = {task.id for task in tasks}
            for line_number, row in enumerate(reader, start=2):
                try:
                    task_id = row["task_id"]
                    config = row["config"]
                    run = int(row["run"])
                except (KeyError, TypeError, ValueError) as exc:
                    p.error(f"cannot resume: invalid row {line_number} in {output_path}: {exc}")
                if task_id not in task_ids or config not in args.configs or not 0 <= run < args.repeats:
                    p.error(
                        f"cannot resume: row {line_number} does not match the requested tasks, "
                        "configs, and repeats"
                    )
                key = (task_id, config, run)
                if key in completed_runs:
                    p.error(f"cannot resume: duplicate task/config/run row at line {line_number}")
                completed_runs.add(key)
    llm = get_client(args.llm)
    if args.llm == "gemini":
        llm = _PacedClient(llm, args.gemini_call_interval)

    file_mode = "a" if args.resume and output_path.exists() and output_path.stat().st_size else "w"
    with output_path.open(file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if file_mode == "w":
            writer.writeheader()
            f.flush()
        completed_tasks = 0
        completed_runs_count = len(completed_runs)
        interrupted = False
        current_run: tuple[str, str, int] | None = None
        try:
            for task in tasks:
                for name in args.configs:
                    for run in range(args.repeats):
                        key = (task.id, name, run)
                        if key in completed_runs:
                            continue
                        current_run = key
                        print(
                            f"starting run {completed_runs_count + 1}/"
                            f"{len(tasks) * len(args.configs) * args.repeats}: "
                            f"{task.id} config {name} repeat {run + 1}/{args.repeats}",
                            flush=True,
                        )
                        call_started = time.perf_counter()
                        config = resolve_models(CONFIGS[name], args.llm)
                        result = run_pipeline(task, config, llm)
                        s = score(task, result)
                        save_trace(result, args.traces, run)
                        writer.writerow({
                            "task_id": task.id, "config": name, "run": run,
                            "key_point_recall": s["key_point_recall"],
                            "rubric_axis_coverage": s["rubric_axis_coverage"],
                            "prompt_tokens": result.usage.prompt_tokens,
                            "completion_tokens": result.usage.completion_tokens,
                            "cost_usd": result.usage.cost_usd,
                            "latency_s": round(result.usage.latency_s, 3),
                        })
                        f.flush()
                        completed_runs.add(key)
                        completed_runs_count += 1
                        current_run = None
                        print(
                            f"completed {completed_runs_count}/"
                            f"{len(tasks) * len(args.configs) * args.repeats} runs: "
                            f"{task.id} config {name} repeat {run + 1}/{args.repeats} "
                            f"({time.perf_counter() - call_started:.1f}s)",
                            flush=True,
                        )
                if all(
                    (task.id, config, run) in completed_runs
                    for config in args.configs
                    for run in range(args.repeats)
                ):
                    completed_tasks += 1
                    print(f"completed task {completed_tasks}/{len(tasks)}: {task.id}", flush=True)
        except OutOfQuotaError as exc:
            p.exit(
                1,
                f"Gemini quota exhausted: {exc}. The current run was not saved; "
                f"completed rows remain in {output_path}. After quota resets, rerun "
                "the same command with --resume.\n",
            )
        except KeyboardInterrupt:
            interrupted = True
            active_run = (
                f" Current unreturned run: {current_run[0]} config {current_run[1]} "
                f"repeat {current_run[2] + 1}."
                if current_run
                else ""
            )
            print(
                f"interrupted; kept {completed_runs_count} completed run(s)."
                f"{active_run} Rerun with --resume to continue.",
                flush=True,
            )
    run_count = completed_runs_count
    status = "stopped" if interrupted else "wrote"
    print(f"{status} {output_path} with {run_count} completed run(s)")


if __name__ == "__main__":
    main()
