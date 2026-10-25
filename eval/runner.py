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
from pathlib import Path

from seminar_critique.configs import CONFIGS
from seminar_critique.llm import get_client
from seminar_critique.orchestrator import run_pipeline
from seminar_critique.run import save_trace
from seminar_critique.schemas import Task

from .scorer import score

FIELDS = [
    "task_id", "config", "run", "key_point_recall", "rubric_axis_coverage",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_s",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="eval/tasks")
    p.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=sorted(CONFIGS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--llm", choices=["fake", "gemini"], default="fake")
    p.add_argument("--out", default="eval/results/results.csv")
    p.add_argument("--traces", default="traces")
    args = p.parse_args()

    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    task_files = sorted(glob.glob(os.path.join(args.tasks, "*.json")))
    if not task_files:
        p.error(f"no task JSON files found in {args.tasks!r}")
    tasks = [Task.from_json(path) for path in task_files]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    llm = get_client(args.llm)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for task in tasks:
            for name in args.configs:
                for run in range(args.repeats):
                    result = run_pipeline(task, CONFIGS[name], llm)
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
    run_count = len(tasks) * len(args.configs) * args.repeats
    print(f"wrote {output_path} with {run_count} run(s) for {len(tasks)} task(s)")


if __name__ == "__main__":
    main()
