"""Experiment runner. Owner: Tam.

    python -m eval.runner --tasks eval/tasks --configs A B C --repeats 3

Runs every task with every config several times, saves each trace under traces/,
and writes one CSV row per run into eval/results/. Commit the CSV, not the traces,
so the numbers in the report can be reproduced from the repository.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

from seminar_critique.configs import CONFIGS
from seminar_critique.llm import get_client
from seminar_critique.orchestrator import run_pipeline
from seminar_critique.run import save_trace
from seminar_critique.schemas import Task

from .scorer import score

FIELDS = [
    "task_id", "config", "run", "key_point_recall",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_s",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="eval/tasks")
    p.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=sorted(CONFIGS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--llm", choices=["fake", "gemini"], default="fake")
    p.add_argument("--out", default="eval/results/results.csv")
    args = p.parse_args()

    llm = get_client(args.llm)
    tasks = [Task.from_json(f) for f in sorted(glob.glob(os.path.join(args.tasks, "*.json")))]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for task in tasks:
            for name in args.configs:
                for run in range(args.repeats):
                    result = run_pipeline(task, CONFIGS[name], llm)
                    save_trace(result, "traces", run)
                    s = score(task, result)
                    writer.writerow({
                        "task_id": task.id, "config": name, "run": run,
                        "key_point_recall": s["key_point_recall"],
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "cost_usd": result.usage.cost_usd,
                        "latency_s": round(result.usage.latency_s, 3),
                    })
    print(f"wrote {args.out} for {len(tasks)} task(s)")


if __name__ == "__main__":
    main()
