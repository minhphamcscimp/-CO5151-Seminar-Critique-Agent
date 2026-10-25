"""Summarize evaluation CSV means by configuration.

Run with ``python -m eval.summarize`` or pass ``--input`` and ``--output``.
Coverage columns are literal phrase-match proxies; they do not establish
semantic correctness.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from seminar_critique.configs import CONFIGS

METRICS = {
    "key_point_recall": "Key-point recall",
    "rubric_axis_coverage": "Rubric-axis coverage",
    "prompt_tokens": "Prompt tokens",
    "completion_tokens": "Completion tokens",
    "cost_usd": "Cost (USD)",
    "latency_s": "Latency (s)",
}
REQUIRED_FIELDS = {"task_id", "config", "run", *METRICS}


def _read_results(path: Path) -> dict[str, list[dict[str, float | str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing CSV columns: {', '.join(sorted(missing))}")
        rows_by_config: dict[str, list[dict[str, float | str]]] = defaultdict(list)
        for line_number, row in enumerate(reader, start=2):
            config = row["config"]
            if config not in CONFIGS:
                raise ValueError(f"{path}:{line_number}: unknown config {config!r}")
            parsed: dict[str, float | str] = {
                "task_id": row["task_id"],
                "run": row["run"],
            }
            for column in METRICS:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid number in {column!r}: {row[column]!r}"
                    ) from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"{path}:{line_number}: {column!r} must be finite and non-negative"
                    )
                parsed[column] = value
            rows_by_config[config].append(parsed)
    if not rows_by_config:
        raise ValueError(f"{path} contains no evaluation rows")
    return rows_by_config


def _complete_task_ids(
    rows_by_config: dict[str, list[dict[str, float | str]]],
    expected_repeats: int,
) -> tuple[set[str], set[str]]:
    task_runs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for config, rows in rows_by_config.items():
        for row in rows:
            task_runs[str(row["task_id"])][config].add(str(row["run"]))

    complete: set[str] = set()
    incomplete = set(task_runs)
    for task_id, configs in task_runs.items():
        if set(configs) != set(CONFIGS):
            continue
        run_sets = list(configs.values())
        expected_runs = {str(run) for run in range(expected_repeats)}
        if all(runs == expected_runs for runs in run_sets):
            complete.add(task_id)
    incomplete.difference_update(complete)
    return complete, incomplete


def _render(
    rows_by_config: dict[str, list[dict[str, float | str]]],
    expected_repeats: int = 3,
) -> str:
    if expected_repeats < 1:
        raise ValueError("expected_repeats must be at least 1")
    complete_task_ids, incomplete_task_ids = _complete_task_ids(
        rows_by_config, expected_repeats
    )
    columns = ["Config", "Tasks", "Runs", *METRICS.values()]
    lines = [
        "# Evaluation summary",
        "",
        "Arithmetic means across runs in `results.csv`. Coverage metrics are literal "
        "phrase-match proxies, not semantic correctness judgments.",
        "",
    ]
    if not complete_task_ids:
        lines.extend([
            "**No complete task available; metrics are withheld.** Each included task "
            "must have matching run numbers for configurations A, B, and C.",
            "",
        ])
        if incomplete_task_ids:
            lines.extend([
                f"Incomplete tasks excluded: {', '.join(sorted(incomplete_task_ids))}.",
                "",
            ])
        return "\n".join(lines) + "\n"

    if len(complete_task_ids) < 5:
        lines.extend([
            f"**Warning:** only {len(complete_task_ids)} complete task(s) are included; "
            "the project plan requires at least 5 for reporting.",
            "",
        ])
    if incomplete_task_ids:
        lines.extend([
            f"Incomplete tasks excluded: {', '.join(sorted(incomplete_task_ids))}.",
            "",
        ])
    lines.extend([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ])
    for config in CONFIGS:
        rows = [
            row for row in rows_by_config.get(config, [])
            if str(row["task_id"]) in complete_task_ids
        ]
        if not rows:
            continue
        task_count = len({str(row["task_id"]) for row in rows})
        values = [
            config,
            str(task_count),
            str(len(rows)),
            *(f"{statistics.fmean(float(row[column]) for row in rows):.3f}" for column in METRICS),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="eval/results/results.csv")
    parser.add_argument("--output", default="eval/results/summary.md")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    summary = _render(_read_results(Path(args.input)), expected_repeats=args.repeats)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
