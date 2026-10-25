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


def _render(rows_by_config: dict[str, list[dict[str, float | str]]]) -> str:
    columns = ["Config", "Tasks", "Runs", *METRICS.values()]
    lines = [
        "# Evaluation summary",
        "",
        "Arithmetic means across runs in `results.csv`. Coverage metrics are literal "
        "phrase-match proxies, not semantic correctness judgments.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for config in CONFIGS:
        rows = rows_by_config.get(config)
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
    args = parser.parse_args()

    summary = _render(_read_results(Path(args.input)))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
