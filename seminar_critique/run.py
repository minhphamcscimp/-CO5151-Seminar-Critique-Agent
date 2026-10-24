"""Run one task with one config and save the trace.

    python -m seminar_critique.run --task eval/tasks/example_task.json --config C

Uses the offline FakeLLM by default, so it works without any API key.
The trace JSON is what the report shows as the "real trace" (Part 2).
"""
from __future__ import annotations

import argparse
import json
import os

from .configs import CONFIGS
from .llm import get_client
from .orchestrator import run_pipeline
from .schemas import Task


def save_trace(result, out_dir: str, run_id: int = 0) -> str:
    path = os.path.join(out_dir, result.config_name)
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, f"{result.task_id}_run{run_id}.json")
    with open(file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--config", choices=sorted(CONFIGS), default="C")
    p.add_argument("--llm", choices=["fake", "gemini"], default="fake")
    p.add_argument("--out", default="traces")
    args = p.parse_args()

    result = run_pipeline(Task.from_json(args.task), CONFIGS[args.config], get_client(args.llm))
    file = save_trace(result, args.out)
    print(f"steps={len(result.trace)} tokens={result.usage.prompt_tokens + result.usage.completion_tokens} "
          f"latency={result.usage.latency_s:.3f}s trace={file}")


if __name__ == "__main__":
    main()
