# Seminar Critique Agent

Multi-agent seminar critique tool using GraphRAG, aligned with the instructor's rubric.
CO5151 (Advanced Agentic AI), HCMUT, Group 7.

The tool helps students prepare a structured critique of a seminar paper along five axes:
assumptions, baseline fairness, data reliability, generalizability, reproducibility.

## Architecture (target, matches the D1 proposal)

```
material -> [retrieval: GraphRAG] -> Summarizer -> Critic (+ benchmark tool) -> Synthesizer
                                                                   |
                                      Reflection loop (max 2 iterations, external feedback)
```

Experiment configurations (see `seminar_critique/configs.py`):

| Config | What it does |
|---|---|
| A | One LLM call, no pipeline (simple baseline) |
| B | 3 agents with retrieval and the benchmark tool, no reflection |
| C | B plus the reflection loop |

## Status

| Part | Status |
|---|---|
| Skeleton, interfaces, contract tests | done (contract tests pass, FakeLLM kept for offline tests) |
| Real LLM client | real: GeminiClient with temperature, per-call model, token usage, latency and cost from price env vars (seminar_critique/llm/) |
| Retrieval (GraphRAG) | real: BM25 + graph expansion, 10 papers, 2 topics (retrieval/stats.json) |
| Tools and permissions | real: read-only benchmark lookup, 13 sourced records, permission allowlist; returns evidence for 4 of 9 eval tasks, the other 5 have no matching record ([details](seminar_critique/tools/README.md)) |
| Eval set and runner | real: runner and scorer, 9 tasks x 3 configs x 3 repeats = 81 runs (eval/results/results.csv, summary.md); scores are phrase-match proxies |
| Threat model | documented; input sanitization and prompt boundaries implemented, live injection evaluation pending ([details](seminar_critique/security/threat_model.md)) |

Update this table as parts become real. Be exact: "stub" means it does not do the real work yet.

## Layout

```
seminar_critique/
  schemas.py        shared data contracts
  llm/              LLM client (FakeLLM and GeminiClient)
  agents/           summarizer, critic, synthesizer, reflection
  orchestrator.py   pipeline and reflection loop
  configs.py        configurations A, B, C
  retrieval/        GraphRAG retrieval
  tools/            tool registry with allowlist, benchmark lookup
  security/         threat model and untrusted-content handling
  run.py            run one task, save the trace
eval/               tasks, runner, scorer, results
tests/              contract tests
third_party/MAD/    code and data from the MAD paper (not ours, see below)
```

## Quick start

```
pip install -r requirements.txt
pytest
python -m seminar_critique.run --task eval/tasks/example_task.json --config C
python -m eval.runner --repeats 3
```

Team workflow: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Third-party code and attribution

`third_party/MAD/` is the code and data of *Encouraging Divergent Thinking in Large Language
Models through Multi-Agent Debate* (Liang et al., 2023,
[Skytliang/Multi-Agents-Debate](https://github.com/Skytliang/Multi-Agents-Debate)),
licensed under GPL-3.0. It is kept as a reference for the retry logic and the debate pattern.
Results and data inside `third_party/` are **not** results of this project.

## Reproducibility

Numbers in the report come from `eval/results/*.csv`, produced by `python -m eval.runner`.
The report states the repository URL and the exact commit hash.