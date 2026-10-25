# Tools and permissions — Tin / Report Part 3

| Tool | Purpose | Implementation | Permission | Enforcement |
|---|---|---|---|---|
| `lookup_benchmark(name, metric)` | Supply a published benchmark score with method and provenance | Local JSON lookup, 13 records | `READ_ONLY` | Fixed `BENCHMARKS_PATH`, `open("r")`, no path argument; registry checks permission before execution |
| Unregistered tools / tools marked write or code execution | Outside D2 scope | Blocked | Denied | `call_tool` raises `PermissionError` before invoking the function |

`seminar_critique/tools/__init__.py::call_tool` is the agent tool entry point.
`TOOL_REGISTRY` contains only the benchmark lookup; `ALLOWED_PERMISSIONS` contains
only `READ_ONLY`. The registered function in `benchmark_lookup.py` loads
`data/benchmarks.json` relative to its own module, independent of the working
directory. It reads no secrets, calls no network services and opens no write mode.
The registry is a code policy; developers remain responsible for registering only
implementations that actually respect their permission label.

```python
from seminar_critique.tools import call_tool

out = call_tool("lookup_benchmark", name="gsm8k", metric="ACCURACY")
assert out["result"]["value"] == 78.2
assert "Table 2" in out["result"]["source"]
```

`call_tool` returns `{tool, args, result}`. A hit includes `found`, `benchmark`,
`metric`, `method`, `value`, and `source`; a miss includes `found: false` and
`source: null`. Matching is exact after stripping surrounding whitespace and
case-folding both arguments. Unknown metrics are not silently substituted. File
and schema errors propagate rather than becoming a fabricated result.

## Handle missing evidence and blocked tools

A missing pair means there is no matching published evidence in the local file;
it must not be interpreted as a score of zero. Check `found` before reading
`value` or citing `source`:

```python
out = call_tool("lookup_benchmark", name="unknown-benchmark", metric="accuracy")
result = out["result"]
if result["found"]:
    print(result["value"], result["source"])
else:
    print("No matching published result; no score available.")
```

A blocked tool raises `PermissionError` instead of returning a lookup result.
Report that error to the caller; do not retry by changing the allowlist:

```python
try:
    call_tool("delete_everything")
except PermissionError as error:
    print(f"Tool call blocked: {error}")
```

Missing files and malformed benchmark data are operational errors, distinct from
a valid lookup miss. Preserve those exceptions so the dataset can be repaired.

## Published data (verified 2026-10-25)

Both papers are already listed in `data/papers.md`. Every record identifies the
paper, pinned version, table, row/model, scale and URL in `source`.

| Paper | Table / selected method | Records |
|---|---|---|
| [Gou et al., CRITIC, v4](https://arxiv.org/html/2305.11738v4) | Table 1, CRITIC with gpt-3.5-turbo, ordinary non-oracle row | AmbigNQ, TriviaQA, HotpotQA: `exact_match` and `f1` (6 records) |
| [Gou et al., CRITIC, v4](https://arxiv.org/html/2305.11738v4) | Table 2, same model/method | GSM8K, SVAMP, TabMWP: `accuracy` (3 records) |
| [Yao et al., ReAct, v3](https://arxiv.org/html/2210.03629v3) | Tables 1, 3, 4, PaLM-540B | FEVER `accuracy`; ALFWorld `success_rate` (average over prompts); WebShop `score` and `success_rate` (4 records) |

All values use a 0–100 scale, not a 0–1 fraction. For CRITIC's math tasks,
`accuracy` means the paper's rounded numeric exact-match score. For QA, use the
separate `exact_match` and `f1` metrics. ALFWorld uses the average row, not the
best-of-six result. These are published external results, not measurements of our
own system; different models/tasks/settings are not directly comparable.

The D2 contract selects one published method per benchmark/metric. Duplicates for
the requested pair raise `ValueError`; adding a method selector requires team
agreement on the shared interface.

## Integration handoff to Van

- Supply a benchmark name and its metric, e.g. `GSM8K` / `accuracy`. The current
  orchestrator passes `task.topic`; topic IDs intentionally return no match.
- `agents.base.call_agent` appends `UNTRUSTED_CONTENT_POLICY` to every system
  message. Keep this policy when replacing the stub prompts.
- Critic and Reflection sanitize the JSON rendering of tool evidence. The raw
  structured result remains in `TraceStep.tool_calls` for debugging and auditing.
- Retrieved text/metadata is sanitized together in Summarizer, at the point it is
  formatted for the prompt. Summaries, critiques and drafts are also wrapped when
  forwarded to another agent. Function signatures and `schemas.py` are unchanged.
- Review these small integration edits with Van before merging. Do not describe
  boundary unit tests as live prompt-injection experiments. See
  [the threat model](../security/threat_model.md) for implementation limits.

Validation: `python -m pytest`; CLI smoke check:
`python -m seminar_critique.run --task eval/tasks/example_task.json --config C`.
