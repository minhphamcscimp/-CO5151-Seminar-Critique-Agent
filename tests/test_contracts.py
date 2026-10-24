"""Contract tests. Run with: pytest

They only check that the four interfaces fit together, using stubs and FakeLLM.
Each owner adds tests for their own module next to this file.
"""
import pytest

from seminar_critique.configs import CONFIGS
from seminar_critique.llm import FakeLLM
from seminar_critique.orchestrator import run_pipeline
from seminar_critique.retrieval import retrieve
from seminar_critique.schemas import Chunk, Task
from seminar_critique.tools import call_tool, TOOL_REGISTRY, Permission

TASK = Task(
    id="t1",
    topic="self-improvement-reflection",
    input="Critique this claim.",
    rubric_axes=["assumptions"],
    key_points=["fake"],
)


@pytest.mark.parametrize("name,steps", [("A", 1), ("B", 3), ("C", 5)])
def test_pipeline_runs_for_each_config(name, steps):
    result = run_pipeline(TASK, CONFIGS[name], FakeLLM())
    assert result.config_name == name
    assert len(result.trace) == steps  # A: 1 call, B: 3 agents, C: 3 agents + 2 reflections
    assert result.usage.prompt_tokens > 0
    assert result.answer


def test_retrieve_returns_chunks():
    chunks = retrieve("q", "topic", k=3)
    assert chunks and all(isinstance(c, Chunk) for c in chunks)


def test_tool_registry_blocks_unknown_tool():
    with pytest.raises(PermissionError):
        call_tool("delete_everything")


def test_tool_registry_blocks_non_read_only(monkeypatch):
    monkeypatch.setitem(TOOL_REGISTRY, "writer", (lambda: None, Permission.WRITE_IRREVERSIBLE))
    with pytest.raises(PermissionError):
        call_tool("writer")


def test_lookup_benchmark_is_allowed():
    out = call_tool("lookup_benchmark", name="GSM8K", metric="accuracy")
    assert out["tool"] == "lookup_benchmark"
