"""The orchestrator must query the benchmark tool by benchmark name, not by topic."""
from seminar_critique.configs import CONFIGS
from seminar_critique.llm import FakeLLM
from seminar_critique.orchestrator import _benchmark_names, run_pipeline
from seminar_critique.schemas import Task


def _task(text: str) -> Task:
    return Task(id="t", topic="self-improvement-reflection", input=text, rubric_axes=["assumptions"])


def test_huang_task_finds_real_records():
    result = run_pipeline(_task("Huang et al. (2024) claim X. Critique it."), CONFIGS["B"], FakeLLM())
    critic = next(s for s in result.trace if s.agent == "critic")
    found = [c for c in critic.tool_calls if c["result"]["found"]]
    assert {c["args"]["name"] for c in found} == {"HotpotQA", "GSM8K"}
    assert all(c["result"]["source"] for c in found)


def test_react_task_maps_to_react_benchmarks():
    assert _benchmark_names(_task("Yao et al. (2023) propose ReAct."))[:2] == ["HotpotQA", "FEVER"]


def test_unknown_paper_keeps_old_topic_lookup():
    result = run_pipeline(_task("Some other paper."), CONFIGS["B"], FakeLLM())
    critic = next(s for s in result.trace if s.agent == "critic")
    assert len(critic.tool_calls) == 1
    assert critic.tool_calls[0]["args"]["name"] == "self-improvement-reflection"
    assert critic.tool_calls[0]["result"]["found"] is False
