"""Read-only benchmark tool checks. Offline; no API key needed."""
import importlib
import json
from pathlib import Path

import pytest

from seminar_critique.tools import Permission, TOOL_REGISTRY, call_tool, lookup_benchmark
from seminar_critique.tools.benchmark_lookup import BENCHMARKS_PATH


def test_lookup_returns_published_value_and_provenance():
    out = call_tool("lookup_benchmark", name=" gsm8k ", metric=" ACCURACY ")
    assert out["result"]["found"] is True
    assert out["result"]["benchmark"] == "GSM8K"
    assert out["result"]["value"] == 78.2
    assert out["result"]["method"] == "CRITIC (gpt-3.5-turbo, non-oracle)"
    assert "Table 2" in out["result"]["source"]
    assert "2305.11738v4" in out["result"]["source"]


def test_dataset_has_10_to_20_unique_sourced_records():
    records = json.loads(BENCHMARKS_PATH.read_text(encoding="utf-8"))
    assert 10 <= len(records) <= 20
    pairs = [(row["benchmark"].casefold(), row["metric"].casefold()) for row in records]
    assert len(set(pairs)) == len(pairs)
    for row in records:
        assert set(row) == {"benchmark", "metric", "method", "value", "source"}
        assert "Table " in row["source"] and "https://arxiv.org/" in row["source"]
        result = lookup_benchmark(row["benchmark"], row["metric"])
        assert result == {**row, "found": True}


@pytest.mark.parametrize("name,metric", [
    ("no-such-benchmark", "accuracy"), ("GSM8K", "no-such-metric"),
    ("", "accuracy"), ("GSM8K", ""), ("../../.env", "accuracy"),
    ("self-improvement-reflection", "accuracy"),
])
def test_unknown_pair_never_fabricates_a_value(name, metric):
    out = lookup_benchmark(name, metric)
    assert out == {"found": False, "source": None}


def test_lookup_uses_fixed_file_in_read_mode_from_any_directory(monkeypatch, tmp_path):
    before = BENCHMARKS_PATH.read_bytes()
    opened = []
    original_open = Path.open

    def read_only(path, mode="r", *args, **kwargs):
        assert path == BENCHMARKS_PATH
        assert mode == "r"
        opened.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", read_only)
        assert lookup_benchmark("GSM8K", "accuracy")["found"]
    assert opened == [BENCHMARKS_PATH]
    assert BENCHMARKS_PATH.read_bytes() == before


def test_agent_cannot_supply_a_file_path(tmp_path):
    with pytest.raises(TypeError):
        call_tool("lookup_benchmark", name="GSM8K", metric="accuracy", path=str(tmp_path))


@pytest.mark.parametrize("permission", [
    Permission.WRITE_REVERSIBLE, Permission.WRITE_IRREVERSIBLE, Permission.CODE_EXECUTION,
])
def test_disallowed_permissions_block_before_execution(monkeypatch, permission):
    def forbidden():
        pytest.fail("a blocked tool was executed")

    monkeypatch.setitem(TOOL_REGISTRY, "forbidden", (forbidden, permission))
    with pytest.raises(PermissionError):
        call_tool("forbidden")


@pytest.mark.parametrize("name,metric", [(None, "accuracy"), ("GSM8K", 1)])
def test_invalid_arguments_are_rejected(name, metric):
    with pytest.raises(TypeError):
        lookup_benchmark(name, metric)


@pytest.mark.parametrize("contents", [
    "{}", "not json", '[{"benchmark": "GSM8K"}]',
    *[json.dumps([{"benchmark": "GSM8K", "metric": "accuracy", "method": "test",
                   "value": value, "source": "paper, Table 1"}])
      for value in (None, True, "78.2", float("nan"), float("inf"), -1, 101, 10**400)],
])
def test_malformed_dataset_fails_instead_of_returning_evidence(monkeypatch, tmp_path, contents):
    path = tmp_path / "benchmarks.json"
    path.write_text(contents, encoding="utf-8")
    module = importlib.import_module("seminar_critique.tools.benchmark_lookup")
    monkeypatch.setattr(module, "BENCHMARKS_PATH", path)
    with pytest.raises(ValueError):
        lookup_benchmark("GSM8K", "accuracy")


def test_duplicate_pair_is_not_silently_selected(monkeypatch, tmp_path):
    row = lookup_benchmark("GSM8K", "accuracy")
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    module = importlib.import_module("seminar_critique.tools.benchmark_lookup")
    monkeypatch.setattr(module, "BENCHMARKS_PATH", path)
    with pytest.raises(ValueError, match="ambiguous"):
        lookup_benchmark("GSM8K", "accuracy")


def test_missing_dataset_is_not_reported_as_no_match(monkeypatch, tmp_path):
    module = importlib.import_module("seminar_critique.tools.benchmark_lookup")
    monkeypatch.setattr(module, "BENCHMARKS_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        lookup_benchmark("GSM8K", "accuracy")
