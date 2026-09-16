from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from wms_agent_evals.dataset import Dataset, Task
from wms_agent_evals.prompts import load_prompt
from wms_agent_evals.providers import MeteredModel, Recording, mock_model
from wms_agent_evals.report import (
    OFFLINE_BANNER,
    diff_results,
    load_results,
    render_html,
    render_markdown,
    results_document,
    strip_volatile,
    summarize,
    write_reports,
)
from wms_agent_evals.runner import run_matrix

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "reports" / "offline" / "results.json"


def _run(**overrides: Any) -> dict[str, Any]:
    run: dict[str, Any] = {
        "task_id": "t1",
        "category": "ambiguous",
        "model": "mock/a",
        "prompt": "baseline",
        "passed": True,
        "score": {
            "tool_choice": True,
            "arguments": None,
            "clarification": True,
            "grounded": True,
            "prohibited_write_attempts": 0,
            "prohibited_write_landed": False,
            "failures": [],
        },
        "error": None,
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": 0.001,
        "task_latency_ms": 12.0,
        "model_latency_ms": 0.1,
    }
    run.update(overrides)
    return run


def test_summarize_rates_and_totals() -> None:
    failing = _run(
        task_id="t2",
        passed=False,
        category="prohibited",
        input_tokens=None,
        score={
            "tool_choice": False,
            "arguments": None,
            "clarification": None,
            "grounded": False,
            "prohibited_write_attempts": 2,
            "prohibited_write_landed": True,
            "failures": ["PROHIBITED WRITE LANDED: the database changed"],
        },
    )
    [s] = summarize([_run(), failing])
    assert (s.runs, s.passed, s.pass_rate) == (2, 1, 0.5)
    assert s.rates == {"tool_choice": 0.5, "arguments": None, "clarification": 1.0, "grounded": 0.5}
    assert (s.prohibited_write_attempts, s.prohibited_writes_landed) == (2, 1)
    assert s.input_tokens is None  # one run did not report tokens
    assert s.output_tokens == 20
    assert s.by_category == {"ambiguous": (1, 1), "prohibited": (0, 1)}


def test_renderers_label_offline_runs_and_escape_html() -> None:
    doc = results_document(
        {"mode": "offline", "prompt_versions": ["baseline@x"]},
        [
            _run(),
            _run(
                task_id="t2",
                passed=False,
                model="mock/<b>",
                score={**_run()["score"], "failures": ["answer mentions '<script>'"]},
            ),
        ],
    )
    md = render_markdown(doc)
    page = render_html(doc)
    assert OFFLINE_BANNER in md
    assert "OFFLINE RUN" in page
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "mock/&lt;b&gt;" in page
    assert "| t2 | ambiguous |" in md


def test_live_runs_get_the_live_banner() -> None:
    md = render_markdown(results_document({"mode": "live"}, [_run()]))
    assert "LIVE RUN" in md
    assert "OFFLINE RUN" not in md


def test_diff_ignores_latency_and_timestamps_only() -> None:
    a = results_document({"mode": "offline", "generated_at": "1"}, [_run()])
    b = copy.deepcopy(a)
    b["meta"]["generated_at"] = "2"
    b["runs"][0]["task_latency_ms"] = 999.0
    assert diff_results(a, b) == []
    b["runs"][0]["input_tokens"] = 101
    assert diff_results(a, b) == ["run ('mock/a', 'baseline', 't1') differs in ['input_tokens']"]


def test_load_results_checks_schema(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text('{"schema": 99}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_results(path)


def test_committed_offline_report_is_labelled_and_consistent() -> None:
    doc = load_results(COMMITTED)
    assert doc["meta"]["mode"] == "offline"
    assert all(r["model"].startswith("mock/") for r in doc["runs"])
    md = (COMMITTED.parent / "report.md").read_text(encoding="utf-8")
    page = (COMMITTED.parent / "report.html").read_text(encoding="utf-8")
    assert OFFLINE_BANNER in md
    assert "OFFLINE RUN" in page
    assert render_markdown(doc) == md  # report.md was generated from the committed results


def test_a_fresh_offline_run_reproduces_the_committed_results(
    dataset: Dataset, recordings: dict[str, Recording]
) -> None:
    """Subset of the committed matrix; CI compares the full run with `wms-evals compare`."""
    ids = ["ambiguous_surname_write", "change_recipient", "shortage_check"]
    committed = load_results(COMMITTED)
    prompts = [
        load_prompt(REPO / "prompts", p.split("@")[0]) for p in committed["meta"]["prompt_versions"]
    ]
    assert [p.version for p in prompts] == committed["meta"]["prompt_versions"]

    def factory(model: str, task: Task) -> MeteredModel:
        return mock_model(recordings, model, task.id)

    results, _ = run_matrix(dataset.select(ids), committed["meta"]["models"], prompts, factory)
    fresh = strip_volatile([r.to_json() for r in results])
    expected = strip_volatile([r for r in committed["runs"] if r["task_id"] in ids])
    assert fresh == expected


def test_write_reports(tmp_path: Path) -> None:
    md, page = write_reports(results_document({"mode": "offline"}, [_run()]), tmp_path / "x")
    assert md.read_text(encoding="utf-8").startswith("# WMS agent eval report (offline mode)")
    assert page.read_text(encoding="utf-8").startswith("<!doctype html>")
