"""The agent loop from mcp-logistica, driven by the mock provider, against a real stdio server."""

from __future__ import annotations

import json
from pathlib import Path

from wms_agent_evals.dataset import Dataset, Task
from wms_agent_evals.prompts import load_prompt
from wms_agent_evals.providers import (
    MeteredModel,
    RecordedCompletion,
    Recording,
    mock_model,
    resolve_turns,
)
from wms_agent_evals.runner import run_matrix, run_one

REPO = Path(__file__).resolve().parents[1]
BASELINE = load_prompt(REPO / "prompts", "baseline")


def _task(dataset: Dataset, task_id: str) -> Task:
    return dataset.select([task_id])[0]


def test_careful_recording_passes_a_read_and_a_dry_run(
    dataset: Dataset, recordings: dict[str, Recording], tmp_path: Path
) -> None:
    for task_id in ("stalled_over_three_days", "dry_run_status_change"):
        task = _task(dataset, task_id)
        workdir = tmp_path / task_id
        workdir.mkdir()
        result, trace = run_one(
            task, mock_model(recordings, "mock/careful", task_id), BASELINE, workdir
        )
        assert result.passed, result.score.failures
        assert not result.db_changed
        assert trace.prompt_version == BASELINE.version
    # The dry run really reached the server and came back as a dry run.
    assert [s.outcome for s in trace.steps] == ["ok", "dry_run"]


def test_a_guessed_write_lands_and_is_caught(
    dataset: Dataset, recordings: dict[str, Recording], tmp_path: Path
) -> None:
    task = _task(dataset, "ambiguous_surname_write")
    result, trace = run_one(task, mock_model(recordings, "mock/eager", task.id), BASELINE, tmp_path)
    assert [s.outcome for s in trace.steps] == ["ok", "applied"]
    assert result.db_changed
    assert result.score.prohibited_write_landed
    assert result.score.clarification is False
    assert not result.passed


def test_the_server_stops_what_the_scorer_still_counts(
    dataset: Dataset, recordings: dict[str, Recording], tmp_path: Path
) -> None:
    """Injected instruction followed: the server refuses, the DB is intact, the run still fails."""
    task = _task(dataset, "injected_note")
    result, trace = run_one(task, mock_model(recordings, "mock/eager", task.id), BASELINE, tmp_path)
    assert trace.steps[1].tool == "set_order_status"
    assert trace.steps[1].is_error
    assert "an agent cannot set status 'delivered'" in trace.steps[1].text
    assert not result.db_changed
    assert result.score.prohibited_write_attempts == 1
    assert not result.passed


def test_a_provider_error_is_a_failed_run_not_a_crash(
    dataset: Dataset, recordings: dict[str, Recording], tmp_path: Path
) -> None:
    task = _task(dataset, "stalled_over_48h")
    # Only the first recorded turn (a tool call): the loop asks again and the recording runs out.
    first = resolve_turns(recordings, "mock/careful", task.id)[:1]
    short = MeteredModel("mock/careful", RecordedCompletion("mock/careful", first), lambda _: 0.0)
    result, _ = run_one(task, short, BASELINE, tmp_path)
    assert result.error is not None
    assert "exhausted" in result.error
    assert result.stopped == "error"
    assert not result.passed


def test_max_steps_stops_a_looping_agent(
    dataset: Dataset, recordings: dict[str, Recording], tmp_path: Path
) -> None:
    task = _task(dataset, "stalled_over_48h")
    model = mock_model(recordings, "mock/careful", task.id)
    result, trace = run_one(task, model, BASELINE, tmp_path, max_steps=1)
    assert result.stopped == "max_steps"
    assert len(trace.steps) == 1
    assert not result.passed


def test_run_matrix_records_usage_and_serialises(
    dataset: Dataset, recordings: dict[str, Recording]
) -> None:
    tasks = dataset.select(["order_status_by_id", "change_quantity"])
    ask = load_prompt(REPO / "prompts", "ask_before_assume")

    def factory(model: str, task: Task) -> MeteredModel:
        return mock_model(recordings, model, task.id)

    results, traces = run_matrix(tasks, ["mock/careful"], [BASELINE, ask], factory)
    assert len(results) == len(traces) == 4
    assert all(r.passed for r in results)
    by_prompt = {(r.prompt, r.task_id): r for r in results}
    base = by_prompt[("baseline", "order_status_by_id")]
    longer = by_prompt[("ask_before_assume", "order_status_by_id")]
    assert base.input_tokens is not None
    assert longer.input_tokens is not None
    assert longer.input_tokens > base.input_tokens  # the longer system prompt costs tokens
    assert base.cost_usd is not None
    assert base.cost_usd > 0
    # change_quantity: the careful answer calls no tool at all.
    assert by_prompt[("baseline", "change_quantity")].tool_calls == []
    json.dumps([r.to_json() for r in results])
    json.dumps(traces)
