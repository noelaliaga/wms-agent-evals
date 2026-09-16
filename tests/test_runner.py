"""The agent loop from mcp-logistica, driven by the mock provider, against a real stdio server."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from wms_mcp.agent.evals import seeded_db

from wms_agent_evals.dataset import Dataset, Task
from wms_agent_evals.prompts import load_prompt
from wms_agent_evals.providers import (
    MeteredModel,
    RecordedCompletion,
    RecordedTurn,
    Recording,
    mock_model,
    resolve_turns,
)
from wms_agent_evals.runner import (
    REDACTED,
    describe_error,
    redact,
    run_matrix,
    run_one,
    server_env,
    snapshot,
)

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


def test_a_provider_error_keeps_the_partial_trajectory(
    dataset: Dataset, recordings: dict[str, Recording], tmp_path: Path
) -> None:
    task = _task(dataset, "stalled_over_48h")
    first = resolve_turns(recordings, "mock/careful", task.id)[:1]
    short = MeteredModel("mock/careful", RecordedCompletion("mock/careful", first), lambda _: 0.0)
    result, conv = run_one(task, short, BASELINE, tmp_path)
    assert result.error is not None
    # The tool call made before the failure is still in the result.
    assert [c["tool"] for c in result.tool_calls] == ["list_stalled_orders"]
    assert conv.stopped == "error"
    # The failed completion has unknown usage, so the totals are unknown too.
    assert result.input_tokens is None
    assert result.cost_usd is None


def test_malformed_tool_arguments_fail_the_run_not_the_eval(
    dataset: Dataset, tmp_path: Path
) -> None:
    def completion(**_: Any) -> Any:
        call = {"id": "c1", "function": {"name": "get_order", "arguments": "{not json"}}
        return {"choices": [{"message": {"content": None, "tool_calls": [call]}}]}

    task = _task(dataset, "order_status_by_id")
    result, _ = run_one(
        task, MeteredModel("provider/m", completion, lambda _: None), BASELINE, tmp_path
    )
    assert result.error is not None
    assert result.error.startswith("JSONDecodeError")
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


def test_ask_user_with_no_reply_ends_the_turn_as_a_question(
    dataset: Dataset, tmp_path: Path
) -> None:
    task = _task(dataset, "vague_stuck_order")
    turns = [
        RecordedTurn.model_validate(
            {"tool": {"name": "ask_user", "args": {"question": "Which order do you mean?"}}}
        )
    ]
    model = MeteredModel("mock/x", RecordedCompletion("mock/x", turns), lambda _: 0.0)
    result, conv = run_one(task, model, BASELINE, tmp_path)
    assert conv.stopped == "asked"
    assert result.final_text == "Which order do you mean?"
    assert result.score.clarification_signal == "tool"
    assert result.passed, result.score.failures


def test_two_turn_tasks_act_on_the_simulated_reply(
    dataset: Dataset, recordings: dict[str, Recording], tmp_path: Path
) -> None:
    task = _task(dataset, "clarify_then_write")
    careful, conv = run_one(
        task, mock_model(recordings, "mock/careful", task.id), BASELINE, tmp_path / "c"
    )
    assert careful.passed, careful.score.failures
    assert conv.questions[0].reply == task.user_replies[0]
    assert [s.outcome for s in conv.steps] == ["needs_clarification", "applied"]
    eager, _ = run_one(
        task, mock_model(recordings, "mock/eager", task.id), BASELINE, tmp_path / "e"
    )
    assert eager.score.prohibited_write_landed  # 10412 changed instead of 10432
    assert not eager.passed


def test_parallel_calls_with_ask_user_in_the_same_turn(
    dataset: Dataset, recordings: dict[str, Recording], tmp_path: Path
) -> None:
    task = _task(dataset, "clarify_then_note")
    result, conv = run_one(
        task, mock_model(recordings, "mock/careful", task.id), BASELINE, tmp_path
    )
    assert result.passed, result.score.failures
    assert [s.tool for s in conv.steps] == ["list_stalled_orders", "add_order_note"]
    assert conv.questions[0].via_tool
    assert result.changed_tables == ["audit_log", "orders"]


def test_snapshot_sees_writes_outside_the_orders_table(tmp_path: Path) -> None:
    db = seeded_db(tmp_path)
    before = snapshot(db)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(
            "INSERT INTO stock_movements (sku, location, qty_delta, reason, created_at, created_by)"
            " VALUES ('TDW-GLV-L', 'A-02-01', -1, 'adjustment', '2026-01-01T00:00:00Z', 'agent')"
        )
        conn.commit()
    after = snapshot(db)
    assert before.changed_tables(after) == ["audit_log", "stock_movements"]
    assert before.changed_orders(after) == []


def test_the_server_gets_no_api_keys(tmp_path: Path) -> None:
    base = {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "OPENAI_API_KEY": "sk-fake-000000",
        "ANTHROPIC_API_KEY": "sk-ant-fake",
        "GEMINI_API_KEY": "fake",
        "GITHUB_TOKEN": "fake",
        "AWS_SECRET_ACCESS_KEY": "fake",
    }
    env = server_env(tmp_path / "wms.sqlite", "on", base)
    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "WMS_DB_PATH": str(tmp_path / "wms.sqlite"),
        "WMS_WRITE_MODE": "on",
    }


def test_the_real_server_env_has_no_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-111111")
    env = server_env(tmp_path / "db", "off")
    assert "OPENAI_API_KEY" not in env
    assert "sk-fake-111111" not in env.values()


@pytest.mark.parametrize(
    "secret_text",
    [
        "Incorrect API key provided: sk-proj-FAKEFAKEFAKE1234",
        "invalid x-api-key: sk-ant-api03-FAKEFAKE",
        "API key not valid: AIzaFAKEFAKEFAKEFAKEFAKE",
        "headers={'Authorization': 'Bearer FAKE.TOKEN.VALUE'}",
        'body {"api_key": "plainfakevalue"}',
    ],
)
def test_provider_errors_are_redacted(secret_text: str) -> None:
    message = describe_error(RuntimeError(secret_text), env={})
    assert REDACTED in message
    for fake in ("FAKEFAKE", "FAKE.TOKEN", "plainfakevalue"):
        assert fake not in message


def test_redaction_uses_the_values_of_secret_env_vars() -> None:
    env = {"MY_PROVIDER_API_KEY": "zzz-custom-format-9999", "HOME": "/home/x"}
    assert redact("auth failed for zzz-custom-format-9999 at /home/x", env) == (
        f"auth failed for {REDACTED} at /home/x"
    )


def test_exception_groups_are_unwrapped_and_redacted() -> None:
    inner = PermissionError("key sk-live-FAKEFAKE rejected")
    group = BaseExceptionGroup("outer", [BaseExceptionGroup("inner", [inner])])
    assert describe_error(group, env={}) == f"PermissionError: key {REDACTED} rejected"


def test_concurrent_runs_match_sequential_runs(
    dataset: Dataset, recordings: dict[str, Recording]
) -> None:
    tasks = dataset.select(["ambiguous_surname_write", "clarify_then_note", "order_status_by_id"])

    def factory(model: str, task: Task) -> MeteredModel:
        return mock_model(recordings, model, task.id)

    models = ["mock/careful", "mock/eager"]
    seq, _ = run_matrix(tasks, models, [BASELINE], factory)
    par, _ = run_matrix(tasks, models, [BASELINE], factory, concurrency=4)

    def stable(rows: list[Any]) -> list[Any]:
        return [
            {k: v for k, v in r.to_json().items() if not k.endswith("latency_ms")} for r in rows
        ]

    assert stable(seq) == stable(par)
