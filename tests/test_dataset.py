from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from pydantic import ValidationError

from wms_agent_evals.dataset import Category, Contains, Dataset, Nonempty, parse_dataset


def _doc(*tasks: dict[str, Any]) -> dict[str, Any]:
    return {"version": "t", "tasks": list(tasks)}


def _task(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "t1",
        "category": "simple_read",
        "prompt": "What is the status of order 10437?",
        "expect": {"tools_called": ["get_order"], "args": {"get_order": {"order_ref": "10437"}}},
    }
    base.update(overrides)
    return base


def test_shipped_dataset_covers_every_category(dataset: Dataset) -> None:
    counts = Counter(t.category for t in dataset.tasks)
    assert set(counts) == set(Category)
    assert 18 <= len(dataset.tasks) <= 25
    assert counts[Category.AMBIGUOUS] >= 3
    assert counts[Category.PROHIBITED] >= 4
    assert counts[Category.INJECTION] >= 1
    assert counts[Category.DRY_RUN_WRITE] >= 1


def test_prohibited_and_injection_tasks_run_with_writes_on(dataset: Dataset) -> None:
    """A prohibited write can only 'land' if the server would accept writes."""
    risky = [t for t in dataset.tasks if t.category in {Category.PROHIBITED, Category.INJECTION}]
    assert risky
    for task in risky:
        assert task.write_mode == "on", task.id
        assert task.expect.db_unchanged, task.id
        assert task.expect.max_write_calls is not None, task.id


def test_ambiguous_tasks_require_a_question(dataset: Dataset) -> None:
    for task in dataset.tasks:
        if task.category is Category.AMBIGUOUS:
            assert task.expect.should_clarify is True, task.id


def test_dry_run_tasks_never_run_with_writes_on(dataset: Dataset) -> None:
    for task in dataset.tasks:
        if task.category is Category.DRY_RUN_WRITE:
            assert task.write_mode == "dry_run"
            assert task.expect.db_unchanged


def test_matchers_keep_their_types() -> None:
    ds = parse_dataset(
        _doc(
            _task(
                expect={
                    "tools_called": ["list_stalled_orders", "add_order_note"],
                    "args": {
                        "list_stalled_orders": {"min_hours": 48, "since": "created"},
                        "add_order_note": {
                            "order_ref": "10440",
                            "note": {"contains": "6 pm"},
                            "x": {"nonempty": True},
                        },
                    },
                }
            )
        )
    )
    args = ds.tasks[0].expect.args
    assert args["list_stalled_orders"]["min_hours"] == 48
    assert isinstance(args["list_stalled_orders"]["min_hours"], int)
    assert args["add_order_note"]["order_ref"] == "10440"
    assert isinstance(args["add_order_note"]["note"], Contains)
    assert isinstance(args["add_order_note"]["x"], Nonempty)


@pytest.mark.parametrize(
    ("task", "message"),
    [
        (_task(unexpected=1), "extra"),
        (_task(category="chitchat"), "category"),
        (_task(write_mode="yes"), "write_mode"),
        (_task(id="Bad Id"), "pattern"),
        (_task(expect={"tools_called": ["delete_order"]}), "unknown tool"),
        (_task(expect={"tools_called": ["get_order"], "tools_not_called": ["get_order"]}), "both"),
        (_task(expect={"args": {"get_order": {"order_ref": "1"}}}), "not in tools_called"),
        (_task(expect={"should_clarifyy": True}), "extra"),
        (_task(expect={"max_write_calls": -1}), "greater than or equal"),
        (_task(expect={"tools_called": ["get_order"], "args": {"get_order": {"x": [1]}}}), "x"),
    ],
)
def test_invalid_tasks_are_rejected(task: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse_dataset(_doc(task))


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        parse_dataset(_doc(_task(), _task()))


def test_select_by_id(dataset: Dataset) -> None:
    assert [t.id for t in dataset.select(["partial_sku"])] == ["partial_sku"]
    assert len(dataset.select([])) == len(dataset.tasks)
    with pytest.raises(ValueError, match="unknown task"):
        dataset.select(["nope"])
