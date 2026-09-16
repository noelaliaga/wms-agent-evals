from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from wms_agent_evals.dataset import Dataset
from wms_agent_evals.providers import (
    MeteredModel,
    MockProviderError,
    Pricing,
    RecordedCompletion,
    RecordedTurn,
    Recording,
    estimate_tokens,
    live_model,
    mock_model,
    resolve_turns,
    synthetic_cost_fn,
)

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
    for n in ("get_order", "set_order_status")
]
MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "Status of 10437?"}]


def turns(*raw: dict[str, Any]) -> list[RecordedTurn]:
    return [RecordedTurn.model_validate(t) for t in raw]


def test_recorded_completion_speaks_the_openai_format() -> None:
    rec = RecordedCompletion(
        "mock/x",
        turns({"tool": {"name": "get_order", "args": {"order_ref": "10437"}}}, {"say": "Packed."}),
    )
    first = rec(model="mock/x", messages=MESSAGES, tools=TOOLS, temperature=0)
    call = first["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_order"
    assert json.loads(call["function"]["arguments"]) == {"order_ref": "10437"}
    assert first["choices"][0]["finish_reason"] == "tool_calls"
    assert first["usage"]["prompt_tokens"] == estimate_tokens(
        {"messages": MESSAGES, "tools": TOOLS}
    )
    second = rec(model="mock/x", messages=MESSAGES, tools=TOOLS)
    assert second["choices"][0]["message"] == {"role": "assistant", "content": "Packed."}
    with pytest.raises(MockProviderError, match="exhausted"):
        rec(model="mock/x", messages=MESSAGES, tools=TOOLS)


def test_recorded_completion_refuses_unadvertised_tools_and_wrong_model() -> None:
    rec = RecordedCompletion("mock/x", turns({"tool": {"name": "drop_table"}}))
    with pytest.raises(MockProviderError, match="not advertised"):
        rec(model="mock/x", messages=MESSAGES, tools=TOOLS)
    with pytest.raises(MockProviderError, match="recording is for"):
        RecordedCompletion("mock/x", [])(model="mock/y", messages=MESSAGES)


def test_recorded_turn_shapes() -> None:
    with pytest.raises(ValueError, match="needs 'tool', 'tools' or 'say'"):
        RecordedTurn.model_validate({})
    with pytest.raises(ValueError, match="not both"):
        RecordedTurn.model_validate({"tool": {"name": "a"}, "tools": [{"name": "b"}]})
    both = RecordedTurn.model_validate({"say": "x", "tool": {"name": "get_order"}})
    assert [c.name for c in both.calls] == ["get_order"]
    assert both.say == "x"


def test_parallel_calls_and_text_alongside_go_through_the_upstream_parser() -> None:
    rec = RecordedCompletion(
        "mock/x",
        turns(
            {
                "say": "Checking both.",
                "tools": [
                    {"name": "get_order", "args": {"order_ref": "10437"}},
                    {"name": "set_order_status", "args": {"order_ref": "10437"}},
                ],
            }
        ),
    )
    model = MeteredModel("mock/x", rec, lambda _: 0.0)
    turn = model.complete(MESSAGES, TOOLS)
    assert turn.text == "Checking both."
    assert [c.name for c in turn.tool_calls] == ["get_order", "set_order_status"]
    assert len({c.id for c in turn.tool_calls}) == 2


def test_malformed_tool_arguments_raise_instead_of_passing_silently() -> None:
    def completion(**_: Any) -> Any:
        call = {"id": "c1", "function": {"name": "get_order", "arguments": '{"order_ref": '}}
        return _response(None, {"prompt_tokens": 1, "completion_tokens": 1}, [call])

    model = MeteredModel("provider/m", completion, lambda _: 0.0)
    with pytest.raises(json.JSONDecodeError):
        model.complete(MESSAGES, TOOLS)


def test_a_failed_call_makes_usage_unknown() -> None:
    responses = iter([_response(None, {"prompt_tokens": 10, "completion_tokens": 2})])

    def flaky(**_: Any) -> Any:
        try:
            return next(responses)
        except StopIteration:
            raise TimeoutError("provider timed out") from None

    model = MeteredModel("provider/m", flaky, lambda _: 0.001)
    model.complete(MESSAGES, TOOLS)
    assert (model.input_tokens, model.cost_usd) == (10, 0.001)
    with pytest.raises(TimeoutError):
        model.complete(MESSAGES, TOOLS)
    assert len(model.calls) == 2
    assert (model.input_tokens, model.output_tokens, model.cost_usd) == (None, None, None)


def test_temperature_none_is_left_out_of_the_request() -> None:
    sent: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return _response("ok", None)

    MeteredModel("provider/m", completion, lambda _: None, temperature=None).complete(
        MESSAGES, TOOLS
    )
    MeteredModel("provider/m", completion, lambda _: None, temperature=0.3).complete(
        MESSAGES, TOOLS
    )
    assert "temperature" not in sent[0]
    assert sent[1]["temperature"] == 0.3


def test_estimate_tokens_is_ceil_chars_over_four() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens({"a": 1}) == estimate_tokens('{"a": 1}')


def _response(content: str | None, usage: Any, tool_calls: Any = None) -> Any:
    return {
        "choices": [{"message": {"content": content, "tool_calls": tool_calls}}],
        "usage": usage,
    }


def test_metered_model_parses_through_the_upstream_adapter_and_records_usage() -> None:
    sent: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> Any:
        sent.append(kwargs)
        tool_call = SimpleNamespace(
            id="c1",
            function=SimpleNamespace(name="get_order", arguments='{"order_ref": "10437"}'),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=15)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    model = MeteredModel("provider/some-model", completion, lambda r: 0.0021, temperature=0.0)
    turn = model.complete(MESSAGES, TOOLS)
    assert turn.tool_calls[0].name == "get_order"
    assert turn.tool_calls[0].arguments == {"order_ref": "10437"}
    assert sent[0]["model"] == "provider/some-model"
    assert sent[0]["tool_choice"] == "auto"
    assert (model.input_tokens, model.output_tokens, model.cost_usd) == (120, 15, 0.0021)
    assert model.model_latency_ms >= 0


def test_missing_usage_or_price_makes_totals_unknown_not_zero() -> None:
    responses = iter(
        [
            _response(None, {"prompt_tokens": 10, "completion_tokens": 2}),
            _response("done", None),
        ]
    )

    def unknown_price(response: Any) -> float | None:
        raise RuntimeError("model not in cost map")

    model = MeteredModel("provider/m", lambda **_: next(responses), unknown_price)
    model.complete(MESSAGES, TOOLS)
    assert (model.input_tokens, model.output_tokens) == (10, 2)
    model.complete(MESSAGES, TOOLS)
    assert (model.input_tokens, model.output_tokens, model.cost_usd) == (None, None, None)
    assert len(model.calls) == 2


def test_synthetic_cost() -> None:
    cost = synthetic_cost_fn(Pricing(input_usd_per_million=1.0, output_usd_per_million=4.0))
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    assert cost(_response("x", usage)) == pytest.approx(3.0)
    assert cost(_response("x", None)) is None


def _rec(model: str, fallback: str | None, tasks: dict[str, Any]) -> Recording:
    return Recording.model_validate(
        {
            "model": model,
            "description": "",
            "fallback": fallback,
            "synthetic_pricing": {"input_usd_per_million": 0, "output_usd_per_million": 0},
            "tasks": tasks,
        }
    )


def test_resolve_turns_follows_fallbacks_and_detects_cycles() -> None:
    recs = {
        "mock/a": _rec("mock/a", None, {"t1": [{"say": "a"}]}),
        "mock/b": _rec("mock/b", "mock/a", {"t2": [{"say": "b"}]}),
    }
    assert resolve_turns(recs, "mock/b", "t1")[0].say == "a"
    assert resolve_turns(recs, "mock/b", "t2")[0].say == "b"
    with pytest.raises(MockProviderError, match="no recorded turns"):
        resolve_turns(recs, "mock/b", "t3")
    loop = {
        "mock/a": _rec("mock/a", "mock/b", {}),
        "mock/b": _rec("mock/b", "mock/a", {}),
    }
    with pytest.raises(MockProviderError, match="cycle"):
        resolve_turns(loop, "mock/a", "t1")


def test_shipped_recordings_cover_every_task(
    dataset: Dataset, recordings: dict[str, Recording]
) -> None:
    assert set(recordings) == {"mock/careful", "mock/eager"}
    task_ids = {t.id for t in dataset.tasks}
    assert set(recordings["mock/careful"].tasks) == task_ids
    assert set(recordings["mock/eager"].tasks) <= task_ids
    for task_id in task_ids:
        model = mock_model(recordings, "mock/eager", task_id)
        assert model.name == "mock/eager"


def test_live_model_refuses_recordings() -> None:
    with pytest.raises(ValueError, match="offline recording"):
        live_model("mock/careful")


class FakeLiteLLM(types.ModuleType):
    """Stands in for the litellm module: no network, no keys."""

    def __init__(self, cost: Any) -> None:
        super().__init__("litellm")
        self.sent: list[dict[str, Any]] = []
        self._cost = cost

    def completion(self, **kwargs: Any) -> Any:
        self.sent.append(kwargs)
        return _response("Packed.", {"prompt_tokens": 50, "completion_tokens": 5})

    def completion_cost(self, completion_response: Any) -> Any:
        if isinstance(self._cost, Exception):
            raise self._cost
        return self._cost


@pytest.mark.parametrize(
    ("cost", "expected"),
    [(0.0004, 0.0004), (None, None), (RuntimeError("model not mapped"), None)],
)
def test_live_model_with_a_fake_litellm(
    monkeypatch: pytest.MonkeyPatch, cost: Any, expected: float | None
) -> None:
    fake = FakeLiteLLM(cost)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    model = live_model("provider/some-model", temperature=None)
    turn = model.complete(MESSAGES, TOOLS)
    assert turn.text == "Packed."
    assert fake.sent[0]["drop_params"] is True
    assert fake.sent[0]["model"] == "provider/some-model"
    assert "temperature" not in fake.sent[0]
    assert model.input_tokens == 50
    assert model.cost_usd == expected
