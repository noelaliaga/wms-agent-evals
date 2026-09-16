"""Model providers: one metered LiteLLM-style adapter, two ways to feed it.

* live:    ``litellm.completion`` with the user's own keys; cost from
           ``litellm.completion_cost``. Never used by tests or CI.
* offline: :class:`RecordedCompletion`, a deterministic stand-in that returns
           hand-written, synthetic turns in the same OpenAI response format.

Both go through :class:`MeteredModel`, which wraps the ``LiteLLMModel`` adapter
from mcp-logistica (response parsing lives there) and records tokens, cost and
latency for every call. Offline token counts are a chars/4 estimate and offline
prices are invented; the report says so.
"""

from __future__ import annotations

import importlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from wms_mcp.agent.models import AssistantTurn, LiteLLMModel, Message, ToolSpec

Completion = Callable[..., Any]
CostFn = Callable[[Any], float | None]

MOCK_PREFIX = "mock/"


# --------------------------------------------------------------------- metering


@dataclass(frozen=True)
class CallUsage:
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: float


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def usage_from_response(response: Any) -> tuple[int | None, int | None]:
    usage = _get(response, "usage")
    if usage is None:
        return None, None
    return _int_or_none(_get(usage, "prompt_tokens")), _int_or_none(
        _get(usage, "completion_tokens")
    )


@dataclass
class MeteredModel:
    """A ``wms_mcp.agent.models.ChatModel`` that records usage per completion call."""

    name: str
    completion: Completion
    cost_fn: CostFn
    # None leaves temperature out of the request: some reasoning models reject any value.
    temperature: float | None = 0.0
    calls: list[CallUsage] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._inner = LiteLLMModel(
            self.name, completion=self._metered_completion, temperature=self.temperature or 0.0
        )

    def _metered_completion(self, **kwargs: Any) -> Any:
        if self.temperature is None:
            kwargs.pop("temperature", None)
        started = time.perf_counter()
        try:
            response = self.completion(**kwargs)
        except BaseException:
            # A failed call has unknown usage, so the run's totals become n/a
            # instead of a partial sum that looks complete.
            latency = round((time.perf_counter() - started) * 1000, 2)
            self.calls.append(CallUsage(None, None, None, latency))
            raise
        latency = round((time.perf_counter() - started) * 1000, 2)
        tokens_in, tokens_out = usage_from_response(response)
        try:
            cost = self.cost_fn(response)
        except Exception:  # cost maps lag behind new models; a missing price is not fatal
            cost = None
        self.calls.append(CallUsage(tokens_in, tokens_out, cost, latency))
        return response

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantTurn:
        return self._inner.complete(messages, tools)

    # Totals are None when any call did not report the figure, so a partial
    # number is never presented as a complete one.
    def _total(self, attr: Literal["input_tokens", "output_tokens"]) -> int | None:
        values = [getattr(c, attr) for c in self.calls]
        return None if any(v is None for v in values) else sum(values)

    @property
    def input_tokens(self) -> int | None:
        return self._total("input_tokens")

    @property
    def output_tokens(self) -> int | None:
        return self._total("output_tokens")

    @property
    def cost_usd(self) -> float | None:
        values = [c.cost_usd for c in self.calls]
        if any(v is None for v in values):
            return None
        return round(sum(v for v in values if v is not None), 8)

    @property
    def model_latency_ms(self) -> float:
        return round(sum(c.latency_ms for c in self.calls), 2)


# ------------------------------------------------------------------------- live


def live_model(model: str, temperature: float | None = 0.0) -> MeteredModel:
    """A model served by LiteLLM with the caller's own API keys (paid calls).

    ``drop_params=True`` asks LiteLLM to drop request parameters a provider does
    not support instead of failing the call; ``temperature=None`` omits it.
    """
    if model.startswith(MOCK_PREFIX):
        raise ValueError(f"{model!r} is an offline recording, not a live model")
    litellm = importlib.import_module("litellm")

    def completion(**kwargs: Any) -> Any:
        return litellm.completion(drop_params=True, **kwargs)

    def cost(response: Any) -> float | None:
        value = litellm.completion_cost(completion_response=response)
        return float(value) if value is not None else None

    return MeteredModel(model, completion, cost, temperature)


# ---------------------------------------------------------------------- offline


class MockProviderError(RuntimeError):
    """The recording cannot answer this call (missing task, exhausted, unknown tool)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecordedToolCall(_Strict):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class RecordedTurn(_Strict):
    """One assistant turn: a tool call, several parallel ones, text, or text plus calls."""

    tool: RecordedToolCall | None = None
    tools: list[RecordedToolCall] = Field(default_factory=list)
    say: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> RecordedTurn:
        if self.tool is not None and self.tools:
            raise ValueError("a recorded turn takes 'tool' or 'tools', not both")
        if self.tool is None and not self.tools and self.say is None:
            raise ValueError("a recorded turn needs 'tool', 'tools' or 'say'")
        return self

    @property
    def calls(self) -> list[RecordedToolCall]:
        return [self.tool] if self.tool is not None else list(self.tools)


class Pricing(_Strict):
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)


class Recording(_Strict):
    model: str = Field(pattern=r"^mock/[a-z0-9-]+$")
    description: str
    synthetic_pricing: Pricing
    fallback: str | None = None
    tasks: dict[str, list[RecordedTurn]]


def load_recording(path: Path) -> Recording:
    with path.open(encoding="utf-8") as fh:
        return Recording.model_validate(yaml.safe_load(fh))


def load_recordings(directory: Path) -> dict[str, Recording]:
    recordings: dict[str, Recording] = {}
    for path in sorted(directory.glob("*.yaml")):
        rec = load_recording(path)
        if rec.model in recordings:
            raise ValueError(f"two recordings for {rec.model}")
        recordings[rec.model] = rec
    for rec in recordings.values():
        if rec.fallback is not None and rec.fallback not in recordings:
            raise ValueError(f"{rec.model}: fallback {rec.fallback!r} has no recording")
    return recordings


def resolve_turns(
    recordings: Mapping[str, Recording], model: str, task_id: str
) -> list[RecordedTurn]:
    """Turns for (model, task), following at most a chain of fallbacks."""
    seen: set[str] = set()
    current: str | None = model
    while current is not None:
        if current in seen:
            raise MockProviderError(f"fallback cycle at {current}")
        seen.add(current)
        rec = recordings.get(current)
        if rec is None:
            raise MockProviderError(f"no recording for {current}")
        if task_id in rec.tasks:
            return list(rec.tasks[task_id])
        current = rec.fallback
    raise MockProviderError(f"{model} has no recorded turns for task {task_id!r}")


def estimate_tokens(payload: Any) -> int:
    """Rough, deterministic token estimate: ceil(chars / 4) of the JSON payload."""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return math.ceil(len(text) / 4)


class RecordedCompletion:
    """Deterministic ``litellm.completion`` stand-in for one (model, task) pair.

    It replays turns in order and ignores tool results, like a script. It does
    check that every recorded tool call targets a tool the server advertised.
    """

    def __init__(self, model: str, turns: Sequence[RecordedTurn]) -> None:
        self.model = model
        self._turns = list(turns)
        self._next = 0

    def __call__(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        **_: Any,
    ) -> dict[str, Any]:
        if model != self.model:
            raise MockProviderError(f"recording is for {self.model}, called as {model}")
        if self._next >= len(self._turns):
            raise MockProviderError(f"{self.model}: recording exhausted after {self._next} turns")
        turn = self._turns[self._next]
        self._next += 1
        advertised = {t["function"]["name"] for t in tools}
        message: dict[str, Any] = {"role": "assistant", "content": turn.say}
        calls = turn.calls
        for call in calls:
            if call.name not in advertised:
                raise MockProviderError(f"{self.model}: tool {call.name!r} not advertised")
        if calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{self._next}_{i}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.args, ensure_ascii=False),
                    },
                }
                for i, call in enumerate(calls)
            ]
        return {
            "id": f"mock-{self._next}",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": estimate_tokens(
                    {"messages": list(messages), "tools": list(tools)}
                ),
                "completion_tokens": estimate_tokens(message),
            },
        }


def synthetic_cost_fn(pricing: Pricing) -> CostFn:
    def cost(response: Any) -> float | None:
        tokens_in, tokens_out = usage_from_response(response)
        if tokens_in is None or tokens_out is None:
            return None
        return (
            tokens_in * pricing.input_usd_per_million + tokens_out * pricing.output_usd_per_million
        ) / 1_000_000

    return cost


def mock_model(recordings: Mapping[str, Recording], model: str, task_id: str) -> MeteredModel:
    turns = resolve_turns(recordings, model, task_id)
    pricing = recordings[model].synthetic_pricing
    return MeteredModel(model, RecordedCompletion(model, turns), synthetic_cost_fn(pricing))
