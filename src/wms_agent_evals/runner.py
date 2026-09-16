"""Run tasks x models x prompt variants against a real mcp-logistica server.

For every run: a freshly seeded SQLite file, a new ``wms-mcp`` stdio
subprocess with the task's write mode, the agent loop from mcp-logistica, a
database snapshot before and after, and a score.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from wms_mcp.agent.evals import read_state, seeded_db
from wms_mcp.agent.loop import Trace, run_agent

from wms_agent_evals.dataset import Task
from wms_agent_evals.prompts import PromptVariant
from wms_agent_evals.providers import MeteredModel
from wms_agent_evals.scorer import Observation, Score, Step, score

ModelFactory = Callable[[str, Task], MeteredModel]
DEFAULT_MAX_STEPS = 6


@dataclass
class RunResult:
    task_id: str
    category: str
    model: str
    prompt: str
    prompt_version: str
    repeat: int
    passed: bool
    score: Score
    # Tool outputs contain timestamps, so they live in traces.jsonl only; this
    # keeps results.json reproducible run to run.
    tool_calls: list[dict[str, Any]]
    final_text: str
    stopped: str
    db_changed: bool
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    model_latency_ms: float
    task_latency_ms: float
    error: str | None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = {**asdict(self.score), "passed": self.score.passed}
        return data


def describe_error(exc: BaseException) -> str:
    """Name the root cause; anyio wraps errors from the MCP client in ExceptionGroups."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe_error(e) for e in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


def observation(task: Task, trace: Trace, db_changed: bool, error: str | None) -> Observation:
    return Observation(
        user_prompt=task.prompt,
        steps=[Step(s.tool, s.arguments, s.is_error, s.outcome, s.text) for s in trace.steps],
        final_text=trace.final_text,
        stopped="error" if error else trace.stopped,
        db_changed=db_changed,
        error=error,
    )


async def _run_loop(
    task: Task,
    model: MeteredModel,
    prompt: PromptVariant,
    db: Path,
    workdir: Path,
    trace: Trace,
    max_steps: int,
) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "wms_mcp.server"],
        env={**os.environ, "WMS_DB_PATH": str(db), "WMS_WRITE_MODE": task.write_mode},
    )
    with (workdir / "server.stderr").open("w", encoding="utf-8") as errlog:
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await run_agent(
                model,
                session,
                task.prompt,
                system_prompt=prompt.text,
                prompt_version=prompt.version,
                max_steps=max_steps,
            )
    trace.steps = result.steps
    trace.final_text = result.final_text
    trace.stopped = result.stopped


def run_one(
    task: Task,
    model: MeteredModel,
    prompt: PromptVariant,
    workdir: Path,
    *,
    repeat: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[RunResult, Trace]:
    db = seeded_db(workdir)
    before = read_state(db)
    trace = Trace(model=model.name, prompt_version=prompt.version, user=task.prompt)
    error: str | None = None
    started = time.perf_counter()
    try:
        anyio.run(_run_loop, task, model, prompt, db, workdir, trace, max_steps)
    except Exception as exc:  # a provider error is a failed run, not a crashed eval
        error = describe_error(exc)
    task_latency = round((time.perf_counter() - started) * 1000, 2)
    after = read_state(db)
    db_changed = before.orders != after.orders or before.audit_actors != after.audit_actors
    s = score(task, observation(task, trace, db_changed, error))
    result = RunResult(
        task_id=task.id,
        category=task.category.value,
        model=model.name,
        prompt=prompt.name,
        prompt_version=prompt.version,
        repeat=repeat,
        passed=s.passed,
        score=s,
        tool_calls=[
            {
                "tool": st.tool,
                "arguments": st.arguments,
                "is_error": st.is_error,
                "outcome": st.outcome,
            }
            for st in trace.steps
        ],
        final_text=trace.final_text,
        stopped="error" if error else trace.stopped,
        db_changed=db_changed,
        input_tokens=model.input_tokens,
        output_tokens=model.output_tokens,
        cost_usd=model.cost_usd,
        model_latency_ms=model.model_latency_ms,
        task_latency_ms=task_latency,
        error=error,
    )
    return result, trace


def run_matrix(
    tasks: Sequence[Task],
    models: Sequence[str],
    prompts: Sequence[PromptVariant],
    make_model: ModelFactory,
    *,
    repeats: int = 1,
    max_steps: int = DEFAULT_MAX_STEPS,
    progress: Callable[[RunResult], None] | None = None,
) -> tuple[list[RunResult], list[dict[str, Any]]]:
    results: list[RunResult] = []
    traces: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="wms-agent-evals-") as tmp:
        n = 0
        for model_name in models:
            for prompt in prompts:
                for task in tasks:
                    for repeat in range(repeats):
                        n += 1
                        workdir = Path(tmp) / f"run-{n:04d}"
                        workdir.mkdir()
                        model = make_model(model_name, task)
                        result, trace = run_one(
                            task, model, prompt, workdir, repeat=repeat, max_steps=max_steps
                        )
                        results.append(result)
                        traces.append(
                            {
                                "task_id": task.id,
                                "model": model_name,
                                "prompt": prompt.version,
                                "repeat": repeat,
                                "trace": trace.to_json(),
                            }
                        )
                        if progress is not None:
                            progress(result)
    return results, traces
