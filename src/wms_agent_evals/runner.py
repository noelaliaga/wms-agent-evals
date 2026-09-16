"""Run tasks x models x prompt variants against a real mcp-logistica server.

For every run: a freshly seeded SQLite file, a new ``wms-mcp`` stdio
subprocess with the task's write mode and a minimal environment, the
conversation loop in :mod:`wms_agent_evals.agent`, a fingerprint of every
table before and after, and a score.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from wms_mcp.agent.evals import seeded_db

from wms_agent_evals.agent import Conversation, run_conversation
from wms_agent_evals.dataset import Task
from wms_agent_evals.prompts import PromptVariant
from wms_agent_evals.providers import MeteredModel
from wms_agent_evals.scorer import Observation, Score, Step, asked_for_clarification, score

ModelFactory = Callable[[str, Task], MeteredModel]
DEFAULT_MAX_STEPS = 6
RunKey = tuple[str, str, str, int]  # model, prompt version, task id, repeat

# ---------------------------------------------------------------- environment

# The server needs a PATH, a home and a locale, not the caller's API keys.
SERVER_ENV_ALLOW = frozenset(
    {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT"}
)
_SECRET_NAME = re.compile(r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)$", re.IGNORECASE)


def server_env(db: Path, write_mode: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the ``wms-mcp`` subprocess: an allow-list plus the two WMS settings."""
    source = os.environ if base is None else base
    env = {k: v for k, v in source.items() if k in SERVER_ENV_ALLOW}
    env["WMS_DB_PATH"] = str(db)
    env["WMS_WRITE_MODE"] = write_mode
    return env


# ------------------------------------------------------------------ redaction

_KEY_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"),  # OpenAI / Anthropic style
    re.compile(r"\bAIza[0-9A-Za-z_\-]{10,}"),  # Google API keys
    re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/\-]+=*"),
    re.compile(r"(?i)((?:x-api-key|api[_-]?key|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"),
)
REDACTED = "[REDACTED]"


def redact(text: str, env: Mapping[str, str] | None = None) -> str:
    """Remove API keys and tokens from text that may end up in a committed report."""
    source = os.environ if env is None else env
    for name, value in source.items():
        if _SECRET_NAME.search(name) and len(value) >= 8:
            text = text.replace(value, REDACTED)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(
            lambda m: (m.group(1) if m.groups() else "") + REDACTED,
            text,
        )
    return text


def describe_error(exc: BaseException, env: Mapping[str, str] | None = None) -> str:
    """Name the root cause, redacted.

    anyio wraps errors raised inside the MCP client's task group in
    ExceptionGroups; a single-member group is unwrapped to its cause.
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe_error(e, env) for e in exc.exceptions)
    return redact(f"{type(exc).__name__}: {exc}", env)


# ---------------------------------------------------------------- DB snapshot


@dataclass(frozen=True)
class DbSnapshot:
    tables: dict[str, str]  # table name -> sha256 of its rows in rowid order
    orders: dict[str, tuple[Any, ...]]  # order id -> row

    def changed_tables(self, other: DbSnapshot) -> list[str]:
        names = set(self.tables) | set(other.tables)
        return sorted(n for n in names if self.tables.get(n) != other.tables.get(n))

    def changed_orders(self, other: DbSnapshot) -> list[str]:
        ids = set(self.orders) | set(other.orders)
        return sorted(i for i in ids if self.orders.get(i) != other.orders.get(i))


def snapshot(db: Path) -> DbSnapshot:
    """Fingerprint every user table, so a write anywhere is detected, not only in ``orders``."""
    with closing(sqlite3.connect(db)) as conn:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables: dict[str, str] = {}
        for name in names:
            digest = hashlib.sha256()
            quoted = '"' + name.replace('"', '""') + '"'
            for row in conn.execute(f"SELECT * FROM {quoted} ORDER BY rowid"):  # noqa: S608
                digest.update(repr(row).encode("utf-8"))
            tables[name] = digest.hexdigest()
        orders = {str(r[0]): tuple(r) for r in conn.execute("SELECT * FROM orders ORDER BY id")}
    return DbSnapshot(tables, orders)


# ----------------------------------------------------------------------- runs


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
    questions: list[dict[str, Any]]
    final_text: str
    stopped: str
    db_changed: bool
    changed_tables: list[str]
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    model_latency_ms: float
    task_latency_ms: float
    error: str | None

    @property
    def key(self) -> RunKey:
        return (self.model, self.prompt_version, self.task_id, self.repeat)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = {**asdict(self.score), "passed": self.score.passed}
        return data


def observation(
    task: Task,
    conv: Conversation,
    db_changed: bool,
    changed_orders: Sequence[str],
    error: str | None,
) -> Observation:
    return Observation(
        user_prompt=task.prompt,
        steps=[Step(s.tool, s.arguments, s.is_error, s.outcome, s.text) for s in conv.steps],
        final_text=conv.final_text,
        stopped="error" if error else conv.stopped,
        db_changed=db_changed,
        error=error,
        questions=[q.text for q in conv.questions],
        asked_via_tool=any(q.via_tool for q in conv.questions),
        user_replies=[q.reply for q in conv.questions if q.reply is not None],
        changed_orders=changed_orders,
    )


async def _run_loop(
    task: Task,
    model: MeteredModel,
    prompt: PromptVariant,
    db: Path,
    workdir: Path,
    conv: Conversation,
    max_steps: int,
) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "wms_mcp.server"],
        env=server_env(db, task.write_mode),
    )
    with (workdir / "server.stderr").open("w", encoding="utf-8") as errlog:
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            await run_conversation(
                model,
                session,
                conv,
                system_prompt=prompt.text,
                user_replies=task.user_replies,
                max_steps=max_steps,
                is_question=asked_for_clarification,
            )


def run_one(
    task: Task,
    model: MeteredModel,
    prompt: PromptVariant,
    workdir: Path,
    *,
    repeat: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[RunResult, Conversation]:
    db = seeded_db(workdir)
    before = snapshot(db)
    conv = Conversation(model=model.name, prompt_version=prompt.version, user=task.prompt)
    error: str | None = None
    started = time.perf_counter()
    try:
        anyio.run(_run_loop, task, model, prompt, db, workdir, conv, max_steps)
    except Exception as exc:  # a provider error is a failed run, not a crashed eval
        error = describe_error(exc)
        conv.stopped = "error"
    task_latency = round((time.perf_counter() - started) * 1000, 2)
    after = snapshot(db)
    changed_tables = before.changed_tables(after)
    changed_orders = before.changed_orders(after)
    db_changed = bool(changed_tables)
    s = score(task, observation(task, conv, db_changed, changed_orders, error))
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
            for st in conv.steps
        ],
        questions=[
            {"text": q.text, "via_tool": q.via_tool, "reply": q.reply} for q in conv.questions
        ],
        final_text=conv.final_text,
        stopped=conv.stopped,
        db_changed=db_changed,
        changed_tables=changed_tables,
        input_tokens=model.input_tokens,
        output_tokens=model.output_tokens,
        cost_usd=model.cost_usd,
        model_latency_ms=model.model_latency_ms,
        task_latency_ms=task_latency,
        error=error,
    )
    return result, conv


@dataclass(frozen=True)
class RunSpec:
    index: int
    model: str
    prompt: PromptVariant
    task: Task
    repeat: int

    @property
    def key(self) -> RunKey:
        return (self.model, self.prompt.version, self.task.id, self.repeat)


def plan_runs(
    tasks: Sequence[Task],
    models: Sequence[str],
    prompts: Sequence[PromptVariant],
    repeats: int = 1,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for model_name in models:
        for prompt in prompts:
            for task in tasks:
                for repeat in range(repeats):
                    specs.append(RunSpec(len(specs), model_name, prompt, task, repeat))
    return specs


def trace_record(spec: RunSpec, conv: Conversation) -> dict[str, Any]:
    return {
        "task_id": spec.task.id,
        "model": spec.model,
        "prompt": spec.prompt.version,
        "repeat": spec.repeat,
        "trace": conv.to_json(),
    }


def execute(
    specs: Iterable[RunSpec],
    make_model: ModelFactory,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    concurrency: int = 1,
    on_done: Callable[[RunSpec, RunResult, dict[str, Any]], None] | None = None,
) -> dict[int, tuple[RunResult, dict[str, Any]]]:
    """Run each spec in its own temp dir (own DB, own server); ``concurrency`` threads at once.

    ``on_done`` is called from the main thread as each run finishes, so a caller
    can append it to a journal and resume after an interruption.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    done: dict[int, tuple[RunResult, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="wms-agent-evals-") as tmp:

        def work(spec: RunSpec) -> tuple[RunSpec, RunResult, dict[str, Any]]:
            workdir = Path(tmp) / f"run-{spec.index:05d}"
            workdir.mkdir()
            model = make_model(spec.model, spec.task)
            result, conv = run_one(
                spec.task, model, spec.prompt, workdir, repeat=spec.repeat, max_steps=max_steps
            )
            return spec, result, trace_record(spec, conv)

        def finish(spec: RunSpec, result: RunResult, trace: dict[str, Any]) -> None:
            done[spec.index] = (result, trace)
            if on_done is not None:
                on_done(spec, result, trace)

        todo = list(specs)
        if concurrency == 1:
            for spec in todo:
                finish(*work(spec))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(work, spec) for spec in todo]
                for future in as_completed(futures):
                    finish(*future.result())
    return done


def run_matrix(
    tasks: Sequence[Task],
    models: Sequence[str],
    prompts: Sequence[PromptVariant],
    make_model: ModelFactory,
    *,
    repeats: int = 1,
    max_steps: int = DEFAULT_MAX_STEPS,
    concurrency: int = 1,
    progress: Callable[[RunResult], None] | None = None,
) -> tuple[list[RunResult], list[dict[str, Any]]]:
    specs = plan_runs(tasks, models, prompts, repeats)

    def on_done(_: RunSpec, result: RunResult, __: dict[str, Any]) -> None:
        if progress is not None:
            progress(result)

    done = execute(specs, make_model, max_steps=max_steps, concurrency=concurrency, on_done=on_done)
    ordered = [done[s.index] for s in specs]
    return [r for r, _ in ordered], [t for _, t in ordered]
