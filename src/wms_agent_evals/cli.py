"""Command line: run, report, compare, validate.

wms-evals run --provider mock --out out/offline
wms-evals run --provider live --models openai/<model>,gemini/<model> --out out/live
wms-evals run --provider live --models ... --concurrency 4 --resume --out out/live
wms-evals report out/offline/results.json --out-dir out/offline
wms-evals compare reports/offline/results.json out/offline/results.json
wms-evals validate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wms_agent_evals import __version__
from wms_agent_evals.dataset import DEFAULT_DATASET, Task, load_dataset
from wms_agent_evals.prompts import DEFAULT_PROMPTS_DIR, DEFAULT_VARIANTS, load_prompt
from wms_agent_evals.providers import (
    MOCK_PREFIX,
    MeteredModel,
    live_model,
    load_recordings,
    mock_model,
    resolve_turns,
)
from wms_agent_evals.report import (
    diff_results,
    load_results,
    results_document,
    write_reports,
)
from wms_agent_evals.runner import (
    DEFAULT_MAX_STEPS,
    RunKey,
    RunResult,
    RunSpec,
    execute,
    plan_runs,
)

DEFAULT_RECORDINGS = Path("recordings")
JOURNAL = "runs.jsonl"


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _temperature(value: str) -> float | None:
    if value.strip().lower() == "none":
        return None
    return float(value)


def _key(row: dict[str, Any]) -> RunKey:
    return (row["model"], row["prompt_version"], row["task_id"], int(row["repeat"]))


def read_journal(path: Path) -> dict[RunKey, tuple[dict[str, Any], dict[str, Any]]]:
    """Finished runs from an interrupted invocation; a torn last line is ignored."""
    done: dict[RunKey, tuple[dict[str, Any], dict[str, Any]]] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[_key(row["result"])] = (row["result"], row["trace"])
    return done


def _server_version() -> str:
    from importlib.metadata import version

    return f"mcp-logistica {version('mcp-logistica')}"


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    tasks = dataset.select(args.tasks)
    prompts = [load_prompt(args.prompts_dir, name) for name in args.prompts]
    recordings = load_recordings(args.recordings)

    if args.provider == "mock":
        models = args.models or sorted(recordings)
        bad = [m for m in models if not m.startswith(MOCK_PREFIX)]
        if bad:
            print(f"mock provider only serves recordings, got {bad}", file=sys.stderr)
            return 2

        def factory(model: str, task: Task) -> MeteredModel:
            return mock_model(recordings, model, task.id)

    else:
        models = args.models
        if not models:
            print("live runs need --models (LiteLLM model names)", file=sys.stderr)
            return 2
        try:
            import litellm  # noqa: F401
        except ImportError:
            print("live runs need LiteLLM: pip install -e '.[live]'", file=sys.stderr)
            return 2
        print(
            f"LIVE run: {len(models)} model(s) x {len(prompts)} prompt(s) x {len(tasks)} task(s)"
            f" x {args.repeats} repeat(s). This calls paid APIs with your own keys.",
            file=sys.stderr,
        )

        def factory(model: str, task: Task) -> MeteredModel:
            return live_model(model, args.temperature)

    specs = plan_runs(tasks, models, prompts, args.repeats)
    args.out.mkdir(parents=True, exist_ok=True)
    journal = args.out / JOURNAL
    previous = read_journal(journal) if args.resume else {}
    if not args.resume:
        journal.unlink(missing_ok=True)
    todo = [s for s in specs if s.key not in previous]
    if previous:
        print(f"resuming: {len(specs) - len(todo)} run(s) already in {journal}", file=sys.stderr)

    with journal.open("a", encoding="utf-8") as jf:

        def on_done(spec: RunSpec, r: RunResult, trace: dict[str, Any]) -> None:
            jf.write(json.dumps({"result": r.to_json(), "trace": trace}, ensure_ascii=False))
            jf.write("\n")
            jf.flush()
            flag = "pass" if r.passed else "FAIL"
            print(f"{flag:4}  {r.model:28} {r.prompt:18} {r.task_id}", file=sys.stderr)

        done = execute(
            todo,
            factory,
            max_steps=args.max_steps,
            concurrency=args.concurrency,
            on_done=on_done,
        )
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for spec in specs:
        if spec.index in done:
            result, trace = done[spec.index]
            rows.append(result.to_json())
            traces.append(trace)
        else:
            row, trace = previous[spec.key]
            rows.append(row)
            traces.append(trace)
    meta: dict[str, Any] = {
        "mode": "offline" if args.provider == "mock" else "live",
        "harness": f"wms-agent-evals {__version__}",
        "server": _server_version(),
        "dataset_version": dataset.version,
        "tasks": len(tasks),
        "models": models,
        "prompt_versions": [p.version for p in prompts],
        "repeats": args.repeats,
        "max_steps": args.max_steps,
        "temperature": None if args.provider == "mock" else args.temperature,
        "concurrency": args.concurrency,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    doc = results_document(meta, rows)
    (args.out / "results.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.out / "traces.jsonl").open("w", encoding="utf-8") as fh:
        for row in traces:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    md, page = write_reports(doc, args.out)
    passed = sum(bool(r["passed"]) for r in rows)
    landed = sum(bool(r["score"]["prohibited_write_landed"]) for r in rows)
    print(f"\n{passed}/{len(rows)} runs passed; prohibited writes landed: {landed}")
    print(f"wrote {args.out / 'results.json'}, {md.name}, {page.name}, traces.jsonl")
    if args.fail_on_landed_write and landed:
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    doc = load_results(args.results)
    md, page = write_reports(doc, args.out_dir)
    print(f"wrote {md} and {page}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    problems = diff_results(load_results(args.expected), load_results(args.actual))
    for p in problems:
        print(p)
    if problems:
        print(f"{len(problems)} difference(s)", file=sys.stderr)
        return 1
    print("results match (latency and timestamps ignored)")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    recordings = load_recordings(args.recordings)
    for name in DEFAULT_VARIANTS:
        load_prompt(args.prompts_dir, name)
    missing = []
    for model in recordings:
        for task in dataset.tasks:
            try:
                resolve_turns(recordings, model, task.id)
            except Exception as exc:
                missing.append(str(exc))
    known = {t.id for t in dataset.tasks}
    extra = [
        f"{r.model}: recording for unknown task {t!r}"
        for r in recordings.values()
        for t in r.tasks
        if t not in known
    ]
    for problem in missing + extra:
        print(problem)
    if missing or extra:
        return 1
    print(
        f"dataset {dataset.version}: {len(dataset.tasks)} tasks; "
        f"recordings: {', '.join(sorted(recordings))}; prompts: {', '.join(DEFAULT_VARIANTS)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wms-evals", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    common.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    common.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)

    run = sub.add_parser("run", parents=[common], help="run the evaluation")
    run.add_argument("--provider", choices=["mock", "live"], default="mock")
    run.add_argument("--models", type=_csv, default=[], help="comma-separated model names")
    run.add_argument("--prompts", type=_csv, default=list(DEFAULT_VARIANTS))
    run.add_argument("--tasks", type=_csv, default=[], help="comma-separated task ids")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    run.add_argument(
        "--temperature",
        type=_temperature,
        default=0.0,
        help="sampling temperature, or 'none' to leave it out (some reasoning models reject it)",
    )
    run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="runs in parallel; each has its own database and server process",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help=f"skip runs already recorded in OUT/{JOURNAL} by an interrupted invocation",
    )
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--fail-on-landed-write",
        action="store_true",
        help="exit 1 if any prohibited write reached the database",
    )
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="render report.md and report.html from results.json")
    rep.add_argument("results", type=Path)
    rep.add_argument("--out-dir", type=Path, required=True)
    rep.set_defaults(func=cmd_report)

    cmp_ = sub.add_parser("compare", help="compare two results.json, ignoring latency")
    cmp_.add_argument("expected", type=Path)
    cmp_.add_argument("actual", type=Path)
    cmp_.set_defaults(func=cmd_compare)

    val = sub.add_parser("validate", parents=[common], help="check dataset, recordings, prompts")
    val.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "repeats", 1) < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2
    if getattr(args, "concurrency", 1) < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        return 2
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
