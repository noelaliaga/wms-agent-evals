"""Aggregate run results and render a comparative Markdown and static HTML report."""

from __future__ import annotations

import html
import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wms_agent_evals.dataset import Category

RESULTS_SCHEMA = 1
CHECKS = ("tool_choice", "arguments", "clarification", "grounded")
# Fields that depend on the machine, the clock or the temp dir, not on behaviour.
VOLATILE_KEYS = frozenset({"model_latency_ms", "task_latency_ms", "generated_at"})

OFFLINE_BANNER = (
    "OFFLINE RUN. The models below are hand-written, synthetic recordings replayed by a "
    "deterministic mock provider. Token counts are a chars/4 estimate and prices are invented. "
    "This report shows that the harness, the scorer and the server work end to end; it says "
    "nothing about how any real model behaves."
)
LIVE_BANNER = (
    "LIVE RUN against real model APIs through LiteLLM. Token counts and costs come from the "
    "provider responses and LiteLLM's cost map. Small dataset, rule-based scorer: read the "
    "per-task failures before drawing conclusions."
)


def results_document(meta: dict[str, Any], runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": RESULTS_SCHEMA, "meta": meta, "runs": list(runs)}


def load_results(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != RESULTS_SCHEMA:
        raise ValueError(f"{path}: unsupported results schema {doc.get('schema')!r}")
    return doc


def strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def diff_results(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Differences in behaviour-relevant fields (latency and timestamps ignored)."""
    a, b = strip_volatile(expected), strip_volatile(actual)
    problems: list[str] = []
    if a["meta"] != b["meta"]:
        problems.append(f"meta differs: {a['meta']} != {b['meta']}")
    if len(a["runs"]) != len(b["runs"]):
        problems.append(f"run count differs: {len(a['runs'])} != {len(b['runs'])}")
    for ra, rb in zip(a["runs"], b["runs"], strict=False):
        if ra != rb:
            key = (ra.get("model"), ra.get("prompt"), ra.get("task_id"))
            fields = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
            problems.append(f"run {key} differs in {fields}")
    return problems


# ------------------------------------------------------------------ aggregation


def _rate(values: Iterable[bool | None]) -> float | None:
    graded = [v for v in values if v is not None]
    return round(sum(graded) / len(graded), 3) if graded else None


def _sum_or_none(values: Sequence[float | int | None]) -> float | None:
    if not values or any(v is None for v in values):
        return None
    return round(sum(v for v in values if v is not None), 6)


@dataclass(frozen=True)
class Summary:
    model: str
    prompt: str
    runs: int
    passed: int
    rates: dict[str, float | None]
    prohibited_write_attempts: int
    prohibited_writes_landed: int
    errors: int
    input_tokens: float | None
    output_tokens: float | None
    cost_usd: float | None
    mean_task_latency_ms: float
    mean_model_latency_ms: float
    by_category: dict[str, tuple[int, int]]

    @property
    def pass_rate(self) -> float:
        return round(self.passed / self.runs, 3) if self.runs else 0.0


def summarize(runs: Sequence[dict[str, Any]]) -> list[Summary]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["model"], run["prompt"]), []).append(run)
    out = []
    for (model, prompt), rows in groups.items():
        by_cat: dict[str, tuple[int, int]] = {}
        for r in rows:
            p, n = by_cat.get(r["category"], (0, 0))
            by_cat[r["category"]] = (p + int(r["passed"]), n + 1)
        out.append(
            Summary(
                model=model,
                prompt=prompt,
                runs=len(rows),
                passed=sum(int(r["passed"]) for r in rows),
                rates={c: _rate(r["score"][c] for r in rows) for c in CHECKS},
                prohibited_write_attempts=sum(
                    r["score"]["prohibited_write_attempts"] for r in rows
                ),
                prohibited_writes_landed=sum(
                    int(r["score"]["prohibited_write_landed"]) for r in rows
                ),
                errors=sum(1 for r in rows if r["error"]),
                input_tokens=_sum_or_none([r["input_tokens"] for r in rows]),
                output_tokens=_sum_or_none([r["output_tokens"] for r in rows]),
                cost_usd=_sum_or_none([r["cost_usd"] for r in rows]),
                mean_task_latency_ms=round(statistics.fmean(r["task_latency_ms"] for r in rows), 1),
                mean_model_latency_ms=round(
                    statistics.fmean(r["model_latency_ms"] for r in rows), 1
                ),
                by_category=by_cat,
            )
        )
    return out


# -------------------------------------------------------------------- rendering


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def _cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _categories(runs: Sequence[dict[str, Any]]) -> list[str]:
    present = {r["category"] for r in runs}
    return [c.value for c in Category if c.value in present]


def _tables(doc: dict[str, Any]) -> tuple[list[str], list[list[str]], list[str], list[list[str]]]:
    runs = doc["runs"]
    sums = summarize(runs)
    head = [
        "Model",
        "Prompt",
        "Pass",
        "Tool choice",
        "Arguments",
        "Clarification",
        "Grounded",
        "Prohibited attempts",
        "Prohibited writes landed",
        "Errors",
        "Tokens in",
        "Tokens out",
        "Cost",
        "Mean task latency (ms)",
    ]
    rows = [
        [
            s.model,
            s.prompt,
            f"{s.passed}/{s.runs} ({_pct(s.pass_rate)})",
            *(_pct(s.rates[c]) for c in CHECKS),
            str(s.prohibited_write_attempts),
            str(s.prohibited_writes_landed),
            str(s.errors),
            _num(s.input_tokens),
            _num(s.output_tokens),
            _cost(s.cost_usd),
            _num(s.mean_task_latency_ms, 1),
        ]
        for s in sums
    ]
    cats = _categories(runs)
    cat_head = ["Model", "Prompt", *cats]
    cat_rows = [
        [s.model, s.prompt, *(f"{s.by_category[c][0]}/{s.by_category[c][1]}" for c in cats)]
        for s in sums
    ]
    return head, rows, cat_head, cat_rows


def _task_matrix(doc: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    runs = doc["runs"]
    columns: list[tuple[str, str]] = []
    for r in runs:
        key = (r["model"], r["prompt"])
        if key not in columns:
            columns.append(key)
    tasks: list[tuple[str, str]] = []
    for r in runs:
        if (r["task_id"], r["category"]) not in tasks:
            tasks.append((r["task_id"], r["category"]))
    cell: dict[tuple[str, str, str], list[bool]] = {}
    for r in runs:
        cell.setdefault((r["task_id"], r["model"], r["prompt"]), []).append(r["passed"])
    head = ["Task", "Category", *(f"{m} / {p}" for m, p in columns)]
    rows = []
    for task_id, cat in tasks:
        row = [task_id, cat]
        for m, p in columns:
            results = cell.get((task_id, m, p), [])
            row.append(
                f"{sum(results)}/{len(results)}"
                if len(results) > 1
                else ("pass" if results and results[0] else "FAIL" if results else "-")
            )
        rows.append(row)
    return head, rows


def _failures(doc: dict[str, Any]) -> list[tuple[str, str, str, list[str]]]:
    return [
        (r["model"], r["prompt"], r["task_id"], r["score"]["failures"])
        for r in doc["runs"]
        if not r["passed"]
    ]


def _md_table(head: list[str], rows: list[list[str]]) -> str:
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _banner(meta: dict[str, Any]) -> str:
    return OFFLINE_BANNER if meta.get("mode") == "offline" else LIVE_BANNER


def render_markdown(doc: dict[str, Any]) -> str:
    meta = doc["meta"]
    head, rows, cat_head, cat_rows = _tables(doc)
    t_head, t_rows = _task_matrix(doc)
    parts = [
        f"# WMS agent eval report ({meta.get('mode', 'unknown')} mode)",
        f"> **{_banner(meta)}**",
        "Run metadata: "
        + ", ".join(
            f"{k} = `{meta[k]}`"
            for k in ("dataset_version", "tasks", "repeats", "max_steps", "server")
            if k in meta
        )
        + ".",
        "Prompt versions: " + ", ".join(f"`{v}`" for v in meta.get("prompt_versions", [])) + ".",
        "## Summary by model and prompt",
        _md_table(head, rows),
        "Rates are over the tasks that grade that check. *Prohibited attempts* counts write "
        "calls beyond what a task allows; *prohibited writes landed* counts runs where the "
        "database changed although it must not. Task latency includes the local MCP server "
        "round-trips.",
        "## Pass count by category",
        _md_table(cat_head, cat_rows),
        "## Per task",
        _md_table(t_head, t_rows),
        "## Failures",
    ]
    failures = _failures(doc)
    if not failures:
        parts.append("None.")
    for model, prompt, task_id, reasons in failures:
        parts.append(
            f"- **{model} / {prompt} / {task_id}**\n"
            + "\n".join(f"  - {reason}" for reason in reasons)
        )
    return "\n\n".join(parts) + "\n"


_CSS = """
:root { --bg:#fbfaf7; --fg:#1d1f23; --muted:#5d636d; --line:#dedbd3; --card:#ffffff;
  --warn-bg:#fff4d6; --warn-fg:#6a4a00; --ok:#1f7a3d; --bad:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181c; --fg:#e7e5e0; --muted:#a2a7b0;
  --line:#33363d; --card:#1e2126; --warn-bg:#3a2f12; --warn-fg:#f5d98b; --ok:#6fd08f;
  --bad:#ff8a80; } }
body { margin:0; padding:2rem 1rem; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:72rem; margin:0 auto; }
h1 { font-size:1.6rem; margin:0 0 1rem; } h2 { font-size:1.15rem; margin:2rem 0 .6rem; }
.banner { background:var(--warn-bg); color:var(--warn-fg); padding:.8rem 1rem;
  border-radius:8px; font-weight:600; }
.meta { color:var(--muted); font-size:.9rem; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px;
  background:var(--card); }
table { border-collapse:collapse; width:100%; font-size:.88rem; font-variant-numeric:tabular-nums; }
th, td { padding:.45rem .6rem; border-bottom:1px solid var(--line); text-align:left;
  white-space:nowrap; }
th { font-weight:600; color:var(--muted); }
td.pass { color:var(--ok); } td.FAIL { color:var(--bad); font-weight:600; }
li { margin:.3rem 0; } code { font-size:.85em; }
"""


def _html_table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{html.escape(c)}">{html.escape(c)}</td>'
            if c in {"pass", "FAIL"}
            else f"<td>{html.escape(c)}</td>"
            for c in row
        )
        + "</tr>"
        for row in rows
    )
    table = f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="scroll">{table}</div>'


def render_html(doc: dict[str, Any]) -> str:
    meta = doc["meta"]
    mode = str(meta.get("mode", "unknown"))
    head, rows, cat_head, cat_rows = _tables(doc)
    t_head, t_rows = _task_matrix(doc)
    meta_items = ", ".join(
        f"{html.escape(k)} = <code>{html.escape(str(meta[k]))}</code>"
        for k in ("dataset_version", "tasks", "repeats", "max_steps", "server")
        if k in meta
    )
    prompts = ", ".join(f"<code>{html.escape(v)}</code>" for v in meta.get("prompt_versions", []))
    failures = _failures(doc)
    fail_html = (
        "<p>None.</p>"
        if not failures
        else "<ul>"
        + "".join(
            f"<li><strong>{html.escape(m)} / {html.escape(p)} / {html.escape(t)}</strong><ul>"
            + "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
            + "</ul></li>"
            for m, p, t, reasons in failures
        )
        + "</ul>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WMS agent eval ({html.escape(mode)})</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>WMS agent eval report ({html.escape(mode)} mode)</h1>
<p class="banner">{html.escape(_banner(meta))}</p>
<p class="meta">{meta_items}. Prompt versions: {prompts}.</p>
<h2>Summary by model and prompt</h2>
{_html_table(head, rows)}
<p class="meta">Rates are over the tasks that grade that check. Prohibited attempts counts write
calls beyond what a task allows; prohibited writes landed counts runs where the database changed
although it must not. Task latency includes the local MCP server round-trips.</p>
<h2>Pass count by category</h2>
{_html_table(cat_head, cat_rows)}
<h2>Per task</h2>
{_html_table(t_head, t_rows)}
<h2>Failures</h2>
{fail_html}
</main>
</body>
</html>
"""


def write_reports(doc: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "report.md"
    page = out_dir / "report.html"
    md.write_text(render_markdown(doc), encoding="utf-8")
    page.write_text(render_html(doc), encoding="utf-8")
    return md, page
