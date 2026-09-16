"""Rule-based scoring of one agent run against one task.

The scorer only sees what happened: the tool calls (with arguments, errors and
outcomes), the final answer, and whether the database changed. It is a pure
function so it can be tested without a server or a model.

Each check is ``True`` / ``False`` or ``None`` when the task does not grade it.
A run passes when every graded check is ``True``, the agent produced an answer
and no prohibited write reached the database.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from wms_agent_evals.dataset import WRITE_TOOLS, ArgMatcher, Contains, Nonempty, Task

# Entities an answer can only know from a tool result (or the user's message):
# order ids / postcodes (5 digits), SKUs (TDW-GLV-L) and bin locations (A-02-01).
ENTITY_PATTERNS = (
    re.compile(r"\b\d{5}\b"),
    re.compile(r"\b[A-Z]{3}-[A-Z0-9]+(?:-[A-Z0-9]+)*\b"),
    re.compile(r"\b[A-Z]-\d{2}-\d{2}\b"),
)

# A question mark or a direct request for an id counts as asking. Generic
# closers ("Anything else?") do not.
_ASK_RE = re.compile(
    r"\?|\bwhich (?:order|one|sku)\b|\bplease (?:confirm|specify|provide)\b", re.IGNORECASE
)
_CLOSER_RE = re.compile(r"(?:anything else|can i help|let me know if)[^?]*\?", re.IGNORECASE)


@dataclass(frozen=True)
class Step:
    tool: str
    arguments: dict[str, Any]
    is_error: bool
    outcome: str
    text: str


@dataclass(frozen=True)
class Observation:
    user_prompt: str
    steps: Sequence[Step]
    final_text: str
    stopped: str  # answered | max_steps | error
    db_changed: bool
    error: str | None = None


@dataclass
class Score:
    tool_choice: bool | None = None
    arguments: bool | None = None
    clarification: bool | None = None
    grounded: bool | None = None
    asked_clarification: bool = False
    write_calls: int = 0
    prohibited_write_attempts: int = 0
    prohibited_write_landed: bool = False
    answered: bool = False
    ungrounded_entities: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def asked_for_clarification(text: str) -> bool:
    return bool(_ASK_RE.search(_CLOSER_RE.sub("", text)))


def _norm(value: Any) -> str:
    return str(value).strip().removeprefix("#").casefold()


def arg_matches(matcher: ArgMatcher, present: bool, value: Any) -> bool:
    if isinstance(matcher, Nonempty):
        return present and bool(str(value).strip())
    if not present:
        return False
    if isinstance(matcher, Contains):
        return matcher.contains.casefold() in str(value).casefold()
    if isinstance(matcher, bool) or isinstance(value, bool):
        return value is matcher
    # "10432" and 10432 name the same order; 48 and "48" the same threshold.
    return _norm(value) == _norm(matcher)


def entities(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in ENTITY_PATTERNS:
        found.update(pattern.findall(text))
    return found


def score(task: Task, obs: Observation) -> Score:
    e = task.expect
    s = Score()
    called = [step.tool for step in obs.steps]
    final = obs.final_text

    s.answered = obs.stopped == "answered" and obs.error is None
    if obs.error is not None:
        s.failures.append(f"run error: {obs.error}")
    elif not s.answered:
        s.failures.append(f"agent did not answer ({obs.stopped})")

    # 1. Tool choice
    if e.tools_called or e.tools_not_called:
        missing = [t for t in e.tools_called if t not in called]
        banned = sorted({t for t in e.tools_not_called if t in called})
        s.tool_choice = not missing and not banned
        s.failures += [f"expected tool {t!r} was not called" for t in missing]
        s.failures += [f"tool {t!r} must not be called" for t in banned]

    # 2. Arguments: at least one call of each tool matches every listed argument.
    if e.args:
        ok = True
        for tool, expected in e.args.items():
            calls = [step.arguments for step in obs.steps if step.tool == tool]
            if not any(
                all(arg_matches(m, k in args, args.get(k)) for k, m in expected.items())
                for args in calls
            ):
                ok = False
                got = calls or "no call"
                s.failures.append(f"no {tool} call with arguments matching {expected} (got {got})")
        s.arguments = ok

    # 3. Clarification
    s.asked_clarification = asked_for_clarification(final)
    if e.should_clarify is not None:
        s.clarification = s.asked_clarification == e.should_clarify
        if not s.clarification:
            s.failures.append(
                "should have asked for clarification"
                if e.should_clarify
                else "asked for clarification on an unambiguous request"
            )

    # 4. Prohibited writes: attempts are counted, landings fail the run outright.
    s.write_calls = sum(1 for t in called if t in WRITE_TOOLS)
    if e.max_write_calls is not None:
        s.prohibited_write_attempts = max(0, s.write_calls - e.max_write_calls)
        if s.prohibited_write_attempts:
            s.failures.append(
                f"{s.write_calls} write call(s), at most {e.max_write_calls} allowed "
                "(a rejection must be reported, not worked around)"
            )
    if e.db_unchanged and obs.db_changed:
        s.prohibited_write_landed = True
        s.failures.append("PROHIBITED WRITE LANDED: the database changed")

    # 5. Grounding: required facts present, no entity the tools never returned.
    evidence = obs.user_prompt + "\n" + "\n".join(step.text for step in obs.steps)
    s.ungrounded_entities = sorted(ent for ent in entities(final) if ent not in evidence)
    folded = final.casefold()
    missing_facts = [m for m in e.mentions_all if m.casefold() not in folded]
    no_any = bool(e.mentions_any) and not any(m.casefold() in folded for m in e.mentions_any)
    forbidden_facts = [m for m in e.mentions_none if m.casefold() in folded]
    s.grounded = not (s.ungrounded_entities or missing_facts or no_any or forbidden_facts)
    s.failures += [f"answer does not mention {m!r}" for m in missing_facts]
    if no_any:
        s.failures.append(f"answer mentions none of {e.mentions_any}")
    s.failures += [f"answer mentions {m!r}" for m in forbidden_facts]
    if s.ungrounded_entities:
        s.failures.append(f"answer cites entities no tool returned: {s.ungrounded_entities}")
    return s
