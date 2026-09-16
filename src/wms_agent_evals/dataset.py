"""Task dataset: requests over the synthetic WMS seed, plus rule-based expectations.

The dataset is YAML validated by strict pydantic models: an unknown key, a
misspelt tool name or a duplicate id fails at load time instead of silently
producing a task that scores nothing.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

READ_TOOLS = frozenset(
    {"find_orders", "get_order", "list_stalled_orders", "get_stock", "get_audit_log"}
)
WRITE_TOOLS = frozenset({"set_order_status", "add_order_note"})
KNOWN_TOOLS = READ_TOOLS | WRITE_TOOLS

DEFAULT_DATASET = Path("datasets/tasks.yaml")


class Category(StrEnum):
    SIMPLE_READ = "simple_read"
    COMPUTED_READ = "computed_read"
    AMBIGUOUS = "ambiguous"
    CLARIFY_THEN_ACT = "clarify_then_act"
    PROHIBITED = "prohibited"
    INJECTION = "injection"
    DRY_RUN_WRITE = "dry_run_write"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Contains(_Strict):
    """Case-insensitive substring match on the string form of an argument."""

    contains: str


class Nonempty(_Strict):
    """The argument is present and not blank; its exact wording is not graded."""

    nonempty: Literal[True]


ArgMatcher = Annotated[
    StrictBool | StrictInt | StrictStr | Contains | Nonempty,
    Field(union_mode="left_to_right"),
]


class Expectations(_Strict):
    # Tool use
    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    args: dict[str, dict[str, ArgMatcher]] = Field(default_factory=dict)
    # Clarification: True = must ask, False = must not ask, None = not graded
    should_clarify: bool | None = None
    # Writes. db_unchanged: every table must be identical after the run.
    # changed_orders: exactly these order rows may change (and must).
    db_unchanged: bool = False
    changed_orders: list[str] | None = None
    max_write_calls: int | None = Field(default=None, ge=0)
    # Final answer
    mentions_all: list[str] = Field(default_factory=list)
    mentions_any: list[str] = Field(default_factory=list)
    mentions_none: list[str] = Field(default_factory=list)
    # Every number in the answer must appear in a tool output or the user's messages.
    grounded_numbers: bool = False

    @field_validator("tools_called", "tools_not_called")
    @classmethod
    def _known_tools(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"unknown tool(s): {unknown}")
        return value

    @field_validator("args")
    @classmethod
    def _known_arg_tools(
        cls, value: dict[str, dict[str, ArgMatcher]]
    ) -> dict[str, dict[str, ArgMatcher]]:
        unknown = sorted(set(value) - KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"args given for unknown tool(s): {unknown}")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Expectations:
        clash = set(self.tools_called) & set(self.tools_not_called)
        if clash:
            raise ValueError(f"tool(s) both required and forbidden: {sorted(clash)}")
        missing = sorted(set(self.args) - set(self.tools_called))
        if missing:
            raise ValueError(f"args expected for tool(s) not in tools_called: {missing}")
        if self.db_unchanged and self.changed_orders is not None:
            raise ValueError("db_unchanged and changed_orders contradict each other")
        return self


class Task(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: Category
    prompt: str = Field(min_length=1)
    write_mode: Literal["off", "dry_run", "on"] = "off"
    notes: str = ""
    # Simulated user: the reply to the agent's first, second... clarifying question.
    user_replies: list[str] = Field(default_factory=list)
    expect: Expectations

    @model_validator(mode="after")
    def _replies_need_a_question(self) -> Task:
        if self.user_replies and self.expect.should_clarify is not True:
            raise ValueError(f"{self.id}: user_replies only make sense with should_clarify: true")
        return self


class Dataset(_Strict):
    version: str
    description: str = ""
    tasks: list[Task]

    @model_validator(mode="after")
    def _unique_ids(self) -> Dataset:
        ids = [t.id for t in self.tasks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate task id(s): {dupes}")
        return self

    def select(self, ids: list[str]) -> list[Task]:
        if not ids:
            return list(self.tasks)
        unknown = sorted(set(ids) - {t.id for t in self.tasks})
        if unknown:
            raise ValueError(f"unknown task id(s): {unknown}")
        return [t for t in self.tasks if t.id in set(ids)]


def parse_dataset(raw: Any) -> Dataset:
    return Dataset.model_validate(raw)


def load_dataset(path: Path) -> Dataset:
    with path.open(encoding="utf-8") as fh:
        return parse_dataset(yaml.safe_load(fh))
