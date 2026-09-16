"""System prompt variants, loaded from ``prompts/<name>.md``.

Each variant's version string includes a short content hash, so a result row
always says exactly which prompt text produced it, even after an edit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPTS_DIR = Path("prompts")
DEFAULT_VARIANTS = ("baseline", "ask_before_assume")


@dataclass(frozen=True)
class PromptVariant:
    name: str
    text: str

    @property
    def version(self) -> str:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:8]
        return f"{self.name}@{digest}"


def load_prompt(directory: Path, name: str) -> PromptVariant:
    path = directory / f"{name}.md"
    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.md"))
        raise ValueError(f"unknown prompt {name!r}; available: {available}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt {name!r} is empty")
    return PromptVariant(name, text + "\n")
