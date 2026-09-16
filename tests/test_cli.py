from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from wms_agent_evals.cli import main

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)


def test_validate_passes_on_the_shipped_files(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate"]) == 0
    assert "21 tasks" in capsys.readouterr().out


def test_run_mock_writes_results_and_reports(tmp_path: Path) -> None:
    code = main(
        [
            "run",
            "--provider",
            "mock",
            "--tasks",
            "partial_sku",
            "--prompts",
            "baseline",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    doc = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert doc["meta"]["mode"] == "offline"
    assert doc["meta"]["models"] == ["mock/careful", "mock/eager"]
    assert len(doc["runs"]) == 2
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "report.html").is_file()
    assert len((tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_fail_on_landed_write(tmp_path: Path) -> None:
    args = ["run", "--models", "mock/eager", "--tasks", "change_recipient", "--prompts", "baseline"]
    assert main([*args, "--out", str(tmp_path / "a")]) == 0
    assert main([*args, "--out", str(tmp_path / "b"), "--fail-on-landed-write"]) == 1


def test_mock_provider_refuses_real_model_names(tmp_path: Path) -> None:
    assert main(["run", "--models", "openai/some-model", "--out", str(tmp_path)]) == 2


def test_live_needs_models(tmp_path: Path) -> None:
    assert main(["run", "--provider", "live", "--out", str(tmp_path)]) == 2


def test_live_without_litellm_stops_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def no_litellm(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("no litellm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_litellm)
    code = main(["run", "--provider", "live", "--models", "openai/x", "--out", str(tmp_path)])
    assert code == 2
    assert "pip install -e '.[live]'" in capsys.readouterr().err
    assert not (tmp_path / "results.json").exists()


def test_compare_and_report_commands(tmp_path: Path) -> None:
    committed = REPO / "reports" / "offline" / "results.json"
    assert main(["compare", str(committed), str(committed)]) == 0
    changed = json.loads(committed.read_text(encoding="utf-8"))
    changed["runs"][0]["passed"] = not changed["runs"][0]["passed"]
    other = tmp_path / "changed.json"
    other.write_text(json.dumps(changed), encoding="utf-8")
    assert main(["compare", str(committed), str(other)]) == 1
    assert main(["report", str(committed), "--out-dir", str(tmp_path / "rep")]) == 0
    assert (tmp_path / "rep" / "report.html").is_file()


def test_repeats_must_be_positive(tmp_path: Path) -> None:
    assert main(["run", "--repeats", "0", "--out", str(tmp_path)]) == 2
