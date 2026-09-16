from __future__ import annotations

from pathlib import Path

import pytest

from wms_agent_evals.dataset import Dataset, load_dataset
from wms_agent_evals.providers import Recording, load_recordings

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return load_dataset(REPO / "datasets" / "tasks.yaml")


@pytest.fixture(scope="session")
def recordings() -> dict[str, Recording]:
    return load_recordings(REPO / "recordings")
