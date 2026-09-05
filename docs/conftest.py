"""pytest configuration for the documentation doctests.

The doctests of the documentation pages (``pytest --doctest-modules docs ...``) read files by
paths relative to the repository root (``examples/data/...``); this fixture makes them independent
of the directory pytest was started from. It is a no-op for every other test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _docs_doctests_run_from_the_repository_root(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = getattr(request.node, "path", None)
    if path is not None and Path(str(path)).is_relative_to(ROOT / "docs"):
        monkeypatch.chdir(ROOT)
