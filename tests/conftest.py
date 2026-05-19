from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_repo_root_cwd() -> None:
    """Normalize the working directory so relative artifact paths resolve.

    Several tests intentionally use repo-relative fixture paths. The suite is
    often launched from the workspace root, so we chdir to the repository root
    for the duration of the test session.
    """
    repo_root = Path(__file__).resolve().parents[1]
    previous_cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
