"""Tests for Pixi configuration invariants."""

import tomllib
from pathlib import Path


def test_pyproject_dependencies_include_required_cli_imports() -> None:
    """Ensure package metadata includes modules imported by the CLI."""
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert any(dependency.startswith("fabric") for dependency in dependencies)
    assert "psutil" in dependencies
    assert not any(dependency.startswith("ray") for dependency in dependencies)
