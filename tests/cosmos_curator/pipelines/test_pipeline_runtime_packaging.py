# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the runtime task with the deployment client unavailable, as in images."""

import json
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_WITHOUT_CLIENT = """
import runpy
import sys

# A checkout or editable install must not supply the package excluded by Docker.
sys.modules["cosmos_curator.client"] = None
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__")
"""


@pytest.mark.parametrize("manifest", ["pixi.toml", "distributable/pixi.toml"])
@pytest.mark.parametrize("help_only", [True, False], ids=["help", "video-transcode-config"])
def test_run_pipeline_without_deployment_client(
    repo_root: Path,
    tmp_path: Path,
    manifest: str,
    *,
    help_only: bool,
) -> None:
    """Both image variants must start and resolve recipe configs without client code."""
    task = tomllib.loads((repo_root / manifest).read_text())["feature"]["core"]["tasks"]["run-pipeline"]
    command = shlex.split(task)
    assert command[:2] == ["python", "-m"]
    config = tmp_path / "video-transcode.json"
    config.write_text(json.dumps({"schema_version": 1, "kind": "video-transcode", "input": {"media_root": "/media"}}))
    # Validate through the selected recipe, stopping before any storage or Ray work.
    arguments = ["--help"] if help_only else [str(config), "--set", "execution.transcode_cpus=0", "--json"]

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _WITHOUT_CLIENT, *command[2:], *arguments],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == (0 if help_only else 2), result.stderr
    if help_only:
        assert "--set" in result.stdout
        assert "--json" in result.stdout
    else:
        assert result.stdout == ""
        error = json.loads(result.stderr)
        assert error["ok"] is False
        assert error["error"] == "invalid"
        assert "transcode_cpus" in error["message"]
