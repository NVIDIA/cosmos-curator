# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery manifests and optional optimization markers."""

import json
import sys
from pathlib import Path

import pytest
from loguru import logger

from cosmos_curator.next.enrichment import workspace as recovery_workspace
from cosmos_curator.next.enrichment.workspace import RecoveryWorkspace


@pytest.mark.parametrize("marker", [None, b"not-json", b"[]", b'{"manifest_digest":"different"}'])
def test_missing_or_invalid_completion_falls_back_to_checkpoints(tmp_path: Path, marker: bytes | None) -> None:
    """A marker is only an optimization and does not make staged results authoritative."""
    workspace = RecoveryWorkspace(str(tmp_path / "workspace"), {"contract": "test"})
    workspace.ensure(storage_profile="default")
    if marker is not None:
        Path(workspace.completion).write_bytes(marker)
    assert not workspace.processing_complete(storage_profile="default")
    Path(workspace.completion).write_text(json.dumps({"manifest_digest": workspace.manifest_digest}), encoding="utf-8")
    assert workspace.processing_complete(storage_profile="default")


def test_manifest_mismatch_preserves_recovery_state(tmp_path: Path) -> None:
    """A new result contract cannot claim or alter another workspace's data."""
    workspace = RecoveryWorkspace(str(tmp_path / "workspace"), {"contract": "original"})
    workspace.ensure(storage_profile="default")
    workspace.record_completion(storage_profile="default")
    other = RecoveryWorkspace(workspace.root, {"contract": "different"})
    with pytest.raises(ValueError, match="workspace does not match"):
        other.ensure(storage_profile="default")
    assert workspace.processing_complete(storage_profile="default")


def test_manifest_tuples_round_trip_on_resume(tmp_path: Path) -> None:
    """Equivalent tuple/list representations retain the same workspace and completion marker."""
    workspace = RecoveryWorkspace(str(tmp_path / "workspace"), {"contract": {"shape": (2, 3)}})
    workspace.ensure(storage_profile="default")
    workspace.record_completion(storage_profile="default")
    workspace.ensure(storage_profile="default")

    resumed = RecoveryWorkspace(workspace.root, {"contract": {"shape": [2, 3]}})
    resumed.ensure(storage_profile="default")
    assert resumed.manifest_digest == workspace.manifest_digest
    assert resumed.processing_complete(storage_profile="default")


def test_workspace_keeps_an_immutable_manifest_snapshot(tmp_path: Path) -> None:
    """Caller mutations cannot rebind a chosen workspace or its completion marker."""
    manifest = {"contract": {"shape": [2, 3]}}
    workspace = RecoveryWorkspace(str(tmp_path / "workspace"), manifest)
    digest = workspace.manifest_digest
    workspace.ensure(storage_profile="default")
    workspace.record_completion(storage_profile="default")
    manifest["contract"]["shape"].append(4)
    workspace.manifest["contract"]["shape"].append(5)
    assert workspace.manifest_digest == digest
    assert workspace.manifest == {"contract": {"shape": [2, 3]}}
    workspace.ensure(storage_profile="default")
    assert workspace.processing_complete(storage_profile="default")


def test_discovery_reads_only_direct_workspace_manifests(tmp_path: Path) -> None:
    """Unmanaged paths and nested result files are never treated as recovery workspaces."""
    workspace = RecoveryWorkspace(str(tmp_path / "selection"), {"contract": "test"})
    workspace.ensure(storage_profile="default")
    nested = Path(workspace.results, "nested")
    nested.mkdir(parents=True)
    (nested / "workspace.json").write_text("not a manifest", encoding="utf-8")
    (tmp_path / "unmanaged").mkdir()
    assert RecoveryWorkspace.discover(str(tmp_path), storage_profile="default") == (workspace,)
    assert RecoveryWorkspace.discover(str(tmp_path / "missing"), storage_profile="default") == ()


def test_marker_write_failure_keeps_durable_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Loss of an optional marker never invalidates durable processing progress."""
    workspace = RecoveryWorkspace(str(tmp_path / "workspace"), {"contract": "test"})
    workspace.ensure(storage_profile="default")
    Path(workspace.results).mkdir()
    result = Path(workspace.results, "result.parquet")
    result.write_bytes(b"durable-result")

    def fail_write(*_args: object, **_kwargs: object) -> bool:
        msg = "marker unavailable"
        raise OSError(msg)

    monkeypatch.setattr(recovery_workspace, "write_if_absent", fail_write)
    sink = logger.add(sys.stderr, level="WARNING")
    try:
        workspace.record_completion(storage_profile="default")
    finally:
        logger.remove(sink)
    assert "marker unavailable" in capfd.readouterr().err
    assert result.read_bytes() == b"durable-result"
    assert not workspace.processing_complete(storage_profile="default")
