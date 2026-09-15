# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the durable contract-scoped caption workspace."""

import json
from pathlib import Path

import pyarrow.fs as pafs
import pytest

from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig, resolve_config_data
from cosmos_curator.next.recipes.video_caption.contracts import (
    CHECKPOINT_ADAPTER_VERSION,
    WORKSPACE_SCHEMA_VERSION,
    CaptionModelSpec,
    normalized_caption_contract,
)
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt
from cosmos_curator.next.recipes.video_caption.workspace import (
    CaptionWorkspace,
    arrow_filesystem,
    cleanup_workspace,
    ensure_workspace,
    filesystem_path,
    phase_a_completion_covers,
    record_phase_a_completion,
    resolve_workspace,
    workspace_manifest_exists,
)


def _config(media_root: Path, staging_root: Path | str) -> ResolvedVideoCaptionConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {"media_root": str(media_root)},
            "model": {"variant": "qwen3_8_27b_fp8"},
            "output": {"staging_root_uri": str(staging_root)},
        }
    )


def _manifest(
    config: ResolvedVideoCaptionConfig,
    spec: CaptionModelSpec,
    digest: str,
) -> dict[str, object]:
    contract = normalized_caption_contract(spec)
    return {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "media_root": config.input.media_root,
        "clips_lance_uri": config.input.clips_lance_uri,
        "caption_field": spec.caption_field_name,
        "metadata_field": spec.metadata_field_name,
        "field_schemas": contract["fields"],
        "model": {
            "id": spec.model_id,
            "revision": spec.revision,
            "runtime_model_dir": str(spec.runtime_model_dir),
        },
        "caption_contract": contract,
        "caption_contract_digest": digest,
        "checkpoint_adapter_version": CHECKPOINT_ADAPTER_VERSION,
    }


def _resolve_workspace(
    config: ResolvedVideoCaptionConfig,
    spec: CaptionModelSpec,
    digest: str,
) -> CaptionWorkspace:
    return resolve_workspace(
        config.output.staging_root_uri,
        spec.caption_field_name,
        digest,
        storage_profile=config.execution.storage_profile,
    )


def _ensure_workspace(
    workspace: CaptionWorkspace,
    config: ResolvedVideoCaptionConfig,
    spec: CaptionModelSpec,
    digest: str,
) -> None:
    ensure_workspace(
        workspace,
        storage_profile=config.execution.storage_profile,
        manifest=_manifest(config, spec, digest),
    )


def test_workspace_layout_and_manifest_are_contract_scoped(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    caption_digest: str,
) -> None:
    """Layout and manifest depend on field set and result contract, not attempts."""
    config = _config(tmp_path / "media", tmp_path / "staging")
    workspace = _resolve_workspace(config, caption_spec, caption_digest)

    _ensure_workspace(workspace, config, caption_spec, caption_digest)
    _ensure_workspace(workspace, config, caption_spec, caption_digest)

    expected_root = tmp_path / "staging" / caption_spec.caption_field_name / caption_digest
    assert workspace.root_uri == str(expected_root)
    assert workspace.results_uri == str(expected_root / "results")
    assert workspace.checkpoints_uri == str(expected_root / "checkpoints")
    assert workspace.phase_a_completion_uri == str(expected_root / "phase-a-complete.json")
    assert isinstance(workspace.filesystem, pafs.LocalFileSystem)
    assert workspace_manifest_exists(workspace) is True
    manifest = json.loads((expected_root / "workspace.json").read_text(encoding="utf-8"))
    assert manifest["caption_contract_digest"] == caption_digest
    assert manifest["caption_field"] == caption_spec.caption_field_name
    assert "lance_version" not in manifest
    assert "fragment" not in manifest


def test_phase_a_completion_marker_uses_fragment_level_coverage(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    caption_digest: str,
) -> None:
    """One small marker covers the completed attempt and later pending subsets."""
    config = _config(tmp_path / "media", tmp_path / "staging")
    workspace = _resolve_workspace(config, caption_spec, caption_digest)
    _ensure_workspace(workspace, config, caption_spec, caption_digest)
    completed = CaptionAttempt(version=3, pending_fragment_ids=(1, 2), complete_fragment_ids=(0,))

    assert phase_a_completion_covers(workspace, completed, caption_digest) is False

    record_phase_a_completion(workspace, completed, caption_digest)

    assert phase_a_completion_covers(workspace, completed, caption_digest) is True
    assert (
        phase_a_completion_covers(
            workspace,
            CaptionAttempt(version=4, pending_fragment_ids=(2,), complete_fragment_ids=(0, 1)),
            caption_digest,
        )
        is True
    )
    assert (
        phase_a_completion_covers(
            workspace,
            CaptionAttempt(version=4, pending_fragment_ids=(1, 2, 3), complete_fragment_ids=(0,)),
            caption_digest,
        )
        is False
    )
    assert phase_a_completion_covers(workspace, completed, "different-digest") is False
    assert (
        phase_a_completion_covers(
            workspace,
            CaptionAttempt(version=2, pending_fragment_ids=(1,), complete_fragment_ids=()),
            caption_digest,
        )
        is False
    )


def test_existing_incompatible_manifest_fails(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    caption_digest: str,
) -> None:
    """A manifest collision cannot silently reuse another workspace contract."""
    config = _config(tmp_path / "media", tmp_path / "staging")
    workspace = _resolve_workspace(config, caption_spec, caption_digest)
    _ensure_workspace(workspace, config, caption_spec, caption_digest)
    manifest_path = Path(workspace.manifest_uri)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clips_lance_uri"] = "/different/table"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible"):
        _ensure_workspace(workspace, config, caption_spec, caption_digest)


def test_cleanup_deletes_only_the_exact_contract_root(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    caption_digest: str,
) -> None:
    """Cleanup removes the verified digest scope without touching siblings."""
    config = _config(tmp_path / "media", tmp_path / "staging")
    workspace = _resolve_workspace(config, caption_spec, caption_digest)
    _ensure_workspace(workspace, config, caption_spec, caption_digest)
    result = Path(workspace.results_uri) / "part.parquet"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"result")
    sibling = Path(config.output.staging_root_uri) / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")

    cleanup_workspace(workspace, storage_profile="default")

    assert not Path(workspace.root_uri).exists()
    assert sibling.read_text(encoding="utf-8") == "keep"


def test_filesystem_path_unwraps_supported_uris() -> None:
    """Arrow receives protocol-free paths for local and S3 filesystems."""
    assert filesystem_path("s3://bucket/a/b") == "bucket/a/b"
    assert filesystem_path("file:///var/lib/caption/a") == "/var/lib/caption/a"
    assert filesystem_path("relative/a") == "relative/a"


def test_filesystem_path_decodes_a_percent_escaped_file_uri(tmp_path: Path) -> None:
    """``%20`` in a ``file://`` URI is an escaped space, not two characters of the name.

    Asserted against a directory that really exists, because the workspace has to
    open what comes back; a string comparison alone would not notice a staging
    root nothing can reach.
    """
    spaced = tmp_path / "my staging root"
    spaced.mkdir()
    assert "%20" in spaced.as_uri()

    assert Path(filesystem_path(spaced.as_uri())) == spaced


def test_filesystem_path_refuses_an_unsupported_scheme() -> None:
    """An unsupported URI must not pass through as the path of a local staging root.

    Which layer answers this is not the recipe's decision: the shared Arrow
    helper owns the URI contract for every caller that pairs a filesystem with a
    path. Pinned here because what matters at this boundary is that the
    workspace refuses the root rather than staging into a relative directory
    named after the scheme.
    """
    with pytest.raises(ValueError, match="unsupported Arrow filesystem URI scheme 'gs'"):
        filesystem_path("gs://bucket/staging")


def test_filesystem_path_refuses_a_nonlocal_file_authority() -> None:
    """A ``file://host`` root names another machine and cannot become a local path."""
    with pytest.raises(ValueError, match="file URI authority must be empty or localhost"):
        filesystem_path("file://server/staging")


def test_arrow_filesystem_refuses_an_unsupported_scheme() -> None:
    """Backend selection fails closed: an unsupported URI never selects local storage.

    The counterpart to the path refusal above, and the one that matters more: a
    silently local filesystem would report every remote object as absent, so the
    run would stage a complete second copy instead of failing.
    """
    with pytest.raises(ValueError, match="unsupported Arrow filesystem URI scheme 'gs'"):
        arrow_filesystem("gs://bucket/staging", storage_profile="default")


def test_file_uri_workspace_round_trip(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    caption_digest: str,
) -> None:
    """Local workspace helpers consistently unwrap file URIs for every filesystem operation."""
    staging_root = tmp_path / "staging"
    config = _config(tmp_path / "media", staging_root.as_uri())
    resolved = _resolve_workspace(config, caption_spec, caption_digest)
    workspace = CaptionWorkspace(
        root_uri=Path(resolved.root_uri).as_uri(),
        manifest_uri=Path(resolved.manifest_uri).as_uri(),
        results_uri=Path(resolved.results_uri).as_uri(),
        checkpoints_uri=Path(resolved.checkpoints_uri).as_uri(),
        filesystem=resolved.filesystem,
    )

    _ensure_workspace(workspace, config, caption_spec, caption_digest)
    _ensure_workspace(workspace, config, caption_spec, caption_digest)

    assert workspace_manifest_exists(workspace) is True
    assert (Path(resolved.root_uri) / "workspace.json").is_file()

    cleanup_workspace(workspace, storage_profile="default")

    assert not Path(resolved.root_uri).exists()
