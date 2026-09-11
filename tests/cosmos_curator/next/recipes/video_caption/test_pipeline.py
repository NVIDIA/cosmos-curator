# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for video-caption orchestration and its no-work fast path."""

from pathlib import Path
from types import SimpleNamespace

import pyarrow.fs as pafs
import pytest

from cosmos_curator.next.recipes.video_caption import pipeline
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig, resolve_config_data
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt
from cosmos_curator.next.recipes.video_caption.publication import PublicationSummary
from cosmos_curator.next.recipes.video_caption.workspace import CaptionWorkspace


def _config(tmp_path: Path) -> ResolvedVideoCaptionConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {
                "media_root": str(tmp_path / "media"),
                "clips_lance_uri": str(tmp_path / "clips.lance"),
            },
            "model": {"variant": "qwen3_8_27b_fp8"},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
        }
    )


def _workspace(tmp_path: Path) -> CaptionWorkspace:
    root = tmp_path / "workspace"
    return CaptionWorkspace(
        root_uri=str(root),
        manifest_uri=str(root / "workspace.json"),
        results_uri=str(root / "results"),
        checkpoints_uri=str(root / "checkpoints"),
        filesystem=pafs.LocalFileSystem(),
    )


def _common_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, attempt: CaptionAttempt) -> list[str]:
    events: list[str] = []
    dataset = SimpleNamespace(version=attempt.version)
    monkeypatch.setattr(pipeline, "get_lance_storage_options", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "ensure_caption_fields", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(pipeline, "validate_caption_input", lambda *_args, **_kwargs: events.append("validate"))
    monkeypatch.setattr(pipeline, "capture_attempt", lambda *_args, **_kwargs: attempt)
    monkeypatch.setattr(pipeline, "resolve_workspace", lambda *_args, **_kwargs: _workspace(tmp_path))
    monkeypatch.setattr(pipeline, "phase_a_completion_covers", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(pipeline.lance, "dataset", lambda *_args, **_kwargs: SimpleNamespace(version=12))
    return events


def test_pending_attempt_runs_one_inference_phase_then_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pending work validates model/workspace, runs one Phase A, then publishes."""
    attempt = CaptionAttempt(version=3, pending_fragment_ids=(0, 1), complete_fragment_ids=(2,))
    events = _common_mocks(monkeypatch, tmp_path, attempt)
    ray_init_calls: list[dict[str, object]] = []
    monkeypatch.setattr(pipeline, "validate_model_directory", lambda *_args: events.append("model"))
    monkeypatch.setattr(pipeline, "ensure_workspace", lambda *_args: events.append("workspace"))
    monkeypatch.setattr(pipeline, "curator_io_resources", lambda: {"curator_io": 16.0})
    monkeypatch.setattr(
        pipeline,
        "s3_compatibility_env_vars",
        lambda *_args, **_kwargs: {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"},
    )

    def initialize_ray(**kwargs: object) -> None:
        ray_init_calls.append(kwargs)
        events.append("ray")

    monkeypatch.setattr(pipeline, "ensure_ray_initialized", initialize_ray)
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", lambda **_kwargs: events.append("progress"))
    monkeypatch.setattr(pipeline, "configure_ray_data_stability", lambda **_kwargs: events.append("stability"))
    monkeypatch.setattr(
        pipeline,
        "configure_ray_data_eager_actor_autoscaling",
        lambda: events.append("autoscaling"),
    )
    monkeypatch.setattr(pipeline, "run_inference_phase", lambda *_args, **_kwargs: events.append("inference"))

    def publish(*_args: object, **_kwargs: object) -> PublicationSummary:
        events.append("publish")
        return PublicationSummary(published_fragments=2, already_committed_fragments=0)

    monkeypatch.setattr(pipeline, "publish_staged_results", publish)
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda *_args, **_kwargs: events.append("cleanup"))

    summary = pipeline.run_config(_config(tmp_path))

    assert events == [
        "validate",
        "workspace",
        "model",
        "ray",
        "progress",
        "stability",
        "autoscaling",
        "inference",
        "publish",
        "cleanup",
    ]
    assert summary["attempt_version"] == 3
    assert summary["pending_fragments"] == 2
    assert summary["published_fragments"] == 2
    assert summary["skipped_complete_fragments"] == 1
    assert summary["clips_lance_version"] == 12
    assert summary["cleanup_complete"] is True
    assert ray_init_calls == [
        {
            "local_resources": {"curator_io": 16.0},
            "runtime_env_vars": {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"},
        }
    ]


def test_storage_compatibility_is_selected_from_every_worker_visible_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray workers are started with the settings every location they open requires.

    The media root, the clip table, and the staging root are each configurable on
    their own, so a selection made from the staging root alone would leave a
    worker reading media or scanning the table under the SDK default.
    """
    attempt = CaptionAttempt(version=9, pending_fragment_ids=(0,), complete_fragment_ids=())
    config = _config(tmp_path)
    workspace = _workspace(tmp_path)
    compatibility_calls: list[tuple[tuple[str, ...], str]] = []
    _common_mocks(monkeypatch, tmp_path, attempt)
    monkeypatch.setattr(pipeline, "phase_a_completion_covers", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(pipeline, "ensure_workspace", lambda *_args: None)
    monkeypatch.setattr(pipeline, "curator_io_resources", lambda: {"curator_io": 16.0})

    def classify_endpoints(*locations: str, storage_profile: str) -> dict[str, str]:
        compatibility_calls.append((locations, storage_profile))
        return {}

    monkeypatch.setattr(pipeline, "s3_compatibility_env_vars", classify_endpoints)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "configure_ray_data_stability", lambda **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "publish_staged_results",
        lambda *_args, **_kwargs: PublicationSummary(published_fragments=1, already_committed_fragments=0),
    )
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda *_args, **_kwargs: None)

    pipeline.run_config(config)

    assert compatibility_calls == [
        (
            (config.input.media_root, config.input.clips_lance_uri, workspace.root_uri),
            config.execution.storage_profile,
        )
    ]


def test_complete_attempt_skips_model_and_ray_and_validates_existing_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete attempt skips model/Ray work but validates reusable staging."""
    attempt = CaptionAttempt(version=8, pending_fragment_ids=(), complete_fragment_ids=(0, 1))
    events = _common_mocks(monkeypatch, tmp_path, attempt)
    monkeypatch.setattr(pipeline, "workspace_manifest_exists", lambda *_args: True)
    monkeypatch.setattr(pipeline, "ensure_workspace", lambda *_args: events.append("workspace"))
    monkeypatch.setattr(
        pipeline,
        "validate_model_directory",
        lambda *_args: pytest.fail("complete attempts must not require model files"),
    )
    monkeypatch.setattr(
        pipeline,
        "ensure_ray_initialized",
        lambda **_kwargs: pytest.fail("complete attempts must not initialize Ray"),
    )
    monkeypatch.setattr(
        pipeline,
        "publish_staged_results",
        lambda *_args, **_kwargs: PublicationSummary(published_fragments=0, already_committed_fragments=0),
    )
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda *_args, **_kwargs: events.append("cleanup"))

    summary = pipeline.run_config(_config(tmp_path))

    assert events == ["validate", "workspace", "cleanup"]
    assert summary["pending_fragments"] == 0
    assert summary["skipped_complete_fragments"] == 2


def test_recovered_phase_a_skips_model_but_initializes_ray_for_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed Phase A marker skips inference while Ray prepares publication."""
    attempt = CaptionAttempt(version=9, pending_fragment_ids=(0,), complete_fragment_ids=())
    events = _common_mocks(monkeypatch, tmp_path, attempt)
    monkeypatch.setattr(pipeline, "phase_a_completion_covers", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(pipeline, "ensure_workspace", lambda *_args: events.append("workspace"))
    monkeypatch.setattr(
        pipeline,
        "validate_model_directory",
        lambda *_args: pytest.fail("complete Phase A must not require model files"),
    )
    monkeypatch.setattr(pipeline, "curator_io_resources", lambda: {"curator_io": 16.0})
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_kwargs: events.append("ray"))
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", lambda **_kwargs: events.append("progress"))
    monkeypatch.setattr(pipeline, "configure_ray_data_stability", lambda **_kwargs: events.append("stability"))
    monkeypatch.setattr(
        pipeline,
        "configure_ray_data_eager_actor_autoscaling",
        lambda: pytest.fail("publication must not enable eager inference actor autoscaling"),
    )
    monkeypatch.setattr(
        pipeline,
        "run_inference_phase",
        lambda *_args, **_kwargs: pytest.fail("complete Phase A must not run inference"),
    )

    def publish(*_args: object, **_kwargs: object) -> PublicationSummary:
        events.append("publish")
        return PublicationSummary(published_fragments=1, already_committed_fragments=0)

    monkeypatch.setattr(pipeline, "publish_staged_results", publish)
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda *_args, **_kwargs: events.append("cleanup"))

    summary = pipeline.run_config(_config(tmp_path))

    assert events == ["validate", "workspace", "ray", "progress", "stability", "publish", "cleanup"]
    assert summary["published_fragments"] == 1


def test_cleanup_failure_does_not_invalidate_canonical_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup is best effort after Lance has been canonically verified."""
    attempt = CaptionAttempt(version=5, pending_fragment_ids=(), complete_fragment_ids=())
    _common_mocks(monkeypatch, tmp_path, attempt)
    monkeypatch.setattr(pipeline, "workspace_manifest_exists", lambda *_args: True)
    monkeypatch.setattr(pipeline, "ensure_workspace", lambda *_args: None)
    monkeypatch.setattr(
        pipeline,
        "publish_staged_results",
        lambda *_args, **_kwargs: PublicationSummary(published_fragments=0, already_committed_fragments=0),
    )

    def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        msg = "temporary object-store outage"
        raise OSError(msg)

    monkeypatch.setattr(pipeline, "cleanup_workspace", fail_cleanup)

    summary = pipeline.run_config(_config(tmp_path))

    assert summary["clips_lance_version"] == 12
    assert summary["cleanup_complete"] is False
