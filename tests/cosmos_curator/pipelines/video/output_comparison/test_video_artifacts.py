# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for output-comparison artifact loaders."""

from pathlib import Path
from typing import Any

import pytest

from cosmos_curator.pipelines.video.output_comparison.caption_comparator import CaptionClipViewLoadWorker
from cosmos_curator.pipelines.video.output_comparison.caption_schema import ClipCaptionView
from cosmos_curator.pipelines.video.output_comparison.video_artifacts import load_clip_artifacts
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec


def _clip_spec(tmp_path: Path, *, clip_id: str = "clip-a", in_a: bool = True, in_b: bool = True) -> ClipComparisonSpec:
    return ClipComparisonSpec(
        video_key="video.mp4",
        clip_id=clip_id,
        output_a=str(tmp_path / "output-a"),
        output_b=str(tmp_path / "output-b"),
        in_a=in_a,
        in_b=in_b,
    )


def test_load_clip_artifacts_loads_only_present_output_sides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Clip side-presence flags decide which metadata objects are read."""
    read_paths: list[str] = []

    def fake_read_json_object(path: object, *, client_params: object) -> dict[str, object]:
        _ = client_params
        read_paths.append(str(path))
        return {"span_uuid": Path(str(path)).stem}

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    artifacts = load_clip_artifacts(_clip_spec(tmp_path, in_a=True, in_b=False), profile_name="default")

    assert artifacts.metadata_a == {"span_uuid": "clip-a"}
    assert artifacts.metadata_b is None
    assert artifacts.metadata_path_a == str(tmp_path / "output-a" / "metas" / "v0" / "clip-a.json")
    assert artifacts.metadata_path_b is None
    assert read_paths == [str(tmp_path / "output-a" / "metas" / "v0" / "clip-a.json")]


def test_load_clip_artifacts_uses_metadata_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Clip metadata loading honors non-default metadata versions."""
    read_paths: list[str] = []

    def fake_read_json_object(path: object, *, client_params: object) -> dict[str, object]:
        _ = client_params
        read_paths.append(str(path))
        return {"span_uuid": Path(str(path)).stem}

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    artifacts = load_clip_artifacts(
        _clip_spec(tmp_path, in_a=True, in_b=False),
        profile_name="default",
        metadata_version="v9",
    )

    assert artifacts.metadata_path_a == str(tmp_path / "output-a" / "metas" / "v9" / "clip-a.json")
    assert read_paths == [str(tmp_path / "output-a" / "metas" / "v9" / "clip-a.json")]


def test_load_clip_artifacts_marks_missing_and_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing and invalid clip metadata are represented without failing the row."""

    def fake_read_json_object(path: object, *, client_params: object) -> dict[str, object]:
        _ = client_params
        path_text = str(path)
        if "/output-a/" in path_text:
            raise FileNotFoundError(path_text)
        error_msg = "invalid metadata"
        raise ValueError(error_msg)

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    artifacts = load_clip_artifacts(_clip_spec(tmp_path), profile_name="default")

    assert artifacts.metadata_a is None
    assert artifacts.metadata_b is None
    assert artifacts.missing_metadata_a is True
    assert artifacts.missing_metadata_b is False
    assert artifacts.invalid_metadata_a is None
    assert artifacts.invalid_metadata_b is not None
    assert "ValueError: invalid metadata" in artifacts.invalid_metadata_b


def test_load_clip_artifacts_marks_client_param_failure_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Storage client setup failures are isolated to the loaded clip side."""

    def fake_get_storage_client(path: str, *, profile_name: str) -> object:
        _ = path, profile_name
        error_msg = "client setup failed"
        raise RuntimeError(error_msg)

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts.storage_utils.get_storage_client",
        fake_get_storage_client,
    )

    artifacts = load_clip_artifacts(_clip_spec(tmp_path, in_a=True, in_b=False), profile_name="default")

    assert artifacts.metadata_a is None
    assert artifacts.missing_metadata_a is False
    assert artifacts.invalid_metadata_a is not None
    assert "RuntimeError: client setup failed" in artifacts.invalid_metadata_a
    assert artifacts.invalid_metadata_b is None


def test_caption_clip_view_load_worker_reuses_client_params_by_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persistent caption view workers create storage client params once per output root."""
    client_calls: list[str] = []
    client_param_calls: list[str] = []
    read_params: list[dict[str, Any]] = []

    def fake_get_storage_client(path: str, *, profile_name: str) -> str:
        assert profile_name == "profile-a"
        client_calls.append(path)
        return f"client:{path}"

    def fake_get_smart_open_client_params(client: str) -> dict[str, Any]:
        client_param_calls.append(client)
        return {"transport_params": {"client": client}}

    def fake_read_json_object(path: object, *, client_params: dict[str, Any]) -> dict[str, object]:
        _ = path
        read_params.append(client_params)
        return {"span_uuid": Path(str(path)).stem}

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts.storage_utils.get_storage_client",
        fake_get_storage_client,
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts.storage_utils.get_smart_open_client_params",
        fake_get_smart_open_client_params,
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    worker = CaptionClipViewLoadWorker(profile_name="profile-a")
    first = ClipCaptionView.from_json_dict(worker(_clip_spec(tmp_path, clip_id="clip-a").to_json_dict()))
    second = ClipCaptionView.from_json_dict(worker(_clip_spec(tmp_path, clip_id="clip-b").to_json_dict()))

    assert first.clip_id == "clip-a"
    assert second.clip_id == "clip-b"
    assert client_calls == [str(tmp_path / "output-a"), str(tmp_path / "output-b")]
    assert client_param_calls == [
        f"client:{tmp_path / 'output-a'}",
        f"client:{tmp_path / 'output-b'}",
    ]
    assert len(read_params) == 4


def test_clip_metadata_load_cache_is_scoped_by_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit loader caches cannot silently reuse client params across profiles."""
    client_calls: list[tuple[str, str]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}

    def fake_get_storage_client(path: str, *, profile_name: str) -> str:
        client_calls.append((profile_name, path))
        return f"{profile_name}:{path}"

    def fake_get_smart_open_client_params(client: str) -> dict[str, Any]:
        return {"transport_params": {"client": client}}

    def fake_read_json_object(path: object, *, client_params: dict[str, Any]) -> dict[str, object]:
        _ = path, client_params
        return {"span_uuid": "clip-a"}

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts.storage_utils.get_storage_client",
        fake_get_storage_client,
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts.storage_utils.get_smart_open_client_params",
        fake_get_smart_open_client_params,
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    load_clip_artifacts(_clip_spec(tmp_path, in_b=False), profile_name="profile-a", client_params_by_output_root=cache)
    load_clip_artifacts(_clip_spec(tmp_path, in_b=False), profile_name="profile-b", client_params_by_output_root=cache)

    assert client_calls == [
        ("profile-a", str(tmp_path / "output-a")),
        ("profile-b", str(tmp_path / "output-a")),
    ]
