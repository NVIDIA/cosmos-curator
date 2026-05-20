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
"""Worker-side artifact loading for video output comparison."""

import json
import time
from collections.abc import Mapping, MutableMapping
from typing import Any, cast

import attrs
import smart_open  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject
from cosmos_curator.pipelines.video.output_comparison.summary_loader import OutputRoot
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec

_SLOW_METADATA_READ_SECONDS = 5.0


@attrs.define(frozen=True)
class LoadedClipArtifacts:
    """Loaded metadata artifacts for one clip comparison row.

    Attributes:
        spec: Clip-level comparison spec that identifies the video, clip, output
            roots, and side presence.
        metadata_a: Parsed output A metadata JSON when the clip exists and the
            metadata was loaded successfully.
        metadata_b: Parsed output B metadata JSON when the clip exists and the
            metadata was loaded successfully.
        metadata_path_a: Expected output A metadata path when the clip exists
            on output A.
        metadata_path_b: Expected output B metadata path when the clip exists
            on output B.
        missing_metadata_a: Whether output A metadata was expected but absent.
        missing_metadata_b: Whether output B metadata was expected but absent.
        invalid_metadata_a: Output A metadata load or parse error, if any.
        invalid_metadata_b: Output B metadata load or parse error, if any.

    """

    spec: ClipComparisonSpec
    metadata_a: JsonDictObject | None
    metadata_b: JsonDictObject | None
    metadata_path_a: str | None
    metadata_path_b: str | None
    missing_metadata_a: bool
    missing_metadata_b: bool
    invalid_metadata_a: str | None
    invalid_metadata_b: str | None


@attrs.define(frozen=True)
class _LoadedClipMetadataSide:
    metadata: JsonDictObject | None
    metadata_path: str | None
    missing_metadata: bool
    invalid_metadata: str | None


@attrs.define(frozen=True)
class _ClipMetadataSideRequest:
    output_label: str
    output_root: OutputRoot
    video_key: str
    clip_id: str
    metadata_version: str


def load_clip_artifacts(
    spec: ClipComparisonSpec,
    *,
    profile_name: str,
    metadata_version: str = "v0",
    client_params_by_output_root: MutableMapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> LoadedClipArtifacts:
    """Load metadata artifacts for one clip comparison row."""
    started_at = time.perf_counter()
    cache = client_params_by_output_root if client_params_by_output_root is not None else {}
    logger.info(
        "Loading clip artifacts: video_key={}, clip_id={}, in_a={}, in_b={}",
        spec.video_key,
        spec.clip_id,
        spec.in_a,
        spec.in_b,
    )
    output_a = (
        _load_clip_metadata_side(
            _ClipMetadataSideRequest(
                output_label="a",
                output_root=spec.output_a,
                video_key=spec.video_key,
                clip_id=spec.clip_id,
                metadata_version=metadata_version,
            ),
            profile_name=profile_name,
            client_params_by_output_root=cache,
        )
        if spec.in_a
        else _missing_side()
    )
    output_b = (
        _load_clip_metadata_side(
            _ClipMetadataSideRequest(
                output_label="b",
                output_root=spec.output_b,
                video_key=spec.video_key,
                clip_id=spec.clip_id,
                metadata_version=metadata_version,
            ),
            profile_name=profile_name,
            client_params_by_output_root=cache,
        )
        if spec.in_b
        else _missing_side()
    )
    artifacts = LoadedClipArtifacts(
        spec=spec,
        metadata_a=output_a.metadata,
        metadata_b=output_b.metadata,
        metadata_path_a=output_a.metadata_path,
        metadata_path_b=output_b.metadata_path,
        missing_metadata_a=output_a.missing_metadata,
        missing_metadata_b=output_b.missing_metadata,
        invalid_metadata_a=output_a.invalid_metadata,
        invalid_metadata_b=output_b.invalid_metadata,
    )
    logger.info(
        "Loaded clip artifacts: video_key={}, clip_id={}, loaded_a={}, loaded_b={}, missing_a={}, missing_b={}, "
        "invalid_a={}, invalid_b={}, elapsed_sec={:.2f}",
        spec.video_key,
        spec.clip_id,
        artifacts.metadata_a is not None,
        artifacts.metadata_b is not None,
        artifacts.missing_metadata_a,
        artifacts.missing_metadata_b,
        artifacts.invalid_metadata_a is not None,
        artifacts.invalid_metadata_b is not None,
        time.perf_counter() - started_at,
    )
    return artifacts


def _load_clip_metadata_side(
    request: _ClipMetadataSideRequest,
    *,
    profile_name: str,
    client_params_by_output_root: MutableMapping[tuple[str, str], Mapping[str, Any]],
) -> _LoadedClipMetadataSide:
    metadata_path = storage_utils.get_full_path(
        request.output_root,
        "metas",
        request.metadata_version,
        f"{request.clip_id}.json",
    )
    metadata_path_str = str(metadata_path)
    try:
        client_params = _client_params_for_output_root(
            request.output_root,
            profile_name=profile_name,
            client_params_by_output_root=client_params_by_output_root,
        )
        read_started_at = time.perf_counter()
        metadata = _read_json_object(metadata_path, client_params=client_params)
        read_elapsed = time.perf_counter() - read_started_at
        if read_elapsed >= _SLOW_METADATA_READ_SECONDS:
            logger.warning(
                "Slow clip metadata read: output={}, video_key={}, clip_id={}, path={}, elapsed_sec={:.2f}",
                request.output_label,
                request.video_key,
                request.clip_id,
                metadata_path_str,
                read_elapsed,
            )
        return _LoadedClipMetadataSide(
            metadata=metadata,
            metadata_path=metadata_path_str,
            missing_metadata=False,
            invalid_metadata=None,
        )
    except FileNotFoundError:
        return _LoadedClipMetadataSide(
            metadata=None,
            metadata_path=metadata_path_str,
            missing_metadata=True,
            invalid_metadata=None,
        )
    except Exception as exc:  # noqa: BLE001
        return _LoadedClipMetadataSide(
            metadata=None,
            metadata_path=metadata_path_str,
            missing_metadata=False,
            invalid_metadata=f"{metadata_path_str}: {exc.__class__.__name__}: {exc}",
        )


def _client_params_for_output_root(
    output_root: OutputRoot,
    *,
    profile_name: str,
    client_params_by_output_root: MutableMapping[tuple[str, str], Mapping[str, Any]],
) -> Mapping[str, Any]:
    output_root_str = str(output_root)
    cache_key = (profile_name, output_root_str)
    if cache_key not in client_params_by_output_root:
        client = storage_utils.get_storage_client(output_root_str, profile_name=profile_name)
        client_params_by_output_root[cache_key] = (
            storage_utils.get_smart_open_client_params(client) if client is not None else {}
        )
    return client_params_by_output_root[cache_key]


def _missing_side() -> _LoadedClipMetadataSide:
    return _LoadedClipMetadataSide(
        metadata=None,
        metadata_path=None,
        missing_metadata=False,
        invalid_metadata=None,
    )


def _read_json_object(path: OutputRoot, *, client_params: Mapping[str, Any]) -> JsonDictObject:
    with smart_open.open(str(path), "r", encoding="utf-8", **client_params) as fp:
        data = json.load(fp)
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        error_msg = "clip artifact metadata must contain a JSON object with string keys"
        raise ValueError(error_msg)
    return cast("JsonDictObject", data)
