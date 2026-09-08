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

"""Ray worker: smart cut + action bin serialization for one ChunkSpanBatch."""

import io
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_curator.core.utils.storage.storage_utils import (
    get_storage_client,
    is_remote_path,
    path_to_prefix,
    read_bytes,
)
from cosmos_curator.next.media.action_binary import encode_action_bin, get_action_binary_spec
from cosmos_curator.next.media.smart_cut import cut_plan
from cosmos_curator.next.recipes.robot_action_split.camera_motion import compute_camera_motion_annotation
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from cosmos_curator.next.recipes.robot_action_split.discovery import ChunkSpanBatch, SpanWorkItem
from cosmos_curator.next.recipes.robot_action_split.identities import make_action_id
from cosmos_curator.next.utils.storage import artifact_uri, write_media

# ---------------------------------------------------------------------------
# Frame-index helpers
# ---------------------------------------------------------------------------


def _abs_frame_range(item: SpanWorkItem) -> tuple[int, int]:
    """Return (abs_start, abs_end_inclusive) in chunk-absolute frame indices.

    ``frame_start``/``frame_end`` in SpanWorkItem are episode-local and
    episode-exclusive respectively.  ``episode_from_timestamp`` is the
    episode's start time within the chunk MP4.  Converting to an integer
    frame offset first (rather than accumulating floats) avoids drift.

    ``cut_plan`` uses **inclusive** end-frame indices.
    """
    offset = round(item.episode_from_timestamp * item.native_fps)
    abs_start = offset + (item.frame_start - item.episode_frame_base)
    abs_end_excl = offset + (item.frame_end - item.episode_frame_base)
    return abs_start, abs_end_excl - 1  # convert to inclusive


# ---------------------------------------------------------------------------
# Action data
# ---------------------------------------------------------------------------

_ACTION_COLS = [
    "episode_index",
    "frame_index",
    "observation.state.hand_left_cam",
    "observation.state.hand_right_cam",
    "observation.state.hand_left_cam_rotation",
    "observation.state.hand_right_cam_rotation",
    "observation.state.camera_position",
    "observation.state.camera_rotation",
    "action",
    "observation.state",
    "actions.joint_position",
    "observation.states.joint_position",
    "observation.states.end_effector",
]

_COL_MAP = {
    "observation.state.hand_left_cam": "hand_left_cam",
    "observation.state.hand_right_cam": "hand_right_cam",
    "observation.state.hand_left_cam_rotation": "hand_left_cam_rotation",
    "observation.state.hand_right_cam_rotation": "hand_right_cam_rotation",
    "observation.state.camera_position": "camera_position",
    "observation.state.camera_rotation": "camera_rotation",
    "action": "action",
    "observation.state": "state",
    "actions.joint_position": "action",
    "observation.states.joint_position": "state_joint_position",
    "observation.states.end_effector": "state_end_effector",
}


def _load_action_arrays(parquet_bytes: bytes, episode_indices: set[int]) -> dict[str, np.ndarray[Any, Any]]:
    schema_names = set(pq.read_schema(io.BytesIO(parquet_bytes)).names)
    cols = [c for c in _ACTION_COLS if c in schema_names]
    ep_filter = pc.field("episode_index").isin(sorted(episode_indices))
    table = pq.read_table(io.BytesIO(parquet_bytes), columns=cols, filters=ep_filter)
    return {col: np.asarray(table.column(col).to_pandas().values) for col in cols}


def _slice_action(
    arrays: dict[str, np.ndarray[Any, Any]],
    episode_index: int,
    frame_start: int,
    frame_end: int,
) -> dict[str, list[Any]]:
    ep_mask = arrays.get("episode_index")
    if ep_mask is None:
        return {}
    ep_rows = np.where(ep_mask == episode_index)[0]
    if len(ep_rows) == 0:
        return {}
    fi = arrays.get("frame_index")
    if fi is not None:
        fi_vals = fi[ep_rows]
        seg_rows = ep_rows[(fi_vals >= frame_start) & (fi_vals < frame_end)]
    else:
        seg_rows = ep_rows[frame_start:frame_end]

    result: dict[str, list[Any]] = {}
    for parquet_col, json_key in _COL_MAP.items():
        arr = arrays.get(parquet_col)
        if arr is None:
            continue
        if json_key in result:
            # Multiple parquet columns alias to the same output key (e.g. "action"
            # and "actions.joint_position" both map to "action" for different
            # dataset families). First match wins; skip subsequent aliases.
            continue
        sliced = arr[seg_rows]
        result[json_key] = [
            r.tolist() if hasattr(r, "tolist") else list(r) if hasattr(r, "__iter__") else r for r in sliced
        ]
    return result


def _serialize_action_bin(action_data: dict[str, list[Any]], source_dataset: str) -> bytes:
    """Serialize action data to ACT2 binary format."""
    return encode_action_bin(action_data, source_dataset)


def _serialize_action_pickle(action_data: dict[str, list[Any]]) -> bytes:
    """Serialize action data as pickle bytes."""
    return pickle.dumps(action_data, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# JSON sidecar
# ---------------------------------------------------------------------------


def _build_sidecar_bytes(item: SpanWorkItem, clip_uri: str, action_uri: str) -> bytes:
    meta = {
        "clip_uuid": item.clip_id,
        "span_group_uuid": item.span_group_id,
        "view_name": item.view_name,
        "clip_url": clip_uri,
        "action_data_url": action_uri,
        "episode_id": item.episode_id,
        "episode_index": item.episode_index,
        "subtask_index": item.subtask_index,
        "subtask_name": item.subtask_name,
        "task_index": item.task_index,
        "task_name": item.task_name,
        "task_description": item.task_description,
        "frame_start": item.frame_start,
        "frame_end": item.frame_end,
        "native_fps": item.native_fps,
        "start_ns": item.start_ns,
        "end_ns": item.end_ns,
        "episode_from_timestamp": item.episode_from_timestamp,
        "camera_intrinsics": item.camera_intrinsics,
        "duration_s": item.duration_s,
    }
    return json.dumps(meta, indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------


def _make_base_fields(item: SpanWorkItem, source_dataset: str) -> dict[str, Any]:
    """Return the fields common to both success and failure outcome rows."""
    return {
        "clip_id": item.clip_id,
        "span_group_id": item.span_group_id,
        "view_name": item.view_name,
        "source_id": item.source_id,
        "source_dataset": source_dataset,
        "episode_id": item.episode_id,
        "episode_index": item.episode_index,
        "subtask_index": item.subtask_index,
        "subtask_name": item.subtask_name,
        "task_index": item.task_index,
        "task_name": item.task_name,
        "task_description": item.task_description,
        "frame_start": item.frame_start,
        "frame_end": item.frame_end,
        "start_ns": item.start_ns,
        "end_ns": item.end_ns,
        "native_fps": item.native_fps,
        "episode_from_timestamp": item.episode_from_timestamp,
    }


# ---------------------------------------------------------------------------
# Main worker function
# ---------------------------------------------------------------------------


def process_batch(  # noqa: C901, PLR0912, PLR0915
    batch: ChunkSpanBatch,
    *,
    config: ResolvedRobotActionSplitConfig,
    staged_chunk_path: str | None = None,
) -> list[dict[str, Any]]:
    """Process all spans in one ChunkSpanBatch using smart cut.

    Downloads the chunk MP4 to a local temp file once, builds the PTS index
    once via ``cut_plan``, then cuts all spans with GOP-aware stream copy +
    head re-encode.  Action data and JSON sidecars are written to their final
    local or S3 locations via ``write_media``.

    ``staged_chunk_path`` may be supplied by the caller when the chunk has
    already been downloaded (e.g. by the sequential pipeline loop that groups
    consecutive batches sharing the same source chunk).  When provided the
    download step is skipped entirely.

    Returns a list of outcome dicts (one per item) with all fields required by
    ``records.CLIP_SCHEMA`` (on success) or ``records.ERROR_SCHEMA`` (on failure).
    """
    media_root = config.output.media_root
    action_format = config.output.action_format
    storage_profile = config.execution.storage_profile
    source_dataset = config.input.source_dataset
    bitrate = config.output.video_bitrate
    tmp_dir = config.execution.tmp_dir or None

    def _batch_failure(stage: str, exc: Exception) -> list[dict[str, Any]]:
        return [
            {
                **_make_base_fields(item, source_dataset),
                "clip_uri": None,
                "action_data_uri": None,
                "camera_motion_annotation": None,
                "status": "failed",
                "error_stage": stage,
                "error_message": str(exc),
            }
            for item in batch.items
        ]

    # Load the data parquet into memory (small) and stream the chunk MP4 to disk
    # (potentially very large — streaming avoids a full-file in-memory copy).
    try:
        parquet_client = get_storage_client(batch.data_parquet_uri, profile_name=storage_profile)
        parquet_bytes = read_bytes(batch.data_parquet_uri, client=parquet_client)
    except Exception as exc:  # noqa: BLE001
        return _batch_failure("load", exc)

    try:
        episode_indices = {item.episode_index for item in batch.items}
        action_arrays = _load_action_arrays(parquet_bytes, episode_indices)
    except Exception as exc:  # noqa: BLE001
        return _batch_failure("parquet-parse", exc)

    outcomes: list[dict[str, Any]] = []
    action_ext = ".bin" if action_format == "bin" else ".pickle"

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        if not is_remote_path(batch.chunk_mp4_uri):
            # Local path (e.g. in tests) — read directly without staging a copy.
            chunk_local = batch.chunk_mp4_uri
        else:
            # Remote: stage to a local file so cut_plan can seek it.
            # If the caller supplies a staged_chunk_path it owns the file's lifetime
            # and we reuse it across batches that share the same source chunk.
            # When None we download into the batch-scoped temp dir and discard after.
            chunk_local = staged_chunk_path or str(Path(tmp) / "chunk.mp4")
            if not Path(chunk_local).exists():
                try:
                    chunk_client = get_storage_client(batch.chunk_mp4_uri, profile_name=storage_profile)
                    if chunk_client is None:
                        msg = f"No storage client available for {batch.chunk_mp4_uri}"
                        raise ValueError(msg)  # noqa: TRY301
                    chunk_client.download_to_path(path_to_prefix(batch.chunk_mp4_uri), chunk_local)
                except Exception as exc:  # noqa: BLE001
                    return _batch_failure("chunk-stage", exc)

        # Build the cut plan for all items in this batch.  cut_plan probes the
        # PTS index once from chunk_local, then runs one ffmpeg per cut.
        clip_local_paths: dict[str, str] = {}
        cut_specs: list[dict[str, Any]] = []
        for item in batch.items:
            abs_start, abs_end_incl = _abs_frame_range(item)
            local_clip = str(Path(tmp) / f"{item.clip_id}.mp4")
            clip_local_paths[item.clip_id] = local_clip
            cut_specs.append({"startFrame": abs_start, "endFrame": abs_end_incl, "output": local_clip})

        try:
            cut_results = cut_plan(chunk_local, cut_specs, bitrate=bitrate, smart_cut=True, tmp_dir=tmp_dir)
        except Exception as exc:  # noqa: BLE001
            return _batch_failure("cut-index", exc)

        cut_by_output = {r["output"]: r for r in cut_results}

        for item in batch.items:
            base = _make_base_fields(item, source_dataset)
            clip_uri = f"{media_root.rstrip('/')}/video/{item.view_name}/{item.clip_id}.mp4"
            action_id = make_action_id(item.span_group_id, action_format, source_dataset)
            action_uri = f"{media_root.rstrip('/')}/action/{action_id}{action_ext}"
            sidecar_uri = f"{media_root.rstrip('/')}/temp_metas/{item.clip_id}.json"
            local_clip = clip_local_paths[item.clip_id]
            cut_rec = cut_by_output.get(local_clip, {})

            if not cut_rec.get("success"):
                err = (
                    cut_rec.get("error")
                    or f"cut produced {cut_rec.get('written_frames')}/{cut_rec.get('expected_frames')} frames"
                )
                outcomes.append(
                    {
                        **base,
                        "clip_uri": None,
                        "action_data_uri": None,
                        "camera_motion_annotation": None,
                        "status": "failed",
                        "error_stage": "cut",
                        "error_message": err,
                    }
                )
                continue

            try:
                write_media(clip_uri, Path(local_clip).read_bytes(), storage_profile=storage_profile)
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    {
                        **base,
                        "clip_uri": None,
                        "action_data_uri": None,
                        "camera_motion_annotation": None,
                        "status": "failed",
                        "error_stage": "clip-write",
                        "error_message": str(exc),
                    }
                )
                continue

            try:
                action_data = _slice_action(action_arrays, item.episode_index, item.frame_start, item.frame_end)
                camera_motion_annotation = compute_camera_motion_annotation(
                    action_data.get("camera_position"),
                    action_data.get("camera_rotation"),
                    item.native_fps,
                    clip_id=item.clip_id,
                )
                if item.camera_intrinsics is not None:
                    # Only inject intrinsics when the dataset's ACT2 spec declares
                    # it as a per-clip field (e.g. mecka). Injecting it for other
                    # datasets (libero, robomind, …) causes encode_action_bin to
                    # raise because the field set would not match the registered spec.
                    if action_format == "bin":
                        try:
                            spec = get_action_binary_spec(source_dataset)
                            if any(f.name == "intrinsics" for f in spec.per_clip_fields):
                                action_data["intrinsics"] = item.camera_intrinsics
                        except ValueError:
                            pass  # unregistered dataset; skip intrinsics
                    else:
                        action_data["intrinsics"] = item.camera_intrinsics
                action_bytes = (
                    _serialize_action_bin(action_data, source_dataset)
                    if action_format == "bin"
                    else _serialize_action_pickle(action_data)
                )
                write_media(action_uri, action_bytes, storage_profile=storage_profile)
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    {
                        **base,
                        "clip_uri": None,
                        "action_data_uri": None,
                        "camera_motion_annotation": None,
                        "status": "failed",
                        "error_stage": "action",
                        "error_message": str(exc),
                    }
                )
                continue

            try:
                sidecar_data = _build_sidecar_bytes(item, artifact_uri(clip_uri), artifact_uri(action_uri))
                write_media(sidecar_uri, sidecar_data, storage_profile=storage_profile)
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    {
                        **base,
                        "clip_uri": None,
                        "action_data_uri": None,
                        "camera_motion_annotation": None,
                        "status": "failed",
                        "error_stage": "sidecar",
                        "error_message": str(exc),
                    }
                )
                continue

            outcomes.append(
                {
                    **base,
                    "clip_uri": artifact_uri(clip_uri),
                    "action_data_uri": artifact_uri(action_uri),
                    "camera_motion_annotation": camera_motion_annotation,
                    "status": "success",
                    "error_stage": None,
                    "error_message": None,
                }
            )

    return outcomes
