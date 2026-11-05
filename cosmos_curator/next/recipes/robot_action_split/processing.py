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

"""Ray worker: cut one source chunk's clips, write their payloads, and return the chunk's terminal outcome.

``cut_chunks`` turns each work item into one ``contracts.CHUNK_SCHEMA`` outcome.
The chunk MP4 downloads once and ``cut_plan`` cuts all of its spans. Every clip
and action payload and sidecar is durable before the outcome is returned, so a
checkpointed outcome never references a missing payload; payload paths are
deterministic, so a replayed chunk rewrites the same objects.

Failure policy: a missing or archived source object, unparseable data parquet,
action data that cannot be sliced or encoded, and a ``MediaError`` from the
shared media contract -- a chunk that is invalid or cannot be indexed, or a cut
the source cannot yield exactly -- are stable properties of immutable input and
become terminal errors. Every other failure raises, leaving the chunk retryable:
an inconclusive probe or ffmpeg run (``ProbeError``, ``FFmpegRunError``), and
chunk downloads and payload writes once their own retries are spent.
"""

import io
import json
import pickle
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage.storage_utils import (
    get_storage_client,
    is_remote_path,
    path_to_prefix,
    read_bytes,
)
from cosmos_curator.next.media.action_binary import encode_action_bin, get_action_binary_spec
from cosmos_curator.next.media.failures import MediaError
from cosmos_curator.next.media.smart_cut import cut_plan
from cosmos_curator.next.recipes.robot_action_split.camera_motion import compute_camera_motion_annotation
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from cosmos_curator.next.recipes.robot_action_split.contracts import CHUNK_SCHEMA, error_details
from cosmos_curator.next.recipes.robot_action_split.discovery import SpanWorkItem, decode_chunk
from cosmos_curator.next.recipes.robot_action_split.identities import make_action_id
from cosmos_curator.next.utils.source_objects import is_unreadable_source
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


def clip_fields(item: SpanWorkItem, source_dataset: str) -> dict[str, Any]:
    """Return the fields every clip outcome carries, finished or failed."""
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


def failed_clip(item: SpanWorkItem, source_dataset: str, error_stage: str, error_message: str) -> dict[str, Any]:
    """Describe a clip that will never be produced: its labels and error, without payloads."""
    return {
        **clip_fields(item, source_dataset),
        "clip_uri": None,
        "action_data_uri": None,
        "camera_motion_annotation": None,
        **error_details(error_stage, error_message),
    }


def chunk_outcome(work_item: dict[str, Any], clips: list[dict[str, Any]]) -> dict[str, Any]:
    """Finish a chunk with an outcome for every clip it owes, each finished or failed."""
    return {
        "chunk_id": work_item["chunk_id"],
        "chunk_mp4_uri": work_item["chunk_mp4_uri"],
        "data_parquet_uri": work_item["data_parquet_uri"],
        "clips": clips,
        "error_stage": None,
        "error_message": None,
    }


def failed_chunk_outcome(
    work_item: dict[str, Any], items: list[SpanWorkItem], source_dataset: str, error_stage: str, error_message: str
) -> dict[str, Any]:
    """Fail the chunk as a whole, and every clip it owes with the same error."""
    clips = [failed_clip(item, source_dataset, error_stage, error_message) for item in items]
    return chunk_outcome(work_item, clips) | error_details(error_stage, error_message)


def _action_bytes(
    action_arrays: dict[str, np.ndarray[Any, Any]], item: SpanWorkItem, *, action_format: str, source_dataset: str
) -> tuple[bytes, str | None]:
    """Slice and encode one span's action data, and describe its camera motion."""
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
    return action_bytes, camera_motion_annotation


# ---------------------------------------------------------------------------
# Main worker functions
# ---------------------------------------------------------------------------


def _download_chunk(chunk_mp4_uri: str, destination: str, *, storage_profile: str, attempts: int) -> None:
    """Stage a remote chunk MP4 locally, retrying failed transfers; a final failure propagates for classification."""
    client = get_storage_client(chunk_mp4_uri, profile_name=storage_profile)
    if client is None:
        msg = f"No storage client available for {chunk_mp4_uri}"
        raise RuntimeError(msg)
    do_with_retries(
        partial(client.download_to_path, path_to_prefix(chunk_mp4_uri), destination),
        max_attempts=attempts,
        backoff_factor=4.0,
        max_wait_time_s=64.0,
        name="robot-action-split-chunk-download",
    )


def _cut_failure(item: SpanWorkItem, source_dataset: str, cut_rec: dict[str, Any]) -> dict[str, Any]:
    """Record a cut the source cannot yield as a failed clip; raise any other failure so the chunk is retried."""
    failure = cut_rec["failure"]
    if isinstance(failure, MediaError):
        return failed_clip(item, source_dataset, "cut", str(failure))
    raise failure


def cut_chunk(work_item: dict[str, Any], *, config: ResolvedRobotActionSplitConfig) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Cut every span of one chunk and return its terminal outcome in ``CHUNK_SCHEMA``.

    Downloads the chunk MP4 to a local temp file once, builds the PTS index
    once via ``cut_plan``, then cuts all spans with GOP-aware stream copy +
    head re-encode, ``cut_cpus`` at a time. As each cut finishes, its clip,
    action data, and JSON sidecar are written to their final local or S3
    locations and the local clip is removed, so local disk holds the chunk and
    only the clips in flight. Every payload is durable before this returns.
    """
    chunk = decode_chunk(work_item)
    items = chunk.items
    media_root = config.output.media_root.rstrip("/")
    action_format = config.output.action_format
    action_ext = ".bin" if action_format == "bin" else ".pickle"
    input_profile = config.input.storage_profile
    output_profile = config.execution.storage_profile
    source_dataset = config.input.source_dataset
    tmp_dir = config.execution.tmp_dir or None
    write = partial(_write_with_retries, storage_profile=output_profile, attempts=config.execution.media_write_attempts)
    failed_chunk = partial(failed_chunk_outcome, work_item, items, source_dataset)

    # Load the data parquet into memory (small) and stream the chunk MP4 to disk
    # (potentially very large — streaming avoids a full-file in-memory copy).
    try:
        parquet_bytes = read_bytes(
            chunk.data_parquet_uri, client=get_storage_client(chunk.data_parquet_uri, profile_name=input_profile)
        )
    except Exception as exc:
        if is_unreadable_source(exc, chunk.data_parquet_uri, storage_profile=input_profile):
            return failed_chunk("load", f"Data parquet unavailable: {exc}")
        raise

    try:
        action_arrays = _load_action_arrays(parquet_bytes, {item.episode_index for item in items})
    except pa.ArrowMemoryError:
        raise  # Worker memory pressure, not a malformed parquet; a retry can succeed.
    except (pa.ArrowException, ValueError) as exc:
        return failed_chunk("parquet-parse", str(exc))

    def finish(cut_rec: dict[str, Any]) -> None:
        """Turn one finished cut into its clip outcome, writing its payloads; runs on cut_plan's worker threads."""
        local_clip = Path(cut_rec["output"])
        item = by_output[cut_rec["output"]]
        try:
            if failures:
                return  # The chunk will raise; spend no more writes on it.
            if not cut_rec.get("success"):
                clips[item.clip_id] = _cut_failure(item, source_dataset, cut_rec)
                return
            try:
                action_bytes, camera_motion_annotation = _action_bytes(
                    action_arrays, item, action_format=action_format, source_dataset=source_dataset
                )
            except (ValueError, KeyError, IndexError) as exc:
                clips[item.clip_id] = failed_clip(item, source_dataset, "action", str(exc))
                return
            clip_uri = f"{media_root}/video/{item.view_name}/{item.clip_id}.mp4"
            action_id = make_action_id(item.span_group_id, action_format, source_dataset)
            action_uri = f"{media_root}/action/{action_id}{action_ext}"
            sidecar_uri = f"{media_root}/temp_metas/{item.clip_id}.json"
            write(clip_uri, local_clip.read_bytes())
            # Views of one span share its action payload; replays and other views rewrite identical bytes.
            write(action_uri, action_bytes)
            write(sidecar_uri, _build_sidecar_bytes(item, artifact_uri(clip_uri), artifact_uri(action_uri)))
            clips[item.clip_id] = {
                **clip_fields(item, source_dataset),
                "clip_uri": artifact_uri(clip_uri),
                "action_data_uri": artifact_uri(action_uri),
                "camera_motion_annotation": camera_motion_annotation,
                "error_stage": None,
                "error_message": None,
            }
        except Exception as exc:  # noqa: BLE001 -- re-raised from the chunk task after cut_plan returns
            failures.append(exc)
        finally:
            local_clip.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        if not is_remote_path(chunk.chunk_mp4_uri):
            # Local path (e.g. in tests) — read directly without staging a copy.
            chunk_local = chunk.chunk_mp4_uri
            if not Path(chunk_local).is_file():
                return failed_chunk("chunk-stage", f"Chunk MP4 unavailable: {chunk_local}")
        else:
            # Remote: stage to a local file so cut_plan can seek it.
            chunk_local = str(Path(tmp) / "chunk.mp4")
            try:
                _download_chunk(
                    chunk.chunk_mp4_uri,
                    chunk_local,
                    storage_profile=input_profile,
                    attempts=config.execution.chunk_download_attempts,
                )
            except Exception as exc:
                if is_unreadable_source(exc, chunk.chunk_mp4_uri, storage_profile=input_profile):
                    return failed_chunk("chunk-stage", f"Chunk MP4 unavailable: {exc}")
                raise

        # cut_plan probes the PTS index once from chunk_local, then runs one ffmpeg per cut
        # and hands each record to `finish` as soon as that cut completes.
        by_output: dict[str, SpanWorkItem] = {}
        cut_specs: list[dict[str, Any]] = []
        clips: dict[str, dict[str, Any]] = {}
        failures: list[Exception] = []
        for item in items:
            abs_start, abs_end_incl = _abs_frame_range(item)
            if abs_start < 0 or abs_end_incl < abs_start:
                # The source data places this span outside the chunk; cut_plan rejects such a plan outright.
                msg = f"span frames {abs_start}-{abs_end_incl} are not a range in the chunk"
                clips[item.clip_id] = failed_clip(item, source_dataset, "cut", msg)
                continue
            local_clip = str(Path(tmp) / f"{item.clip_id}.mp4")
            by_output[local_clip] = item
            cut_specs.append({"startFrame": abs_start, "endFrame": abs_end_incl, "output": local_clip})

        try:
            cut_plan(
                chunk_local,
                cut_specs,
                bitrate=config.output.video_bitrate,
                workers=max(1, int(config.execution.cut_cpus)),
                smart_cut=True,
                tmp_dir=tmp_dir,
                on_cut=finish,
            )
        except MediaError as exc:
            # The chunk is invalid, or its packet index is not a frame index; a ProbeError propagates.
            return failed_chunk("cut-index", str(exc))
        if failures:
            raise failures[0]

    return chunk_outcome(work_item, [clips[item.clip_id] for item in items])


def _write_with_retries(location: str, data: bytes, *, storage_profile: str, attempts: int) -> None:
    """Write one deterministic payload, retrying transient failures and then raising."""
    do_with_retries(
        partial(write_media, location, data, storage_profile=storage_profile),
        max_attempts=attempts,
        name="robot-action-split-payload-write",
    )


def cut_chunks(batch: pa.Table, *, config: ResolvedRobotActionSplitConfig) -> pa.Table:
    """Cut each chunk in a block of work items; only terminal outcomes reach checkpoints.

    A chunk that raises blocks its whole selection until it succeeds, so the
    error names the chunk for the operator who has to find it.
    """
    outcomes = []
    for item in batch.to_pylist():
        try:
            outcomes.append(cut_chunk(item, config=config))
        except Exception as exc:
            exc.add_note(f"robot-action-split chunk_id={item['chunk_id']} chunk_mp4_uri={item['chunk_mp4_uri']}")
            raise
    return pa.Table.from_pylist(outcomes, schema=CHUNK_SCHEMA)
