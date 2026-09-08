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

"""Parquet-driven span provider for ``robot-action-split``.

Reads LeRobot/Mecka dataset shards and emits one SpanWorkItem per (subtask
span, camera view).  No video is decoded during discovery.

Discovery phases
----------------
1. Shard enumeration  -- list shard_*/ dirs under each input URI.
2. Per-shard prep     -- read meta/info.json, subtasks.parquet, tasks.parquet;
                         discover videos/ view dirs; list data chunk files.
3. Segment build      -- per data file: read data parquet + episode parquets;
                         project episode metadata; numpy diff-based subtask
                         boundary detection.
4. Filter + dedup     -- duration bounds, label skip, per-episode dedup.
"""

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import get_storage_client, path_exists, read_bytes
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig, SpanFilterConfig
from cosmos_curator.next.recipes.robot_action_split.identities import (
    make_clip_id,
    make_source_id,
    make_span_group_id,
)

_CHUNK_FILE_RE = re.compile(r"^chunk-(\d+)/file-(\d+)\.parquet$")
_VIEW_PREFIX = "observation.images."

# Accepted label column names per meta parquet, in precedence order: the
# canonical name, its sibling spelling, then the name pandas gives an unnamed
# index it preserved into a parquet, which is how the LIBERO exports carry their
# instruction text. The preserved index comes last so a real named column always
# wins. A candidate must also hold text: that same index name carries numbers
# just as readily, and a number is an identifier, never a label.
_TASK_LABEL_COLUMNS = ("task", "task_name", "__index_level_0__")
_SUBTASK_LABEL_COLUMNS = ("subtask", "subtask_name", "__index_level_0__")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class SpanWorkItem:
    """One (subtask span, camera view) to cut and register."""

    # Identity
    source_id: str
    span_group_id: str
    clip_id: str
    view_name: str

    # Source location
    chunk_mp4_uri: str
    data_parquet_uri: str

    # Span geometry (integer frames)
    episode_index: int
    episode_frame_base: int
    frame_start: int
    frame_end: int
    native_fps: float
    episode_from_timestamp: float

    # Labels
    subtask_index: int
    subtask_name: str
    task_index: int
    task_name: str
    task_description: str | None
    # False when subtask_name stands in from task_name because the shard offers no
    # subtask label. Such spans share one name without describing one action, so
    # per-subtask-label dedup must not read them as repeats of each other.
    subtask_label_resolved: bool

    # Episode identity
    episode_id: str

    # Camera
    camera_intrinsics: list[float] | None

    @property
    def duration_s(self) -> float:
        """Span duration in seconds."""
        return (self.frame_end - self.frame_start) / self.native_fps

    @property
    def start_ns(self) -> int:
        """Span start as absolute nanoseconds within the chunk MP4."""
        chunk_frame_offset = round(self.episode_from_timestamp * self.native_fps)
        abs_start = chunk_frame_offset + (self.frame_start - self.episode_frame_base)
        return round(abs_start / self.native_fps * 1e9)

    @property
    def end_ns(self) -> int:
        """Span end (exclusive) as absolute nanoseconds within the chunk MP4."""
        chunk_frame_offset = round(self.episode_from_timestamp * self.native_fps)
        abs_end = chunk_frame_offset + (self.frame_end - self.episode_frame_base)
        return round(abs_end / self.native_fps * 1e9)


@attrs.define
class ChunkSpanBatch:
    """All SpanWorkItems sharing the same source chunk MP4 and data parquet."""

    chunk_mp4_uri: str
    data_parquet_uri: str
    items: list[SpanWorkItem]


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _get_client(uri: str, storage_profile: str) -> StorageClient:
    """Return a storage client for *uri*, raising if the path is not remote."""
    client = get_storage_client(uri, profile_name=storage_profile)
    if client is None:
        msg = f"No storage client available for remote URI: {uri}"
        raise RuntimeError(msg)
    return client


def _s3_list_relative(uri: str, storage_profile: str) -> list[str]:
    """Return keys under *uri* relative to that prefix, via recursive S3 listing."""
    client = _get_client(uri, storage_profile)
    s3p = S3Prefix(uri.rstrip("/") + "/")
    root_key = s3p.prefix.rstrip("/") + "/"
    objects = client.list_recursive(s3p)
    result = []
    for obj in objects:
        key = obj["Key"]
        if key.startswith(root_key):
            result.append(key[len(root_key) :])
    return result


# ---------------------------------------------------------------------------
# Shard/filesystem helpers
# ---------------------------------------------------------------------------


def _list_shard_dirs(
    root: str,
    *,
    storage_profile: str = "default",
) -> list[tuple[str, str]]:
    """Return (path, name) pairs for shard directories under *root*.

    Each returned path has a ``meta/`` subdirectory (or S3 prefix).
    """
    if _is_s3(root):
        client = _get_client(root, storage_profile)
        s3p = S3Prefix(root.rstrip("/") + "/")
        root_key = s3p.prefix.rstrip("/") + "/"
        bucket = s3p.bucket
        try:
            all_objects = client.list_recursive(s3p)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to list objects under {root!r} (profile={storage_profile!r}): {exc}")
            return []

        subdir_names: dict[str, set[str]] = {}
        for obj in all_objects:
            key = obj["Key"]
            rel = key.removeprefix(root_key)
            parts = rel.split("/", 1)
            if len(parts) >= 2:  # noqa: PLR2004
                subdir_names.setdefault(parts[0], set()).add(parts[1])

        result = []
        for name in sorted(subdir_names):
            info_key = f"s3://{bucket}/{root_key}{name}/meta/info.json"
            if client.object_exists(S3Prefix(info_key)):
                uri = f"s3://{bucket}/{root_key}{name}"
                result.append((uri, name))

        # Flat layout: root itself is the shard.
        if not result:
            flat_info = f"s3://{bucket}/{root_key}meta/info.json"
            if client.object_exists(S3Prefix(flat_info)):
                name = root.rstrip("/").rsplit("/", 1)[-1]
                result.append((root.rstrip("/"), name))

        return result

    root_path = Path(root)
    dirs = [(str(p), p.name) for p in sorted(root_path.iterdir()) if p.is_dir() and (p / "meta").is_dir()]
    # Flat layout: root itself is the shard.
    if not dirs and (root_path / "meta").is_dir():
        dirs.append((root, root_path.name))
    return dirs


def _holds_text(field_type: pa.DataType) -> bool:
    """Return whether an Arrow type carries text usable as a label.

    Every Arrow string encoding survives a parquet round trip as itself, so all
    of them are accepted. ``binary`` is deliberately not: its values arrive as
    ``bytes``, which stringify to ``"b'text'"`` rather than to the text.
    """
    if pa.types.is_dictionary(field_type):
        return _holds_text(field_type.value_type)
    return bool(
        pa.types.is_string(field_type) or pa.types.is_large_string(field_type) or pa.types.is_string_view(field_type)
    )


def _read_label_map(
    source: str | io.BytesIO,
    key: str,
    labels: tuple[str, ...],
    *,
    uri: str,
) -> dict[int, str]:
    """Return ``index -> label`` from a meta parquet, skipping blank labels.

    *labels* are the accepted label columns, tried in order; a candidate must
    also hold text. *uri* names the file in errors, since *source* may be a
    buffer.

    Raises:
        ValueError: If *key* is absent, no candidate holds text, one index
            carries conflicting labels, or a populated file yields no label.

    """
    # Resolve against the Arrow schema, never through to_pandas(): a preserved
    # pandas index is an ordinary Arrow column here, but there it moves into
    # DataFrame.index where a column scan cannot see it.
    table = pq.read_table(source)
    label = next(
        (name for name in labels if name in table.column_names and _holds_text(table.schema.field(name).type)),
        None,
    )
    # Reported separately: naming both halves when only one is missing sends the
    # operator looking for the wrong column.
    observed = ", ".join(f"{field.name}: {field.type}" for field in table.schema)
    if key not in table.column_names:
        msg = f"{uri}: expected an index column named {key!r}, found [{observed}]"
        raise ValueError(msg)
    if label is None:
        msg = (
            f"{uri}: expected a text label column named one of {list(labels)} "
            f"(the last being how pandas preserves an unnamed index), found [{observed}]"
        )
        raise ValueError(msg)

    label_map: dict[int, str] = {}
    for raw_index, raw_label in zip(table.column(key).to_pylist(), table.column(label).to_pylist(), strict=True):
        if raw_index is None or raw_label is None or not str(raw_label).strip():
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            # The bare coercion error names no file, and this raise crosses a
            # thread pool spanning every shard in the run.
            msg = f"{uri}: {key} value {raw_index!r} is not an integer"
            raise ValueError(msg) from exc
        text = str(raw_label)
        previous = label_map.get(index)
        if previous is not None and previous != text:
            msg = f"{uri}: {key} {index} maps to both {previous!r} and {text!r}"
            raise ValueError(msg)
        label_map[index] = text

    # Rows that all resolve to nothing mean a corrupt table, not an empty one:
    # left alone it would silently drop every span that referenced this file.
    if table.num_rows and not label_map:
        msg = f"{uri}: {table.num_rows} row(s) but none carry a label under {label!r}"
        raise ValueError(msg)
    return label_map


def _read_shard_meta(
    shard_path: str,
    *,
    storage_profile: str = "default",
) -> tuple[dict[int, str] | None, dict[int, str], float]:
    """Read subtask map, task map, and fps from a shard's meta directory.

    The two storage backends differ only in how they locate bytes; labels are
    resolved identically -- see "Label Resolution" in
    ``docs/curator/design/curator-next-robot-action-split.md``.

    Returns:
        ``(subtask_map, task_map, fps)``, where *subtask_map* is ``None`` when the
        shard ships no subtask labels at all, as opposed to an empty mapping.

    Raises:
        ValueError: If a meta parquet carries no usable label column, or if
            ``tasks.parquet`` yields no labels.

    """
    subtask_source: str | io.BytesIO | None
    if _is_s3(shard_path):
        client = _get_client(shard_path, storage_profile)
        meta_uri = shard_path.rstrip("/") + "/meta"

        info = json.loads(read_bytes(f"{meta_uri}/info.json", client=client).decode("utf-8"))
        subtask_uri = f"{meta_uri}/subtasks.parquet"
        subtask_source = (
            io.BytesIO(read_bytes(subtask_uri, client=client)) if client.object_exists(S3Prefix(subtask_uri)) else None
        )
        tasks_uri = f"{meta_uri}/tasks.parquet"
        tasks_source: str | io.BytesIO = io.BytesIO(read_bytes(tasks_uri, client=client))
    else:
        meta_dir = Path(shard_path) / "meta"

        info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
        subtask_file = meta_dir / "subtasks.parquet"
        subtask_uri = str(subtask_file)
        subtask_source = subtask_uri if subtask_file.is_file() else None
        tasks_uri = str(meta_dir / "tasks.parquet")
        tasks_source = tasks_uri

    # A table that resolves to nothing carries no more information than an absent
    # file: either way this shard has no subtask label to offer.
    subtask_map: dict[int, str] | None = None
    if subtask_source is not None:
        subtask_map = _read_label_map(subtask_source, "subtask_index", _SUBTASK_LABEL_COLUMNS, uri=subtask_uri) or None

    task_map = _read_label_map(tasks_source, "task_index", _TASK_LABEL_COLUMNS, uri=tasks_uri)
    if not task_map:
        msg = f"{tasks_uri}: no task labels, so every span in this shard would be dropped"
        raise ValueError(msg)

    return subtask_map, task_map, float(info.get("fps", 30))


def _local_data_file_indices(data_dir: Path) -> set[tuple[int, int]]:
    """Scan a local ``data/`` directory for ``(chunk_index, file_index)`` pairs."""
    pairs: set[tuple[int, int]] = set()
    for chunk_dir in data_dir.iterdir():
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        for fpath in chunk_dir.iterdir():
            if fpath.name.startswith("file-") and fpath.suffix == ".parquet":
                try:
                    ci = int(chunk_dir.name.split("-")[1])
                    fi = int(fpath.stem.split("-")[1])
                    pairs.add((ci, fi))
                except (IndexError, ValueError):
                    pass
    return pairs


def _list_data_files(
    shard_path: str,
    *,
    storage_profile: str = "default",
) -> list[tuple[int, int]]:
    """Return sorted (chunk_index, file_index) pairs under data/."""
    if _is_s3(shard_path):
        data_uri = shard_path.rstrip("/") + "/data/"
        try:
            relative_keys = _s3_list_relative(data_uri, storage_profile)
        except Exception:  # noqa: BLE001
            return []
        pairs: set[tuple[int, int]] = set()
        for rel in relative_keys:
            m = _CHUNK_FILE_RE.match(rel)
            if m:
                pairs.add((int(m.group(1)), int(m.group(2))))
        return sorted(pairs)

    data_dir = Path(shard_path) / "data"
    if not data_dir.is_dir():
        return []
    return sorted(_local_data_file_indices(data_dir))


def _discover_view_names(  # noqa: C901
    shard_path: str,
    requested: tuple[str, ...],
    *,
    storage_profile: str = "default",
) -> tuple[list[str], bool]:
    """Return (selected_views, source_is_multiview)."""
    if _is_s3(shard_path):
        videos_uri = shard_path.rstrip("/") + "/videos/"
        try:
            relative_keys = _s3_list_relative(videos_uri, storage_profile)
        except Exception:  # noqa: BLE001
            relative_keys = []
        views: set[str] = set()
        for rel in relative_keys:
            first = rel.split("/", 1)[0]
            if first.startswith(_VIEW_PREFIX):
                views.add(first)
        available = sorted(views) or ["observation.images.main"]
        source_is_multiview = len(available) > 1
        if requested:
            selected = [v for v in requested if v in set(available)]
            if not selected:
                msg = (
                    f"None of the requested views {list(requested)} are available "
                    f"in {shard_path}; available: {available}"
                )
                raise ValueError(msg)
            return (selected, source_is_multiview)
        return (available, source_is_multiview)

    vid_dir = Path(shard_path) / "videos"
    if not vid_dir.is_dir():
        return (["observation.images.main"], False)
    available = sorted(p.name for p in vid_dir.iterdir() if p.is_dir() and p.name.startswith("observation.images."))
    if not available:
        available = ["observation.images.main"]
    source_is_multiview = len(available) > 1
    if requested:
        selected = [v for v in requested if v in set(available)]
        if not selected:
            msg = f"None of the requested views {list(requested)} are available in {shard_path}; available: {available}"
            raise ValueError(msg)
        return (selected, source_is_multiview)
    return (available, source_is_multiview)


def _read_episode_meta(
    shard_path: str,
    view_names: list[str],
    *,
    storage_profile: str = "default",
) -> dict[int, dict[str, Any]]:
    """Build episode_index to metadata from meta/episodes/ parquets."""
    shard_name = shard_path.rstrip("/").rsplit("/", 1)[-1]
    meta: dict[int, dict[str, Any]] = {}
    if _is_s3(shard_path):
        client = _get_client(shard_path, storage_profile)
        ep_uri = shard_path.rstrip("/") + "/meta/episodes/"
        s3p = S3Prefix(ep_uri)
        root_key = s3p.prefix.rstrip("/") + "/"
        bucket = s3p.bucket
        try:
            relative_keys = _s3_list_relative(ep_uri, storage_profile)
        except Exception:  # noqa: BLE001
            return meta
        for rel in sorted(relative_keys):
            if not rel.endswith(".parquet"):
                continue
            uri = f"s3://{bucket}/{root_key}{rel}"
            data = read_bytes(uri, client=client)
            meta = _parse_episode_parquet(io.BytesIO(data), view_names, meta, shard_name)
        return meta

    ep_root = Path(shard_path) / "meta" / "episodes"
    if not ep_root.is_dir():
        return meta
    for fpath in sorted(ep_root.rglob("*.parquet")):
        meta = _parse_episode_parquet(str(fpath), view_names, meta, shard_name)
    return meta


def _synthetic_episode_id(shard_name: str, episode_index: int) -> str:
    """Return a synthetic episode_id when the column is absent.

    episode_id is a derived field added by the vendor to their parquet schema;
    it is not part of the standard LeRobot format. When absent, falling back
    to str(episode_index) alone would cause cross-shard collisions since
    episode_index is only unique within a shard. This synthetic id uses
    ``"{shard_name}:{episode_index}"`` which is unique within a dataset root:
    shards have distinct names and episode_index is unique within each shard.
    Cross-root uniqueness is already handled by source_id (hash of the dataset
    root URI) being part of span_group_id.
    """
    return f"{shard_name}:{episode_index}"


def _validate_task_description_field(
    schema: pa.Schema,
    source: str | io.BytesIO,
    shard_name: str,
) -> None:
    """Require an optional ``task_description`` field to carry Arrow text."""
    if "task_description" not in schema.names:
        return
    description_type = schema.field("task_description").type
    if _holds_text(description_type):
        return
    source_name = source if isinstance(source, str) else f"{shard_name}/meta/episodes parquet"
    msg = (
        f"{source_name}: task_description must use an Arrow text type "
        f"(string, large_string, string_view, or dictionary thereof), got {description_type}"
    )
    raise ValueError(msg)


def _parse_episode_parquet(
    source: str | io.BytesIO,
    view_names: list[str],
    meta: dict[int, dict[str, Any]],
    shard_name: str = "",
) -> dict[int, dict[str, Any]]:
    """Parse one episode parquet file and merge its rows into *meta*.

    Preserve nullable ``task_description`` text exactly and reject non-text columns.
    """
    schema = pq.read_schema(source)
    schema_names = set(schema.names)
    _validate_task_description_field(schema, source, shard_name)
    if isinstance(source, io.BytesIO):
        source.seek(0)
    cols = [
        c
        for c in ("episode_index", "episode_id", "camera_intrinsics", "dataset_from_index", "task_description")
        if c in schema_names
    ]
    for v in view_names:
        for suffix in ("chunk_index", "file_index", "from_timestamp"):
            col = f"videos/{v}/{suffix}"
            if col in schema_names:
                cols.append(col)
    if "from_timestamp" in schema_names:
        cols.append("from_timestamp")
    if "episode_index" not in cols:
        return meta
    d = pq.read_table(source, columns=cols).to_pydict()
    n = len(d["episode_index"])
    for i in range(n):
        ep = int(d["episode_index"][i])
        intrinsics = d.get("camera_intrinsics", [None] * n)[i]
        views_info: dict[str, dict[str, Any]] = {}
        for v in view_names:
            cc, fc, tc = f"videos/{v}/chunk_index", f"videos/{v}/file_index", f"videos/{v}/from_timestamp"
            views_info[v] = {
                "chunk_index": int(d[cc][i]) if cc in d and d[cc][i] is not None else None,
                "file_index": int(d[fc][i]) if fc in d and d[fc][i] is not None else None,
                "from_timestamp": (
                    float(d[tc][i])
                    if tc in d and d[tc][i] is not None
                    else float(d["from_timestamp"][i])
                    if "from_timestamp" in d
                    else 0.0
                ),
            }
        meta[ep] = {
            "episode_id": str(d["episode_id"][i]) if "episode_id" in d else _synthetic_episode_id(shard_name, ep),
            "camera_intrinsics": [float(x) for x in intrinsics] if intrinsics is not None else None,
            "dataset_from_index": int(d["dataset_from_index"][i]) if "dataset_from_index" in d else 0,
            "task_description": d["task_description"][i] if "task_description" in d else None,
            "views": views_info,
        }
    return meta


def _view_mp4_path(shard_path: str, view: str, meta: dict[str, Any], data_chunk_index: int) -> str:
    vinfo = meta.get("views", {}).get(view, {})
    ci = vinfo.get("chunk_index") if vinfo.get("chunk_index") is not None else data_chunk_index
    fi = vinfo.get("file_index") if vinfo.get("file_index") is not None else 0
    if _is_s3(shard_path):
        return f"{shard_path.rstrip('/')}/videos/{view}/chunk-{ci:03d}/file-{fi:03d}.mp4"
    return str(Path(shard_path) / "videos" / view / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4")


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------


def _locate_data_parquet(
    shard_path: str,
    ci: int,
    fi: int,
    *,
    storage_profile: str = "default",
) -> tuple[str | io.BytesIO, str] | None:
    """Return one data file's parquet source and URI, or ``None`` when it is absent."""
    if _is_s3(shard_path):
        data_uri = f"{shard_path.rstrip('/')}/data/chunk-{ci:03d}/file-{fi:03d}.parquet"
        client = _get_client(shard_path, storage_profile)
        if not path_exists(data_uri, client=client):
            return None
        return io.BytesIO(read_bytes(data_uri, client=client)), data_uri

    data_path = Path(shard_path) / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
    if not data_path.is_file():
        return None
    return str(data_path), str(data_path)


def _read_span_columns(
    parquet_source: str | io.BytesIO,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], bool] | None:
    """Return a data file's span columns, sorted by (episode, frame).

    Returns:
        ``(episode, frame, task_index, subtask_index, has_subtask)``, or ``None``
        when the file holds no rows. Without a ``subtask_index`` column the task
        index stands in, so a constant task bounds one span per episode.

    """
    schema_names = set(pq.read_schema(parquet_source).names)
    if isinstance(parquet_source, io.BytesIO):
        parquet_source.seek(0)
    has_subtask = "subtask_index" in schema_names
    cols = ["episode_index", "frame_index", "task_index", *(["subtask_index"] if has_subtask else [])]
    t = pq.read_table(parquet_source, columns=cols)
    ep_arr = np.asarray(t.column("episode_index").to_numpy())
    fr_arr = np.asarray(t.column("frame_index").to_numpy())
    ti_arr = np.asarray(t.column("task_index").to_numpy())
    si_arr = np.asarray(t.column("subtask_index").to_numpy()) if has_subtask else ti_arr.copy()

    if len(ep_arr) == 0:
        return None

    # Both keys must be ordered, not just the episode: runs are found on adjacent
    # rows and a span's bounds come from its run's first and last row, so frames
    # out of order inside an ordered episode fragment the run and mis-bound what
    # is left. Sorting is skipped only when it would change nothing.
    order = np.lexsort((fr_arr, ep_arr))
    if not np.array_equal(order, np.arange(len(order))):
        ep_arr, fr_arr, ti_arr, si_arr = ep_arr[order], fr_arr[order], ti_arr[order], si_arr[order]

    return ep_arr, fr_arr, ti_arr, si_arr, has_subtask


def _warn_dropped_spans(
    data_parquet_uri: str,
    dropped: int,
    unresolved_tasks: set[int],
    unresolved_subtasks: set[int],
    unknown_episodes: set[int],
) -> None:
    """Report what one data file dropped, one warning per cause.

    The run-level failure in ``discover_spans`` names the causes but not the
    indices, so these warnings are what an operator reads to locate the fault.
    """
    if dropped:
        logger.warning(
            f"{data_parquet_uri}: dropped {dropped} span(s) carrying no label in meta/; "
            f"unresolved task_index={sorted(unresolved_tasks)}, "
            f"subtask_index={sorted(unresolved_subtasks)}"
        )
    if unknown_episodes:
        logger.warning(
            f"{data_parquet_uri}: dropped every span of {len(unknown_episodes)} episode(s) missing from "
            f"meta/episodes/; episode_index={sorted(unknown_episodes)}"
        )


def _build_segments_for_file(  # noqa: PLR0913
    shard_path: str,
    source_id: str,
    ci: int,
    fi: int,
    subtask_map: dict[int, str] | None,
    task_map: dict[int, str],
    fps: float,
    episode_meta: dict[int, dict[str, Any]],
    view_names: list[str],
    video_bitrate: str,
    *,
    source_is_multiview: bool,
    storage_profile: str = "default",
) -> list[SpanWorkItem]:
    """Build SpanWorkItems from one data parquet file.

    A span whose task or subtask index resolves to no label is dropped rather
    than labelled with its index; *subtask_map* being ``None`` instead means the
    shard ships no subtask labels, and the task label stands in for all spans.

    Raises:
        ValueError: If the data file listed under ``data/`` cannot be read.

    """
    located = _locate_data_parquet(shard_path, ci, fi, storage_profile=storage_profile)
    if located is None:
        # (ci, fi) came from listing data/, so the file was there and is not now.
        # Skipping would drop its spans and still report the run successful. The
        # listing accepts any digit width but every path here is built with three,
        # so an unconventionally padded name reaches this branch for every file.
        msg = (
            f"{shard_path}: data/chunk-{ci:03d}/file-{fi:03d}.parquet was listed under data/ but cannot "
            "be read. It was removed mid-run, or its name on disk is padded to another width."
        )
        raise ValueError(msg)
    parquet_source, data_parquet_uri = located

    columns = _read_span_columns(parquet_source)
    if columns is None:
        # Warned rather than returned silently: a run of only empty files fails in
        # discover_spans, which can only point at the warnings each file left.
        logger.warning(f"{data_parquet_uri}: holds no rows, so it yields no span")
        return []
    ep_arr, fr_arr, ti_arr, si_arr, has_subtask = columns

    # None wherever nothing finer than the task label exists: either this file
    # declares no subtask_index, or the shard ships no labels to resolve one
    # against. Collapsing both into one local keeps the loop to one condition.
    subtask_labels = subtask_map if has_subtask else None

    ep_bounds = np.concatenate(([0], np.where(np.diff(ep_arr) != 0)[0] + 1, [len(ep_arr)]))
    items: list[SpanWorkItem] = []
    dropped = 0
    unresolved_tasks: set[int] = set()
    unresolved_subtasks: set[int] = set()
    unknown_episodes: set[int] = set()

    for i in range(len(ep_bounds) - 1):
        lo, hi = int(ep_bounds[i]), int(ep_bounds[i + 1])
        ep_idx = int(ep_arr[lo])
        meta = episode_meta.get(ep_idx)
        if meta is None:
            unknown_episodes.add(ep_idx)
            continue

        frames = fr_arr[lo:hi]
        subtasks = si_arr[lo:hi]
        tasks = ti_arr[lo:hi]
        ep_frame_base = int(frames[0])

        run_bounds = np.concatenate(([0], np.where(np.diff(subtasks) != 0)[0] + 1, [len(subtasks)]))
        for j in range(len(run_bounds) - 1):
            rs, re = int(run_bounds[j]), int(run_bounds[j + 1])
            st_idx = int(subtasks[rs])
            t_idx = int(tasks[rs])
            frame_s, frame_e = int(frames[rs]), int(frames[re - 1]) + 1
            task_name = task_map.get(t_idx)
            if task_name is None:
                unresolved_tasks.add(t_idx)
                dropped += 1
                continue
            # Only an index missing from a populated map is a data fault, and that
            # span is dropped below.
            subtask_name = subtask_labels.get(st_idx) if subtask_labels is not None else task_name
            if subtask_name is None:
                unresolved_subtasks.add(st_idx)
                dropped += 1
                continue
            span_group_id = make_span_group_id(source_id, meta["episode_id"], st_idx, frame_s)
            for view in view_names:
                vinfo = meta.get("views", {}).get(view, {})
                items.append(
                    SpanWorkItem(
                        source_id=source_id,
                        span_group_id=span_group_id,
                        clip_id=make_clip_id(
                            span_group_id, view, video_bitrate, source_is_multiview=source_is_multiview
                        ),
                        view_name=view,
                        chunk_mp4_uri=_view_mp4_path(shard_path, view, meta, ci),
                        data_parquet_uri=data_parquet_uri,
                        episode_index=ep_idx,
                        episode_frame_base=ep_frame_base,
                        frame_start=frame_s,
                        frame_end=frame_e,
                        native_fps=fps,
                        episode_from_timestamp=float(vinfo.get("from_timestamp", 0.0)),
                        subtask_index=st_idx,
                        subtask_name=subtask_name,
                        task_index=t_idx,
                        task_name=task_name,
                        task_description=meta.get("task_description"),
                        subtask_label_resolved=subtask_labels is not None,
                        episode_id=str(meta["episode_id"]),
                        camera_intrinsics=meta.get("camera_intrinsics"),
                    )
                )

    _warn_dropped_spans(data_parquet_uri, dropped, unresolved_tasks, unresolved_subtasks, unknown_episodes)
    return items


# ---------------------------------------------------------------------------
# Filter + dedup
# ---------------------------------------------------------------------------


def _filter_by_label(items: list[SpanWorkItem], cfg: SpanFilterConfig) -> list[SpanWorkItem]:
    skip_set = frozenset(s.lower() for s in cfg.skip_labels)
    prefixes = tuple(p.lower() for p in cfg.skip_label_prefixes)
    substrings = tuple(s.lower() for s in cfg.skip_label_substrings)
    return [
        item
        for item in items
        if (name := item.subtask_name.strip().lower()) not in skip_set
        and not (prefixes and name.startswith(prefixes))
        and not (substrings and any(sub in name for sub in substrings))
    ]


def _filter_by_duration(items: list[SpanWorkItem], cfg: SpanFilterConfig) -> list[SpanWorkItem]:
    return [
        item
        for item in items
        if item.duration_s >= cfg.min_duration_s
        and (cfg.max_duration_s is None or item.duration_s <= cfg.max_duration_s)
    ]


def _dedup(items: list[SpanWorkItem], cfg: SpanFilterConfig) -> list[SpanWorkItem]:
    """Keep at most ``max_keep_per_description`` spans per episode and subtask label.

    Two subtask indices sharing one label are one dedup group and compete for the
    same allowance. A span whose label only stands in for a missing subtask label
    describes nothing, so it competes with no one.
    """

    def ep_key(x: SpanWorkItem) -> tuple[str, str]:
        # episode_id is the stable vendor-assigned identifier; episode_index is
        # shard-local and is not unique across shards.
        return (x.source_id, x.episode_id)

    result: list[SpanWorkItem] = []
    for _, ep_iter in groupby(sorted(items, key=ep_key), key=ep_key):
        ep_items = list(ep_iter)
        by_desc: dict[tuple[str, int | None], list[SpanWorkItem]] = {}
        for item in ep_items:
            # A stand-in label is shared by every span in the episode, so bucketing
            # on it alone would cap unrelated actions against one allowance and
            # discard the rest. Their distinct identity is the subtask index.
            key = (item.subtask_name, None if item.subtask_label_resolved else item.subtask_index)
            by_desc.setdefault(key, []).append(item)

        for group in by_desc.values():
            span_groups: dict[str, list[SpanWorkItem]] = {}
            for item in group:
                span_groups.setdefault(item.span_group_id, []).append(item)
            span_keys = list(dict.fromkeys(item.span_group_id for item in group))

            if len(span_keys) <= cfg.max_keep_per_description:
                result.extend(group)
                continue

            def _max_dur(k: str, _sg: dict[str, list[SpanWorkItem]] = span_groups) -> float:
                return max(i.duration_s for i in _sg[k])

            preferred = sorted(
                [
                    k
                    for k in span_keys
                    if any(cfg.dedup_prefer_min_s <= i.duration_s <= cfg.dedup_prefer_max_s for i in span_groups[k])
                ],
                key=lambda k: (-_max_dur(k), k),
            )
            if len(preferred) < cfg.max_keep_per_description:
                remaining = sorted(
                    [k for k in span_keys if k not in set(preferred)],
                    key=lambda k: (-_max_dur(k), k),
                )
                preferred = [*preferred, *remaining[: cfg.max_keep_per_description - len(preferred)]]

            for k in preferred[: cfg.max_keep_per_description]:
                result.extend(span_groups[k])

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def discover_spans(config: ResolvedRobotActionSplitConfig) -> list[ChunkSpanBatch]:  # noqa: C901
    """Run span discovery for all dataset roots in the config.

    Returns a list of ChunkSpanBatch objects ready for Ray Data processing.
    Each batch groups all SpanWorkItems that share the same source chunk MP4
    and data parquet, so each Ray worker opens one video file.
    """
    all_items: list[SpanWorkItem] = []
    workers = config.execution.discovery_workers
    requested_views = config.output.views
    storage_profile = config.execution.storage_profile
    video_bitrate = config.output.video_bitrate
    chunk_limit = config.input.limit
    # seen_chunks is global across all input URIs so limit applies to the total
    # number of chunks processed, not per-URI.
    seen_chunks: set[str] = set()
    data_files_read = 0

    for dataset_root in config.input.uris:
        source_id = make_source_id(dataset_root)
        logger.info(f"Listing shard dirs under {dataset_root!r} (profile={storage_profile!r})")
        shard_dirs = _list_shard_dirs(dataset_root, storage_profile=storage_profile)
        logger.info(f"Found {len(shard_dirs)} shard dir(s)")
        if not shard_dirs:
            continue

        def _prep(
            args: tuple[str, str],
        ) -> tuple[
            str,
            dict[int, str] | None,
            dict[int, str],
            float,
            list[tuple[int, int]],
            list[str],
            bool,
            dict[int, dict[str, Any]],
        ]:
            shard_path, _ = args
            subtask_map, task_map, fps = _read_shard_meta(shard_path, storage_profile=storage_profile)
            views, source_is_multiview = _discover_view_names(
                shard_path, requested_views, storage_profile=storage_profile
            )
            data_files = _list_data_files(shard_path, storage_profile=storage_profile)
            episode_meta = _read_episode_meta(shard_path, views, storage_profile=storage_profile)
            return shard_path, subtask_map, task_map, fps, data_files, views, source_is_multiview, episode_meta

        with ThreadPoolExecutor(max_workers=min(len(shard_dirs), workers)) as pool:
            futures = [pool.submit(_prep, sd) for sd in shard_dirs]
            shard_infos = [f.result() for f in futures]

        file_tasks = [
            (sp, ci, fi, stm, tm, fps, views, sim, ep_meta)
            for sp, stm, tm, fps, data_files, views, sim, ep_meta in shard_infos
            for ci, fi in data_files
        ]
        if chunk_limit is not None:
            filtered: list[tuple[Any, ...]] = []
            for task in file_tasks:
                key = f"{task[0]}/chunk-{task[1]:03d}"
                if key not in seen_chunks and len(seen_chunks) < chunk_limit:
                    seen_chunks.add(key)
                if key in seen_chunks:
                    filtered.append(task)
            file_tasks = filtered

        def _build(args: tuple[Any, ...], _sid: str = source_id, _vb: str = video_bitrate) -> list[SpanWorkItem]:
            sp, ci, fi, stm, tm, fps, views, sim, ep_meta = args
            return _build_segments_for_file(
                sp,
                _sid,
                ci,
                fi,
                stm,
                tm,
                fps,
                ep_meta,
                views,
                _vb,
                source_is_multiview=sim,
                storage_profile=storage_profile,
            )

        data_files_read += len(file_tasks)
        with ThreadPoolExecutor(max_workers=min(len(file_tasks) or 1, workers)) as pool:
            all_items.extend(
                item for r in as_completed({pool.submit(_build, t): t for t in file_tasks}) for item in r.result()
            )

    # Data files that yield no span at all are corrupt, not empty. Returning []
    # here is indistinguishable from a dataset with nothing to do, and the caller
    # reports that as a successful run -- so total loss would exit 0. Checked
    # before filtering, because filtering everything out IS a legitimate outcome.
    if data_files_read and not all_items:
        msg = (
            f"{data_files_read} data file(s) yielded no span: every span was dropped for an unresolvable "
            "task_index/subtask_index or an episode_index absent from meta/episodes/, or the files hold "
            "no rows at all. The per-file warnings above name the cause."
        )
        raise ValueError(msg)

    cfg = config.split
    all_items = _filter_by_label(all_items, cfg)
    all_items = _filter_by_duration(all_items, cfg)
    all_items = _dedup(all_items, cfg)

    by_chunk: dict[tuple[str, str], list[SpanWorkItem]] = {}
    for item in all_items:
        by_chunk.setdefault((item.chunk_mp4_uri, item.data_parquet_uri), []).append(item)

    max_per_batch = config.execution.max_segments_per_batch
    return [
        ChunkSpanBatch(
            chunk_mp4_uri=mp4_uri, data_parquet_uri=parquet_uri, items=chunk_items[start : start + max_per_batch]
        )
        for (mp4_uri, parquet_uri), chunk_items in by_chunk.items()
        for start in range(0, len(chunk_items), max_per_batch)
    ]
