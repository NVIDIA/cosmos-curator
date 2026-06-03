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
"""Persistence for split-comparison reports.

One report is written per run, in the format chosen by ``report_format``:

- ``json`` -- single file, human-readable, ``details`` rehydrated to nested
  dicts. Written via :mod:`smart_open` so cloud URLs (``s3://``, ``gs://``,
  ``az://``, ``sftp://``, ...) work alongside local paths.
- ``lance`` -- columnar Lance dataset (directory), zero-copy from the
  in-memory ``pa.Table``. Lance handles cloud backends natively via its
  ``object_store`` integration (``s3://`` / ``gs://`` / ``az://`` -- not
  ``sftp://``; mount via sshfs if you need that).

``report_path`` is used verbatim; the format comes from ``report_format``, not
the extension. See ``docs/curator/design/split-comparison.md``.
"""

import json
from collections import Counter
from pathlib import Path
from typing import assert_never

import lance
import pyarrow as pa
import smart_open  # type: ignore[import-untyped]

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.split_comparison.config import DEFAULT_PROFILE_NAME, ReportFormat
from cosmos_curator.pipelines.video.split_comparison.result_model import Report

_LANCE_SUFFIX = ".lance"


def _is_local_path(path: str) -> bool:
    """Heuristic: a path with no ``<scheme>://`` is treated as a local filesystem path."""
    return "://" not in path


def _ensure_local_parent(path: str) -> None:
    """For local paths, create the parent directory so the writer doesn't have to."""
    if _is_local_path(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_report(report: Report, path: str, *, report_format: ReportFormat) -> str:
    """Persist ``report`` to ``path`` in ``report_format``; return the path written.

    ``report_format`` is ``"json"`` or ``"lance"`` (validated upstream by
    ``SplitComparisonConfig``). ``path`` is used verbatim.
    """
    profile_name = report.config.profile_name if report.config is not None else DEFAULT_PROFILE_NAME
    if report_format == "json":
        _write_json(report, path, profile_name=profile_name)
    elif report_format == "lance":
        _write_lance(report, path)
    else:
        assert_never(report_format)
    return path


def _write_json(report: Report, path: str, *, profile_name: str) -> None:
    """Pretty-printed JSON with ``details`` rehydrated to nested dicts for humans.

    Routes through ``smart_open`` so cloud URLs work; transport params come
    from the project's storage profile (same plumbing the stage actors use).
    """
    rows: list[dict[str, object]] = []
    for row in report.issues.to_pylist():
        if row.get("details") is not None:
            row["details"] = json.loads(row["details"])
        rows.append({key: value for key, value in row.items() if value is not None})
    payload: dict[str, object] = {
        "passed": report.passed,
        "output_a": report.output_a,
        "output_b": report.output_b,
        "source_a": report.source_a,
        "source_b": report.source_b,
        "summary": build_summary(report),
        "videos": report.videos.to_pylist(),
        "issues": rows,
    }
    config_dict = serialize_config(report)
    if config_dict is not None:
        payload["config"] = config_dict
    body = json.dumps(payload, indent=2, sort_keys=True)
    _ensure_local_parent(path)
    transport_params = storage_utils.get_smart_open_params(path, profile_name=profile_name)
    with smart_open.open(path, "w", encoding="utf-8", **transport_params) as handle:
        handle.write(body)


def _write_lance(report: Report, path: str) -> None:
    """Lance dataset of the issue table plus a sidecar dataset of run-level metadata.

    Summary and config live in the sidecar as JSON-encoded string columns so the
    schema stays flat regardless of how many features/codes turn up or what
    config shape evolves into: read with
    ``json.loads(table['summary'][0].as_py())`` and
    ``json.loads(table['config'][0].as_py())`` (config column is nullable).

    Lance handles its own URL parsing; cloud backends (``s3://``, ``gs://``,
    ``az://``) go through ``object_store`` natively. For local paths we
    materialize the parent directory so callers don't have to mkdir first.
    """
    _ensure_local_parent(path)
    lance.write_dataset(report.issues, path, mode="overwrite")
    config_dict = serialize_config(report)
    sidecar = pa.Table.from_pylist(
        [
            {
                "passed": report.passed,
                "output_a": report.output_a,
                "output_b": report.output_b,
                "source_a": report.source_a,
                "source_b": report.source_b,
                "summary": json.dumps(build_summary(report), sort_keys=True),
                "config": json.dumps(config_dict, sort_keys=True) if config_dict is not None else None,
            }
        ],
    )
    sidecar_path = path.removesuffix(_LANCE_SUFFIX) + ".meta" + _LANCE_SUFFIX
    _ensure_local_parent(sidecar_path)
    lance.write_dataset(sidecar, sidecar_path, mode="overwrite")
    videos_path = path.removesuffix(_LANCE_SUFFIX) + ".videos" + _LANCE_SUFFIX
    _ensure_local_parent(videos_path)
    lance.write_dataset(report.videos, videos_path, mode="overwrite")


def serialize_config(report: Report) -> dict[str, object] | None:
    """Render ``report.config`` as a JSON-friendly dict; returns ``None`` if no config recorded."""
    if report.config is None:
        return None
    return report.config.model_dump(mode="json")


def build_summary(report: Report) -> dict[str, object]:
    """Aggregate per-feature / per-code issue counts into the report summary block.

    Public so ``compare_split_outputs`` callers (and tests) can render the same
    summary without writing a report to disk first.
    """
    issues = report.issues
    features = issues["feature"].to_pylist() if issues.num_rows else []
    codes = issues["code"].to_pylist() if issues.num_rows else []
    videos = issues["video"].to_pylist() if issues.num_rows else []
    clips = issues["clip"].to_pylist() if issues.num_rows else []

    clip_keys: set[tuple[str | None, str]] = set()
    clips_by_feature: dict[str, set[tuple[str | None, str]]] = {}
    for feature, video, clip in zip(features, videos, clips, strict=True):
        if clip is None:
            continue
        key = (video, clip)
        clip_keys.add(key)
        if feature is not None:
            clips_by_feature.setdefault(feature, set()).add(key)

    return {
        "stages_run": sorted(report.stages_run),
        "runtime_sec": report.runtime_sec,
        "total_clips_compared": report.clip_count,
        "clips_in_a": report.clips_in_a,
        "clips_in_b": report.clips_in_b,
        "clips_in_both": report.clips_in_both,
        "total_issues": issues.num_rows,
        "clips_with_issues": len(clip_keys),
        "issues_by_feature": dict(Counter(f for f in features if f is not None)),
        "clips_with_issues_by_feature": {feature: len(keys) for feature, keys in clips_by_feature.items()},
        "issues_by_code": dict(Counter(codes)),
    }
