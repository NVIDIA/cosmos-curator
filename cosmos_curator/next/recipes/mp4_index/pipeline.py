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

"""Ray Data entry point for the ``mp4-index`` recipe -- the only module here allowed to import Ray."""

import os
from datetime import UTC, datetime
from typing import Any, cast

import ray
from loguru import logger  # type: ignore[import-untyped]

from cosmos_curator.core.utils import environment
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.mp4_index import lance_sink
from cosmos_curator.next.recipes.mp4_index.config import Mp4IndexInputConfig, Mp4IndexRunConfig, ResolvedMp4IndexConfig
from cosmos_curator.next.recipes.mp4_index.contracts import (
    RECORD_SCHEMA_VERSION,
    VIDEO_MANIFEST_SCHEMA,
    derive_session_and_camera,
    make_session_camera_id,
    make_video_id,
)
from cosmos_curator.next.recipes.mp4_index.discovery import resolve_input_selection
from cosmos_curator.next.recipes.mp4_index.processing import Mp4ExtensionFactory, _ProcessOneVideo

_BYTES_PER_GIB = 1024**3


class SessionCameraIdCollisionError(ValueError):
    """Two distinct video_ids derive the same (source_session_id, camera_key)."""


def _check_no_session_camera_id_collisions(
    rows: list[dict[str, str]],
    *,
    session_id_path_depth: int,
    log_prefix: str,
    committed_owners: dict[str, tuple[str, str]] | None = None,
) -> None:
    """Raise if two distinct video_ids -- within *rows*, or against *committed_owners* -- derive one session_camera_id.

    An in-batch-only check misses a collision that spans separate runs.

    Raises:
        SessionCameraIdCollisionError: see class docstring.

    """
    session_camera_id_owners: dict[str, tuple[str, str]] = dict(committed_owners or {})
    for row in rows:
        source_session_id, camera_key = derive_session_and_camera(
            row["media_location"], session_id_path_depth=session_id_path_depth
        )
        session_camera_id = make_session_camera_id(source_session_id, camera_key)
        owner = session_camera_id_owners.get(session_camera_id)
        if owner is not None and owner[0] != row["video_id"]:
            msg = (
                f"{log_prefix}: media_location {owner[1]!r} and {row['media_location']!r} both derive "
                f"source_session_id={source_session_id!r}, camera_key={camera_key!r} -- each would publish "
                "a distinct manifest row with ambiguous session/camera ownership. "
                "Adjust execution.session_id_path_depth, or the input corpus's "
                "own layout, so each session appears once."
            )
            raise SessionCameraIdCollisionError(msg)
        session_camera_id_owners[session_camera_id] = (row["video_id"], row["media_location"])


def discover_rows(input_config: Mp4IndexInputConfig, *, storage_profile: str, log_prefix: str) -> list[dict[str, str]]:
    """Discover candidate videos and compute each one's ``video_id``."""
    sources = resolve_input_selection(input_config, storage_profile=storage_profile)
    logger.info("{}: {} candidate video(s) discovered", log_prefix, len(sources))
    return [
        {"video_id": make_video_id(source.media_location), "media_location": source.media_location}
        for source in sources
    ]


def _empty_summary(
    *, manifest_lance_uri: str, errors_lance_uri: str, discovered: int, already: int, skipped: int
) -> dict[str, Any]:
    return {
        "videos_discovered": discovered,
        "videos_already_extracted": already,
        "videos_skipped_known_failure": skipped,
        "videos_extracted": 0,
        "videos_failed": 0,
        "manifest_lance_uri": manifest_lance_uri,
        "errors_lance_uri": errors_lance_uri,
        "indices_built": [],
        "error_indices_built": [],
    }


def _indexing_io_resources() -> dict[str, float]:
    """Honor explicit IO budgets; otherwise use one slot per available CPU."""
    if environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR in os.environ:
        return curator_io_resources()
    slots = os.process_cpu_count() or environment.DEFAULT_CURATOR_IO_SLOTS_PER_NODE
    return {environment.CURATOR_IO_RESOURCE_NAME: float(slots)}


def _run_manifest(
    rows: list[dict[str, str]],
    config: Mp4IndexRunConfig,
    *,
    extension_factory: Mp4ExtensionFactory | None = None,
) -> dict[str, Any]:
    """Reconcile *rows* against already-committed/known-failed work, dispatch, and publish.

    Reserves one ``curator_io`` slot per actor task unconditionally.
    Extensions contribute frame columns within the fixed shared output schema.
    """
    manifest_lance_uri = config.output.manifest_lance_uri
    errors_lance_uri = config.output.errors_lance_uri
    storage_profile = config.execution.storage_profile
    log_prefix = config.kind

    if not rows:
        return _empty_summary(
            manifest_lance_uri=manifest_lance_uri, errors_lance_uri=errors_lance_uri, discovered=0, already=0, skipped=0
        )

    # Bootstrapped before dispatch: a reconciliation scan needs the table to already exist.
    manifest_dataset = lance_sink.open_or_create_manifest_table(
        uri=manifest_lance_uri, storage_profile=storage_profile, schema=VIDEO_MANIFEST_SCHEMA, kind=config.kind
    )
    lance_sink.validate_committed_versions(
        manifest_dataset,
        extraction_contract_id=config.extraction_contract_id,
        record_schema_version=RECORD_SCHEMA_VERSION,
    )

    candidate_session_camera_ids = {
        make_session_camera_id(
            *derive_session_and_camera(
                row["media_location"], session_id_path_depth=config.execution.session_id_path_depth
            )
        )
        for row in rows
    }
    committed_owners = lance_sink.committed_session_camera_owners(manifest_dataset, candidate_session_camera_ids)
    _check_no_session_camera_id_collisions(
        rows,
        session_id_path_depth=config.execution.session_id_path_depth,
        log_prefix=log_prefix,
        committed_owners=committed_owners,
    )

    candidate_video_ids = {row["video_id"] for row in rows}
    committed_ids = lance_sink.committed_video_ids(manifest_dataset, candidate_video_ids)
    remaining = [row for row in rows if row["video_id"] not in committed_ids]
    already = len(rows) - len(remaining)
    if already:
        logger.info("{}: {} of {} candidate video(s) already committed, skipping", log_prefix, already, len(rows))

    skipped = 0
    if config.execution.skip_known_failures:
        failed_ids = lance_sink.known_failed_video_ids(
            errors_lance_uri, storage_profile, {row["video_id"] for row in remaining}
        )
        new_rows = [row for row in remaining if row["video_id"] not in failed_ids]
        skipped = len(remaining) - len(new_rows)
        if skipped:
            logger.info(
                "{}: {} candidate video(s) already have a failed attempt, skipping (set "
                "execution.skip_known_failures=false to retry)",
                log_prefix,
                skipped,
            )
    else:
        new_rows = remaining

    if not new_rows:
        return _empty_summary(
            manifest_lance_uri=manifest_lance_uri,
            errors_lance_uri=errors_lance_uri,
            discovered=len(rows),
            already=already,
            skipped=skipped,
        )

    ensure_ray_initialized(local_resources=_indexing_io_resources())
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability()
    ray.data.DataContext.get_current().execution_no_progress_timeout_s = config.execution.no_progress_timeout_s

    results = ray.data.from_items(new_rows, override_num_blocks=len(new_rows)).map_batches(
        _ProcessOneVideo,
        batch_format="pyarrow",
        batch_size=1,
        compute=ray.data.ActorPoolStrategy(min_size=config.execution.min_actors, max_size=config.execution.max_actors),
        # Left unset, Ray Data reserves ~2.4 GiB per CPU a task asks for.
        memory=(
            None if config.execution.actor_memory_gb is None else int(config.execution.actor_memory_gb * _BYTES_PER_GIB)
        ),
        # One slot per actor task -- the concurrency bound, not the node's own total budget
        # (curator_io_resources(), passed above to ensure_ray_initialized instead).
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1.0},
        fn_constructor_kwargs={"config": config, "extension_factory": extension_factory},
    )

    # Batched, not one commit per video -- repartitions into blocks of videos_per_fragment and
    # stages one Lance fragment per block, then the driver commits one fragment per block.
    publication_batches = results.repartition(
        target_num_rows_per_block=config.execution.videos_per_fragment, strict=True
    )
    staged = publication_batches.map_batches(
        cast("Any", lance_sink.stage_manifest_batch),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs={"uri": manifest_lance_uri, "storage_profile": storage_profile, "schema": VIDEO_MANIFEST_SCHEMA},
    )

    attempted_at = datetime.now(UTC)
    extracted = 0
    failures: list[dict[str, Any]] = []
    # prefetch_batches=0, not iter_rows() -- read-ahead would delay every commit.
    for control_batch in staged.iter_batches(prefetch_batches=0, batch_size=None, batch_format="pyarrow"):
        for result in cast("Any", control_batch).to_pylist():
            if result["result_type"] == "failure":
                failures.append(
                    {
                        "video_id": result["video_id"],
                        "media_location": result["media_location"],
                        "error_type": result["error_type"],
                        "error_message": result["error_message"],
                        "attempted_at": attempted_at,
                    }
                )
                continue
            lance_sink.append_manifest_fragment(
                result["candidate_json"],
                uri=manifest_lance_uri,
                storage_profile=storage_profile,
                schema=VIDEO_MANIFEST_SCHEMA,
                kind=config.kind,
            )
            extracted += int(result["extracted_count"])
            logger.info("{}: committed {} video(s)", log_prefix, result["extracted_count"])

    indices_built: list[str] = []
    if extracted:
        indices_built = lance_sink.rebuild_manifest_indices(manifest_lance_uri, storage_profile)
        logger.info("{}: rebuilt scalar index on {}", log_prefix, indices_built)

    error_indices_built = lance_sink.append_error_log(
        failures, errors_lance_uri=errors_lance_uri, storage_profile=storage_profile
    )

    summary: dict[str, Any] = {
        "videos_discovered": len(rows),
        "videos_already_extracted": already,
        "videos_skipped_known_failure": skipped,
        "videos_extracted": extracted,
        "videos_failed": len(failures),
        "manifest_lance_uri": manifest_lance_uri,
        "errors_lance_uri": errors_lance_uri,
        "indices_built": indices_built,
        "error_indices_built": error_indices_built,
    }
    logger.info(
        "{}: processed {}/{} new video(s) ({} already committed, {} skipped known failures), {} failure(s)",
        log_prefix,
        extracted,
        len(new_rows),
        already,
        skipped,
        len(failures),
    )
    return summary


def run_config(config: ResolvedMp4IndexConfig) -> dict[str, Any]:
    """Discover, reconcile against already-indexed work, extract (with caching), index, and publish.

    Raises:
        SessionCameraIdCollisionError: see that class's own docstring.

    """
    rows = discover_rows(config.input, storage_profile=config.execution.storage_profile, log_prefix=config.kind)
    return _run_manifest(rows, config)
