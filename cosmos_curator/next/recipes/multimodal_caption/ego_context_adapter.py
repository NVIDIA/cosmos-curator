# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin caption adapter for deterministic episode-level EgoTrajectory context."""

from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from cosmos_curator.core.sensors.exceptions import AlignmentError
from cosmos_curator.next.recipes.multimodal_caption.contracts import EGO_PROMPT_SENTINEL
from cosmos_curator.next.recipes.multimodal_caption.ego_context import (
    EgoContextBuilder,
    EgoTrajectoryContextPolicy,
)
from cosmos_curator.next.recipes.video_caption.inference import (
    ItemMediaError,
    read_and_validate_media,
)

_EGO_CONTEXT_INPUT_COLUMNS = (
    "start_ns",
    "exclusive_end_ns",
    "ego_media_location",
    "ego_media_size_bytes",
    "ego_media_sha256",
    "download_error_type",
)


def _durable_alignment_error(exc: AlignmentError) -> str:
    """Render safe structured diagnostics without copying source-clock values."""
    measurements = (
        ("delta_ns", exc.delta_ns),
        ("max_delta_ns", exc.max_delta_ns),
        ("gap_ns", exc.gap_ns),
        ("max_gap_ns", exc.max_gap_ns),
    )
    details = ", ".join(f"{name}={value}" for name, value in measurements if value is not None)
    return exc.reason.value if not details else f"{exc.reason.value}: {details}"


def append_ego_context_batch(  # noqa: PLR0913 -- the adapter receives one fully resolved context
    batch: pa.Table,
    *,
    storage_profile: str,
    attempts: int,
    schema_name: str,
    protobuf_mapping: Mapping[str, Any],
    topic: str,
    frame: str,
    policy: EgoTrajectoryContextPolicy,
    prompt_template: str,
    build_context: EgoContextBuilder,
) -> pa.Table:
    """Fetch, verify, and render Ego context after the camera fetch stage."""
    error_types = batch.column("download_error_type").to_pylist()
    error_messages = batch.column("download_error_message").to_pylist()
    rendered_prompts: list[str | None] = [None] * batch.num_rows
    context_digests: list[str | None] = [None] * batch.num_rows
    for index, row in enumerate(batch.select(_EGO_CONTEXT_INPUT_COLUMNS).to_pylist()):
        if row.get("download_error_type") is not None:
            continue

        media_location = str(row["ego_media_location"])
        expected_size = int(row["ego_media_size_bytes"])
        expected_sha256 = str(row["ego_media_sha256"])
        episode_start_ns = int(row["start_ns"])
        exclusive_end_ns = int(row["exclusive_end_ns"])
        try:
            payload = read_and_validate_media(
                media_location,
                storage_profile=storage_profile,
                attempts=attempts,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                media_label="Ego MCAP",
            )
        except ItemMediaError as exc:
            error_types[index] = type(exc).__name__
            error_messages[index] = str(exc)
            continue

        try:
            context_bytes, context_digest = build_context(
                payload,
                episode_start_ns=episode_start_ns,
                exclusive_end_ns=exclusive_end_ns,
                topic=topic,
                schema_name=schema_name,
                protobuf_mapping=protobuf_mapping,
                frame=frame,
                policy=policy,
            )
        except AlignmentError as exc:
            error_types[index] = type(exc).__name__
            error_messages[index] = _durable_alignment_error(exc)
            continue
        except (ValueError, TypeError) as exc:
            error_types[index] = type(exc).__name__
            error_messages[index] = str(exc)
            continue

        rendered_prompts[index] = prompt_template.replace(EGO_PROMPT_SENTINEL, context_bytes.decode("utf-8"), 1)
        context_digests[index] = context_digest

    error_type_index = batch.schema.get_field_index("download_error_type")
    error_message_index = batch.schema.get_field_index("download_error_message")
    return (
        batch.set_column(
            error_type_index,
            batch.schema.field(error_type_index),
            pa.array(error_types, type=pa.string()),
        )
        .set_column(
            error_message_index,
            batch.schema.field(error_message_index),
            pa.array(error_messages, type=pa.large_string()),
        )
        .append_column(
            pa.field("rendered_prompt", pa.large_string()),
            pa.array(rendered_prompts, type=pa.large_string()),
        )
        .append_column(
            pa.field("context_digest", pa.string()),
            pa.array(context_digests, type=pa.string()),
        )
    )
