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
r"""Typed settings for the video shard pipeline (CLI / NVCF).

Shard-only CLI flags use ``metadata[\"cli\"]`` on :class:`ShardPipelineSettings`; register them with
:func:`add_shard_args`. Shared flags are registered separately via
:func:`~cosmos_curator.pipelines.pipeline_args.add_common_args` from the shard parser setup.

Construct :class:`ShardPipelineSettings` in :func:`~cosmos_curator.pipelines.video.sharding_pipeline.shard`
via :func:`~cosmos_curator.pipelines.common_pipeline_settings.composite_from_namespace`.
"""

import argparse

import attrs
from attrs import NOTHING, validators

from cosmos_curator.pipelines.common_pipeline_settings import (
    CommonPipelineSettings,
    add_settings_cli_arguments,
    cli,
)
from cosmos_curator.pipelines.video.splitting_pipeline import ALL_CAPTION_ALGOS

_CAPTION_CHOICES = frozenset(ALL_CAPTION_ALGOS)

# Defaults shared by argparse and call sites (e.g. grouping helpers).
MAX_TARS_PER_PART_DEFAULT = 100
TARGET_TAR_SIZE_MB_DEFAULT = 500
MIN_CLIPS_PER_TAR_DEFAULT = 1


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    r"""Register shard-only flags (fields with ``metadata[\"cli\"]`` on :class:`ShardPipelineSettings`)."""
    add_settings_cli_arguments(parser, ShardPipelineSettings)


@attrs.define(slots=False)
class ShardPipelineSettings:
    """Configuration for :func:`~cosmos_curator.pipelines.video.sharding_pipeline.shard`.

    Shared CLI flags live under ``common`` (:class:`CommonPipelineSettings`); use ``settings.common.*``
    (e.g. ``settings.common.verbose``).
    """

    common: CommonPipelineSettings
    input_clip_path: str = attrs.field(
        validator=validators.min_len(1),
        metadata=cli(
            help="S3 or local path which has input processed clips",
            required=True,
            default=NOTHING,
        ),
    )
    output_dataset_path: str = attrs.field(
        validator=validators.min_len(1),
        metadata=cli(
            help="S3 or local path to store output webdataset",
            required=True,
            default=NOTHING,
        ),
    )
    captioning_algorithm: str = attrs.field(
        validator=validators.in_(_CAPTION_CHOICES),
        metadata=cli(
            help="Captioning algorithm used in annotation pipeline.",
            default="qwen",
            choices=_CAPTION_CHOICES,
        ),
    )
    annotation_version: str = attrs.field(
        validator=validators.min_len(1),
        metadata=cli(help="Annotation version to use for clip metadata", default="v0"),
    )
    input_semantic_dedup_s3_profile_name: str = attrs.field(
        validator=validators.min_len(1),
        metadata=cli(
            help="S3 profile name to use for input semantic dedup S3 path.",
            default="default",
        ),
    )
    semantic_dedup_epsilon: float = attrs.field(
        validator=validators.ge(0.0),
        metadata=cli(
            help=(
                "Epsilon threshold for semantic dedup (default: 0.01). "
                "Clips with cosine similarity ≥ (1 - epsilon) will be considered duplicates."
            ),
            default=0.01,
        ),
    )
    max_tars_per_part: int = attrs.field(
        validator=validators.ge(1),
        metadata=cli(
            help=f"Maximum number of tar archives per part (default: {MAX_TARS_PER_PART_DEFAULT}).",
            default=MAX_TARS_PER_PART_DEFAULT,
        ),
    )
    target_tar_size_mb: int = attrs.field(
        validator=validators.ge(1),
        metadata=cli(
            help=(f"Target size in MB for each tar archive (default: {TARGET_TAR_SIZE_MB_DEFAULT})."),
            default=TARGET_TAR_SIZE_MB_DEFAULT,
        ),
    )
    min_clips_per_tar: int = attrs.field(
        validator=validators.ge(1),
        metadata=cli(
            help=(f"Minimum number of clips required per tar archive (default: {MIN_CLIPS_PER_TAR_DEFAULT})."),
            default=MIN_CLIPS_PER_TAR_DEFAULT,
        ),
    )
    drop_small_shards: bool = attrs.field(
        validator=validators.instance_of(bool),
        metadata=cli(
            help=("Drop shards that have fewer than --min-clips-per-tar clips (default: False)."),
            default=False,
            arg_type=None,
            action=argparse.BooleanOptionalAction,
        ),
    )
    # Optional CLI arg must be last: attrs forbids mandatory fields after any defaulted field.
    input_semantic_dedup_path: str | None = attrs.field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        metadata=cli(
            help=(
                "S3 or local path to the dedup pipeline output root containing parquet files with semantically"
                " deduplicated clip IDs (pass the directory containing 'extraction/', not the extraction/ path itself)"
            ),
            default=None,
        ),
    )
