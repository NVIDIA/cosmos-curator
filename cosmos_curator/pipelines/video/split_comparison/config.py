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
"""Input contract for split output comparison -- one ``SplitComparisonConfig``.

Pydantic v2 models with ``frozen=True``, ``strict=True``, ``extra="forbid"``:
no field coercion, no silent typos, immutable once constructed. A single
config is a fully self-describing audit spec -- it holds the comparison
targets (``output_a`` / ``output_b``), the report destination
(``report_path`` / ``report_format``), and every tuning knob. The CLI loads
one of these via ``--config PATH`` and runs.
"""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PROFILE_NAME = "default"

# Output format for the persisted comparison report. The CLI writes exactly one
# report in exactly one of these formats. Keep in sync with
# ``report_io.write_report``'s dispatcher.
ReportFormat = Literal["json", "lance"]

# Project precedent: every config model uses this triple.
#   frozen=True       -- immutability by design
#   strict=True       -- no "5" -> 5 coercion; YAML/JSON typos in value types fail loudly
#   extra="forbid"    -- typos in field names fail at load (instead of silently ignored)
_CONFIG_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")


class SummaryPolicy(BaseModel):
    """Knobs for :mod:`summary_compare`.

    ``token_count_abs_tolerance`` / ``token_count_rel_tolerance`` apply to the
    ``total_prompt_tokens`` and ``total_output_tokens`` summary fields. Everything
    else is compared by strict equality.
    """

    model_config = _CONFIG_MODEL_CONFIG

    token_count_abs_tolerance: float = Field(default=0.0, ge=0.0)
    token_count_rel_tolerance: float = Field(default=0.0, ge=0.0)


class ScoreTolerance(BaseModel):
    """Abs / rel tolerance for scalar per-clip score comparisons (aesthetic, motion)."""

    model_config = _CONFIG_MODEL_CONFIG

    abs_tolerance: float = Field(default=1e-6, ge=0.0)
    rel_tolerance: float = Field(default=1e-6, ge=0.0)


class CaptionPolicy(BaseModel):
    """Knobs for :func:`compare_captions`: which embedding model to load and how strict the similarity check is.

    ``encode_batch_size`` is the chunk size handed to
    :func:`SentenceTransformer.encode` inside the cross-clip batched caption
    path; the optimum depends on machine and model, so it's tuneable. Default
    128 is CPU-friendly for BGE-small at audit batch sizes.
    """

    model_config = _CONFIG_MODEL_CONFIG

    model_id: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    min_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    encode_batch_size: int = Field(default=128, ge=1)


_DEFAULT_METADATA_FIELDS: tuple[str, ...] = (
    "codec_name",
    "codec_max_bframes",
    "codec_profile",
    "container_format",
    "height",
    "width",
    "avg_frame_rate_numerator",
    "avg_frame_rate_denominator",
    "pix_fmt",
    "bit_rate_bps",
)


class VideoIndexPolicy(BaseModel):
    """Tolerance + field-set knobs for clip MP4 ``VideoIndex`` comparison."""

    model_config = _CONFIG_MODEL_CONFIG

    compare_metadata_fields: tuple[str, ...] = Field(default=_DEFAULT_METADATA_FIELDS)
    int_tolerance: int = Field(default=0, ge=0)
    float_rtol: float = Field(default=1e-5, ge=0.0)
    float_atol: float = Field(default=1e-8, ge=0.0)


def _default_metadata_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


class SplitComparisonConfig(BaseModel):
    """Top-level configuration for ``compare_split_outputs``.

    Holds the full audit spec: which outputs to compare (``output_a`` /
    ``output_b``), where and how to persist the report (``report_path`` /
    ``report_format``), and every tuning knob. Designed for JSON / YAML round-trip via
    :meth:`model_validate_json` / :meth:`model_dump_json`.
    """

    model_config = _CONFIG_MODEL_CONFIG

    # Comparison targets. Required for a real run; ``--print-default-config``
    # emits placeholder values that the user must replace.
    output_a: str = Field(min_length=1)
    output_b: str = Field(min_length=1)

    # Where to persist the report, and in which format. Exactly one report is
    # written. ``report_format`` selects the writer: ``json`` (human-readable,
    # via smart_open so s3://, gs://, az://, ... work) or ``lance`` (columnar
    # dataset, native multi-backend via object_store). ``report_path`` is used
    # verbatim -- match the extension to the format if you like, but it is not
    # required and the format is never inferred from it.
    report_path: str = Field(default="report.json", min_length=1)
    report_format: ReportFormat = Field(default="json")

    # Storage profile used when reading both outputs.
    profile_name: str = Field(default=DEFAULT_PROFILE_NAME, min_length=1)

    # Comparison policies.
    summary: SummaryPolicy = Field(default_factory=SummaryPolicy)
    aesthetic: ScoreTolerance = Field(default_factory=ScoreTolerance)
    motion: ScoreTolerance = Field(default_factory=ScoreTolerance)
    caption: CaptionPolicy = Field(default_factory=CaptionPolicy)
    video_index: VideoIndexPolicy = Field(default_factory=VideoIndexPolicy)

    # Feature toggles.
    compare_video_index: bool = True
    compare_captions: bool = True

    # Run-scope filters.
    clip_limit: int | None = Field(default=None, ge=1)
    video_key: str | None = Field(default=None, min_length=1)

    # Ray / actor pool tuning.
    # Each stage carries a worker count, a CPU reservation per worker, and a
    # target rows-per-batch knob. Batch size is what map_batches actually receives
    # per __call__; block count is derived as ceil(num_rows / batch_size), floored
    # at pool_size so every worker gets at least one block.
    metadata_workers: int = Field(default_factory=_default_metadata_workers, ge=1)
    metadata_cpus_per_worker: float = Field(default=1.0, gt=0.0)
    # Stage 1 batches drive cross-clip caption-encode amortization: one
    # ``model.encode()`` per batch. Bigger batches amortize tokenizer + framework
    # dispatch better; memory grows with rows x caption-windows-per-row.
    metadata_batch_size: int = Field(default=128, ge=1)
    # Stage 2 actor count is derived as ``floor(cpu_count / video_index_cpus_per_worker)``
    # at run time, so this knob fully determines Stage 2 parallelism without a separate
    # worker-count field. The default reserves a whole CPU per actor (one actor per core):
    # earlier 0.25 / 4-actors-per-core defaults thrashed on real workloads -- MP4
    # indexing does enough header-parse + array-build work that packing 4 actors per
    # core fought for CPU. Lower (e.g. 0.5) if your indexer is dominantly IO-bound on
    # a given storage backend; raise above 1 only if profiling shows multi-core gains
    # per index call.
    video_index_cpus_per_worker: float = Field(default=1.0, gt=0.0)
    # Stage 2 batches the row-loop only; MP4 reads don't batch usefully. Smaller
    # batches give the scheduler more rebalancing points and shorten the tail.
    video_index_batch_size: int = Field(default=2, ge=1)


def example_default_config() -> SplitComparisonConfig:
    """Construct a config with placeholder targets, for ``--print-default-config``.

    ``output_a`` / ``output_b`` carry sentinel values that the user must replace
    before the resulting JSON is usable -- the strings are valid (so the model
    constructs), but they obviously aren't real paths.
    """
    return SplitComparisonConfig(
        output_a="REPLACE_WITH_OUTPUT_A_PATH",
        output_b="REPLACE_WITH_OUTPUT_B_PATH",
    )
