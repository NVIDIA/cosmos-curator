# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict configuration and deterministic resolution for ``robot-action-caption``.

The sections split along exactly one line, and the split is the point::

    input            source location and row scope
    output           staging location
    model            SEMANTIC - reaches the contract digest
    generation       SEMANTIC - reaches the contract digest
    video            SEMANTIC - reaches the contract digest
    media            SEMANTIC - reaches the contract digest
    execution        NON-SEMANTIC - never reaches the digest

A field belongs in a semantic section if changing it changes what a run
publishes, and in ``execution`` otherwise. ``execution``'s invariant is what
makes retuning a run safe: it never invalidates the completed GPU work a resume
would reuse. Raising the media byte cap, changing sampling, or changing the frame
budget has the opposite effect on purpose, so each recaptions the configured
input scope.
``input.selected_only`` and ``input.shard`` instead define which source rows this
attempt owes - the first by row scope, the second by dividing the table's
fragments across independently launched jobs. Neither changes a row's caption, so
neither reaches the caption digest: rows a shard does not own are simply left
owed, to be captioned identically by whichever job does own them.

Prompt text and fixed request-shape invariants remain frozen beside the code that
owns them; legitimate experiment controls are explicit typed config values.
"""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Self, cast
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from cosmos_curator.core.utils.config.pydantic_config import STRICT_CONFIG_MODEL_CONFIG
from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.core.sharding import ShardConfig
from cosmos_curator.next.utils.local_uri import local_media_path
from cosmos_curator.next.utils.uris import normalize_s3_uri

_YAML_SUFFIXES: Final = frozenset({".yaml", ".yml"})

# Rows per Ray Data vLLM batch, and the payload multiplier in the per-worker heap
# reservations ``stage_heap_reservations()`` declares on the three media-carrying
# CPU stages. Those reservations bound how many such workers Ray places per node,
# so this value trades engine batching efficiency against the parallelism those
# stages get for frame decoding. It ALSO moves total media bytes in flight, which
# a purely proportional reservation would not: admissible workers are node memory
# over `resident + payload`, so the payload term no longer cancels and a smaller
# batch admits proportionally less media at once. Lowering it therefore does buy
# headroom, at the cost of paying the resident term over more workers.
DEFAULT_INFERENCE_BATCH_SIZE: Final = 16
DEFAULT_MAX_CONCURRENT_BATCHES: Final = 8

# Per-object media size cap. Refused off the store's own stat() before the
# transfer starts, and re-checked against the delivered length because that stat
# is advisory and a backend may omit it entirely.
#
# This is a SEMANTIC value, not a throughput knob: raising it converts terminal
# `error` rows into captions, so two runs with different caps produce different
# populations from the same table. Changing it therefore recaptions the
# configured input scope under latest-contract-wins - it is a deployment-wide setting, not a per-run
# throughput choice.
#
# The bounds below are arithmetic, not decoration. A clip encodes to base64 at
# ceil(n/3) * 4, and that string is what travels to the engine, so in-flight host
# memory per engine replica is `execution.inference_batch_size` x
# `execution.max_concurrent_batches` of them, with Ray's high-memory detector
# switched off in this recipe's driver. At their defaults and eight replicas per
# node, the aggregate at MAX_SUPPORTED_CLIP_BYTES is roughly one third of a
# 1 TB host in base64 strings alone, before vLLM's retained original bytes and
# decoded frames.
#
# The per-cap table lives in "Why the byte cap has a ceiling" in
# docs/curator/design/curator-next-robot-action-caption.md. This module enforces
# the per-clip cap only: raising either execution multiplier or placing more than
# eight replicas on a node can exceed that documented default-geometry aggregate
# without a config error. The values remain execution-only so retuning throughput
# cannot alter caption identity.
#
# These are worst cases that only materialize if the corpus holds clips near the
# cap; the current upstream splitter writes clips of a few megabytes. That is a
# PROPERTY OF THE INPUT DATA, not a guarantee this code makes. Bounding the
# aggregate is the producer's job, not captioning's.
DEFAULT_MAX_CLIP_BYTES: Final = 128 * 1024 * 1024
MAX_SUPPORTED_CLIP_BYTES: Final = 256 * 1024 * 1024

# Clips one media actor reads at once. Chosen as a PARTITION of the node's
# ``curator_io`` budget, never an addition to it: each concurrent read reserves
# one slot, so a node advertising N slots sustains N reads whether they are
# N actors of one thread or N/4 actors of four. Four buys most of the win in
# process count - it quarters the actors, their storage clients and their CPU
# reservations for the same read capacity - while staying low enough that one
# actor's in-flight clips cannot dominate a worker's memory, since each holds
# its encoded bytes and base64 string at once. Raising total read capacity is a
# different action: it needs --ray-io-slots-per-node, not this value.
DEFAULT_MEDIA_FETCH_WORKERS: Final = 4

# One decoded frame from a Mecka ACT2 camera, at 1280x720. Sizes the heap
# reservation of the stages that carry decoded media, so it tracks the SOURCE
# resolution: the pixel caps in the video config are applied by the processor
# inside the engine, long after the frames have crossed three stages at full
# size. A corpus from wider cameras must raise it, which is why it is a config
# field and not a constant folded into the arithmetic.
DEFAULT_SOURCE_FRAME_PIXELS: Final = 1280 * 720

# The checkpoint a run captions with unless it says otherwise. This is a global
# ``all_models.json`` key, not a recipe-local compatibility declaration.
DEFAULT_MODEL_VARIANT: Final = "qwen2.5_vl"

type SchemaVersion = Literal[1]
type RobotActionCaptionKind = Literal["robot-action-caption"]

KIND_NAME: Final = "robot-action-caption"


def _normalize_location(location: str, *, label: str) -> str:
    """Normalize one local-or-S3 location without changing an S3 object key."""
    if not location or location != location.strip():
        msg = f"{label} must be non-empty and cannot have surrounding whitespace"
        raise ValueError(msg)
    parsed = urlsplit(location)
    scheme = parsed.scheme.lower()
    if scheme == "s3":
        return normalize_s3_uri(location, strip_trailing_slash=True)
    if scheme == "file":
        # Validated by the shared local-URI gate rather than re-derived here, so
        # this config accepts exactly the file:// forms the writers and readers
        # that later open the location accept. Re-deriving it would drop the
        # delimiter check: ``?`` and ``#`` end a URI's path, so
        # file:///data/run#2/clips.lance normalizes to /data/run - a shorter
        # location that may well exist and is not the one that was written.
        try:
            path = local_media_path(location)
        except ValueError as e:
            msg = f"Unsupported local file URI for {label}: {location!r}"
            raise ValueError(msg) from e
        return str(path.expanduser().resolve())
    if scheme:
        msg = f"{label} must be a local path or s3:// URI, got {location!r}"
        raise ValueError(msg)
    return str(Path(location).expanduser().resolve())


class RobotActionCaptionInputConfig(BaseModel):
    """Define the clip table and the source rows this attempt may consume."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    clips_lance_uri: str = Field(
        min_length=1,
        examples=["s3://example-bucket/robot_clips/lance/clips.lance"],
    )
    selected_only: bool = Field(
        default=True,
        description=(
            "Caption only rows whose curate_selection_reason is 'selected'. Requires the Curate-owned column; "
            "set false to caption every otherwise-pending row without requiring curation output. This is input "
            "scope, not caption semantics, and never reaches the caption contract digest."
        ),
    )
    shard: ShardConfig | None = Field(
        default=None,
        description=(
            "Divide the table's fragments across independently launched jobs, this job taking the shard "
            "at the given zero-based index. Absent means one job owns every fragment. This is execution "
            "scope, not caption semantics: it never reaches the caption contract digest, so which job "
            "captioned a clip cannot change what that caption says. It IS part of the recovery namespace, "
            "which is what keeps concurrent shards from sharing a staging directory - so hold count fixed "
            "for a campaign: changing it re-partitions the table and strands any uncommitted progress "
            "staged under the old count, and shards of unequal counts silently claim the same fragments."
        ),
    )

    @field_validator("clips_lance_uri")
    @classmethod
    def _normalize_clips_uri(cls, location: str) -> str:
        return _normalize_location(location, label="input.clips_lance_uri")


class RobotActionCaptionModelConfig(BaseModel):
    """Select a pinned artifact through the global model registry."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    variant: str = Field(
        default=DEFAULT_MODEL_VARIANT,
        min_length=1,
        description=(
            "Nonblank key in cosmos_curator/configs/all_models.json. The resolved model id, "
            "revision, and precision enter the caption digest; vLLM, not the key spelling, "
            "determines model and video compatibility."
        ),
    )

    @field_validator("variant")
    @classmethod
    def _require_registered_variant(cls, variant: str) -> str:
        """Require a nonblank key present in the global model registry."""
        if not variant.strip() or variant != variant.strip():
            msg = "model.variant must be nonblank and cannot have surrounding whitespace"
            raise ValueError(msg)
        try:
            resolve_registered_model(variant)
        except KeyError as e:
            raise ValueError(str(e)) from e
        return variant


class RobotActionCaptionGenerationConfig(BaseModel):
    """Control semantic generation behavior recorded in the contract digest."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    temperature: float = Field(default=0.7, ge=0.0, allow_inf_nan=False)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0, allow_inf_nan=False)
    max_output_tokens: int = Field(default=512, ge=1)

    # The engine's context length is the SECOND bound on where a generation
    # stops, and the quieter of the two. A prompt plus its video that exceeds it
    # makes the request raise, which ends the attempt honestly; a prompt that
    # merely leaves less than max_output_tokens of headroom does not raise. The
    # engine stops at the context limit, so `caption_status` measures prompt plus
    # generation against this exact configured value rather than the output cap
    # alone. That second mode is also why the value is semantic and digested: it
    # can shorten a caption while nothing fails.
    max_model_len: int = Field(default=32_768, ge=2)

    @model_validator(mode="after")
    def _require_output_headroom(self) -> Self:
        """Require the context window to exceed the generation token cap."""
        if self.max_output_tokens >= self.max_model_len:
            msg = "generation.max_output_tokens must be smaller than generation.max_model_len"
            raise ValueError(msg)
        return self


class RobotActionCaptionVideoConfig(BaseModel):
    """Define recipe-owned sampling and optional processor-native overrides.

    All fields are semantic: the exact mappings emitted by this class enter both
    vLLM construction and the caption contract digest. Curator validates value
    categories and ordering only; vLLM owns processor compatibility and
    token/context admission.
    """

    model_config = STRICT_CONFIG_MODEL_CONFIG

    fps: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Optional target frames per second. Set it together with min_frames and "
            "max_frames; null delegates temporal behavior to the selected backend."
        ),
    )
    min_frames: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional positive sampling floor. Set it together with fps and max_frames; "
            "changing it changes captions and the digest."
        ),
    )
    max_frames: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional positive sampling ceiling. Set it together with fps and min_frames; "
            "changing it changes captions and the digest."
        ),
    )
    min_pixels: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional processor-native pixel floor. Set it together with max_pixels; "
            "null delegates pixel behavior to the selected backend."
        ),
    )
    max_pixels: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional processor-native pixel ceiling. Set it together with min_pixels; "
            "null delegates pixel behavior to the selected backend."
        ),
    )

    @model_validator(mode="after")
    def _require_ordered_bounds(self) -> Self:
        """Require complete optional groups and ordered bounds."""
        temporal_values = (self.fps, self.min_frames, self.max_frames)
        if any(value is None for value in temporal_values) and any(value is not None for value in temporal_values):
            msg = "video.fps, video.min_frames, and video.max_frames must be set together or all be null"
            raise ValueError(msg)
        if self.min_frames is not None and self.max_frames is not None and self.max_frames < self.min_frames:
            msg = f"video.max_frames must be at least video.min_frames, got {self.max_frames} < {self.min_frames}"
            raise ValueError(msg)
        if (self.min_pixels is None) != (self.max_pixels is None):
            msg = "video.min_pixels and video.max_pixels must be set together or both be null"
            raise ValueError(msg)
        if self.min_pixels is not None and self.max_pixels is not None and self.max_pixels < self.min_pixels:
            msg = f"video.max_pixels must be at least video.min_pixels, got {self.max_pixels} < {self.min_pixels}"
            raise ValueError(msg)
        return self

    def media_io_kwargs(self) -> dict[str, Any]:
        """Return a complete temporal group, or delegate to the backend."""
        # ``num_frames`` is absent because it would COEXIST with fps rather than
        # refine it. vLLM does have a rule making the two exclusive, but it fires
        # only when a per-request mapping overrides a config-level one, and this
        # mapping is config-level with no per-request counterpart. Emitting both
        # would leave num_frames bound to VideoMediaIO's named parameter and fps
        # loose in its **kwargs, and which one decides sampling would then be the
        # resolved backend's business rather than this config's.
        #
        # The empty mapping below is delegation, not "no sampling": VideoMediaIO
        # applies its own num_frames default to whichever backend resolves.
        if self.fps is None:
            return {}
        return {
            "video": {
                "fps": self.fps,
                "min_frames": self.min_frames,
                "max_frames": self.max_frames,
            }
        }

    def mm_processor_kwargs(self) -> dict[str, Any]:
        """Return the configured pixel pair, or explicit processor delegation."""
        if self.min_pixels is None:
            return {}
        return {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}


class RobotActionCaptionMediaConfig(BaseModel):
    """Define media admission policy whose fields all reach the digest.

    Deliberately not part of ``execution``. The byte cap decides whether a clip
    becomes a caption or a terminal ``error`` row, so two runs with different caps
    produce different populations from the same table - which is the definition of
    a digest input, and the opposite of ``execution``'s contract.
    """

    model_config = STRICT_CONFIG_MODEL_CONFIG

    max_clip_bytes: int = Field(
        default=DEFAULT_MAX_CLIP_BYTES,
        gt=0,
        le=MAX_SUPPORTED_CLIP_BYTES,
        description=(
            "Per-clip byte cap; larger clips become terminal error rows. Digested, "
            "so changing it recaptions the configured input scope. See the cap constants in this "
            "module for the host-memory arithmetic behind the ceiling."
        ),
    )


class RobotActionCaptionOutputConfig(BaseModel):
    """Durable staging root, derived from the clip table's parent by default."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    staging_root_uri: str = Field(
        default="",
        description="Defaults to <clips_lance_uri>__caption_staging.",
    )

    @field_validator("staging_root_uri")
    @classmethod
    def _normalize_staging_uri(cls, location: str) -> str:
        if not location:
            return location
        return _normalize_location(location, label="output.staging_root_uri")


class RobotActionCaptionExecutionConfig(BaseModel):
    """Define non-semantic execution policy excluded from caption identity."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    storage_profile: str = Field(default="default", min_length=1)
    inference_concurrency: int | Literal["auto"] = Field(
        default="auto",
        description="Maximum vLLM replicas, or auto to let Ray scale with pending work and live GPUs.",
    )
    inference_batch_size: int = Field(
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        ge=1,
        description=(
            "Rows per Ray Data vLLM batch. Larger batches can improve GPU utilization "
            "but increase the host-memory reservation of every media-carrying stage, "
            "reducing how many of its CPU workers fit on a node."
        ),
    )
    max_concurrent_batches: int = Field(
        default=DEFAULT_MAX_CONCURRENT_BATCHES,
        ge=1,
        description=(
            "In-flight Ray batch calls per vLLM replica. Raises overlap and host-memory "
            "demand without changing the GPU count."
        ),
    )
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Tensor-parallel factor within each vLLM replica.",
    )
    pipeline_parallel_size: int = Field(
        default=1,
        ge=1,
        description=("Pipeline stages per vLLM replica. Multiplies tensor_parallel_size into the replica's GPU width."),
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "vLLM scheduler token budget per step. Null preserves vLLM's own default, which at the "
            "pinned version is 2048 - small enough that a multimodal prompt of tens of thousands of "
            "tokens is prefilled over several steps, so this is a throughput knob on any "
            "large-prompt run rather than a rarely-needed override. Affects capacity and memory "
            "headroom but not completed caption identity."
        ),
    )
    gpu_memory_utilization: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "Fraction of each GPU vLLM may claim. Null preserves vLLM's own default. What is left "
            "after the model weights becomes KV cache, which sets how many requests a replica can "
            "hold at once - so on a large model this is the knob that decides engine concurrency, "
            "and vLLM prints the resulting ceiling as 'Maximum concurrency for N tokens per "
            "request'. Raising it trades startup headroom for residency and can fail at graph "
            "capture; it never changes a caption."
        ),
    )
    enable_prefix_caching: bool = Field(
        default=False,
        description=(
            "Whether vLLM retains KV blocks for prompt-prefix reuse. Off because "
            "contracts.REQUEST_CONTENT_TYPES puts the clip before the text, so every prompt opens "
            "with its own vision tokens and no two requests share a prefix - measured at a 0.0% "
            "hit rate. Worth enabling only once the request leads with shared text, which is a "
            "change to REQUEST_CONTENT_TYPES rather than to this setting."
        ),
    )
    media_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Transport read attempts per clip before the reader gives up. The only lever "
            "over an intermittently failing object store. Belongs here and not under "
            "media because a spent budget raises and ends the attempt rather than "
            "publishing an error row: retrying more never changes what a caption says, "
            "only whether a flaky store yields one at all."
        ),
    )
    media_fetch_workers: int = Field(
        default=DEFAULT_MEDIA_FETCH_WORKERS,
        ge=1,
        description=(
            "Clips one media actor fetches at once. Each concurrent read reserves one "
            "node-advertised curator_io slot, so this is a PARTITION of that budget "
            "rather than an addition to it: a node publishing N slots sustains N reads "
            "however they are divided into actors. Raise the node budget with "
            "--ray-io-slots-per-node to increase total read capacity; raise this to buy "
            "the same capacity with fewer processes. Must not exceed any node's "
            "advertised budget, or no media actor can ever be placed."
        ),
    )
    # Per concurrent READ, not per actor, and multiplied by the fetch width at
    # the call site. Each fetch thread spills its clip to a temp file, forks
    # ffprobe over it, and base64-encodes the bytes, so CPU demand tracks reads
    # in flight rather than processes. A flat per-actor figure would silently
    # shrink the stage's declared share as the width grew, since a wider actor
    # serves the same node budget with fewer processes.
    media_cpus_per_fetch: float = Field(
        default=0.25,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "CPU reservation per concurrent clip read. The actor reservation multiplies "
            "this value by media_fetch_workers, so widening an actor never silently reduces "
            "the CPU share declared for each read."
        ),
    )
    lance_read_batch_size: int = Field(
        default=256,
        ge=1,
        description="Rows per Lance scan batch before media fetching.",
    )
    parquet_rows_per_file: int = Field(
        default=256,
        ge=1,
        description=(
            "Minimum and maximum staged rows requested per Parquet file. Ray's sink adopts the "
            "minimum as the write bundle's row floor and one bundle is one checkpoint, so this is "
            "the recovery granularity: a preemption re-generates at most this many rows per bundle "
            "in flight, and generation is unseeded. The bundle can never be smaller than the block "
            "reaching the sink, so lowering this below lance_read_batch_size only splits files. "
            "Raise the two together to trade recovery granularity for fewer staged objects."
        ),
    )
    source_frame_pixels: int = Field(
        default=DEFAULT_SOURCE_FRAME_PIXELS,
        gt=0,
        description=(
            "Pixels in one decoded source frame, used with the video frame ceiling to "
            "size the heap reservation of the stages that carry decoded media. Belongs "
            "here rather than under video because the engine's own processor resizes "
            "every frame before inference, so this changes scheduling density only and "
            "never a caption. Set it from the widest camera in the corpus: understating "
            "it packs more actors per node than their working set fits."
        ),
    )
    progress: bool = False

    @field_validator("storage_profile")
    @classmethod
    def _reject_surrounding_whitespace(cls, value: str) -> str:
        """Reject padded credential profile names without normalizing them."""
        if value != value.strip():
            msg = "storage_profile cannot have surrounding whitespace"
            raise ValueError(msg)
        return value

    @field_validator("inference_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int | Literal["auto"]) -> int | Literal["auto"]:
        if isinstance(value, int) and value < 1:
            msg = f"inference_concurrency must be 'auto' or a positive integer, got {value}"
            raise ValueError(msg)
        return value


class ResolvedRobotActionCaptionConfig(BaseModel):
    """Represent the canonical v1 Curator Next ``robot-action-caption`` config."""

    model_config = STRICT_CONFIG_MODEL_CONFIG

    schema_version: SchemaVersion
    kind: RobotActionCaptionKind
    input: RobotActionCaptionInputConfig
    output: RobotActionCaptionOutputConfig = Field(default_factory=RobotActionCaptionOutputConfig)
    model: RobotActionCaptionModelConfig = Field(default_factory=RobotActionCaptionModelConfig)
    generation: RobotActionCaptionGenerationConfig = Field(default_factory=RobotActionCaptionGenerationConfig)
    video: RobotActionCaptionVideoConfig = Field(default_factory=RobotActionCaptionVideoConfig)
    media: RobotActionCaptionMediaConfig = Field(default_factory=RobotActionCaptionMediaConfig)
    execution: RobotActionCaptionExecutionConfig = Field(default_factory=RobotActionCaptionExecutionConfig)

    @property
    def staging_root(self) -> str:
        """Return the configured staging root, or the default beside the clip table.

        Derived on read rather than in a ``mode="before"`` validator so the default
        is computed from the ALREADY-normalized table URI. Deriving it earlier
        would join a raw user string, letting the same table reached by two
        spellings produce two staging roots and therefore two workspaces that
        cannot see each other's completed work.
        """
        if self.output.staging_root_uri:
            return self.output.staging_root_uri
        return f"{self.input.clips_lance_uri}__caption_staging"


_TEMPLATE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "kind": KIND_NAME,
    "input": {"clips_lance_uri": "s3://example-bucket/robot_clips/lance/clips.lance"},
    "output": {},
    "model": {},
    "generation": {},
    "video": {},
    "media": {},
    "execution": {"storage_profile": "default"},
}
_TEMPLATE = ResolvedRobotActionCaptionConfig.model_validate(_TEMPLATE_BASE).model_dump(mode="json")
_TEMPLATE_PREAMBLE = """\
# All supported settings and their defaults are shown. Unchanged settings may be removed.
# staging_root_uri defaults to <clips_lance_uri>__caption_staging.
# Model weights must already be staged under /config/models on every inference worker.
#
# `input.selected_only` is INPUT SCOPE: it limits the source rows this attempt
# owes and never reaches the caption contract digest.
# `model`, `generation`, `video`, and `media` are SEMANTIC: they reach the
# caption contract digest, so changing them recaptions stale rows.
# `execution` is NON-SEMANTIC: retuning it never discards completed work that a
# resume can reuse.
# Prompt text and fixed request-shape invariants remain frozen in code.
#
# Temporal and pixel controls are independent optional groups.
# Omitting a complete group delegates that behavior to the selected backend.
# vLLM remains the authority on model compatibility and control interpretation.
"""


def load_config_data(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML config as a top-level mapping."""
    path = Path(config_path)
    if not path.exists():
        msg = f"Pipeline config file not found: {path}"
        raise FileNotFoundError(msg)
    try:
        with path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file) if path.suffix.lower() in _YAML_SUFFIXES else json.load(config_file)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"Failed to parse config {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)
    return cast("dict[str, Any]", loaded)


def resolve_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> ResolvedRobotActionCaptionConfig:
    """Load, override, and validate one config file."""
    return resolve_config_data(load_config_data(config_path), overrides=overrides)


def resolve_config_data(
    raw_data: Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
) -> ResolvedRobotActionCaptionConfig:
    """Apply dotted YAML-valued overrides and return canonical typed config."""
    data = copy.deepcopy(dict(raw_data))
    apply_dotted_overrides(data, overrides)
    return ResolvedRobotActionCaptionConfig.model_validate(data)


def resolved_config_to_json(config: ResolvedRobotActionCaptionConfig, *, indent: int = 2) -> str:
    """Render canonical resolved JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent, ensure_ascii=False) + "\n"


def config_schema_json(*, indent: int = 2) -> str:
    """Return JSON Schema for v1 configs."""
    return json.dumps(ResolvedRobotActionCaptionConfig.model_json_schema(), indent=indent, ensure_ascii=False) + "\n"


def config_template() -> dict[str, Any]:
    """Return a complete editable config template with every default."""
    return copy.deepcopy(_TEMPLATE)


def config_template_yaml() -> str:
    """Render the complete template as annotated YAML."""
    return _TEMPLATE_PREAMBLE + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, Any]:
    """Return structured template metadata for agents."""
    return {
        "kind": KIND_NAME,
        "description": ("Caption egocentric robot-action clips and atomically enrich their canonical Lance rows."),
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": KIND_NAME},
            {
                "path": "input.clips_lance_uri",
                "example": "s3://example-bucket/robot_clips/lance/clips.lance",
            },
        ],
        "config": config_template(),
    }
