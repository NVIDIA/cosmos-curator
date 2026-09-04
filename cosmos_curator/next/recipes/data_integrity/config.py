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

"""Strict config and deterministic resolution for Curator Next ``data-integrity``.

Four blocks, split by what they decide rather than by what they configure:

* ``input`` -- which sessions the run covers.
* ``checks`` -- what the verdicts mean. Every field here reaches a measurement or a
  threshold, so changing one changes what the stored rows say.
* ``output`` -- where the rows land.
* ``execution`` -- credentials and concurrency. Nothing here changes a verdict, which
  is why ``session_concurrency`` sits beside ``s3_profile_name`` rather than in
  ``checks``.

Deliberately Ray-free: the CLI, ``pipeline template`` and ``pipeline validate`` all
resolve a config without starting a cluster.
"""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.core.sensors.data_integrity.instruments import MAX_PERCENT, Thresholds
from cosmos_curator.next.core.config import apply_dotted_overrides

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")

SchemaVersion = Literal[1]
DataIntegrityKind = Literal["data-integrity"]

#: The kind name, hyphenated, used in ``kind:``, as the CLI's kind argument, and as the
#: registered name. The underscored spelling is a different kind and is not accepted.
KIND_NAME = "data-integrity"

#: What the store's ``run.lance`` row and ``manifest.json`` record as the writer,
#: alongside ``di-check`` / ``di-session`` / ``di-reevaluate``.
TOOL_NAME = KIND_NAME


class DataIntegrityInputConfig(BaseModel):
    """Which sessions the run covers, as any combination of three forms.

    Combinable rather than exclusive because they answer different questions -- a
    dataset root plus a handful of extra sessions is a normal thing to want, and the
    union is deduplicated anyway (see :func:`.sessions.expand_sessions`).
    """

    model_config = _MODEL_CONFIG

    sessions: tuple[str, ...] = Field(
        default=(),
        description="Explicit session paths: local directories, or s3:// / az:// prefixes.",
        examples=[["s3://example-bucket/clips/0a1b2c/"]],
    )
    session_list_uri: str | None = Field(
        default=None,
        min_length=1,
        description="File of session paths, one per line or as a JSON array.",
        examples=["s3://example-bucket/manifests/sessions.txt"],
    )
    session_roots: tuple[str, ...] = Field(
        default=(),
        description="Dataset roots whose child prefixes name sessions.",
        examples=[["s3://example-bucket/clips/"]],
    )
    session_depth: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Prefix levels below each root that name a session. Only 1 is supported today.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Max streams per session; null means no cap. Caps streams, never sessions.",
    )

    @field_validator("sessions", "session_roots", mode="before")
    @classmethod
    def _coerce_sequence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("sessions", "session_roots")
    @classmethod
    def _reject_blank_entries(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        if any(not path.strip() for path in paths):
            msg = "session paths and roots must be non-empty"
            raise ValueError(msg)
        return paths

    @model_validator(mode="after")
    def _at_least_one_form(self) -> Self:
        if not self.sessions and self.session_list_uri is None and not self.session_roots:
            msg = "input must set at least one of 'sessions', 'session_list_uri' or 'session_roots'"
            raise ValueError(msg)
        return self


class ThresholdsConfig(BaseModel):
    """The pass/fail policy, mirroring the kernel's :class:`Thresholds` defaults.

    Restated as a Pydantic model rather than reused because ``Thresholds`` is an attrs
    class: this gives the config block strictness and a JSON schema, and
    :meth:`to_thresholds` converts. The defaults are the kernel's, so an omitted block
    is the kernel's policy.
    """

    model_config = _MODEL_CONFIG

    max_strict_violations: int = Field(default=0, ge=0)
    max_rate_deviation_percent: float = Field(default=5.0, ge=0.0, allow_inf_nan=False)
    max_gaps: int = Field(default=0, ge=0)
    max_jitter_percent: float = Field(default=10.0, ge=0.0, allow_inf_nan=False)
    allow_frame_reordering: bool = False
    # Session-grain, judging the rig rather than one stream. Here rather than only on
    # di-session because a pipeline run measures sessions too, and a policy it cannot
    # state is a policy it is stuck with.
    max_sensor_spread_ns: int = Field(default=1_000_000_000, ge=0)
    max_non_overlap_percent: float = Field(default=5.0, ge=0.0, le=MAX_PERCENT, allow_inf_nan=False)

    def to_thresholds(self) -> Thresholds:
        """Convert to the kernel policy object the instruments and store take."""
        return Thresholds(
            max_strict_violations=self.max_strict_violations,
            max_rate_deviation_percent=self.max_rate_deviation_percent,
            max_gaps=self.max_gaps,
            max_jitter_percent=self.max_jitter_percent,
            allow_frame_reordering=self.allow_frame_reordering,
            max_sensor_spread_ns=self.max_sensor_spread_ns,
            max_non_overlap_percent=self.max_non_overlap_percent,
        )


class DataIntegrityChecksConfig(BaseModel):
    """What the metrics measure and how their results are judged.

    Only the knobs ``run_checks`` already takes. This is not the check-declaration
    format that ``data-integrity-design.md`` leaves open, and the name should not be
    read as a position on it.
    """

    model_config = _MODEL_CONFIG

    expected_hz: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Expected sample rate for every stream. Null takes each stream's own nominal rate, "
            "which is the right default across a heterogeneous dataset; where a container reports "
            "none, the rate, gap and jitter metrics come back undefined for that stream."
        ),
    )
    batch_size: int = Field(
        default=0,
        ge=0,
        description="Timestamps per metric update; 0 feeds the whole array at once.",
    )
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)


class DataIntegrityOutputConfig(BaseModel):
    """Where the run's rows land."""

    model_config = _MODEL_CONFIG

    store_root: str = Field(min_length=1, examples=["s3://example-bucket/di_store/"])

    @field_validator("store_root")
    @classmethod
    def _validate_store_root(cls, store_root: str) -> str:
        # The same rule store_cli.validate_store_path enforces for the two CLIs, and for
        # the same reason: the store is written last, so a root that cannot be written
        # must fail before any stream is read rather than after all of them.
        value = store_root.strip()
        if not value:
            msg = "output.store_root is empty; give a local directory or an s3:// prefix"
            raise ValueError(msg)
        if value.startswith("az://"):
            # get_lance_storage_options raises on az:// rather than write an
            # unauthenticated store. Reading *sessions* from az:// is unaffected.
            msg = f"the data-integrity store does not support az:// yet: {value!r}; use a local path or s3://"
            raise ValueError(msg)
        if value.startswith("s3://"):
            if not value.removeprefix("s3://").strip(" /"):
                msg = f"output.store_root {value!r} names no bucket; use s3://bucket/prefix"
                raise ValueError(msg)
            return value
        if "://" in value:
            # Anything else carrying a scheme would be taken for a local path and
            # quietly create a directory named after it, so refuse rather than guess.
            msg = f"unsupported output.store_root {value!r}; use a local path or an s3:// prefix"
            raise ValueError(msg)
        return str(Path(value).expanduser())


class DataIntegrityExecutionConfig(BaseModel):
    """Credentials and concurrency: nothing here changes a stored verdict.

    The three credential fields are named individually rather than folded into one
    ``storage_profile`` because ``run_checks``, ``discover_streams`` and
    ``content_identity`` already take them that way and this recipe passes them
    through. ``s3_profile_name`` is an AWS named profile, which is not the same thing
    as the Curator profile that ``storage_profile`` selects elsewhere.

    There is one concurrency knob rather than two because there is one stage: a task
    lists its session and then measures it, so listing is not separately schedulable.
    """

    model_config = _MODEL_CONFIG

    s3_profile_name: str | None = Field(default=None, min_length=1)
    azure_profile_name: str = Field(default="default", min_length=1)
    endpoint_url: str | None = Field(default=None, min_length=1)
    # One task per session, each decoding its streams in sequence on one CPU, so this
    # is how many sessions are in flight at once. A plain default rather than one
    # derived from the live cluster: the helpers that would size it are private to the
    # legacy Ray Data tree, which next/AGENTS.md forbids importing from here.
    session_concurrency: int = Field(
        default=8,
        ge=1,
        description="Concurrent session-measurement tasks.",
    )
    # Attempts per stream when the failure looks like a transport hiccup. Retried in the
    # worker rather than by Ray: nothing escapes the map function, so Ray's own map retry
    # never sees a failure here. Exhausting these attempts makes the stream unreachable
    # rather than unreadable -- see session_runner.InfrastructureError.
    stream_attempts: int = Field(default=3, ge=1)
    # Sessions per driver-side append. Bounds how much row payload the driver holds at
    # once, and how many Lance fragments a run leaves behind.
    append_batch_size: int = Field(default=512, ge=1)
    progress: bool = False


class ResolvedDataIntegrityConfig(BaseModel):
    """Canonical v1 execution contract for Curator Next ``data-integrity``."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: DataIntegrityKind
    input: DataIntegrityInputConfig
    checks: DataIntegrityChecksConfig = Field(default_factory=DataIntegrityChecksConfig)
    output: DataIntegrityOutputConfig
    execution: DataIntegrityExecutionConfig = Field(default_factory=DataIntegrityExecutionConfig)


_TEMPLATE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "kind": KIND_NAME,
    "input": {"sessions": ["s3://example-bucket/clips/0a1b2c/"]},
    "output": {"store_root": "s3://example-bucket/di_store/"},
}
_TEMPLATE = ResolvedDataIntegrityConfig.model_validate(_TEMPLATE_BASE).model_dump(mode="json")
_TEMPLATE_PREAMBLE = """\
# All supported settings and their defaults are shown. Unchanged settings may be removed.
# `input` accepts any combination of `sessions`, `session_list_uri` and `session_roots`;
# at least one must be non-empty, and an expansion yielding zero sessions is an error.
# `output.store_root` must be a local directory or an s3:// prefix; az:// is not supported.
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
) -> ResolvedDataIntegrityConfig:
    """Load, override, and validate one config file."""
    return resolve_config_data(load_config_data(config_path), overrides=overrides)


def resolve_config_data(
    raw_data: Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
) -> ResolvedDataIntegrityConfig:
    """Apply dotted YAML-valued overrides and return canonical typed config."""
    data = copy.deepcopy(dict(raw_data))
    apply_dotted_overrides(data, overrides)
    return ResolvedDataIntegrityConfig.model_validate(data)


def resolved_config_to_json(config: ResolvedDataIntegrityConfig, *, indent: int = 2) -> str:
    """Render canonical resolved JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent, ensure_ascii=False) + "\n"


def config_schema_json(*, indent: int = 2) -> str:
    """Return JSON Schema for v1 configs."""
    return json.dumps(ResolvedDataIntegrityConfig.model_json_schema(), indent=indent, ensure_ascii=False) + "\n"


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
        "description": (
            "Measure data-integrity metrics across many sessions with Ray Data and persist "
            "the measurements and verdicts into one Lance store root."
        ),
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": KIND_NAME},
            {
                "path": "input.sessions|input.session_list_uri|input.session_roots",
                "example": ["s3://example-bucket/clips/0a1b2c/"],
            },
            {"path": "output.store_root", "example": "s3://example-bucket/di_store/"},
        ],
        "config": config_template(),
    }
