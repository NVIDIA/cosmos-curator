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

"""Config for discovering, extracting, and indexing MP4 video files, fragmented or not."""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self, cast
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.recipes.mp4_index.contracts import KIND
from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.uris import normalize_location

_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})

SchemaVersion = Literal[1]
Mp4IndexKind = Literal["mp4-index"]
ContainerKind = Literal["fragmented", "nonfragmented"]


def _join(root: str, name: str) -> str:
    """Append *name* as a path segment onto *root*, preserving a URI's scheme separators."""
    if "://" not in root:
        return f"{root.rstrip('/')}/{name}"
    parsed = urlsplit(root)
    return parsed._replace(path=f"{parsed.path.rstrip('/')}/{name}").geturl()


class Mp4IndexInputConfig(BaseModel):
    """Exactly one of an explicit object set or a discovered root+glob set."""

    model_config = _MODEL_CONFIG

    uris: tuple[str, ...] | None = Field(
        default=None,
        min_length=1,
        description="Explicit media locations -- no discovery, no globs, no limit.",
        examples=[["s3://example-bucket/raw/a.mp4"]],
    )
    path_prefix: str | None = Field(
        default=None,
        min_length=1,
        description="Local directory or s3:// prefix scanned for candidate videos.",
        examples=["s3://example-bucket/raw"],
    )
    globs: tuple[str, ...] = Field(
        default=(),
        description=(
            "Required when path_prefix is set. A video matches if any pattern matches, applied "
            "against each object's path relative to path_prefix."
        ),
        examples=[["unit-*/*.mp4"]],
    )
    exclude_globs: tuple[str, ...] | None = Field(
        default=None,
        description="Only meaningful with path_prefix. A video is dropped if any pattern matches a globs hit.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Only meaningful with path_prefix. Optional cap applied after discovery.",
    )
    container_kind: ContainerKind = Field(
        description="mp4 container shape (fragmented/non-fragmented)",
    )

    @field_validator("uris", "globs", "exclude_globs", mode="before")
    @classmethod
    def _coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("uris")
    @classmethod
    def _validate_uris(cls, uris: tuple[str, ...] | None) -> tuple[str, ...] | None:
        return None if uris is None else tuple(normalize_location(uri) for uri in uris)

    @field_validator("path_prefix")
    @classmethod
    def _validate_path_prefix(cls, prefix: str | None) -> str | None:
        return None if prefix is None else normalize_location(prefix)

    @model_validator(mode="after")
    def _validate_input_mode(self) -> Self:
        if (self.uris is None) == (self.path_prefix is None):
            msg = "input must set exactly one of 'uris' or 'path_prefix'"
            raise ValueError(msg)
        if self.path_prefix is not None and not self.globs:
            msg = "input.globs is required when path_prefix is set"
            raise ValueError(msg)
        if self.uris is not None and (self.globs or self.exclude_globs is not None or self.limit is not None):
            msg = "input.globs/exclude_globs/limit are only meaningful with path_prefix, not uris"
            raise ValueError(msg)
        return self


class Mp4IndexOutputConfig(BaseModel):
    """Canonical datasets and raw-box cache paths under one output root."""

    model_config = _MODEL_CONFIG

    path: str = Field(
        min_length=1, description="Root that will contain manifest.lance, errors.lance, and raw moov/moof caches."
    )

    @property
    def manifest_lance_uri(self) -> str:
        """``{path}/manifest.lance``."""
        return _join(self.path, "manifest.lance")

    @property
    def moov_sidecar_root(self) -> str:
        """``{path}/moov``."""
        return _join(self.path, "moov")

    @property
    def moof_table_root(self) -> str:
        """``{path}/moof``."""
        return _join(self.path, "moof")

    @property
    def errors_lance_uri(self) -> str:
        """``{path}/errors.lance``."""
        return _join(self.path, "errors.lance")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, path: str) -> str:
        return normalize_location(path)


class Mp4IndexExecutionConfig(BaseModel):
    """Settings that decide how a run is carried out -- mostly never what it produces.

    One exception: ``session_id_path_depth`` changes ``source_session_id``/``camera_key`` for
    identical input bytes. It lives here, not in ``input``, because it configures *how to parse*
    an already-chosen path, not *which* videos are chosen.
    """

    model_config = _MODEL_CONFIG

    session_id_path_depth: int = Field(
        default=2,
        ge=1,
        description=(
            "How many directories up from media_location the session segment sits, for deriving "
            "source_session_id/camera_key generically from the path. Defaults to 2, matching a "
            "{prefix}/{session}/unit-NN/camera_X.mp4 layout; a layout one directory deeper "
            "({prefix}/{session}/unit-NN/extra/camera_X.mp4) needs depth=3 set explicitly. One "
            "run still assumes a single, uniform depth across every video it indexes -- a corpus "
            "mixing both layouts in one run is not yet supported."
        ),
    )
    storage_profile: str = Field(
        default="default",
        min_length=1,
        description="Profile name used to build the storage client for reading moov/moof-table sidecars.",
    )
    min_actors: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum Ray Data actor pool size. Ray Data's own unconstrained autoscaling shrinks "
            "the pool as the remaining queue thins, serializing a batch's last items onto a "
            "single actor; a floor above 1 keeps that from happening."
        ),
    )
    max_actors: int | None = Field(
        default=None,
        ge=1,
        description="Maximum Ray Data actor pool size, or null to let Ray Data bound it by available resources.",
    )
    actor_memory_gb: float | None = Field(
        default=0.5,
        gt=0.0,
        description="Amount of heap one indexing actor reserves, in GiB, or null to accept Ray Data's own default.",
    )
    skip_known_failures: bool = Field(
        default=True,
        description="Skip a candidate whose video_id already has a row in errors_lance_uri, rather than retrying.",
    )
    videos_per_fragment: int = Field(
        default=1_000,
        ge=1,
        description=(
            "Manifest rows per staged Lance fragment, and therefore per driver-side commit. "
            "Smaller means more frequent commits (less recomputation lost to a mid-run crash), "
            "at the cost of more, smaller Lance fragments to compact later."
        ),
    )
    progress: bool = Field(default=False, description="Draw Ray Data progress bars. Off by default.")
    skip_existing_sidecars: bool = Field(
        default=True,
        description=(
            "Caching guard for moov/moof sidecars. Off forces a full re-extraction of every "
            "candidate; metadata and frame indexes are rebuilt into Lance records for uncommitted videos."
        ),
    )
    chunk_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1,
        description=(
            "Bytes fetched per request while walking a fragmented file for moof boxes. Only "
            "exercised when input.container_kind == 'fragmented' and the moof sidecar is not "
            "already cached."
        ),
    )
    no_progress_timeout_s: float = Field(
        default=3600.0,
        ge=1.0,
        description=(
            "Ray Data's execution_no_progress_timeout_s: the conditional moof-walk branch can "
            "run for tens of minutes on a single large fragmented file with no block boundary "
            "crossed, indistinguishable from a hang to Ray Data's own 1800s default."
        ),
    )


class Mp4IndexRunConfig(BaseModel):
    """Shared settings and extraction contract for MP4 indexing recipes."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: str
    input: Mp4IndexInputConfig
    output: Mp4IndexOutputConfig
    execution: Mp4IndexExecutionConfig = Field(default_factory=Mp4IndexExecutionConfig)

    @classmethod
    def resolve_config(cls, config_path: str | Path, *, overrides: Sequence[str] = ()) -> Self:
        """Load, override, and validate one config file for this recipe."""
        return cls.resolve_config_data(load_config_data(config_path), overrides=overrides)

    @classmethod
    def resolve_config_data(cls, config_data: Mapping[str, Any], *, overrides: Sequence[str] = ()) -> Self:
        """Apply dotted YAML-valued overrides without modifying the supplied config."""
        data = copy.deepcopy(dict(config_data))
        apply_dotted_overrides(data, overrides)
        return cls.model_validate(data)

    @classmethod
    def config_schema_json(cls, *, indent: int = 2) -> str:
        """Render JSON Schema using this recipe's concrete model."""
        return json.dumps(cls.model_json_schema(), indent=indent, ensure_ascii=False) + "\n"

    @property
    def extraction_contract(self) -> dict[str, object]:
        """Describe extraction-time output semantics, independently of execution tuning.

        Only fields that change what gets extracted/written for identical input bytes belong
        here -- read-time-only interpretation behavior and execution-only tuning are both
        excluded, since neither changes the output.
        """
        return {
            "kind": self.kind,
            "extraction_version": 1,  # bumped manually when extraction semantics change
            "container_kind": self.input.container_kind,
            "session_id_path_depth": self.execution.session_id_path_depth,
        }

    @property
    def extraction_contract_id(self) -> str:
        """Identify compatible extraction work."""
        return canonical_digest(self.extraction_contract)


class ResolvedMp4IndexConfig(Mp4IndexRunConfig):
    """Canonical execution contract for the ``mp4-index`` recipe."""

    kind: Mp4IndexKind


_TEMPLATE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "kind": KIND,
    "input": {"uris": ["s3://example-bucket/raw/example.mp4"], "container_kind": "nonfragmented"},
    "output": {"path": "s3://example-bucket/mp4-index/"},
}
_TEMPLATE = ResolvedMp4IndexConfig.model_validate(_TEMPLATE_BASE).model_dump(mode="json", exclude_none=True)
_TEMPLATE_PREAMBLE = """\
# All supported settings and their defaults are shown. Unchanged settings may be removed.
# To discover a prefix recursively instead, replace `uris` with `path_prefix` + `globs`
# (e.g. `path_prefix: s3://example-bucket/raw`, `globs: ["unit-*/*.mp4"]`).
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


resolve_config = ResolvedMp4IndexConfig.resolve_config
resolve_config_data = ResolvedMp4IndexConfig.resolve_config_data
config_schema_json = ResolvedMp4IndexConfig.config_schema_json


def resolved_config_to_json(config: Mp4IndexRunConfig, *, indent: int = 2) -> str:
    """Render canonical resolved JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent, ensure_ascii=False) + "\n"


def config_template() -> dict[str, Any]:
    """Return a complete editable config template with every default."""
    return copy.deepcopy(_TEMPLATE)


def config_template_yaml() -> str:
    """Render the complete template as annotated YAML."""
    return _TEMPLATE_PREAMBLE + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, Any]:
    """Return structured template metadata for agents."""
    return {
        "kind": KIND,
        "description": (
            "Discover source video files and build complete per-video metadata/frame-index Lance records for "
            "either fragmented or non-fragmented MP4s, extracting moov/moof directly (with a "
            "caching guard that skips already-extracted videos)."
        ),
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": KIND},
            {"path": "input.uris|input.path_prefix+input.globs", "example": ["s3://example-bucket/raw/example.mp4"]},
            {"path": "input.container_kind", "example": "nonfragmented"},
            {"path": "output.path", "example": "s3://example-bucket/mp4-index/"},
        ],
        "config": config_template(),
    }
