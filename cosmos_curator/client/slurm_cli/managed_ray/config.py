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
"""Configuration models and resolution for managed Ray clusters on Slurm."""

import copy
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.client.slurm_cli.slurm_common import (
    _DEFAULT_CACHE_PATH,
    _DEFAULT_CONTAINER_IMAGE,
    _SLURM_ACCOUNT_ENV_VAR,
    _is_valid_slurm_memory,
)
from cosmos_curator.core.utils import environment

_CONFIG_MODEL = ConfigDict(extra="forbid", frozen=True, strict=True)
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_STARTUP_TIMEOUT_PATTERN = re.compile(r"^(?P<value>[1-9][0-9]*)(?P<unit>[smhd])$")

# The six walltime spellings sbatch accepts. A leading ``D-`` makes the first field days; otherwise the number of
# colons decides, so `30` is minutes, `30:00` is minutes:seconds, and `4:00:00` is hours:minutes:seconds.
_SLURM_TIME_PATTERN = re.compile(r"^(?:(?P<days>\d+)-)?(?P<first>\d+)(?::(?P<second>\d+))?(?::(?P<third>\d+))?$")
_DEFAULT_WALLTIME = "7-00:00:00"
DEFAULT_STATE_DIR = "~/slurm-ray"
STATE_DIR_ENV_VAR = "COSMOS_CURATOR_SLURM_RAY_STATE_DIR"

SchemaVersion = Literal[1]
# The version a generated template and the rendered schema are for. Adding a version widens ``SchemaVersion``
# to keep accepting the old one; this names the one the launcher writes.
CURRENT_SCHEMA_VERSION: SchemaVersion = 1
MountMode = Literal["ro", "rw"]


class SlurmRayConfigError(ValueError):
    """Raised when a Slurm-Ray config cannot be loaded or resolved."""


def _default_host_path(path: Path) -> str:
    expanded = path.expanduser()
    try:
        relative = expanded.relative_to(Path.home())
    except ValueError:
        return str(expanded)
    return f"~/{relative}"


def expand_host_path(value: str, home: Path) -> Path:
    """Resolve a configured host path against the cluster account's home.

    Every path in this config is absolute or ``~/``-relative by validation, so expansion is the same operation
    wherever it happens.
    """
    if value.startswith("~/"):
        return home / value[2:]
    return Path(value)


def resolve_container_image(value: str, home: Path) -> str:
    """Resolve the container image reference for the node runtime.

    Host paths follow the normal expansion rules; registry references (anything else) pass through
    untouched for the node runtime to pull.
    """
    if value.startswith(("~/", "/")):
        return str(expand_host_path(value, home))
    return value


def _host_path(value: str, *, field_name: str) -> str:
    value = value.strip()
    if not value:
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)
    if "\n" in value or "\r" in value:
        msg = f"{field_name} must not contain a newline"
        raise ValueError(msg)
    if "," in value or ":" in value:
        msg = f"{field_name} must not contain ',' or ':' because it is used in a container mount"
        raise ValueError(msg)
    if not (value.startswith("~/") or Path(value).is_absolute()):
        msg = f"{field_name} must be absolute or start with '~/'"
        raise ValueError(msg)
    return value


def slurm_time_seconds(value: str) -> int:
    """Convert an sbatch walltime to seconds.

    The head's walltime is the run's lifetime and the ratio of the two decides how many allocations a lane renews
    through, so ``UNLIMITED`` and Slurm's ``0`` spelling of it are rejected.
    """
    match = _SLURM_TIME_PATTERN.fullmatch(value.strip())
    if match is None:
        msg = f"Walltime must be a finite sbatch duration such as 7-00:00:00 or 04:00:00, got {value!r}"
        raise ValueError(msg)

    fields = [int(match.group(name) or 0) for name in ("first", "second", "third")]
    if match.group("days") is not None:
        days = int(match.group("days"))
        hours, minutes, seconds = fields
    elif match.group("third") is not None:
        days = 0
        hours, minutes, seconds = fields
    elif match.group("second") is not None:
        days, hours = 0, 0
        minutes, seconds = fields[0], fields[1]
    else:
        days, hours, seconds = 0, 0, 0
        minutes = fields[0]

    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        msg = f"Walltime must be greater than zero; Slurm reads {value!r} as no limit"
        raise ValueError(msg)
    return total


def _required_directive(value: Any) -> str:  # noqa: ANN401
    if not isinstance(value, str):
        msg = "Slurm directive values must be strings"
        raise ValueError(msg)  # noqa: TRY004
    normalized = value.strip()
    if not normalized:
        msg = "Slurm directive values must not be empty"
        raise ValueError(msg)
    return normalized


def _optional_directive(value: Any) -> str | None:  # noqa: ANN401
    if value is None:
        return None
    if not isinstance(value, str):
        msg = "Slurm directive values must be strings or null"
        raise ValueError(msg)  # noqa: TRY004
    normalized: str = value.strip()
    if not normalized:
        return None
    if "\n" in normalized or "\r" in normalized:
        msg = "Slurm directive values must not contain a newline"
        raise ValueError(msg)
    if any(character.isspace() for character in normalized) or "#" in normalized:
        msg = "Slurm directive values must not contain whitespace or '#'"
        raise ValueError(msg)
    return normalized


class SlurmRayMount(BaseModel):
    """One host-to-container bind mount."""

    model_config = _CONFIG_MODEL

    source: str = Field(description="Absolute or home-relative host path.")
    destination: str = Field(description="Absolute path inside the container.")
    mode: MountMode = Field(default="rw", description="Read-only or read-write mount mode.")

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        return _host_path(value, field_name="mount source")

    @field_validator("destination")
    @classmethod
    def _validate_destination(cls, value: str) -> str:
        value = value.strip()
        if not value or not Path(value).is_absolute():
            msg = "mount destination must be an absolute path"
            raise ValueError(msg)
        if "\n" in value or "\r" in value:
            msg = "mount destination must not contain a newline"
            raise ValueError(msg)
        if "," in value or ":" in value:
            msg = "mount destination must not contain ',' or ':'"
            raise ValueError(msg)
        return value


class SlurmRayAllocationConfig(BaseModel):
    """Independent head and worker allocations share resource fields and defaults.

    By default each allocation takes a whole CPU or GPU node. Set ``exclusive: false`` to request a slice
    with the configured CPU and memory limits instead.
    """

    model_config = _CONFIG_MODEL

    partition: str | None = Field(default=None, description="Partition; null uses the Slurm default.")
    qos: str | None = Field(default=None, description="Quality of service; null uses the Slurm default.")
    time: str = Field(
        default=_DEFAULT_WALLTIME,
        description=(
            "Walltime in a format accepted by sbatch. The head's bounds the whole run; a worker walltime shorter "
            "than the head's makes each lane renew through as many allocations as it takes to cover it."
        ),
    )

    cpus: int = Field(
        default=16, ge=1, description="Cores per allocation (--cpus-per-task). Ignored when exclusive=true."
    )
    memory: str = Field(
        default="64G",
        description="Memory per allocation (sbatch --mem format, e.g. 64G). Ignored when exclusive=true.",
    )
    exclusive: bool = Field(
        default=True,
        description="Take the entire node exclusively (--exclusive --mem=0). Set false to request cpus/memory.",
    )
    gpus: int | None = Field(default=None, ge=1, description="Number of GPUs requested per node (--gpus, job-scoped).")
    gpus_per_node: int | None = Field(
        default=None,
        ge=1,
        description=(
            "GPUs per node (--gpus-per-node). Use on clusters that prohibit job-scoped GPU requests (e.g. NVL72)."
        ),
    )

    _normalize_directives = field_validator("partition", "qos", mode="before")(_optional_directive)
    _normalize_time = field_validator("time", mode="before")(_required_directive)

    @field_validator("time")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        slurm_time_seconds(value)
        return value

    @field_validator("memory")
    @classmethod
    def _validate_memory(cls, value: str) -> str:
        if not _is_valid_slurm_memory(value):
            msg = f"memory must be a positive sbatch size such as 64G or 65536M, got {value!r}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_gpu_fields(self) -> Self:
        if self.gpus is not None and self.gpus_per_node is not None:
            msg = "gpus and gpus_per_node are mutually exclusive; set only one"
            raise ValueError(msg)
        return self


class SlurmRayHeadConfig(SlurmRayAllocationConfig):
    """Allocation for the Ray head, pipeline driver, and application tasks."""


class SlurmRayWorkerConfig(SlurmRayAllocationConfig):
    """Allocation settings shared by all additional Ray worker lanes."""

    max_lane_allocations: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cap on the worker lane array size; null derives it from the head/worker walltime ratio. "
            "Use on busy clusters where the head (CPU) starts quickly but the worker (GPU) queues for hours: "
            "a short worker walltime aids backfill scheduling but the resulting large array may exceed the "
            "cluster's per-user job quota. The lane may stop renewing before the head completes."
        ),
    )


def _default_slurm_account() -> str | None:
    """Default the billing account to the launching environment, which most sites already export."""
    return os.getenv(_SLURM_ACCOUNT_ENV_VAR)


class SlurmRaySlurmConfig(BaseModel):
    """Slurm settings for a managed Ray cluster."""

    model_config = _CONFIG_MODEL

    account: str | None = Field(
        default_factory=_default_slurm_account,
        # Normalize the environment-derived default too, so a blank variable means "use the Slurm default".
        validate_default=True,
        description="Slurm billing account.",
    )
    head: SlurmRayHeadConfig = Field(default_factory=SlurmRayHeadConfig)
    worker: SlurmRayWorkerConfig = Field(default_factory=SlurmRayWorkerConfig)

    _normalize_account = field_validator("account", mode="before")(_optional_directive)


class SlurmRayRuntimeConfig(BaseModel):
    """Container runtime shared by the Ray head and workers."""

    model_config = _CONFIG_MODEL

    container_image: str = Field(
        default=_DEFAULT_CONTAINER_IMAGE,
        description=(
            "Host path to a squashfs image, or a registry reference (Pyxis enroot format: `host#org/image:tag`) "
            "pulled at job start by the node runtime. Registry pulls happen inside the `ray.startup_timeout` "
            "window; increase it for large images."
        ),
    )
    curator_path: str | None = Field(default=None, description="Optional host path to a Cosmos Curator checkout.")
    workspace_path: str = Field(
        default=_default_host_path(environment.LOCAL_WORKSPACE_PATH),
        description="Host workspace mounted at /config.",
    )
    cache_path: str = Field(
        default=_default_host_path(_DEFAULT_CACHE_PATH),
        description="Host cache mounted at /cache.",
    )
    mount_s3_creds: bool = Field(default=True, description="Mount ~/.aws/credentials when it exists.")
    mount_azure_creds: bool = Field(default=False, description="Mount ~/.azure/credentials when it exists.")
    mounts: list[SlurmRayMount] = Field(default_factory=list, description="Additional shared host mounts.")
    node_local_mounts: list[SlurmRayMount] = Field(
        default_factory=list,
        description="Mounts whose source exists only on allocated compute nodes.",
    )
    prepare_node_local_mounts: bool = Field(
        default=False,
        description="Create node-local mount sources before starting the container.",
    )
    environment: list[str] = Field(
        default_factory=list,
        description="Names of host environment variables to forward into the container.",
    )
    pixi_envs: list[str] | None = Field(
        default=None,
        description="Optional Pixi environment warmup override.",
    )

    @field_validator("container_image")
    @classmethod
    def _validate_container_image(cls, value: str) -> str:
        value = value.strip()
        # Host paths keep the existing validation; a value with a '#' is a registry reference the
        # node runtime pulls. Other shapes (colon-registry refs, relative paths) fall through to
        # host-path validation and fail there.
        if value.startswith(("~/", "/")) or "#" not in value:
            return _host_path(value, field_name="container_image")
        return value

    @field_validator("workspace_path", "cache_path")
    @classmethod
    def _validate_required_paths(cls, value: str, info: Any) -> str:  # noqa: ANN401
        return _host_path(value, field_name=str(info.field_name))

    @field_validator("curator_path")
    @classmethod
    def _validate_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _host_path(value, field_name="curator_path")

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            normalized_value = value.strip()
            if not _ENVIRONMENT_NAME_PATTERN.fullmatch(normalized_value):
                msg = f"runtime.environment entries must be variable names, got {normalized_value!r}"
                raise ValueError(msg)
            normalized.append(normalized_value)
        if len(normalized) != len(set(normalized)):
            msg = "runtime.environment must not contain duplicate names"
            raise ValueError(msg)
        return normalized

    @field_validator("pixi_envs")
    @classmethod
    def _validate_pixi_envs(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        normalized = [value.strip() for value in values]
        if not normalized or any(not value for value in normalized):
            msg = "runtime.pixi_envs must contain at least one non-empty environment name"
            raise ValueError(msg)
        if len(normalized) != len(set(normalized)):
            msg = "runtime.pixi_envs must not contain duplicates"
            raise ValueError(msg)
        return normalized


class SlurmRayRayConfig(BaseModel):
    """Ray bootstrap settings for a managed cluster."""

    model_config = _CONFIG_MODEL

    startup_timeout: str = Field(
        default="10m",
        description="Timeout for submission completion and Ray bootstrap after allocation (for example, 10m).",
        pattern=_STARTUP_TIMEOUT_PATTERN.pattern,
    )
    temp_dir: str | None = Field(default=None, description="Optional node-local root for Ray temporary files.")
    io_slots_per_node: int = Field(
        default=environment.DEFAULT_CURATOR_IO_SLOTS_PER_NODE,
        ge=1,
        description=(
            "Per-node curator_io slots on the head and workers limiting concurrent large-source downloads. "
            "Each download task must reserve one slot."
        ),
    )

    @field_validator("temp_dir")
    @classmethod
    def _validate_temp_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _host_path(value, field_name="ray.temp_dir")


class SlurmRayConfig(BaseModel):
    """Canonical version 1 managed Ray-on-Slurm submission config."""

    model_config = _CONFIG_MODEL

    schema_version: SchemaVersion
    job_name: str = Field(default="cosmos_curator", min_length=1, max_length=64)
    worker_lanes: int = Field(default=1, ge=0, description="Initial number of worker lanes in addition to the head.")
    slurm: SlurmRaySlurmConfig = Field(default_factory=SlurmRaySlurmConfig)
    runtime: SlurmRayRuntimeConfig = Field(default_factory=SlurmRayRuntimeConfig)
    ray: SlurmRayRayConfig = Field(default_factory=SlurmRayRayConfig)

    @field_validator("job_name")
    @classmethod
    def _validate_job_name(cls, value: str) -> str:
        value = value.strip()
        if not _JOB_NAME_PATTERN.fullmatch(value):
            msg = "job_name may contain only letters, digits, '.', '_', and '-'"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_reserved_mount(self) -> Self:
        reserved_root = PurePosixPath("/run/cosmos-curator/slurm-ray")
        for mount in (*self.runtime.mounts, *self.runtime.node_local_mounts):
            if PurePosixPath(mount.destination).is_relative_to(reserved_root):
                msg = (
                    "mount destination /run/cosmos-curator/slurm-ray and its descendants "
                    "are reserved for managed cluster state"
                )
                raise ValueError(msg)
        return self


def default_state_dir() -> str:
    """Return the state directory lifecycle commands search when one is not given."""
    configured = os.getenv(STATE_DIR_ENV_VAR)
    return configured.strip() if configured and configured.strip() else DEFAULT_STATE_DIR


def validate_state_dir(value: str) -> str:
    """Validate the launcher-wide shared root used to locate and render run state."""
    try:
        value = _host_path(value, field_name="state_dir")
    except ValueError as exc:
        raise SlurmRayConfigError(str(exc)) from exc
    if any(character.isspace() for character in value) or any(character in value for character in "#%"):
        msg = "state_dir must not contain whitespace, '#', or '%' because it prefixes Slurm log paths"
        raise SlurmRayConfigError(msg)
    return value


def slurm_ray_config_template() -> dict[str, Any]:
    """Return an editable config template containing every setting of the current schema version."""
    return SlurmRayConfig(schema_version=CURRENT_SCHEMA_VERSION).model_dump(mode="json")


def slurm_ray_template_yaml() -> str:
    """Render the editable template as YAML."""
    return yaml.safe_dump(slurm_ray_config_template(), sort_keys=False)


def slurm_ray_schema_json(*, indent: int = 2) -> str:
    """Render JSON Schema for user-authored configs."""
    return json.dumps(SlurmRayConfig.model_json_schema(), indent=indent) + "\n"


def load_slurm_ray_config_data(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config mapping."""
    path = Path(config_path)
    if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        msg = f"Config file must use a .json, .yaml, or .yml suffix: {path}"
        raise SlurmRayConfigError(msg)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise SlurmRayConfigError(msg)
    return cast("dict[str, Any]", loaded)


def resolve_slurm_ray_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> SlurmRayConfig:
    """Resolve defaults, a user config, and repeated ``--set`` overrides."""
    return resolve_slurm_ray_config_data(load_slurm_ray_config_data(config_path), overrides=overrides)


def resolve_slurm_ray_config_data(
    raw_data: Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
) -> SlurmRayConfig:
    """Resolve a raw mapping into the canonical config.

    ``--set`` is applied first because it addresses the user's document rather than the validated model.
    """
    resolved = copy.deepcopy(dict(raw_data))
    _apply_cli_overrides(resolved, overrides)
    return SlurmRayConfig.model_validate(resolved)


def slurm_ray_config_to_json(config: SlurmRayConfig, *, indent: int = 2) -> str:
    """Render the canonical resolved config as JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent) + "\n"


def lane_allocations(config: SlurmRayConfig) -> int:
    """Return how many worker allocations one lane renews through to cover the head's walltime.

    A worker walltime at least as long as the head's needs a single allocation. A shorter one needs a chain,
    because Slurm does not requeue a job that hit its time limit. Rounding up only over-provisions: allocations
    the run outlives are canceled with the rest of the lane at teardown. ``worker.max_lane_allocations`` caps
    the chain for clusters that limit a user's job count; when it binds, lane coverage ends before the head.
    """
    head_seconds = slurm_time_seconds(config.slurm.head.time)
    worker_seconds = slurm_time_seconds(config.slurm.worker.time)
    derived = max(1, math.ceil(head_seconds / worker_seconds))
    cap = config.slurm.worker.max_lane_allocations
    if cap is not None and derived > cap:
        return cap
    return derived


def startup_timeout_seconds(value: str) -> int:
    """Convert a validated startup timeout to seconds."""
    match = _STARTUP_TIMEOUT_PATTERN.fullmatch(value)
    if match is None:
        msg = f"Invalid startup timeout: {value!r}"
        raise SlurmRayConfigError(msg)
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return int(match.group("value")) * multipliers[match.group("unit")]


def _apply_cli_overrides(data: dict[str, Any], overrides: Sequence[str]) -> None:
    for raw_override in overrides:
        path, value = _parse_cli_override(raw_override)
        target = data
        for key in path[:-1]:
            # A config need not mention a section for --set to reach into it, so an absent one is created and
            # left for the model to fill in. Only a section that exists as something other than an object fails.
            next_value = target.setdefault(key, {})
            if not isinstance(next_value, dict):
                msg = f"--set path {'.'.join(path)} cannot descend into non-object key {key!r}"
                raise SlurmRayConfigError(msg)
            target = cast("dict[str, Any]", next_value)
        target[path[-1]] = value


def _parse_cli_override(raw_override: str) -> tuple[list[str], Any]:
    if "=" not in raw_override:
        msg = f"--set override must be PATH=VALUE, got {raw_override!r}"
        raise SlurmRayConfigError(msg)
    raw_path, raw_value = raw_override.split("=", maxsplit=1)
    path = raw_path.split(".")
    if any(not part for part in path):
        msg = f"--set override path must contain non-empty keys, got {raw_path!r}"
        raise SlurmRayConfigError(msg)
    try:
        value = yaml.safe_load(raw_value) if raw_value else ""
    except yaml.YAMLError as exc:
        msg = f"Failed to parse --set value for {raw_path}: {exc}"
        raise SlurmRayConfigError(msg) from exc
    return path, value
