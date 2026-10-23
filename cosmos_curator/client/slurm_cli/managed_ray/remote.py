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
"""Remote run state and file transfer for managed Slurm-Ray runs.

Scheduler queries live in :mod:`.scheduler`; this module owns the run directory the client reaches over the
login-node connection.
"""

import json
import re
import shlex
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from invoke.runners import Result as InvokeResult

from cosmos_curator.client.slurm_cli.managed_ray.config import validate_state_dir
from cosmos_curator.client.slurm_cli.managed_ray.onnode import slurm_ray_runtime, slurm_ray_state
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_runtime import RUNTIME_MODULE_FILENAME
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    MANIFEST_FILENAME,
    MANIFEST_REVISION_CONFLICT_EXIT_CODE,
    MINIMUM_PYTHON_VERSION,
    STATE_MODULE_FILENAME,
    JsonObject,
    SlurmRayManifest,
    validate_manifest,
)
from cosmos_curator.client.slurm_cli.slurm_common import _get_username, expand_host_path
from cosmos_curator.client.slurm_cli.slurm_submit import (
    ConnectionProtocol,
    connect,
    upload_text,
)
from cosmos_curator.client.slurm_cli.slurm_submit import (
    remote_home as _remote_home,
)

_RUN_ID_PATTERN = re.compile(r"^cc-ray-[0-9a-f]{8,32}$")
# Bounding each remote call keeps one unreachable login node from hanging a lifecycle command indefinitely.
_REMOTE_COMMAND_TIMEOUT_SECONDS = 5 * 60


class SlurmRayOperationError(RuntimeError):
    """Raised when a remote Slurm-Ray lifecycle operation cannot complete."""


class RemoteManifestRevisionConflictError(SlurmRayOperationError):
    """Raised when a conditional mutation was based on an obsolete manifest revision."""


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        msg = f"Invalid managed Ray run ID: {run_id!r}"
        raise SlurmRayOperationError(msg)


def run_remote(connection: ConnectionProtocol, command: str, **kwargs: object) -> InvokeResult:
    """Run one bounded remote command."""
    kwargs.setdefault("timeout", _REMOTE_COMMAND_TIMEOUT_SECONDS)
    return connection.run(command, **kwargs)


def slurm_cluster_name(connection: ConnectionProtocol) -> str:
    """Return the scheduler identity whose job IDs the connected login node addresses."""
    result = run_remote(connection, "scontrol show config", hide=True, warn=True)
    if not result.ok:
        msg = f"Could not identify the Slurm cluster reached through {connection.host}"
        raise SlurmRayOperationError(msg)
    for line in str(result.stdout).splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "ClusterName" and value.strip():
            return value.strip()
    msg = f"Slurm on {connection.host} did not report ClusterName"
    raise SlurmRayOperationError(msg)


def _verify_manifest_cluster(connection: ConnectionProtocol, manifest: SlurmRayManifest) -> None:
    """Refuse to interpret cluster-local job IDs through a different Slurm controller."""
    actual = slurm_cluster_name(connection)
    expected = manifest["slurm_cluster_name"]
    if actual != expected:
        msg = (
            f"Run {manifest['run_id']} belongs to Slurm cluster {expected!r}, but {connection.host} "
            f"reaches {actual!r}; use a login node for {expected!r}"
        )
        raise SlurmRayOperationError(msg)


def remote_home(connection: ConnectionProtocol) -> Path:
    """Resolve the absolute home directory of the connected cluster account."""
    try:
        return _remote_home(connection)
    except ValueError as exc:
        raise SlurmRayOperationError(str(exc)) from exc


def verify_remote_python(connection: ConnectionProtocol) -> None:
    """Reject a cluster whose ``python3`` cannot run the modules this submission is about to upload.

    Those modules run outside any container, so checking once here turns a later ``ImportError`` from an uploaded
    file nobody recognizes into a message naming both versions. It cannot speak for a compute node whose image
    differs from the login node's, which stays a site assumption.
    """
    minimum = "{}.{}".format(*MINIMUM_PYTHON_VERSION)
    command = shlex.join(["python3", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
    result = run_remote(connection, command, hide=True, warn=True)
    if not result.ok:
        msg = f"No usable python3 on {connection.host}; managed Ray runs its own state module there and needs {minimum}"
        raise SlurmRayOperationError(msg)

    reported = str(result.stdout).strip()
    try:
        version = tuple(int(part) for part in reported.split("."))
    except ValueError:
        msg = f"Could not read a python3 version from {connection.host}: {reported!r}"
        raise SlurmRayOperationError(msg) from None
    if version < MINIMUM_PYTHON_VERSION:
        msg = (
            f"python3 on {connection.host} is {reported}, but managed Ray runs its own state module there and "
            f"needs at least {minimum}"
        )
        raise SlurmRayOperationError(msg)


def expand_remote_path(connection: ConnectionProtocol, path: str) -> Path:
    """Resolve a configured host path against the connected account's home.

    Resolving that home costs a round trip, so an already-absolute path skips it.
    """
    if not path.startswith("~/"):
        return Path(path)
    return expand_host_path(path, remote_home(connection))


def run_manifest_path(connection: ConnectionProtocol, state_dir: str, run_id: str) -> Path:
    """Return the manifest path a state directory and run ID imply, without consulting any index."""
    return expand_remote_path(connection, validate_state_dir(state_dir)) / run_id / MANIFEST_FILENAME


def read_remote_json_files(connection: ConnectionProtocol, paths: list[Path]) -> dict[str, JsonObject]:
    """Read several small JSON files in one round trip, skipping any that are missing or damaged.

    Each file is flattened onto a single output line prefixed by its path, so one command can carry many
    documents back without a second connection per run.
    """
    if not paths:
        return {}
    quoted = " ".join(shlex.quote(str(path)) for path in paths)
    result = run_remote(
        connection,
        f'for path in {quoted}; do [ -f "$path" ] || continue; '
        'printf "%s\\t" "$path"; tr -d "\\n" < "$path"; printf "\\n"; done',
        hide=True,
        warn=True,
    )
    if not result.ok:
        return {}

    documents: dict[str, JsonObject] = {}
    for line in result.stdout.splitlines():
        path, separator, raw = line.partition("\t")
        if not separator:
            continue
        with suppress(SlurmRayOperationError):
            documents[path] = _decode_remote_json(raw, path=Path(path))
    return documents


def list_run_manifest_paths(connection: ConnectionProtocol, state_dir: str) -> dict[str, Path]:
    """Map every run directory under one state directory to its manifest."""
    directory = expand_remote_path(connection, validate_state_dir(state_dir))
    listing = run_remote(
        connection,
        f"ls -1 -- {shlex.quote(str(directory))} 2>/dev/null",
        hide=True,
        warn=True,
    )
    if not listing.ok:
        return {}
    return {
        name: directory / name / MANIFEST_FILENAME for name in listing.stdout.split() if _RUN_ID_PATTERN.fullmatch(name)
    }


def _state_module_path(run_dir: Path) -> Path:
    return run_dir / STATE_MODULE_FILENAME


def upload_state_module(connection: ConnectionProtocol, run_dir: Path) -> None:
    """Upload this run's copy of the standard-library-only state module."""
    source_path = Path(slurm_ray_state.__file__)
    upload_text(connection, [(source_path.read_text(encoding="utf-8"), _state_module_path(run_dir), 0o700)])


def upload_runtime_modules(connection: ConnectionProtocol, run_dir: Path) -> None:
    """Upload the exact state and container runtime implementation owned by this submission."""
    upload_state_module(connection, run_dir)
    runtime_source = Path(slurm_ray_runtime.__file__)
    upload_text(
        connection,
        [(runtime_source.read_text(encoding="utf-8"), run_dir / RUNTIME_MODULE_FILENAME, 0o700)],
    )


def _atomic_upload_text(
    connection: ConnectionProtocol,
    path: Path,
    text: str,
    *,
    mode: int,
) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    upload_text(connection, [(text, temporary_path, mode)])
    run_remote(
        connection,
        f"mv -f -- {shlex.quote(str(temporary_path))} {shlex.quote(str(path))} && "
        f"chmod {mode:o} {shlex.quote(str(path))}",
    )


def atomic_upload_json(
    connection: ConnectionProtocol,
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int = 0o600,
) -> None:
    """Upload one JSON document so that readers never observe a partial write."""
    _atomic_upload_text(
        connection,
        path,
        json.dumps(value, indent=2) + "\n",
        mode=mode,
    )


def read_remote_json(connection: ConnectionProtocol, path: Path) -> JsonObject:
    """Read and decode one remote JSON object."""
    result = run_remote(connection, f"cat -- {shlex.quote(str(path))}", hide=True)
    return _decode_remote_json(result.stdout, path=path)


def _decode_remote_json(raw: str, *, path: Path) -> JsonObject:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON state in {path}: {exc}"
        raise SlurmRayOperationError(msg) from exc
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path}"
        raise SlurmRayOperationError(msg)
    return cast("JsonObject", value)


def _validated_manifest(value: JsonObject, *, path: Path) -> SlurmRayManifest:
    """Validate a manifest the client just read, reporting damaged state as a lifecycle error."""
    try:
        return validate_manifest(value)
    except TypeError as exc:
        msg = f"Invalid managed Ray run state in {path}: {exc}"
        raise SlurmRayOperationError(msg) from exc


def _read_remote_manifest(connection: ConnectionProtocol, manifest_path: Path) -> SlurmRayManifest:
    """Read one manifest snapshot.

    This takes no lock. Every writer publishes through an atomic rename, so a reader sees either the previous
    revision or the next one and never a partial write.
    """
    result = run_remote(connection, f"cat -- {shlex.quote(str(manifest_path))}", hide=True, warn=True)
    if not result.ok:
        # An unknown run ID and a wrong state directory look identical here, so name the path that was searched
        # rather than reporting whatever cat said about it.
        msg = f"No managed Ray run state at {manifest_path}; check the run ID and --state-dir"
        raise SlurmRayOperationError(msg)
    return _validated_manifest(_decode_remote_json(result.stdout, path=manifest_path), path=manifest_path)


def mutate_remote_manifest(
    connection: ConnectionProtocol,
    manifest_path: Path,
    mutation: Mapping[str, object],
    *,
    run_id: str,
    expected_revision: int | None = None,
) -> SlurmRayManifest:
    """Apply one named mutation to freshly read, exclusively locked state on the login node.

    ``expected_revision`` provides compare-and-set semantics for a mutation derived from an earlier snapshot.
    """
    arguments = ["mutate", str(manifest_path), run_id, json.dumps(mutation, sort_keys=True)]
    if expected_revision is not None:
        arguments.extend(["--expected-revision", str(expected_revision)])
    command = shlex.join(["python3", str(_state_module_path(manifest_path.parent)), *arguments])
    result = run_remote(connection, command, hide=True, warn=True)
    if not result.ok:
        if result.exited == MANIFEST_REVISION_CONFLICT_EXIT_CODE:
            actual = str(result.stdout).strip() or "unknown"
            msg = f"Manifest revision no longer matches; current revision is {actual}"
            raise RemoteManifestRevisionConflictError(msg)
        stderr = str(result.stderr).strip()
        raise SlurmRayOperationError(stderr or str(result.stdout).strip() or "remote run state command failed")
    return _validated_manifest(_decode_remote_json(result.stdout, path=manifest_path), path=manifest_path)


@dataclass(frozen=True)
class RemoteRun:
    """A resolved remote run and one manifest snapshot."""

    connection: ConnectionProtocol
    manifest_path: Path
    manifest: SlurmRayManifest

    def mutate(self, mutation: Mapping[str, object]) -> SlurmRayManifest:
        """Apply one mutation to this run, judged against the state the writer finds under the lock."""
        return mutate_remote_manifest(
            self.connection,
            self.manifest_path,
            mutation,
            run_id=self.manifest["run_id"],
        )


def _read_run_manifest(connection: ConnectionProtocol, manifest_path: Path, run_id: str) -> SlurmRayManifest:
    """Read one manifest snapshot and confirm it describes the run that was asked for."""
    manifest = _read_remote_manifest(connection, manifest_path)
    if manifest["run_id"] != run_id:
        msg = f"Manifest belongs to another run: {manifest['run_id']!r}"
        raise SlurmRayOperationError(msg)
    return manifest


@contextmanager
def open_remote_run(
    run_id: str,
    *,
    login_node: str,
    username: str | None,
    state_dir: str,
) -> Iterator[RemoteRun]:
    """Resolve a run on the cluster it belongs to and read one manifest snapshot.

    The snapshot is what a command checks its preconditions against; it is not a claim on the run. Every write
    re-reads state under the run lock.
    """
    _validate_run_id(run_id)
    connection = connect(login_node, username or _get_username())
    try:
        manifest_path = run_manifest_path(connection, state_dir, run_id)
        manifest = _read_run_manifest(connection, manifest_path, run_id)
        _verify_manifest_cluster(connection, manifest)
        yield RemoteRun(connection=connection, manifest_path=manifest_path, manifest=manifest)
    finally:
        connection.close()
