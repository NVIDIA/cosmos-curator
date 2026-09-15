# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract-scoped durable workspace creation, validation, and cleanup."""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pyarrow.fs as pafs
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix
from cosmos_curator.core.utils.storage.storage_utils import get_storage_client
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, filesystem_path

_PRECONDITION_FAILED = 412
_PHASE_A_COMPLETION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaptionWorkspace:
    """Resolved paths for one field-set/contract recovery scope."""

    root_uri: str
    manifest_uri: str
    results_uri: str
    checkpoints_uri: str
    filesystem: pafs.FileSystem

    @property
    def phase_a_completion_uri(self) -> str:
        """Return the small marker written after one complete inference pass."""
        return _join(self.root_uri, "phase-a-complete.json")


def resolve_workspace(
    staging_root_uri: str,
    caption_field_name: str,
    digest: str,
    *,
    storage_profile: str,
) -> CaptionWorkspace:
    """Derive the contract workspace and its credentialed Arrow filesystem."""
    root_uri = _join(staging_root_uri, caption_field_name, digest)
    filesystem = arrow_filesystem(root_uri, storage_profile=storage_profile)
    return CaptionWorkspace(
        root_uri=root_uri,
        manifest_uri=_join(root_uri, "workspace.json"),
        results_uri=_join(root_uri, "results"),
        checkpoints_uri=_join(root_uri, "checkpoints"),
        filesystem=filesystem,
    )


def ensure_workspace(
    workspace: CaptionWorkspace,
    *,
    storage_profile: str,
    manifest: dict[str, object],
) -> None:
    """Create ``workspace.json`` once or require exact compatibility."""
    payload = manifest
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    created = _write_if_absent(
        workspace.manifest_uri,
        encoded,
        storage_profile=storage_profile,
    )
    if created:
        return
    try:
        existing = json.loads(_read_bytes(workspace.manifest_uri, storage_profile=storage_profile).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"Existing caption workspace manifest is unreadable: {workspace.manifest_uri}"
        raise ValueError(msg) from exc
    if existing != payload:
        msg = f"Existing caption workspace is incompatible with this contract: {workspace.manifest_uri}"
        raise ValueError(msg)


def workspace_manifest_exists(workspace: CaptionWorkspace) -> bool:
    """Return whether this exact contract workspace already has a manifest."""
    info = workspace.filesystem.get_file_info(filesystem_path(workspace.manifest_uri))
    return bool(info.type == pafs.FileType.File)


def phase_a_completion_covers(
    workspace: CaptionWorkspace,
    attempt: CaptionAttempt,
    digest: str,
) -> bool:
    """Return whether a cheap completion marker covers every pending fragment."""
    marker_path = filesystem_path(workspace.phase_a_completion_uri)
    info = workspace.filesystem.get_file_info(marker_path)
    if info.type != pafs.FileType.File:
        return False
    try:
        with workspace.filesystem.open_input_file(marker_path) as source:
            payload = json.loads(source.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    raw_fragment_ids = payload.get("pending_fragment_ids")
    marker_attempt_version = payload.get("attempt_version")
    if (
        payload.get("schema_version") != _PHASE_A_COMPLETION_SCHEMA_VERSION
        or payload.get("caption_contract_digest") != digest
        or not isinstance(marker_attempt_version, int)
        or isinstance(marker_attempt_version, bool)
        or marker_attempt_version > attempt.version
        or not isinstance(raw_fragment_ids, list)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in raw_fragment_ids)
        or len(set(raw_fragment_ids)) != len(raw_fragment_ids)
    ):
        return False
    return set(attempt.pending_fragment_ids).issubset(raw_fragment_ids)


def record_phase_a_completion(
    workspace: CaptionWorkspace,
    attempt: CaptionAttempt,
    digest: str,
) -> None:
    """Record fragment-level coverage after Ray's checkpointed write succeeds."""
    payload = {
        "schema_version": _PHASE_A_COMPLETION_SCHEMA_VERSION,
        "caption_contract_digest": digest,
        "attempt_version": attempt.version,
        "pending_fragment_ids": list(attempt.pending_fragment_ids),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with workspace.filesystem.open_output_stream(filesystem_path(workspace.phase_a_completion_uri)) as output:
        output.write(encoded)


def cleanup_workspace(workspace: CaptionWorkspace, *, storage_profile: str) -> None:
    """Delete one verified contract workspace, never its staging parent."""
    if workspace.root_uri.lower().startswith("s3://"):
        client = get_storage_client(workspace.root_uri, profile_name=storage_profile, can_delete=True)
        if not isinstance(client, S3Client):
            msg = f"Could not create an S3 cleanup client for {workspace.root_uri}"
            raise TypeError(msg)
        s3_root = S3Prefix(workspace.root_uri.rstrip("/") + "/")
        for item in client.list_recursive_directory(s3_root):
            client.delete_object(item)
        return
    local_root = Path(filesystem_path(workspace.root_uri))
    if local_root.exists():
        shutil.rmtree(local_root)


def _join(root: str, *parts: str) -> str:
    suffix = "/".join(part.strip("/") for part in parts)
    return f"{root.rstrip('/')}/{suffix}"


def _read_bytes(location: str, *, storage_profile: str) -> bytes:
    if location.lower().startswith("s3://"):
        client = get_storage_client(location, profile_name=storage_profile)
        if not isinstance(client, S3Client):
            msg = f"Could not create an S3 client for {location}"
            raise TypeError(msg)
        return client.download_object_as_bytes(S3Prefix(location))
    return Path(filesystem_path(location)).read_bytes()


def _write_if_absent(location: str, payload: bytes, *, storage_profile: str) -> bool:
    """Atomically create one object, returning false when it already exists."""
    if location.lower().startswith("s3://"):
        prefix = S3Prefix(location)
        client = get_storage_client(location, profile_name=storage_profile)
        if not isinstance(client, S3Client):
            msg = f"Could not create an S3 client for {location}"
            raise TypeError(msg)
        try:
            client.s3.put_object(Bucket=prefix.bucket, Key=prefix.prefix, Body=payload, IfNoneMatch="*")
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = exc.response.get("Error", {}).get("Code")
            if status == _PRECONDITION_FAILED or code in {"PreconditionFailed", "ConditionalRequestConflict"}:
                return False
            raise
        return True

    path = Path(filesystem_path(location))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    finally:
        temporary.unlink(missing_ok=True)
