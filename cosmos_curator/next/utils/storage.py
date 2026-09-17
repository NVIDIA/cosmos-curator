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

"""Stateless location and byte helpers for Curator Next recipes.

Every helper here is single-shot: it names or normalizes a URI, or moves the
complete bytes of one object, and lets a backend failure reach the caller
unchanged. Reading a per-row media object under a size ceiling, with retries and
a classified verdict on failure, is a stateful job that lives next door in
:mod:`cosmos_curator.next.utils.media_reader`.

Importing this module costs a cloud SDK (``botocore``, and ``azure.identity``
transitively), which a client-only install does not have. Pure URI parsing
therefore lives in :mod:`cosmos_curator.next.utils.local_uri`, with location
normalization in :mod:`cosmos_curator.next.utils.uris`. Config and CLI callers
must import those modules directly rather than this one.
"""

import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import tenacity
from botocore.exceptions import ClientError
from loguru import logger

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix, get_s3_client_config, is_s3path
from cosmos_curator.core.utils.storage.storage_utils import (
    get_storage_client,
    is_missing_object_error,
    is_remote_path,
    path_to_prefix,
)
from cosmos_curator.next.utils.local_uri import (
    LOCAL_SCHEMES,
    local_media_path,
    local_path,
    uri_scheme,
)

_PRECONDITION_FAILED = 412
_CONDITIONAL_REQUEST_CONFLICT = 409

# A conditional PUT that collides with another conditional PUT on the same key
# answers 409, which S3 documents as retryable and which proves nothing about
# whether the object exists. Retried here rather than propagated because
# ``write_if_absent`` promises the loser of a race a verdict, and botocore does
# not cover this code: its S3 retry policy is socket errors, 5xx, throttling,
# ``RequestTimeout`` and ``BadDigest``. Bounded, and exhaustion re-raises the
# botocore error rather than reporting a create that never happened.
_S3_CONFLICT_MAX_ATTEMPTS = 5
_S3_CONFLICT_RETRY_MULTIPLIER_S = 0.1
_S3_CONFLICT_RETRY_MAX_WAIT_S = 2.0


def artifact_uri(location: str) -> str:
    """Return the normalized durable URI recorded in manifests, rows, and receipts."""
    if is_s3path(location):
        return S3Prefix(location).path
    if location.startswith("file://"):
        return local_path(location).resolve().as_uri()
    return Path(location).expanduser().resolve().as_uri()


@lru_cache(maxsize=8)
def _s3_client(profile: str, *, overwrite: bool = False) -> S3Client:
    return S3Client(get_s3_client_config(profile, can_overwrite=overwrite))


def download_bytes(source_uri: str, *, storage_profile: str) -> bytes:
    """Read local or S3 object bytes with cached worker credentials."""
    if source_uri.startswith("s3://"):
        return _s3_client(storage_profile).download_object_as_bytes(S3Prefix(source_uri))
    return local_media_path(source_uri).read_bytes()


def upload_bytes(payload: bytes, destination_uri: str, *, storage_profile: str) -> None:
    """Atomically replace a local or S3 destination on task replay."""
    if destination_uri.startswith("s3://"):
        _s3_client(storage_profile, overwrite=True).upload_bytes(S3Prefix(destination_uri), payload)
    else:
        write_media(destination_uri, payload, storage_profile=storage_profile)


def write_media(location: str, data: bytes, *, storage_profile: str = "default") -> None:
    """Atomically replace a deterministic local/S3 media object with complete bytes."""
    if is_s3path(location):
        client = get_storage_client(location, profile_name=storage_profile, can_overwrite=True)
        if client is None:
            msg = f"Could not create an S3 client for {location}"
            raise RuntimeError(msg)
        client.upload_bytes(S3Prefix(location), data)
        return

    # ``artifact_uri`` in this same module hands back file:// URIs, and recipes
    # accept a file:// output prefix, so a location can arrive as one. ``Path``
    # would read the scheme as a directory name and write the object into a
    # relative ``file:/`` tree beside whatever the worker's cwd was -- quietly,
    # so the run reports success and the bytes are somewhere else.
    destination = local_path(location)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_media_if_present(location: str, *, storage_profile: str = "default") -> bytes | None:
    """Read a media object's bytes, or ``None`` when it does not exist.

    The read-direction counterpart of :func:`write_media` and the same
    absence-is-a-no-op contract as :func:`remove_object`: a missing target returns
    ``None`` so a best-effort artifact (for example a Curate run report) collapses
    to "unavailable" instead of crashing, while a permission, network, or any
    other non-absence failure propagates. Local paths and remote object stores
    (``s3://`` / ``az://``) are handled through the same storage client as the
    other helpers in this module.

    Args:
        location: Local path or remote (``s3://`` / ``az://``) object URI.
        storage_profile: Credential profile for the remote read.

    Returns:
        The object's bytes when present, otherwise ``None``.

    Raises:
        ValueError: If ``location`` is remote but no storage client can be built.

    """
    if is_remote_path(location):
        client = get_storage_client(location, profile_name=storage_profile)
        if client is None:
            msg = f"no storage client available to read remote object {location}"
            raise ValueError(msg)
        prefix = path_to_prefix(location)
        # object_exists maps a remote 404 to False and lets every other error
        # (permission, network) propagate, so absence is the only path to None -
        # a raw download would raise a backend-specific not-found exception that
        # the recipe layer must not know about.
        if not client.object_exists(prefix):
            return None
        try:
            return client.download_object_as_bytes(prefix)
        except Exception as e:
            if is_missing_object_error(e):
                return None
            raise

    try:
        return Path(location).read_bytes()
    except FileNotFoundError:
        # Same contract as the removal helpers: an absent object is not an error,
        # every other failure (PermissionError, IsADirectoryError) propagates.
        return None


def remove_prefix(uri: str, *, storage_profile: str = "default") -> None:
    """Delete everything under a prefix, local or remote; absence is a no-op.

    Removes a whole staging directory / object-store prefix: a recursive object
    delete under ``uri`` for remote stores, a directory ``rmtree`` for a local
    path.

    Args:
        uri: Local path or remote (``s3://`` / ``az://``) prefix to remove.
        storage_profile: Credential profile for remote deletion.

    Raises:
        ValueError: If ``uri`` is remote but no delete-capable storage client
            can be built for it.

    """
    if is_remote_path(uri):
        client = get_storage_client(uri, profile_name=storage_profile, can_delete=True)
        if client is None:
            msg = f"no storage client available to remove remote prefix {uri}"
            raise ValueError(msg)
        deleted = 0
        for obj in client.list_recursive_directory(path_to_prefix(uri)):
            try:
                client.delete_object(obj)
            except Exception as e:
                if is_missing_object_error(e):
                    continue
                raise
            deleted += 1
        logger.info(f"Removed remote prefix {uri} ({deleted} object(s)).")
        return

    try:
        shutil.rmtree(uri)
    except FileNotFoundError:
        # Absence is not an error: a re-run that already cleaned up is a no-op.
        # Every OTHER filesystem failure - a PermissionError, or a partially
        # deleted directory - must propagate rather than be logged as a
        # successful removal, which is why ignore_errors is NOT used here.
        return
    logger.info(f"Removed local prefix {uri}.")


def remove_object(uri: str, *, storage_profile: str = "default") -> None:
    """Delete a single object/file, local or remote; absence is a no-op.

    Removes one durable artifact (for example a stale report or a copied-out
    centroid file), never a tree - use :func:`remove_prefix` for a whole staging
    prefix. A missing target is not an error; a local path that is a directory,
    or any other non-absence failure, propagates.

    Args:
        uri: Local path or remote (``s3://`` / ``az://``) object URI.
        storage_profile: Credential profile for remote deletion.

    Raises:
        ValueError: If ``uri`` is remote but no delete-capable storage client
            can be built for it.

    """
    if is_remote_path(uri):
        client = get_storage_client(uri, profile_name=storage_profile, can_delete=True)
        if client is None:
            msg = f"no storage client available to remove remote object {uri}"
            raise ValueError(msg)
        prefix = path_to_prefix(uri)
        # object_exists maps a remote 404 to False and lets every other error
        # propagate, so absence is the only path to a no-op - Azure DELETE on a
        # missing blob raises ResourceNotFoundError whereas S3 DELETE is idempotent.
        if not client.object_exists(prefix):
            return
        try:
            client.delete_object(prefix)
        except Exception as e:
            if is_missing_object_error(e):
                return
            raise
        logger.info(f"Removed remote object {uri}.")
        return

    try:
        Path(uri).unlink()
    except FileNotFoundError:
        # Same contract as remove_prefix: an absent object is a no-op, but a
        # PermissionError or IsADirectoryError must surface, not be swallowed.
        return
    logger.info(f"Removed local object {uri}.")


def write_if_absent(uri: str, data: bytes, *, storage_profile: str = "default") -> bool:
    """Create one object exactly once, reporting whether this call created it.

    The create is atomic against a concurrent caller, which is what makes it
    usable as the claim on a shared durable namespace: two workers racing to
    initialize the same manifest cannot both believe they won, and the loser
    reads the winner's bytes instead of overwriting them.

    Args:
        uri: Local path, ``file://`` URI, or ``s3://`` object URI.
        data: Complete object contents.
        storage_profile: Credential profile for the remote write.

    Returns:
        ``True`` when this call created the object, ``False`` when it already
        existed and was left untouched.

    Raises:
        ValueError: If the URI scheme has no atomic create-if-absent primitive,
            or if a local target already exists and is not a regular file, so
            no create there could ever produce the object.
        ClientError: If a remote conditional write keeps losing to a concurrent
            one until the bounded retries are spent, leaving it unknown whether
            the object exists.

    """
    if is_s3path(uri):
        return _write_s3_if_absent(uri, data, storage_profile=storage_profile)
    scheme = uri_scheme(uri)
    if scheme in LOCAL_SCHEMES:
        return _write_local_if_absent(local_media_path(uri), data)
    msg = (
        f"atomic create-if-absent is unavailable for scheme {scheme!r} ({uri}); "
        f"supported schemes are local paths, file://, and s3://"
    )
    raise ValueError(msg)


def _write_local_if_absent(path: Path, data: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A regular file is the only existing entry that answers "another caller
    # created this object". A directory, a device, or a dangling symlink at the
    # same name can never BECOME that object, so reporting it as an existing one
    # hands the caller the losing-the-race verdict and sends it on to read bytes
    # that are not there - surfacing as an errno from the read-back rather than
    # as the name that is occupied. ``is_file`` follows symlinks, so a link to a
    # real file is still an existing object.
    if path.is_file():
        return False
    if path.exists() or path.is_symlink():
        msg = f"cannot create {path} exactly once: the path exists and is not a regular file"
        raise ValueError(msg)

    while True:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        break

    try:
        with os.fdopen(descriptor, "wb") as artifact:
            artifact.write(data)
            artifact.flush()
            os.fsync(artifact.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            # A concurrent caller won the race between the exists() check above
            # and this link, so this call created nothing.
            return False
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def _is_conditional_write_conflict(error: BaseException) -> bool:
    """Return whether an S3 error reports a concurrent conditional write, not an existing object."""
    if not isinstance(error, ClientError):
        return False
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    code = error.response.get("Error", {}).get("Code")
    return status == _CONDITIONAL_REQUEST_CONFLICT or code == "ConditionalRequestConflict"


def _put_s3_object_if_absent(client: S3Client, prefix: S3Prefix, data: bytes) -> bool:
    """Issue one conditional PUT, reporting whether it created the object."""
    try:
        client.s3.put_object(Bucket=prefix.bucket, Key=prefix.prefix, Body=data, IfNoneMatch="*")
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = exc.response.get("Error", {}).get("Code")
        # 412 is the only answer that means "already there": IfNoneMatch="*" was
        # refused because the key is taken, so the object exists and this call
        # left it untouched. A 409 is NOT that answer - it reports a conflicting
        # conditional write in flight, whose outcome is still unknown, so
        # reading it as "already there" would tell a caller an object exists
        # that may never be written. It leaves here as the botocore error the
        # retry above selects on.
        if status == _PRECONDITION_FAILED or code == "PreconditionFailed":
            return False
        raise
    return True


def _write_s3_if_absent(location: str, data: bytes, *, storage_profile: str) -> bool:
    prefix = S3Prefix(location)
    client = get_storage_client(location, profile_name=storage_profile)
    if not isinstance(client, S3Client):
        msg = f"Could not create an S3 client for {location}"
        raise TypeError(msg)
    retrying = tenacity.Retrying(
        stop=tenacity.stop_after_attempt(_S3_CONFLICT_MAX_ATTEMPTS),
        wait=tenacity.wait_exponential(
            multiplier=_S3_CONFLICT_RETRY_MULTIPLIER_S,
            max=_S3_CONFLICT_RETRY_MAX_WAIT_S,
        ),
        retry=tenacity.retry_if_exception(_is_conditional_write_conflict),
        reraise=True,
    )
    return retrying(_put_s3_object_if_absent, client, prefix, data)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
