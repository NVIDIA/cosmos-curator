# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Base storage client interface for various storage systems.

This module provides a base interface for storage client implementations
(S3, Azure Blob Storage, etc.) with common operations for interacting
with cloud storage systems.
"""

import abc
import concurrent.futures
import datetime
import pathlib
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

# Constants for chunk sizes
DOWNLOAD_CHUNK_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
UPLOAD_CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def join_under(root: pathlib.Path, relative: str) -> pathlib.Path:
    """Join ``relative`` onto ``root`` and require the result stay inside ``root``.

    Used by S3 and Azure ``sync_remote_to_local`` so a key containing ``..`` or an
    absolute path cannot write outside the destination directory.

    Object-store keys have no symlink file type, unlike zip members. An already
    planted dest symlink is followed by ``resolve()`` and rejected when the
    target escapes ``root``.

    Args:
        root: Already-resolved destination directory.
        relative: Object-key remainder after the storage prefix is stripped.

    Returns:
        The resolved path under ``root``.

    Raises:
        ValueError: If the resolved path escapes ``root``.

    """
    dest = (root / relative).resolve()
    if not dest.is_relative_to(root):
        error_msg = f"Refusing to write path outside {root}: {relative}"
        raise ValueError(error_msg)
    return dest


@attrs.define
class BaseClientConfig:
    """Base configuration class for storage clients.

    Attributes:
        max_concurrent_threads (int): Maximum number of concurrent threads (default: 100).
        operation_timeout_s (int): Timeout for operations in seconds (default: 180).
        can_overwrite (bool): Whether the client can overwrite existing objects (default: False).
        can_delete (bool): Whether the client can delete objects (default: False).

    """

    max_concurrent_threads: int = attrs.field(default=100)
    operation_timeout_s: int = attrs.field(default=180)
    can_overwrite: bool = attrs.field(default=False)
    can_delete: bool = attrs.field(default=False)


@attrs.define
class StoragePrefix:
    """Base class for representing a storage path prefix.

    This is extended by specific implementations like S3Prefix and AzurePrefix.
    """

    _input: str = attrs.field()

    @property
    @abc.abstractmethod
    def path(self) -> str:
        """Return the full path for this prefix."""

    @property
    def prefix(self) -> str:
        """Return the prefix for this storage path.

        Returns:
            The prefix for this storage path.

        """
        parts = self._input.split("/", 1)
        return parts[1] if len(parts) > 1 else ""

    def __str__(self) -> str:
        """Return a string representation."""
        return self.path


@attrs.define(frozen=True)
class StorageStat:
    """What one metadata lookup says about a single stored object.

    Named for the storage layer rather than for the cloud because the same facts
    describe a local file: ``os.stat`` supplies the size and the modification time,
    and only the entity tag has no local analogue. A caller reporting progress over
    a mix of local paths and cloud URIs then needs one type instead of two.

    Every field is optional because each value is whatever the backend chose to
    report. Both supported backends do report size and last-modified for an object
    that exists, so a ``None`` means the store omitted it rather than that the
    object is absent -- a missing object raises instead of yielding an empty stat.

    Attributes:
        size_bytes (int | None): Object size as the store reports it.
        last_modified (datetime.datetime | None): Server-side modification time,
            timezone-aware.
        etag (str | None): The backend's entity tag, unquoted so a value recorded
            through one client compares cleanly against one from another. Treat it
            as an opaque change token, not a checksum: S3 returns an MD5 for a
            single-part upload but a ``<hash>-<part-count>`` composite for a
            multipart one, so identical bytes can carry different tags. A changed
            tag reliably means something happened; an unchanged tag alongside an
            unchanged size is good evidence nothing did.

    """

    size_bytes: int | None = attrs.field(default=None)
    last_modified: datetime.datetime | None = attrs.field(default=None)
    etag: str | None = attrs.field(default=None)


class BackgroundUploader(abc.ABC):
    """Abstract base class for background uploaders.

    Attributes:
        client: The storage client instance.
        chunk_size_bytes (int): The size of chunks for uploading.
        executor (ThreadPoolExecutor): The thread pool executor for background tasks.
        futures (List[Future]): List of futures for tracking upload tasks.

    """

    def __init__(self, client: object, chunk_size_bytes: int) -> None:
        """Initialize the BackgroundUploader with the given client and chunk size.

        Args:
            client: The storage client instance.
            chunk_size_bytes: The size of chunks to use for uploading.

        """
        self.client = client
        self.chunk_size_bytes = chunk_size_bytes
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.futures: list[concurrent.futures.Future[None]] = []

    @abc.abstractmethod
    def add_task_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        """Add a file upload task to the background uploader.

        Args:
            local_path: Path to the local file to upload.
            remote_path: Path in the storage system where the file will be uploaded.

        """

    def block_until_done(self) -> None:
        """Wait for all background tasks to complete."""
        concurrent.futures.wait(self.futures)
        self.executor.shutdown(wait=True)


class StorageClient(abc.ABC):
    """Abstract base class for storage clients.

    This class defines the interface for storage client operations such as checking object
    existence, uploading and downloading objects, and listing objects.
    """

    @abc.abstractmethod
    def stat(self, dest: StoragePrefix) -> StorageStat:
        """Return what one metadata lookup says about the object at ``dest``.

        Follows ``os.stat`` semantics: a missing object is an error rather than an
        empty result, so a caller that wants the metadata never has to guess whether
        an all-``None`` answer means "absent" or "the store said little". Callers
        that only want the boolean use :meth:`object_exists`.

        Implementations must NOT wrap the lookup in ``do_with_retries``. The read
        paths in this package back off for up to 256 seconds, which is right for a
        transfer that has to succeed but wrong here: absence is a normal answer for
        this call, so a progress display walking a session's objects would stall for
        minutes on the first one that is legitimately not there.

        Args:
            dest: The storage prefix of the object to describe.

        Returns:
            The metadata the backend reported for the object.

        Raises:
            FileNotFoundError: If no object exists at ``dest``. Matches
                ``s3_client.list_child_prefixes``, which reports an absent prefix
                the same way.
            TypeError: If ``dest`` is not the prefix type this client handles, such
                as an ``AzurePrefix`` handed to an ``S3Client``.

        """

    def object_exists(self, dest: StoragePrefix) -> bool:
        """Check if an object exists at the specified path.

        Defined in terms of :meth:`stat`, as ``pathlib.Path.exists`` is defined in
        terms of ``Path.stat``: one lookup answers both questions, so a caller that
        also wants the size pays for a single round trip rather than two.

        Only "not found" is translated. Anything else -- the 403 a bucket the
        credentials cannot read answers with, a connection failure -- propagates,
        because reporting those as "does not exist" turns a permissions problem into
        silently missing data.

        Args:
            dest: The storage prefix of the object to check.

        Returns:
            bool: True if the object exists, False otherwise.

        """
        try:
            self.stat(dest)
        except FileNotFoundError:
            return False
        return True

    @abc.abstractmethod
    def upload_bytes(self, dest: StoragePrefix, data: "bytes | npt.NDArray[np.uint8]") -> None:
        """Upload binary data to the specified storage path.

        Accepts ``bytes`` or ``numpy.ndarray[uint8]``.  Implementations
        wrap the data in ``io.BytesIO`` for streaming upload, reading
        directly from the buffer protocol - no separate ``.tobytes()``
        allocation needed.

        Args:
            dest: The storage prefix where the object will be stored.
            data: Binary data to upload (bytes or uint8 numpy array).

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """

    @abc.abstractmethod
    def upload_bytes_uri(self, uri: str, data: bytes, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> None:
        """Upload bytes data to the specified URI.

        Args:
            uri: The URI where the object will be stored.
            data: The bytes data to upload.
            chunk_size_bytes: The size of chunks to use for uploading.

        """

    def download_to_path(
        self, uri: StoragePrefix, dest: str | pathlib.Path, chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES
    ) -> None:
        """Stream an object to a local file without loading it fully into memory.

        The default implementation falls back to ``download_object_as_bytes``; subclasses
        should override to use backend-native streaming downloads (e.g. boto3
        ``download_fileobj``) so that the full file is never held in memory.

        Args:
            uri: The storage prefix of the object to download.
            dest: Local file path to write to.
            chunk_size_bytes: Hint for internal streaming chunk size.

        """
        dest = pathlib.Path(dest)
        partial = dest.with_suffix(dest.suffix + ".partial")
        try:
            partial.write_bytes(self.download_object_as_bytes(uri, chunk_size_bytes=chunk_size_bytes))
            partial.replace(dest)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    @abc.abstractmethod
    def download_object_as_bytes(self, uri: StoragePrefix, chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES) -> bytes:
        """Download an object as bytes from the specified storage path.

        Args:
            uri: The storage prefix of the object to download.
            chunk_size_bytes: The size of chunks to use for downloading.

        Returns:
            bytes: The object's content as bytes.

        """

    @abc.abstractmethod
    def download_objects_as_bytes(self, uris: list[StoragePrefix]) -> list[bytes]:
        """Download multiple objects as bytes from the specified URIs.

        Args:
            uris: A list of URIs of the objects to download.

        Returns:
            A list of bytes containing the object contents.

        """

    @abc.abstractmethod
    def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
        """List all objects recursively, starting from the given prefix.

        Args:
            uri: The storage prefix to list objects from.
            limit: Maximum number of objects to return.

        Returns:
            A list of storage prefixes for all objects found.

        """

    @abc.abstractmethod
    def list_recursive(self, prefix: StoragePrefix, limit: int = 0) -> list[dict[str, Any]]:
        """List all objects recursively, starting from the given prefix.

        Args:
            prefix: Storage prefix to list objects from.
            limit: Maximum number of objects to return.

        Returns:
            A list of dictionaries with object metadata.

        """

    @abc.abstractmethod
    def list_recursive_with_suffixes(
        self,
        uri: StoragePrefix,
        suffixes: tuple[str, ...],
        limit: int = 0,
    ) -> list[StoragePrefix]:
        """List objects under ``uri`` whose key ends in one of ``suffixes``.

        ``limit`` counts matches rather than listed objects, which is the whole point
        of having this beside :meth:`list_recursive`. That method stops after ``limit``
        objects of any kind, so a prefix carrying one sidecar file per video hands back
        half the videos a caller asked for and reports no problem. Both the filter and
        the cap therefore belong inside the pagination loop: filtering afterwards would
        turn a request for five videos from a production prefix into a full enumeration
        of it.

        Stopping early is sound here in a way it is not for
        ``s3_client.list_child_prefixes``, whose docstring declines a limit for what
        looks like the opposite reason. That function reads a *delimited* listing whose
        callers deduplicate and sort the children afterwards, so an early exit there
        would change which children are selected. This is a flat listing returned in
        the store's own key order, so the first ``limit`` matches are the same set
        however many pages it took to find them.

        Args:
            uri: The storage prefix to list under.
            suffixes: Non-empty suffixes to match. Compared case-insensitively at both
                ends, so a caller need not pre-lowercase them.
            limit: Maximum number of matches to return; ``0`` (the default) means
                unlimited, as it does for :meth:`list_recursive`.

        Returns:
            Storage prefixes for the matching objects, in listing order.

        Raises:
            ValueError: If ``suffixes`` is empty, or contains an empty string. Both
                defeat the filter silently and in opposite directions:
                ``str.endswith(())`` is False, so an empty tuple matches nothing at
                all, while ``str.endswith("")`` is True for every key, so one stray
                empty element turns a filtered listing into the first ``limit``
                objects of any type.
            TypeError: If ``uri`` is not the prefix type this client handles.

        """

    @staticmethod
    def _lowercased_suffixes(suffixes: tuple[str, ...]) -> tuple[str, ...]:
        """Validate and normalize the suffix filter shared by every implementation.

        Lowercasing once here rather than per key keeps the comparison in the
        pagination loop to a single ``str.endswith`` against a tuple.
        """
        if not suffixes:
            error_msg = "suffixes must not be empty; an empty filter would match no objects at all"
            raise ValueError(error_msg)
        if any(not suffix for suffix in suffixes):
            error_msg = "suffixes must not contain an empty string; it would match every object, filtering nothing"
            raise ValueError(error_msg)
        return tuple(suffix.lower() for suffix in suffixes)

    @abc.abstractmethod
    def upload_file(
        self,
        local_path: str,
        remote_path: StoragePrefix,
        chunk_size: int = UPLOAD_CHUNK_SIZE_BYTES,
    ) -> None:
        """Upload a file to the specified path.

        Args:
            local_path: The local path of the file to upload.
            remote_path: The URI where the file will be uploaded.
            chunk_size: The size of chunks to use for uploading.

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """

    @abc.abstractmethod
    def sync_remote_to_local(
        self,
        remote_prefix: StoragePrefix,
        local_dir: pathlib.Path,
        *,
        delete: bool = False,
        chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES,
    ) -> None:
        """Sync contents of a remote prefix with a local directory.

        Args:
            remote_prefix: The remote prefix to sync from.
            local_dir: The local directory path to sync to.
            delete: If True, delete local files that don't exist in the remote prefix.
            chunk_size_bytes: The size of chunks to use for downloading.

        """

    @abc.abstractmethod
    def make_background_uploader(self, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> BackgroundUploader:
        """Create and return a BackgroundUploader instance.

        Args:
            chunk_size_bytes: The size of chunks to use for uploading.

        Returns:
            An initialized BackgroundUploader instance.

        """

    @abc.abstractmethod
    def delete_object(self, dest: StoragePrefix) -> None:
        """Delete an object at the specified path.

        Args:
            dest: The storage prefix of the object to delete.

        Raises:
            ValueError: If deletion is not allowed by the client configuration.

        """


def is_storage_path(path: str | None, protocol: str) -> bool:
    """Check if a path string is a storage path with the given protocol.

    Args:
        path: The path to check.
        protocol: The protocol to check for (e.g., "s3", "azure").

    Returns:
        bool: True if the path is a storage path with the given protocol, False otherwise.

    """
    if path is None:
        return False
    return path.startswith(f"{protocol}://")
