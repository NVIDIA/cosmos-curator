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

"""Storage Utilities."""

import contextlib
import hashlib
import json
import os
import pathlib
import re
import shutil
import tempfile
from collections.abc import Generator, Mapping
from typing import IO, Any

import attrs
import botocore.exceptions
from loguru import logger

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage import azure_client, s3_client
from cosmos_curator.core.utils.storage.azure_client import AzurePrefix, is_azure_path
from cosmos_curator.core.utils.storage.s3_client import S3Prefix, is_s3path
from cosmos_curator.core.utils.storage.storage_client import StorageClient, StoragePrefix


def is_remote_path(path: str | None) -> bool:
    """Check if a path is on a remote storage system (S3 or Azure)."""
    if path is None:
        return False
    return is_s3path(path) or is_azure_path(path)


def get_storage_client(
    target_path: str,
    *,
    profile_name: str = "default",
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> StorageClient | None:
    """Get the appropriate storage client for a path (S3 or Azure).

    Args:
        target_path: The path to get a client for.
        profile_name: The profile name to use.
        can_overwrite: Whether the client can overwrite existing objects.
        can_delete: Whether the client can delete objects.

    Returns:
        A storage client instance if the path is a remote path, None otherwise.

    """
    if is_s3path(target_path):
        return s3_client.create_s3_client(
            target_path,
            profile_name,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    if is_azure_path(target_path):
        return azure_client.create_azure_client(
            target_path,
            profile_name,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    return None


def get_lance_storage_options(path: str, *, profile_name: str = "default") -> dict[str, str] | None:
    """Build storage options for Lance based on configured profiles."""
    if is_s3path(path):
        s3_cfg = s3_client.get_s3_client_config(profile_name, can_overwrite=True)
        options = {
            k: v
            for k, v in {
                "aws_access_key_id": s3_cfg.aws_access_key_id,
                "aws_secret_access_key": s3_cfg.aws_secret_access_key,
                "aws_session_token": s3_cfg.aws_session_token,
                "aws_region": s3_cfg.region,
                "aws_endpoint": s3_cfg.endpoint_url,
            }.items()
            if v
        }
        return options or None

    if is_azure_path(path):
        azure_cfg = azure_client.get_azure_client_config(profile_name=profile_name, can_overwrite=True)
        options = {
            k: v
            for k, v in {
                "account_name": azure_cfg.account_name,
                "account_key": azure_cfg.account_key,
                "account_url": azure_cfg.account_url,
                "connection_string": azure_cfg.connection_string,
            }.items()
            if v
        }
        return options or None

    return None


def path_to_prefix(path: str) -> StoragePrefix:
    """Convert a path string to the appropriate storage prefix object.

    Args:
        path: The path to convert.

    Returns:
        A StoragePrefix instance for the path.

    Raises:
        ValueError: If the path is not a valid remote storage path.

    """
    if is_s3path(path):
        return S3Prefix(path)
    if is_azure_path(path):
        return AzurePrefix(path)
    error_msg = f"Path {path} is not a valid remote storage path"
    raise ValueError(error_msg)


def read_bytes(
    filepath: StoragePrefix | pathlib.Path | str,
    client: StorageClient | None = None,
    max_attempts: int = 5,
) -> bytes:
    """Read bytes from a file, whether local or on a remote storage system.

    Args:
        filepath: The path to the file to read.
        client: The storage client to use for remote paths.
        max_attempts: The maximum number of attempts for reading from remote storage.

    Returns:
        The file contents as bytes.

    """
    # Convert string paths to the appropriate type
    if isinstance(filepath, str):
        filepath = path_to_prefix(filepath) if is_remote_path(filepath) else pathlib.Path(filepath)

    # Handle remote storage paths
    if isinstance(filepath, StoragePrefix):
        if client is None:
            client = get_storage_client(str(filepath))

        def func_to_call() -> bytes:
            assert client is not None
            return client.download_object_as_bytes(filepath)

        return do_with_retries(func_to_call, max_attempts=max_attempts, backoff_factor=4.0, max_wait_time_s=256.0)
    # Handle local paths
    with filepath.open("rb") as fp:
        return fp.read()


def read_text(
    filepath: StoragePrefix | pathlib.Path | str,
    client: StorageClient | None = None,
    max_attempts: int = 5,
) -> str:
    """Read text from a file, whether local or on a remote storage system.

    Args:
        filepath: The path to the file to read.
        client: The storage client to use for remote paths.
        max_attempts: The maximum number of attempts for reading from remote storage.

    Returns:
        The file contents as a string.

    """
    return read_bytes(filepath, client, max_attempts).decode("utf-8")


def read_json_file(
    filepath: StoragePrefix | pathlib.Path | str,
    client: StorageClient | None = None,
    max_attempts: int = 5,
) -> dict[Any, Any]:
    """Read a JSON file, whether local or on a remote storage system.

    Args:
        filepath: The path to the file to read.
        client: The storage client to use for remote paths.
        max_attempts: The maximum number of attempts for reading from remote storage.

    Returns:
        The parsed JSON content.

    """
    # Convert string paths to the appropriate type
    if isinstance(filepath, str):
        filepath = path_to_prefix(filepath) if is_remote_path(filepath) else pathlib.Path(filepath)

    if isinstance(filepath, StoragePrefix):
        buffer = read_bytes(filepath, client, max_attempts)
        return json.loads(buffer.decode("utf-8"))  # type: ignore[no-any-return]
    with filepath.open() as fp:
        return json.load(fp)  # type: ignore[no-any-return]


def path_exists(
    path: str | StoragePrefix | pathlib.Path,
    client: StorageClient | None = None,
) -> bool:
    """Check if a path exists, whether local or on a remote storage system.

    Args:
        path: The path to check.
        client: The storage client to use for remote paths.

    Returns:
        True if the path exists, False otherwise.

    """
    # Convert string paths to the appropriate type
    if isinstance(path, str):
        path = path_to_prefix(path) if is_remote_path(path) else pathlib.Path(path)

    if isinstance(path, StoragePrefix):
        if client is None:
            client = get_storage_client(str(path))
        assert client is not None
        return client.object_exists(path)
    return pathlib.Path(path).exists()


def verify_path(path: str, level: int = 0) -> None:
    """Verify that a path exists.

    Args:
        path: The path to verify.
        level: For local paths, how many levels up to check.

    Raises:
        FileNotFoundError: If the path does not exist.

    """
    if not is_remote_path(path):
        path_to_check = pathlib.Path(path)
        while level > 0:
            path_to_check = path_to_check.parent
            level -= 1
        if not pathlib.Path(path_to_check).exists():
            error_msg = f"Local path {path_to_check} does not exist."
            raise FileNotFoundError(error_msg)


def create_path(path: str) -> None:
    """Create a path if it does not exist.

    Args:
        path: The path to create.

    """
    if not is_remote_path(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def is_path_nested(path1: str, path2: str) -> bool:
    """Check if one path is nested inside another.

    Args:
        path1: The first path.
        path2: The second path.

    Returns:
        True if one path is nested inside the other, False otherwise.

    """
    _path1 = path1.rstrip("/") + "/"
    _path2 = path2.rstrip("/") + "/"
    return _path1.startswith(_path2) or _path2.startswith(_path1)


def get_full_path(path: str | StoragePrefix | pathlib.Path, *args: str) -> StoragePrefix | pathlib.Path:
    """Construct a full path from a base path and additional components.

    Args:
        path: The base path.
        *args: Additional path components.

    Returns:
        The full path as a StoragePrefix or Path object.

    """
    # Convert string paths to the appropriate type
    if isinstance(path, str):
        if is_remote_path(path):
            if is_s3path(path):
                return _get_s3_prefix(path, *args)
            if is_azure_path(path):
                return _get_azure_prefix(path, *args)
        else:
            return _get_local_path(pathlib.Path(path), *args)
    elif isinstance(path, StoragePrefix):
        if isinstance(path, S3Prefix):
            return _get_s3_prefix(path, *args)
        if isinstance(path, AzurePrefix):
            return _get_azure_prefix(path, *args)
    assert isinstance(path, pathlib.Path)
    return _get_local_path(path, *args)


def _get_s3_prefix(s3path: str | S3Prefix, *args: str) -> S3Prefix:
    """Construct a full S3 path from a base path and additional components.

    Args:
        s3path: The base S3 path.
        *args: Additional path components.

    Returns:
        The full S3 path as an S3Prefix object.

    """
    prefix = [str(s3path).rstrip("/")]
    if args:
        prefix += [x.strip("/") for x in args]
    return S3Prefix("/".join(prefix))


def _get_azure_prefix(azurepath: str | AzurePrefix, *args: str) -> AzurePrefix:
    """Construct a full Azure path from a base path and additional components.

    Args:
        azurepath: The base Azure path.
        *args: Additional path components.

    Returns:
        The full Azure path as an AzurePrefix object.

    """
    prefix = [str(azurepath).rstrip("/")]
    if args:
        prefix += [x.strip("/") for x in args]
    return AzurePrefix("/".join(prefix))


def _get_local_path(localpath: pathlib.Path, *args: str) -> pathlib.Path:
    """Construct a full local path from a base path and additional components.

    Args:
        localpath: The base local path.
        *args: Additional path components.

    Returns:
        The full local path as a Path object.

    """
    return pathlib.Path(localpath, *args)


def _get_objects_from_storage_prefix_relative(
    prefix: StoragePrefix,
    client: StorageClient,
    limit: int = 0,
) -> list[str]:
    """Get a list of objects from a storage prefix, with paths relative to the prefix.

    Args:
        prefix: The storage prefix to list objects from.
        client: The storage client to use.
        limit: Max numbers to iterate

    Returns:
        A list of object paths relative to the prefix.

    """

    def func_to_call() -> list[StoragePrefix]:
        return client.list_recursive_directory(prefix, limit)

    objects = do_with_retries(func_to_call)
    prefix_path = str(prefix).rstrip("/") + "/"
    objects_relative = []
    for x in objects:
        if not str(x).startswith(prefix_path):
            continue
        if str(x).endswith("/"):
            logger.warning(f"Skipping empty prefix {x}")
            continue
        objects_relative.append(str(x).replace(prefix_path, ""))
    objects_relative = sorted(objects_relative)
    # Defensive clamp: some storage clients may over-return despite a limit.
    # Keep caller-visible semantics stable by enforcing the cap here as well.
    if limit > 0:
        objects_relative = objects_relative[:limit]
    return objects_relative


def _get_files_from_localpath_relative(localpath: pathlib.Path, limit: int = 0) -> list[str]:
    """Get a list of files from a local path, with paths relative to the local path.

    Args:
        localpath: The local path to list files from.
        limit: Max numbers to iterate; 0 means no limit.

    Returns:
        A list of file paths relative to the local path.

    """
    files = sorted([str(x.relative_to(localpath)) for x in localpath.rglob("*") if x.is_file()])
    if limit > 0:
        files = files[:limit]
    return files


def get_files_relative(path: str, client: StorageClient | None = None, limit: int = 0) -> list[str]:
    """Get a list of files from a path, with paths relative to the base path.

    Args:
        path: The path to list files from.
        client: The storage client to use for remote paths.
        limit: Max numbers to iterate

    Returns:
        A list of file paths relative to the base path.

    """
    if is_remote_path(path):
        prefix = path_to_prefix(path)
        if client is None:
            client = get_storage_client(path)
        assert client is not None
        return _get_objects_from_storage_prefix_relative(prefix, client, limit)
    return _get_files_from_localpath_relative(pathlib.Path(path), limit)


def get_directories_relative(path: str, client: StorageClient | None = None) -> list[str]:
    """Get a list of top-level directories from a path.

    Args:
        path: The path to list directories from.
        client: The storage client to use for remote paths.

    Returns:
        A list of directory names at the top level of the path.

    """
    objects = get_files_relative(path, client)
    dirs = set()
    for obj in objects:
        dirs.add(obj.split("/")[0])
    return sorted(dirs)


def get_next_file(
    filename: str,
    filetype: str,
    output_path: str,
    client: StorageClient | None = None,
) -> StoragePrefix | pathlib.Path:
    """Get the path for the next file in a sequence.

    Args:
        filename: The base filename.
        filetype: The file extension.
        output_path: The output directory.
        client: The storage client to use for remote paths.

    Returns:
        The path for the next file in the sequence.

    """
    next_index = 0
    existing_files = get_files_relative(output_path, client)
    for item in existing_files:
        match = re.match(rf"{filename}_(\d+)\.{filetype}", item)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)
    next_file_name = f"{filename}_{next_index}.{filetype}"
    return get_full_path(output_path, next_file_name)


def backup_file(  # noqa: C901
    dest: StoragePrefix | pathlib.Path | str,
    client: StorageClient | None = None,
) -> None:
    """Create a backup of a file.

    Args:
        dest: The path to the file to back up.
        client: The storage client to use for remote paths.

    """
    # Convert string paths to the appropriate type
    if isinstance(dest, str):
        dest = path_to_prefix(dest) if is_remote_path(dest) else pathlib.Path(dest)

    # Handle remote storage paths
    if isinstance(dest, StoragePrefix):
        if client is None:
            client = get_storage_client(str(dest))

        assert client is not None
        if client.object_exists(dest):

            def func_to_call() -> None:
                idx = 1
                data = client.download_object_as_bytes(dest)
                bak_dest: StoragePrefix | None = None
                while True:
                    if isinstance(dest, S3Prefix):
                        bak_dest = S3Prefix(str(dest) + f".bak{idx}")
                    elif isinstance(dest, AzurePrefix):  # AzurePrefix
                        bak_dest = AzurePrefix(str(dest) + f".bak{idx}")
                    assert bak_dest is not None
                    if not client.object_exists(bak_dest):
                        break
                    idx += 1
                client.upload_bytes(bak_dest, data)

            do_with_retries(func_to_call)
    # Handle local paths
    elif dest.exists():
        idx = 1
        while True:
            bak_dest = dest.with_suffix(f"{dest.suffix}.bak{idx}")
            if not bak_dest.exists():
                break
            idx += 1
        dest.rename(bak_dest)


def is_parquet_file(filename: str) -> bool:
    """Check if a file is a parquet file."""
    return filename.lower().endswith(".parquet")


def extract_parquet_files(
    input_path: str,
    profile_name: str,
    limit: int = 0,
    *,
    verbose: bool = False,
) -> list[StoragePrefix | pathlib.Path]:
    """List parquet files under an input path.

    Args:
        input_path: Base path to search (local or remote like s3://...).
        profile_name: Profile name to use when accessing remote storage.
        limit: Optional maximum number of parquet files to return; 0 means no limit.
        verbose: If True, log each discovered parquet file.

    Returns:
        A list of fully-qualified paths (StoragePrefix for remote, Path for local) to parquet files.

    """
    client_input = get_storage_client(input_path, profile_name=profile_name)

    # List all files relative to the base path then filter parquet files
    all_items = get_files_relative(input_path, client_input)
    parquet_items = [item for item in all_items if is_parquet_file(item)]

    if limit > 0:
        parquet_items = parquet_items[:limit]

    if verbose:
        logger.debug(f"Found {len(parquet_items)} parquet files under {input_path}")
        for item in parquet_items:
            logger.debug(item)

    return [get_full_path(input_path, item) for item in parquet_items]


@attrs.define(frozen=True)
class WritablePath(os.PathLike[str]):
    """Writable local path returned by :meth:`StorageWriter.resolve_path`.

    Implements ``os.PathLike[str]`` so it can be passed directly to any
    API that accepts a file path (``open()``, ``str()``, etc.).  Call
    :meth:`close` when writing is complete to finalize the file.

    For **local** destinations :meth:`close` is a no-op (the file is
    already at its final location).  For **remote** destinations it
    uploads the file and removes the local staging copy.

    Attributes:
        local: The underlying ``pathlib.Path`` on the local filesystem.
        writer: Back-reference to the owning :class:`StorageWriter`.
        sub: The relative sub-path used by :meth:`StorageWriter.close`.

    """

    local: pathlib.Path
    writer: "StorageWriter"
    sub: str

    # -- os.PathLike protocol --------------------------------------------------

    def __fspath__(self) -> str:
        """Return the filesystem path as a string."""
        return os.fspath(self.local)

    def __str__(self) -> str:
        """Return the string representation of the local path."""
        return str(self.local)

    # -- Proxied pathlib.Path operations ---------------------------------------

    def open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        """Open the local file (delegates to ``pathlib.Path.open``)."""
        return self.local.open(mode, buffering, encoding, errors, newline)

    def exists(self) -> bool:
        """Return ``True`` if the local file exists."""
        return self.local.exists()

    # -- Lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Finalize: upload to remote if needed, no-op for local.

        Delegates to :meth:`StorageWriter.close` with the sub-path
        that was used to create this ``WritablePath``.
        """
        self.writer.close(self.sub)


class StorageWriter:
    """Unified write abstraction for local and remote (S3/Azure) storage.

    Provides a single interface for writing data to local filesystem,
    S3, or Azure storage.  Resolves the storage backend once at
    construction time and reuses it for all subsequent writes.

    **When to use StorageWriter:**

    Use ``StorageWriter`` whenever code needs to persist files to a
    destination path that may be local *or* remote.  The writer is
    the **abstraction boundary** between your code and the storage
    backend -- callers should never inspect the destination path to
    decide how to write.  Pass the path to ``StorageWriter`` and let
    it resolve the backend transparently.

    ::

        +--------------------+
        |   Consumer code    |  <-- knows nothing about S3/Azure/local
        +--------------------+
                 |
                 v
        +--------------------+
        |   StorageWriter    |  <-- resolves backend once at construction
        +--------------------+
           /         |         \
          v          v          v
       local       S3        Azure

    **When NOT to use StorageWriter:**

    * When the destination is always a local ``pathlib.Path`` by
      contract (e.g. a temp staging dir used within a single
      function).  Use ``pathlib.Path`` directly in that case.
    * When you need *read* access.  ``StorageWriter`` is write-only;
      contributions to add read support are welcome!

    **Anti-pattern -- do NOT do this:**

    .. code-block:: python

        # WRONG: consumer inspects the path to branch on local vs remote.
        # This leaks storage internals into the consumer layer.
        if is_remote_path(dest):
            upload_to_s3(dest, data)
        else:
            pathlib.Path(dest).write_bytes(data)

        # CORRECT: let StorageWriter handle it.
        writer = StorageWriter(dest)
        writer.write(data)

    **Two usage patterns:**

    1. **Single-file** -- *base_path* is the full file path.  Use
       :meth:`write` / :meth:`write_str` to write directly.
    2. **Directory** -- *base_path* is a directory.  Use
       :meth:`write_bytes_to` / :meth:`write_str_to` to write
       multiple files under that directory, each identified by a
       *sub_path*.

    The ``_to`` suffix signals "a destination sub_path follows" so
    the two patterns are visually distinct at every call site.

    **Which method to use:**

    * :meth:`write` -- single-file pattern, payload as ``bytes``.
    * :meth:`write_str` -- single-file pattern, payload as ``str``.
    * :meth:`write_bytes_to` -- directory pattern, payload as ``bytes``,
      written to ``<base_path>/<sub_path>``.
    * :meth:`write_str_to` -- directory pattern, payload as ``str``,
      written to ``<base_path>/<sub_path>``.
    * :meth:`upload_file_to` -- directory pattern, data already on
      disk as a local file.  Streams without loading the full file
      into memory (handles multi-GB artifacts with constant memory).
    * :meth:`open_writer` -- a library accepts a **writable file-like
      object** (e.g. an ``outfile`` parameter).  Streams to disk and
      closes on context exit.
    * :meth:`resolve_path` -- consumer code requires a **local file
      path** (e.g. a library, module, or API that writes to disk).
      Returns a :class:`WritablePath` (``os.PathLike[str]``) that can
      be used like a regular path.  Call ``path.close()`` when done.

    Example -- single-file pattern::

        writer = StorageWriter("s3://bucket/output/report.bin")
        writer.write(raw_bytes)
        writer.write_str(json_text)

    Example -- directory pattern::

        writer = StorageWriter("s3://bucket/output/profiles")

        # You have bytes/str in memory:
        writer.write_bytes_to("cpu/stage_1.html", html_bytes)
        writer.write_str_to("timeline/trace.json", json_text)

        # A renderer accepts a file-like outfile:
        with writer.open_writer("memory/stage_1.html") as buf:
            reporter.render(outfile=buf, ...)  # closed automatically

        # Consumer code requires a local file path:
        path = writer.resolve_path("memory/stage.bin")
        tracker.start(str(path))
        ...  # write spans start/stop lifecycle
        path.close()  # uploads if remote, no-op if local

    Args:
        base_path: Root directory (local path, ``s3://...``, or ``az://...``).
        profile_name: Named credential profile for S3/Azure access.
        tmp_dir: Staging directory for local files when *base_path* is
            remote.  When *None* (default) the system temp directory is
            used.  Set explicitly to a volume with enough space for very
            large files (1 TB+) that may exceed ``/tmp`` capacity.
            Ignored when *base_path* is a local path.

    """

    _UPLOAD_MAX_ATTEMPTS: int = 3
    _UPLOAD_BACKOFF_FACTOR: float = 16.0
    _UPLOAD_MAX_WAIT_S: float = 256.0

    def __init__(
        self,
        base_path: str,
        *,
        profile_name: str = "default",
        tmp_dir: str | pathlib.Path | None = None,
    ) -> None:
        """Initialize the writer, resolving the storage backend once.

        Args:
            base_path: Root directory (local, ``s3://...``, or ``az://...``).
            profile_name: Named credential profile for remote access.
            tmp_dir: Staging directory for local files when *base_path*
                is remote.  When *None* the system temp directory is
                used.  Ignored for local *base_path* values.

        """
        self._base_path = base_path
        self._client: StorageClient | None = get_storage_client(
            base_path,
            profile_name=profile_name,
            can_overwrite=True,
        )

        # Staging directory for remote destinations.  For local
        # destinations we write directly to the final path, so no
        # staging is needed.
        self._tmp_dir: pathlib.Path | None = pathlib.Path(tmp_dir) if tmp_dir is not None else None

    @property
    def base_path(self) -> str:
        """Return the base path this writer was constructed with."""
        return self._base_path

    @property
    def is_remote(self) -> bool:
        """Return True when writing to a remote backend (S3 / Azure)."""
        return self._client is not None

    def _resolve_local(self, sub_path: str | None = None) -> pathlib.Path:
        """Resolve to a local ``pathlib.Path`` and ensure parents exist.

        Only valid when :pyattr:`is_remote` is False.  Creates parent
        directories as needed so the caller can write directly.

        Args:
            sub_path: Relative path under *base_path*.  When *None*,
                resolves to *base_path* itself (single-file pattern).

        """
        base = pathlib.Path(self._base_path)
        dest = base / sub_path if sub_path is not None else base
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    def _resolve_remote(self, sub_path: str | None = None) -> StoragePrefix:
        """Resolve to a ``StoragePrefix``.

        Only valid when :pyattr:`is_remote` is True.  Narrows the
        return type for mypy so callers can pass the result directly
        to ``StorageClient`` methods without casting.

        Args:
            sub_path: Relative path under *base_path*.  When *None*,
                resolves to *base_path* itself (single-file pattern).

        """
        dest = get_full_path(self._base_path, sub_path) if sub_path is not None else get_full_path(self._base_path)
        if not isinstance(dest, StoragePrefix):
            msg = f"Expected remote path, got {type(dest)}: {dest}"
            raise TypeError(msg)
        return dest

    def _upload_file(self, sub_path: str, local_path: pathlib.Path) -> None:
        """Upload a local file to the remote ``<base_path>/<sub_path>``.

        Private helper used by :meth:`close`.  Streams via
        ``StorageClient.upload_file()`` which performs multipart upload
        on S3 and streaming blob on Azure, so arbitrarily large files
        (1 TB+) are handled without loading the full content into
        memory.

        Args:
            sub_path: Relative path under *base_path*.
            local_path: Local file to upload.

        """
        dest = self._resolve_remote(sub_path)

        def _upload() -> None:
            self._client.upload_file(str(local_path), dest)  # type: ignore[union-attr]

        do_with_retries(
            _upload,
            max_attempts=self._UPLOAD_MAX_ATTEMPTS,
            backoff_factor=self._UPLOAD_BACKOFF_FACTOR,
            max_wait_time_s=self._UPLOAD_MAX_WAIT_S,
        )

    def _upload_bytes(self, dest: StoragePrefix, data: bytes) -> None:
        """Upload *data* to a remote *dest* with automatic retries.

        Private helper that wraps ``StorageClient.upload_bytes`` in
        ``do_with_retries``.  Used by :meth:`write` and
        :meth:`write_bytes_to` to avoid duplicating the retry closure.
        """

        def _upload() -> None:
            self._client.upload_bytes(dest, data)  # type: ignore[union-attr]

        do_with_retries(
            _upload,
            max_attempts=self._UPLOAD_MAX_ATTEMPTS,
            backoff_factor=self._UPLOAD_BACKOFF_FACTOR,
            max_wait_time_s=self._UPLOAD_MAX_WAIT_S,
        )

    def write(self, data: bytes) -> None:
        """Write raw bytes directly to *base_path*.

        Use when the writer was constructed with a full file path
        (single-file pattern)::

            writer = StorageWriter("s3://bucket/output/report.bin")
            writer.write(raw_bytes)

        For writing multiple files under a shared directory, use
        :meth:`write_bytes_to` instead.

        For remote destinations the upload is retried automatically.
        For local destinations parent directories are created as needed.

        Args:
            data: Raw bytes to write.

        """
        if not self.is_remote:
            self._resolve_local().write_bytes(data)
            return
        self._upload_bytes(self._resolve_remote(), data)

    def write_str(self, text: str, encoding: str = "utf-8") -> None:
        """Write a string directly to *base_path*.

        Use when the writer was constructed with a full file path
        (single-file pattern)::

            writer = StorageWriter("s3://bucket/output/report.json")
            writer.write_str(json_text)

        Convenience wrapper: encodes *text* with *encoding* and
        delegates to :meth:`write`.

        For writing multiple files under a shared directory, use
        :meth:`write_str_to` instead.

        Args:
            text: Text content to write.
            encoding: Character encoding (default ``"utf-8"``).

        """
        self.write(text.encode(encoding))

    def write_bytes_to(self, sub_path: str, data: bytes) -> None:
        """Write raw bytes to ``<base_path>/<sub_path>``.

        Use when writing multiple files under a shared *base_path*
        directory and you already have the complete payload as ``bytes``
        in memory.  The entire *data* buffer is held in memory for the
        duration of the upload.  For large files prefer
        :meth:`resolve_path` which streams without full buffering.

        For a single-file write (no *sub_path*), use :meth:`write`
        instead.

        For remote destinations the upload is retried automatically.
        For local destinations parent directories are created as needed.

        Args:
            sub_path: Relative path under *base_path* (e.g. ``"cpu/stage_1.html"``).
            data: Raw bytes to write.

        """
        if not self.is_remote:
            self._resolve_local(sub_path).write_bytes(data)
            return
        self._upload_bytes(self._resolve_remote(sub_path), data)

    def write_str_to(self, sub_path: str, text: str, encoding: str = "utf-8") -> None:
        """Write a string to ``<base_path>/<sub_path>``.

        Use when writing multiple files under a shared *base_path*
        directory and you already have the complete payload as a
        ``str`` (e.g. JSON, HTML, CSV that is already serialized).
        Convenience wrapper: encodes *text* with *encoding* and
        delegates to :meth:`write_bytes_to`.

        For a single-file write (no *sub_path*), use :meth:`write_str`
        instead.

        Args:
            sub_path: Relative path under *base_path*.
            text: Text content to write.
            encoding: Character encoding (default ``"utf-8"``).

        """
        self.write_bytes_to(sub_path, text.encode(encoding))

    def upload_file_to(self, sub_path: str, local_path: pathlib.Path) -> None:
        """Stream a local file to ``<base_path>/<sub_path>`` without buffering.

        Use when the data already exists as a local file and loading it
        entirely into memory via :meth:`write_bytes_to` would be
        prohibitively expensive (multi-GB artifacts).

        For remote destinations this delegates to
        ``StorageClient.upload_file()`` which performs multipart upload
        (S3) or streaming blob upload (Azure) with constant memory.

        For local destinations this uses ``shutil.copy2`` to stream
        file-to-file with a fixed OS buffer.

        Args:
            sub_path: Relative path under *base_path*.
            local_path: Path to the local source file.

        """
        if not self.is_remote:
            dest = self._resolve_local(sub_path)
            shutil.copy2(local_path, dest)
            return
        self._upload_file(sub_path, local_path)

    @contextlib.contextmanager
    def open_writer(
        self,
        sub_path: str,
        *,
        mode: str = "w",
        encoding: str | None = "utf-8",
    ) -> Generator[IO[Any], None, None]:
        """Context manager: write to a file-like object, close on exit.

        Resolves a local path via :meth:`resolve_path`, opens it for
        writing, and yields the file handle.  On normal context exit
        the file is closed and uploaded via :meth:`close` (no-op for
        local destinations).

        On exception the file handle is closed but the remote upload
        is skipped.  The staging file is intentionally **not** deleted
        so it remains available for debugging or retry.  The next
        :meth:`resolve_path` call for the same *sub_path* will unlink
        it automatically, so there is no permanent leak.

        Use when a library accepts a writable file-like object (e.g.
        an ``outfile`` parameter) rather than returning a string.

        Example -- text mode (default)::

            with writer.open_writer("memory/stage_1.html") as f:
                reporter.render(outfile=f, ...)

        Example -- binary mode::

            with writer.open_writer("data/chunk.bin", mode="wb") as f:
                f.write(raw_bytes)

        Args:
            sub_path: Relative path under *base_path*.
            mode: File open mode (default ``"w"``).  Use ``"wb"`` for
                binary writes.  Must be a write mode (e.g. ``"w"``,
                ``"wb"``, ``"wt"``).
            encoding: Character encoding (default ``"utf-8"``).
                Ignored when *mode* is binary (contains ``"b"``).

        Yields:
            A writable file handle (:class:`typing.TextIO` for text
            modes, :class:`typing.BinaryIO` for binary modes).

        """
        wpath = self.resolve_path(sub_path)
        open_kwargs: dict[str, Any] = {}
        if "b" not in mode:
            open_kwargs["encoding"] = encoding
        with wpath.open(mode, **open_kwargs) as f:
            yield f
        # Intentionally outside try/finally: on exception the upload
        # is skipped and the staging file is preserved for
        # debugging/retry.
        wpath.close()

    def _staging_root(self) -> pathlib.Path:
        """Return the staging root directory for remote destinations.

        Uses *tmp_dir* (if provided at construction) or falls back to
        the system temp directory.  A short hash of *base_path* is
        appended so that two writers targeting different remote
        destinations never collide on the same staging file when they
        share a *sub_path*.
        """
        root = self._tmp_dir if self._tmp_dir is not None else pathlib.Path(tempfile.gettempdir())
        tag = hashlib.sha256(self._base_path.encode()).hexdigest()[:12]
        return root / f"cosmos_staging_{tag}"

    def resolve_path(self, sub_path: str) -> WritablePath:
        """Return a writable :class:`WritablePath` for *sub_path*.

        Use when consumer code requires a **local file path** up front
        and the write may span multiple method calls (e.g. a library
        that is opened once and written to across start/stop calls, or
        a streaming file that is appended to over time).

        The returned :class:`WritablePath` implements
        ``os.PathLike[str]`` so it can be passed directly to any API
        accepting a file path.  Call :meth:`WritablePath.close` when
        done writing to finalize the file (uploads to remote if
        needed, no-op for local).

        For **local** destinations the underlying path is the final
        destination (``<base_path>/<sub_path>``).  For **remote**
        destinations it is a staging path under the staging directory
        (``<staging_root>/<sub_path>``), preserving the directory
        structure for easy inspection.

        Parent directories are created as needed.  Any pre-existing
        file at the path is removed to guarantee a **fresh** path
        (safe for tools that refuse to overwrite existing files).

        Example::

            path = writer.resolve_path("data/output.bin")
            tracker.start(str(path))
            ...
            path.close()

        Args:
            sub_path: Relative path under *base_path*.

        Returns:
            A :class:`WritablePath` wrapping a local ``pathlib.Path``.

        """
        logger.debug(f"StorageWriter.resolve_path: sub_path={sub_path!r}, remote={self.is_remote}")

        if not self.is_remote:
            dest = self._resolve_local(sub_path)
            # Guarantee a fresh (non-existing) file so tools that
            # refuse to overwrite existing files work without
            # pre-cleanup from the caller.
            dest.unlink(missing_ok=True)
            return WritablePath(local=dest, writer=self, sub=sub_path)

        # Remote destination -- stage under the staging directory,
        # mirroring the sub_path structure for easy debugging.
        staging = self._staging_root() / sub_path
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.unlink(missing_ok=True)
        return WritablePath(local=staging, writer=self, sub=sub_path)

    def close(self, sub_path: str) -> None:
        """Close (finalize) the file at *sub_path*.

        For **local** destinations this is a no-op (the file is
        already at its final path).  For **remote** destinations the
        staging file (produced by :meth:`resolve_path`) is uploaded
        via ``StorageClient.upload_file()`` with automatic retries,
        then the local staging copy is deleted to reclaim disk space.

        If the staging file does not exist (e.g. the write was never
        started or already cleaned up), the call is silently skipped.

        Consumers typically call :meth:`WritablePath.close` instead
        of this method directly.

        Args:
            sub_path: Relative path under *base_path* (must match
                the value passed to :meth:`resolve_path`).

        """
        if not self.is_remote:
            return

        staging = self._staging_root() / sub_path
        if not staging.exists():
            logger.debug(f"StorageWriter.close: no file at {staging}, skipping")
            return

        logger.debug(f"StorageWriter.close: uploading {staging} -> {self._base_path}/{sub_path}")
        self._upload_file(sub_path, staging)
        staging.unlink(missing_ok=True)


def get_smart_open_client_params(client: StorageClient) -> dict[str, Any]:
    """Return smart_open-compatible ``client_params`` for a ``StorageClient``.

    The returned dict can be passed directly to
    ``cosmos_curator.core.sensors.utils.io.open_data_source`` (or ``open_file``)
    as ``client_params=``, bridging the storage-client abstraction to the
    ``smart_open`` transport layer.
    """
    if isinstance(client, s3_client.S3Client):
        return {"transport_params": {"client": client.s3}}
    if isinstance(client, azure_client.AzureClient):
        return {"transport_params": {"client": client.service_client}}
    msg = f"Unsupported StorageClient type for smart_open bridge: {type(client)}"
    raise TypeError(msg)


def get_smart_open_params(output_root: str, *, profile_name: str | None = None) -> Mapping[str, Any]:
    """Resolve ``smart_open`` ``client_params`` for ``output_root`` in one call.

    Convenience over ``get_storage_client`` + ``get_smart_open_client_params``
    when the caller only needs the params (not the client itself). Returns an
    empty mapping for local paths.
    """
    profile_name = profile_name or "default"
    client = get_storage_client(output_root, profile_name=profile_name)
    return get_smart_open_client_params(client) if client is not None else {}


def is_missing_object_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` represents 'file/object not found' for a supported backend.

    Covers two cases:

    * Local filesystem -- Python raises :class:`FileNotFoundError` directly.
    * S3 via ``smart_open`` -- ``smart_open`` wraps boto3's NoSuchKey
      ``botocore.exceptions.ClientError`` in a plain :class:`OSError` and
      preserves the original via ``__cause__`` / ``__context__``.

    Lets callers do EAFP without an extra existence probe round-trip.
    """
    if isinstance(exc, FileNotFoundError):
        return True
    cause = exc.__cause__ or exc.__context__
    if isinstance(cause, botocore.exceptions.ClientError):
        response = cause.response or {}
        return response.get("Error", {}).get("Code") == "NoSuchKey"
    return False
