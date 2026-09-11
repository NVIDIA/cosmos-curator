# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""S3 client implementation using boto3.

This module provides a simple S3 client with common operations for interacting
with S3-compatible object storage systems, including chunked downloads and uploads.
"""

import configparser
import contextlib
import io
import os
import pathlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

import attrs
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from loguru import logger
from tqdm import tqdm

from cosmos_curator.core.utils.environment import S3_PROFILE_PATH
from cosmos_curator.core.utils.storage import s3_validation
from cosmos_curator.core.utils.storage.storage_client import (
    DOWNLOAD_CHUNK_SIZE_BYTES,
    UPLOAD_CHUNK_SIZE_BYTES,
    BackgroundUploader,
    BaseClientConfig,
    StorageClient,
    StoragePrefix,
    StorageStat,
    is_storage_path,
    join_under,
)

# Backward-compatible exports for callers that imported these from s3_client.
MAX_S3_KEY_LENGTH_BYTES = s3_validation.MAX_S3_KEY_LENGTH_BYTES
validate_configured_s3_location = s3_validation.validate_configured_s3_location


@attrs.define
class S3ClientConfig(BaseClientConfig):
    """Configuration class for S3 client.

    Attributes:
        aws_access_key_id (str): AWS access key ID. Optional if AWS config file will be parsed directly by boto
        aws_secret_access_key (str): AWS secret access key. Optional if AWS config file will be parsed directly by boto
        endpoint_url (str): S3 endpoint URL.
        region (str): AWS region (default: "").
        aws_session_token (str | None): AWS session token (default: None).
        profile_name (str | None): AWS profile name, honoured only when no explicit
            credentials are supplied. It names a section of the standard AWS
            configuration (``~/.aws/credentials``, ``AWS_CONFIG_FILE``), not of
            Curator's own ``COSMOS_S3_PROFILE_PATH`` file: that file is parsed by
            :func:`get_s3_client_config`, which resolves it into explicit keys and so
            takes the branch above. Naming a profile rather than resolving it here is
            what keeps the resulting client's credentials refreshable.

    """

    # profile related
    aws_access_key_id: str | None = attrs.field(default=None)
    aws_secret_access_key: str | None = attrs.field(default=None)
    aws_session_token: str | None = attrs.field(default=None)
    endpoint_url: str | None = attrs.field(default=None)
    region: str | None = attrs.field(default=None)
    profile_name: str | None = attrs.field(default=None)


@attrs.define
class S3Prefix(StoragePrefix):
    """Represents an S3 prefix (bucket and key).

    Attributes:
        _input (str): The input S3 path.

    Properties:
        bucket (str): The S3 bucket name.
        prefix (str): The S3 key prefix.
        path (str): The full S3 path (s3://bucket/prefix).

    """

    def __attrs_post_init__(self) -> None:
        """Post init."""
        # Remove 's3://' prefix if present
        self._input = self._input.removeprefix("s3://")

        # Split into bucket and key
        parts = self._input.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        s3_validation.validate_s3_prefix(bucket, key)

    @property
    def bucket(self) -> str:
        """Return the bucket name for this S3 prefix.

        Returns:
            The bucket name.

        """
        return self._input.split("/", 1)[0]

    @property
    def path(self) -> str:
        """Return the full S3 path for this S3 prefix.

        Returns:
            The full S3 path.

        """
        return f"s3://{self.bucket}/{self.prefix}"


def make_s3_session(config: S3ClientConfig) -> boto3.Session:
    """Build the boto3 session ``config`` describes, without an S3 service client.

    Split out of :class:`S3Client` for callers that need credential and region
    resolution but issue no request. ``storage_cli.get_lance_storage_options`` is the
    one in tree: it freezes the chain's answer into options for Lance's own object
    store, so a service client built for it is measurable work with no user.

    Args:
        config: Credentials, profile and region to resolve from. ``endpoint_url`` is
            not read here -- it applies to a service client, not to a session.

    Returns:
        A session whose credentials stay refreshable, rather than frozen keys: a
        long-running job outlives the SSO or instance credentials it started with.

    """
    if config.aws_access_key_id is not None:
        # region_name must be passed explicitly: supplying credentials bypasses
        # boto3's profile lookup, so the profile's `region` is otherwise ignored
        # and botocore falls back to us-east-1. None keeps boto3's own resolution.
        return boto3.Session(
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            aws_session_token=config.aws_session_token,
            region_name=config.region,
        )
    # ``profile_name=None`` is exactly what a bare ``boto3.Session()`` does, so one call
    # covers both a named AWS profile and no profile at all, and the chain
    # (``~/.aws/credentials``, ``AWS_PROFILE``, the environment, SSO, an instance role)
    # resolves it. ``region_name=None`` likewise defers to the profile and environment.
    return boto3.Session(profile_name=config.profile_name, region_name=config.region)


class S3Client(StorageClient):
    """S3 client for interacting with S3-compatible object storage systems.

    This class provides methods for common S3 operations such as checking object
    existence, uploading and downloading objects, and listing objects in a bucket.

    Attributes:
        session (boto3.Session): The boto3 session.
        s3 (boto3.client): The boto3 S3 client.
        can_overwrite (bool): Whether the client can overwrite existing objects.
        can_delete (bool): Whether the client can delete objects.

    """

    def __init__(self, config: S3ClientConfig) -> None:
        """Initialize the S3 client with the given configuration.

        Args:
            config (S3ClientConfig): Configuration object containing AWS credentials and settings.

        """
        boto_config = BotoConfig(
            max_pool_connections=config.max_concurrent_threads,
            retries={"max_attempts": 3},
            connect_timeout=config.operation_timeout_s,
            read_timeout=config.operation_timeout_s,
        )
        self.session = make_s3_session(config)
        self.s3 = self.session.client("s3", endpoint_url=config.endpoint_url, config=boto_config)
        self.can_overwrite = config.can_overwrite
        self.can_delete = config.can_delete

    def stat(self, dest: StoragePrefix) -> StorageStat:
        """Return what one ``HeadObject`` says about the object at ``dest``.

        This is the request ``object_exists`` has always issued and thrown away; the
        response it already had is now returned, so a caller that needs the size gets
        it for the round trip it was making anyway.

        Deliberately not wrapped in ``do_with_retries`` -- see
        :meth:`StorageClient.stat` for why absence must answer promptly.

        Args:
            dest (S3Prefix): The S3 prefix of the object to describe.

        Returns:
            Size, last-modified and entity tag as the store reported them.

        Raises:
            FileNotFoundError: If no object exists at ``dest``.
            ClientError: For any other store error, notably the ``403`` returned for a
                bucket the credentials cannot read. Only the literal ``404`` that a
                HeadObject answers with is translated: a HEAD carries no body for
                botocore to read a richer code out of, so an S3-compatible store that
                replies ``NoSuchKey`` or ``NotFound`` instead surfaces raw. Callers
                depend on that to tell such a store apart from AWS itself.

        """
        if not isinstance(dest, S3Prefix):
            error_msg = f"S3Client requires an S3Prefix, got {type(dest).__name__}: {dest}"
            raise TypeError(error_msg)
        try:
            head = self.s3.head_object(Bucket=dest.bucket, Key=dest.prefix)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                error_msg = f"Object does not exist: {dest.path}"
                raise FileNotFoundError(error_msg) from e
            # If it's not a 404 error, re-raise the exception
            raise
        size = head.get("ContentLength")
        etag = head.get("ETag")
        return StorageStat(
            size_bytes=int(size) if size is not None else None,
            last_modified=head.get("LastModified"),
            # boto3 hands the tag back quoted, as it appears on the wire.
            etag=str(etag).strip('"') if etag is not None else None,
        )

    def upload_bytes(self, dest: StoragePrefix, data: "bytes | npt.NDArray[np.uint8]") -> None:
        """Upload binary data to the specified S3 prefix.

        Accepts ``bytes`` or ``numpy.ndarray[uint8]``.
        ``io.BytesIO`` reads the data via the buffer protocol,
        avoiding a separate ``.tobytes()`` allocation.

        Args:
            dest: The S3 prefix where the object will be stored.
            data: Binary data to upload (bytes or uint8 numpy array).

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """
        assert isinstance(dest, S3Prefix)
        if not self.can_overwrite and self.object_exists(dest):
            error_msg = f"Object {dest.prefix} already exists in bucket {dest.bucket} and overwriting is not allowed."
            raise ValueError(error_msg)

        fileobj = io.BytesIO(data)
        self.s3.upload_fileobj(
            fileobj,
            dest.bucket,
            dest.prefix,
            Config=TransferConfig(multipart_threshold=UPLOAD_CHUNK_SIZE_BYTES, max_concurrency=10),
        )

    def upload_bytes_uri(self, uri: str, data: bytes, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> None:  # noqa: ARG002
        """Upload bytes data to the specified S3 URI.

        Args:
            uri: The S3 URI where the object will be stored (e.g., 's3://bucket-name/key').
            data: The bytes data to upload.
            chunk_size_bytes: unused

        """
        self.upload_bytes(S3Prefix(uri), data)

    def download_to_path(
        self, uri: StoragePrefix, dest: str | Path, chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES
    ) -> None:
        """Stream an S3 object to a local file using boto3 multipart download.

        Uses ``download_fileobj`` with a ``TransferConfig`` so large files are
        streamed in parallel chunks rather than buffered entirely in memory.
        The download is atomic: data is written to ``dest + ".partial"`` and
        renamed to ``dest`` only on success, so a failed or interrupted download
        never leaves a truncated file that a subsequent caller could mistake for
        a complete one.
        """
        assert isinstance(uri, S3Prefix)
        dest = Path(dest)
        partial = dest.with_suffix(dest.suffix + ".partial")
        try:
            with partial.open("wb") as fout:
                self.s3.download_fileobj(
                    uri.bucket,
                    uri.prefix,
                    fout,
                    Config=TransferConfig(multipart_threshold=chunk_size_bytes, max_concurrency=10),
                )
            partial.replace(dest)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    def download_object_as_bytes(self, uri: StoragePrefix, chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES) -> bytes:
        """Download an object as bytes, by the transfer shape its size warrants.

        ``chunk_size_bytes`` is the multipart threshold and so also the branch point.
        Below it one ``get_object`` both measures and delivers, since the response
        reports ``ContentLength`` before the body is consumed. At or above it the
        managed transfer's parallel ranged GETs take over, at the cost of an extra
        round trip for the size probe. An object whose response omits
        ``ContentLength`` is unmeasured and takes the managed path.

        Both paths return identical bytes, but they are NOT exception-type
        equivalent: a missing object raises ``NoSuchKey`` here and a ``404``
        ``ClientError`` from the managed transfer's internal HeadObject, and the
        managed transfer may surface its own wrapper types besides. A caller that
        retries broadly - as ``do_with_retries`` does - behaves the same on either
        path; a caller that narrows to its own exception tuple cannot assume the two
        paths agree on which type it will see, and has to cover both. The managed
        transfer additionally retries a mid-stream failure in place; the small path
        leaves that class to the caller's own retry wrapper.

        Args:
            uri (S3Prefix): The S3 prefix of the object to download.
            chunk_size_bytes (int): Multipart threshold, and so also the size at or
                above which this hands over to the managed transfer.

        Returns:
            bytes: The object's content as bytes.

        """
        assert isinstance(uri, S3Prefix)
        response = self.s3.get_object(Bucket=uri.bucket, Key=uri.prefix)
        # A response that omits ContentLength leaves the object unmeasured, so it
        # takes the managed-transfer path rather than being read blind.
        content_length = response.get("ContentLength")
        with contextlib.closing(response["Body"]) as body:
            if content_length is not None and content_length < chunk_size_bytes:
                return cast("bytes", body.read())
        # The probe's stream is discarded unread and nothing from it is reused: the
        # managed transfer below re-resolves the size itself via HeadObject.
        fileobj = io.BytesIO()
        self.s3.download_fileobj(
            uri.bucket,
            uri.prefix,
            fileobj,
            Config=TransferConfig(multipart_threshold=chunk_size_bytes, max_concurrency=10),
        )
        return fileobj.getvalue()

    def download_objects_as_bytes(self, uris: list[StoragePrefix]) -> list[bytes]:
        """Download multiple objects as bytes from the specified URIs.

        Args:
            uris: A list of S3 URIs of the objects to download.

        Returns:
            A list of bytes containing the object contents.

        """
        return [self.download_object_as_bytes(uri) for uri in uris]

    def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
        """List all objects in a bucket recursively, starting from the given prefix.

        Args:
            uri (S3Prefix): The S3 prefix to list objects from.
            limit (int): Limit of list to be returned

        Returns:
            List[S3Prefix]: A list of S3 prefixes for all objects found.

        """
        assert isinstance(uri, S3Prefix)
        objects = self.list_recursive(uri, limit)
        results: list[StoragePrefix] = []
        for obj in objects:
            path = f"s3://{uri.bucket}/{obj['Key']}"
            results.append(S3Prefix(path))
        return results

    def list_child_prefixes(self, uri: StoragePrefix) -> list[str]:
        """List the immediate child prefix names under ``uri``, without descending.

        Convenience wrapper over :func:`list_child_prefixes` for callers that
        already hold an ``S3Client``, which is every in-tree caller.

        Args:
            uri: The S3 prefix whose children to list.

        Returns:
            The child prefix names, without a trailing delimiter.

        """
        assert isinstance(uri, S3Prefix)
        return list_child_prefixes(self.s3, bucket=uri.bucket, prefix=uri.prefix)

    def list_recursive_with_suffixes(
        self,
        uri: StoragePrefix,
        suffixes: tuple[str, ...],
        limit: int = 0,
    ) -> list[StoragePrefix]:
        """List objects under ``uri`` matching ``suffixes``, stopping at ``limit`` matches.

        Both the filter and the cap sit inside the pagination loop, so a small ``limit``
        against a huge prefix fetches only the pages it takes to fill it. See
        :meth:`StorageClient.list_recursive_with_suffixes` for why counting matches
        rather than listed objects is the point, and why the early exit that
        :func:`list_child_prefixes` declines is sound for a flat listing.

        Args:
            uri (S3Prefix): The S3 prefix to list under.
            suffixes: Non-empty suffixes to match, compared case-insensitively.
            limit: Maximum number of matches; ``0`` (the default) means unlimited.

        Returns:
            S3 prefixes for the matching objects, in listing order.

        Raises:
            ValueError: If ``suffixes`` is empty.

        """
        if not isinstance(uri, S3Prefix):
            error_msg = f"S3Client requires an S3Prefix, got {type(uri).__name__}: {uri}"
            raise TypeError(error_msg)
        lowered = self._lowercased_suffixes(suffixes)
        results: list[StoragePrefix] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=uri.bucket, Prefix=uri.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(lowered):
                    continue
                results.append(S3Prefix(f"s3://{uri.bucket}/{key}"))
                if 0 < limit <= len(results):
                    return results
        return results

    def list_recursive(self, s3_prefix: StoragePrefix, limit: int = 0) -> list[dict[str, Any]]:
        """List all objects in a bucket recursively, starting from the given prefix.

        Args:
            s3_prefix: The S3 prefix to list objects from.
            limit (int): Limit of list to be returned

        Returns:
            A list of dictionaries with object metadata.

        """
        paginator = self.s3.get_paginator("list_objects_v2")
        objects = []
        assert isinstance(s3_prefix, S3Prefix)
        for page in paginator.paginate(Bucket=s3_prefix.bucket, Prefix=s3_prefix.prefix):
            if "Contents" in page:
                objects.extend(page["Contents"])
            if limit > 0 and len(objects) >= limit:
                logger.info(f"Truncated list of objects in S3 prefix to {len(objects)} as limit={limit}")
                objects = objects[:limit]
                break
        return objects

    def upload_file(
        self,
        local_path: str,
        remote_path: StoragePrefix,
        chunk_size: int = UPLOAD_CHUNK_SIZE_BYTES,
    ) -> None:
        """Upload a file to the specified S3 path.

        Args:
            local_path: The local path of the file to upload.
            remote_path: The S3 URI where the file will be uploaded (e.g., 's3://bucket-name/key').
            chunk_size: The size of chunks to use for uploading (default: UPLOAD_CHUNK_SIZE_BYTES).

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """
        assert isinstance(remote_path, S3Prefix)
        if not self.can_overwrite and self.object_exists(remote_path):
            error_msg = f"Object {remote_path.path} already exists and overwriting is not allowed."
            raise ValueError(error_msg)

        self.s3.upload_file(
            local_path,
            remote_path.bucket,
            remote_path.prefix,
            Config=TransferConfig(multipart_threshold=chunk_size, max_concurrency=10),
        )
        logger.trace(f"Uploaded {local_path} to {remote_path}")

    def sync_remote_to_local(
        self,
        s3_prefix: StoragePrefix,
        local_dir: pathlib.Path,
        *,
        delete: bool = False,
        chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES,
    ) -> None:
        """Sync contents of an S3 prefix with a local directory.

        Args:
            s3_prefix (S3Prefix): The S3 prefix to sync from.
            local_dir (pathlib.Path): The local directory path to sync to.
            delete (bool): If True, delete local files that don't exist in the S3 prefix.
            chunk_size_bytes (int): The size of chunks to use for downloading.

        """
        print(f"Syncing {s3_prefix} to {local_dir}")  # noqa: T201
        local_dir_path = Path(local_dir)
        local_dir_path.mkdir(parents=True, exist_ok=True)
        dest_root = local_dir_path.resolve()

        assert isinstance(s3_prefix, S3Prefix)
        # List all objects in the S3 prefix
        s3_objects = self.list_recursive(s3_prefix)

        # Download or update files
        with tqdm(
            total=len(s3_objects),
            desc="Syncing",
            unit="file",
            ncols=70,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        ) as pbar:
            planned: list[tuple[dict[str, Any], pathlib.Path]] = []
            for obj in s3_objects:
                relative_path = obj["Key"][len(s3_prefix.prefix) :].lstrip("/")
                planned.append((obj, join_under(dest_root, relative_path)))
            for obj, local_file_path in planned:
                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if local file exists and has the same size and last modified time
                s3_mtime = (
                    obj["LastModified"].timestamp()
                    if isinstance(obj["LastModified"], datetime)
                    else datetime.strptime(obj["LastModified"], "%Y-%m-%dT%H:%M:%S.%fZ %z").timestamp()
                )
                if local_file_path.exists():
                    local_mtime = local_file_path.stat().st_mtime
                    if local_mtime == s3_mtime and local_file_path.stat().st_size == obj["Size"]:
                        pbar.update(1)
                        continue

                self.s3.download_file(
                    s3_prefix.bucket,
                    obj["Key"],
                    str(local_file_path),
                    Config=TransferConfig(multipart_threshold=chunk_size_bytes, max_concurrency=10),
                )
                # Set the local file's modification time to match the S3 object
                os.utime(local_file_path, (s3_mtime, s3_mtime))
                pbar.update(1)

        if delete:
            # Remove local files that don't exist in S3
            local_files = list(dest_root.rglob("*"))
            with tqdm(
                total=len(local_files),
                desc="Cleaning",
                unit="file",
                ncols=70,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            ) as pbar:
                for local_file in local_files:
                    if local_file.is_file():
                        relative_path = local_file.relative_to(dest_root)
                        s3_key = f"{s3_prefix.prefix}{relative_path}".rstrip("/")
                        if not any(obj["Key"] == s3_key for obj in s3_objects):
                            local_file.unlink()
                    pbar.update(1)

        logger.info(f"\nSync completed: {s3_prefix} -> {local_dir}")

    def make_background_uploader(self, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> "S3BackgroundUploader":
        """Create and return a BackgroundUploader instance.

        Args:
            chunk_size_bytes: The size of chunks to use for uploading (default: UPLOAD_CHUNK_SIZE_BYTES).

        Returns:
            An initialized BackgroundUploader instance.

        """
        return S3BackgroundUploader(self, chunk_size_bytes)

    def delete_object(self, dest: StoragePrefix) -> None:
        """Delete an object at the specified S3 URI.

        Args:
            dest (S3Prefix): The S3 prefix of the object to delete.

        Raises:
            ValueError: If deletion is not allowed by the client configuration.

        """
        assert isinstance(dest, S3Prefix)
        if not self.can_delete:
            error_msg = f"Deletion not allowed for object {dest.prefix} in bucket {dest.bucket}."
            raise ValueError(error_msg)

        self.s3.delete_object(Bucket=dest.bucket, Key=dest.prefix)


class S3BackgroundUploader(BackgroundUploader):
    """Handles background uploads to S3."""

    def __init__(self, client: S3Client, chunk_size_bytes: int) -> None:
        """Initialize the BackgroundUploader with the given S3 client and chunk size.

        Args:
            client: The S3 client instance.
            chunk_size_bytes: The size of chunks to use for uploading.

        """
        super().__init__(client, chunk_size_bytes)

    def add_task_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        """Add a file upload task to the background uploader.

        Args:
            local_path: Path to the local file to upload.
            remote_path: Path in the storage system where the file will be uploaded.

        """
        future = self.executor.submit(self._upload_file, local_path, remote_path)
        self.futures.append(future)

    def _upload_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        """Upload a file to S3.

        Args:
            local_path: Path to the local file to upload.
            remote_path: S3 path where the file will be uploaded.

        """
        remote_prefix = S3Prefix(remote_path)
        self.client.upload_file(str(local_path), remote_prefix, self.chunk_size_bytes)  # type: ignore[attr-defined]


def list_child_prefixes(s3_client: BaseClient, *, bucket: str, prefix: str) -> list[str]:
    """List child prefixes with a delimited listing so the store does the scoping.

    A recursive listing would return every object beneath every child, which is
    unusable when a prefix holds millions of them. ``Delimiter="/"`` makes S3
    collapse each child into one ``CommonPrefixes`` entry instead, and leaves
    loose objects sitting directly under the prefix in ``Contents``, where they
    are correctly ignored.

    The boto3 client is supplied by the caller rather than built here, so that
    listing uses the same credentials as the reads that follow it. Every in-tree
    caller now holds an ``S3Client`` and reaches this through
    :meth:`S3Client.list_child_prefixes`; the free function stays for a caller
    that holds a boto3 client and no ``S3Client`` around it.

    No limit is accepted. A caller that caps its selection should do so after
    deduplicating and sorting, since stopping the listing early would change
    which children are selected rather than just how many pages are fetched.

    Args:
        s3_client: A boto3 S3 client, however the caller chose to build it.
        bucket: The bucket to list.
        prefix: The key prefix whose immediate children to list. Trailing
            delimiters are normalized, and an empty prefix lists the bucket root.

    Returns:
        The child prefix names, without a trailing delimiter, in listing order.

    Raises:
        FileNotFoundError: If the prefix holds no keys at all. S3 answers a
            listing of a nonexistent prefix with 200 and no keys, so a typo is
            otherwise indistinguishable from an empty result and the caller
            would proceed with nothing without ever reporting an error.

    """
    listing_key = f"{prefix.rstrip('/')}/" if prefix else ""

    children: list[str] = []
    prefix_exists = False
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=listing_key, Delimiter="/"):
        common_prefixes = page.get("CommonPrefixes", [])
        prefix_exists = prefix_exists or bool(common_prefixes) or bool(page.get("Contents"))
        for common_prefix in common_prefixes:
            child = common_prefix["Prefix"][len(listing_key) :].rstrip("/")
            if child:
                children.append(child)

    if not prefix_exists:
        msg = f"Prefix does not exist or contains no objects: s3://{bucket}/{listing_key}"
        raise FileNotFoundError(msg)
    return children


def resolve_s3_endpoint_url(explicit: str | None = None, *, profile_endpoint_url: str | None = None) -> str | None:
    """Resolve which S3 endpoint URL to use, or ``None`` for boto3's default AWS endpoint.

    boto3 does not read the ``endpoint_url`` that the ``awscli_plugin_endpoint`` plugin
    nests under the ``s3`` / ``s3api`` sections of ``~/.aws/config`` -- that is a
    CLI-only plugin feature -- so an S3-compatible store reachable only through such a
    profile has to be told its endpoint here.

    Resolution order, first non-empty value winning:

    1. ``explicit`` -- typically an ``--endpoint-url`` CLI argument
    2. ``AWS_ENDPOINT_URL_S3`` -- the standard AWS SDK S3-specific variable
    3. ``AWS_ENDPOINT_URL`` -- the standard AWS SDK all-services variable
    4. ``profile_endpoint_url`` -- what a profile file said, supplied by the caller
    5. ``None`` -- boto3's default AWS endpoint

    Levels 2 and 3 are not merely a reimplementation of what boto3 already does with
    those variables for a client built with ``endpoint_url=None``. They are here so the
    precedence against level 4 is explicit -- boto3 never sees a Curator profile file --
    and so consumers that are not boto3 clients, notably Lance's object store, resolve
    the same endpoint the S3 client would.

    The profile-file value arrives as an argument instead of being read here, which is
    what keeps :func:`get_s3_client_config` returning the Curator profile file's
    ``endpoint_url`` verbatim for its existing callers. Folding the environment
    variables into that factory would let an ``AWS_ENDPOINT_URL`` exported for some
    unrelated tool silently redirect every pipeline whose credentials come from
    ``COSMOS_S3_PROFILE_PATH`` -- which is every deployed one. A caller that does want
    the environment to outrank a profile file passes that file's value in.

    Args:
        explicit: An endpoint the caller was told to use, which outranks everything.
        profile_endpoint_url: An endpoint read from a profile file, consulted only
            after the environment.

    Returns:
        The endpoint URL to hand to ``session.client("s3", endpoint_url=...)``, or
        ``None`` to leave boto3 on the default AWS endpoint.

    """
    candidates = (
        explicit,
        os.getenv("AWS_ENDPOINT_URL_S3"),
        os.getenv("AWS_ENDPOINT_URL"),
        profile_endpoint_url,
    )
    return next((candidate for candidate in candidates if candidate), None)


def _make_s3_client_config(
    profile_path: pathlib.Path,
    profile_name: str = "default",
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> S3ClientConfig:
    """Create and return an S3 client configuration from a profile file.

    Args:
        profile_path (pathlib.Path): Path to the S3 profile file.
        profile_name (str): The name of the S3 profile to use (default: "default").
        can_overwrite (bool): Whether the client can overwrite existing objects.
        can_delete (bool): Whether the client can delete objects.

    Returns:
        S3ClientConfig: An S3ClientConfig instance.

    Raises:
        FileNotFoundError: If the profile file does not exist.
        ValueError: If the specified profile is not found in the config file.

    """
    config = configparser.ConfigParser()
    if not profile_path.exists():
        error_msg = f"S3 profile file {profile_path} does not exist."
        raise FileNotFoundError(error_msg)
    config.read(profile_path)

    # find the target profile section
    profile_key = None
    item_len: int = 2
    for section in config.sections():
        if section == profile_name:
            profile_key = section
            break
        if section.startswith("profile "):
            items = section.split()
            if len(items) == item_len and items[1] == profile_name:
                profile_key = section
                c = config[profile_key]
                break

    if profile_key is None:
        error_msg = f"Profile {profile_name} not found in config file {profile_path}"
        raise ValueError(error_msg)
    c = config[profile_key]

    return S3ClientConfig(
        aws_access_key_id=c.get("aws_access_key_id"),
        aws_secret_access_key=c.get("aws_secret_access_key"),
        aws_session_token=c.get("aws_session_token", None),
        endpoint_url=c.get("endpoint_url", None),
        region=c.get("region", None),
        can_overwrite=can_overwrite,
        can_delete=can_delete,
    )


def get_s3_client_config(
    profile_name: str = "default",
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> S3ClientConfig:
    """Create and return an S3ClientConfig instance.

    Args:
        profile_name (str): The name of the S3 profile to use (default: "default").
        can_overwrite (bool): Whether the client is allowed to overwrite existing objects.
        can_delete (bool): Whether the client is allowed to delete objects.

    Returns:
        S3ClientConfig: An initialized S3ClientConfig instance.

    """
    # Deferred: importing it here keeps the rest of this module -- notably
    # ``S3Prefix``, which is pure string validation -- usable from the CLI without a ray install.
    from cosmos_curator.core.cf.nvcf_secrets import (  # noqa: PLC0415
        NVCF_SECRETS_PATH,
        get_secrets_from_nvcf_secret_store,
    )

    if S3_PROFILE_PATH.exists():
        # first try s3 profile
        return _make_s3_client_config(
            S3_PROFILE_PATH,
            profile_name,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    # then try secrets from NVCF secret store
    data = get_secrets_from_nvcf_secret_store()
    if data:
        aws_access_key_id = data.get("aws_access_key_id", None)
        aws_secret_access_key = data.get("aws_secret_access_key", None)
        endpoint_url = data.get("endpoint_url", None)
        region = data.get("region", None)
        return S3ClientConfig(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
            region=region,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    error_msg = f"Not found S3 creds from {S3_PROFILE_PATH} or {NVCF_SECRETS_PATH}"
    raise ValueError(error_msg)


def is_s3path(path: str | None) -> bool:
    """Check if a path string is an S3 path.

    Args:
        path: The path to check.

    Returns:
        bool: True if the path is an S3 path, False otherwise.

    """
    return is_storage_path(path, "s3")


def create_s3_client(
    target_path: str | None = None,
    profile_name: str = "default",
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> S3Client | None:
    """Create and return an S3Client instance if the target path is an S3 path.

    Args:
        target_path: The target path to check. If it's an S3 path, an S3Client is created.
        profile_name: The name of the S3 profile to use (default: "default").
        can_overwrite: Whether the client is allowed to overwrite existing objects.
        can_delete: Whether the client is allowed to delete objects.

    Returns:
        An initialized S3Client instance if the target path is an S3 path, None otherwise.

    """
    if is_s3path(target_path):
        return S3Client(get_s3_client_config(profile_name, can_overwrite=can_overwrite, can_delete=can_delete))
    return None
