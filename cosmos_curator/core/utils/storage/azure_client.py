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
"""Azure Blob Storage client implementation using azure-storage-blob.

This module provides a simple Azure client with common operations for interacting
with Azure Blob Storage, including chunked downloads and uploads.
"""

import configparser
import io
import os
import pathlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

import attrs
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from loguru import logger
from tqdm import tqdm

from cosmos_curator.core.cf.nvcf_utils import NVCF_SECRETS_PATH, get_secrets_from_nvcf_secret_store
from cosmos_curator.core.utils.environment import AZURE_PROFILE_PATH
from cosmos_curator.core.utils.storage.storage_client import (
    DOWNLOAD_CHUNK_SIZE_BYTES,
    UPLOAD_CHUNK_SIZE_BYTES,
    BackgroundUploader,
    BaseClientConfig,
    StorageClient,
    StoragePrefix,
    is_storage_path,
)


@attrs.define
class AzureClientConfig(BaseClientConfig):
    """Configuration class for Azure client.

    Attributes:
        connection_string (str): Azure storage connection string. Optional if managed identity is used.
        account_url (str): Azure storage account URL. Used if connection_string is not provided.
        account_name (str): Azure storage account name. Used with account key if connection_string is not provided.
        account_key (str): Azure storage account key. Used with account name if connection_string is not provided.
        use_managed_identity (bool): Whether to use Azure managed identity for authentication. Default: False.

    """

    # profile related
    connection_string: str | None = attrs.field(default=None)
    account_url: str | None = attrs.field(default=None)
    account_name: str | None = attrs.field(default=None)
    account_key: str | None = attrs.field(default=None)
    use_managed_identity: bool = attrs.field(default=False)


@attrs.define
class AzurePrefix(StoragePrefix):
    """Represents an Azure Blob Storage prefix (container and blob).

    Attributes:
        _input (str): The input Azure path.

    Properties:
        container (str): The Azure container name.
        blob (str): The Azure blob name (path).
        path (str): The full Azure path (az://container/blob).

    """

    def __attrs_post_init__(self) -> None:
        """Post-init."""
        # Remove 'az://' prefix if present
        self._input = self._input.removeprefix("az://")

        # Validate input format
        if not re.match(r"^[a-z0-9](?!.*--)[a-z0-9-]{1,61}[a-z0-9]/.*$", self._input, re.IGNORECASE):
            error_msg = "Invalid Azure path format. Expected 'container/blob' or 'az://container/blob'"
            raise ValueError(error_msg)

    @property
    def container(self) -> str:
        """Get the container name from the Azure path.

        Returns:
            The container name.

        """
        return self._input.split("/", 1)[0]

    @property
    def blob(self) -> str:
        """Get the blob name from the Azure path.

        Returns:
            The blob name.

        """
        parts = self._input.split("/", 1)
        return parts[1] if len(parts) > 1 else ""

    @property
    def path(self) -> str:
        """Get the full Azure path.

        Returns:
            The full Azure path.

        """
        return f"az://{self.container}/{self.blob}"


class AzureBackgroundUploader(BackgroundUploader):
    """Handles background uploads to Azure Blob Storage."""

    def __init__(self, client: "AzureClient", chunk_size_bytes: int) -> None:
        """Initialize the BackgroundUploader with the given Azure client and chunk size.

        Args:
            client: The Azure client instance.
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
        """Upload a file to Azure Blob Storage.

        Args:
            local_path: Path to the local file to upload.
            remote_path: Azure path where the file will be uploaded.

        """
        remote_prefix = AzurePrefix(remote_path)
        self.client.upload_file(str(local_path), remote_prefix, self.chunk_size_bytes)  # type: ignore[attr-defined]


class AzureClient(StorageClient):
    """Azure client for interacting with Azure Blob Storage.

    This class provides methods for common Azure operations such as checking object
    existence, uploading and downloading objects, and listing objects.

    Attributes:
        service_client (BlobServiceClient): The Azure Blob Service client.
        can_overwrite (bool): Whether the client can overwrite existing objects.
        can_delete (bool): Whether the client can delete objects.

    """

    def __init__(self, config: AzureClientConfig) -> None:
        """Initialize the Azure client with the given configuration.

        Args:
            config (AzureClientConfig): Configuration object containing Azure credentials and settings.

        """
        # Create the BlobServiceClient
        if config.connection_string:
            self.service_client = BlobServiceClient.from_connection_string(config.connection_string)
        elif config.use_managed_identity:
            # Use managed identity (for Azure services)
            credential = DefaultAzureCredential()
            assert config.account_url is not None
            self.service_client = BlobServiceClient(account_url=config.account_url, credential=credential)
        elif config.account_name and config.account_key:
            # Use account name and key
            account_url = config.account_url or f"https://{config.account_name}.blob.core.windows.net"
            self.service_client = BlobServiceClient(
                account_url=account_url,
                credential={
                    "account_name": config.account_name,
                    "account_key": config.account_key,
                },
            )
        else:
            error_msg = (
                "Azure client requires either connection_string, managed identity, or account_name and account_key"
            )
            raise ValueError(error_msg)

        self.can_overwrite = config.can_overwrite
        self.can_delete = config.can_delete

    def _get_container_client(self, container_name: str) -> ContainerClient:
        """Get a container client for the specified container."""
        return self.service_client.get_container_client(container_name)

    def _get_blob_client(self, container_name: str, blob_name: str) -> BlobClient:
        """Get a blob client for the specified container and blob."""
        return self.service_client.get_blob_client(container_name, blob_name)

    def object_exists(self, dest: StoragePrefix) -> bool:
        """Check if an object exists at the specified Azure URI.

        Args:
            dest (AzurePrefix): The Azure prefix of the object to check.

        Returns:
            bool: True if the object exists, False otherwise.

        """
        assert isinstance(dest, AzurePrefix)
        blob_client = self._get_blob_client(dest.container, dest.blob)
        try:
            blob_client.get_blob_properties()
        except ResourceNotFoundError:
            return False
        else:
            return True

    def upload_bytes(self, dest: StoragePrefix, data: "bytes | npt.NDArray[np.uint8]") -> None:
        """Upload binary data to the specified Azure prefix.

        Accepts ``bytes`` or ``numpy.ndarray[uint8]``.  Non-bytes data
        is wrapped in ``io.BytesIO`` because Azure SDK's ``upload_blob``
        only accepts bytes, str, Iterable[bytes], or IO[bytes].

        Args:
            dest: The Azure prefix where the object will be stored.
            data: Binary data to upload (bytes or uint8 numpy array).

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """
        assert isinstance(dest, AzurePrefix)
        if not self.can_overwrite and self.object_exists(dest):
            error_msg = (
                f"Object {dest.blob} already exists in container {dest.container} and overwriting is not allowed."
            )
            raise ValueError(error_msg)

        blob_client = self._get_blob_client(dest.container, dest.blob)
        upload_data: bytes | io.BytesIO = data if isinstance(data, bytes) else io.BytesIO(data)
        blob_client.upload_blob(upload_data, overwrite=self.can_overwrite)

    def upload_bytes_uri(self, uri: str, data: bytes, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> None:  # noqa: ARG002
        """Upload bytes data to the specified Azure URI.

        Args:
            uri: The Azure URI where the object will be stored (e.g., 'az://container-name/blob').
            data: The bytes data to upload.
            chunk_size_bytes: unused

        """
        self.upload_bytes(AzurePrefix(uri), data)

    def download_object_as_bytes(self, uri: StoragePrefix, chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES) -> bytes:  # noqa: ARG002
        """Download an object as bytes from the specified Azure prefix.

        Args:
            uri (AzurePrefix): The Azure prefix of the object to download.
            chunk_size_bytes (int): unused

        Returns:
            bytes: The object's content as bytes.

        """
        assert isinstance(uri, AzurePrefix)
        blob_client = self._get_blob_client(uri.container, uri.blob)
        # Azure handles chunking internally, so we use a straightforward download
        download = blob_client.download_blob()
        return download.readall()

    def download_objects_as_bytes(self, uris: list[StoragePrefix]) -> list[bytes]:
        """Download multiple objects as bytes from the specified URIs.

        Args:
            uris: A list of Azure URIs of the objects to download.

        Returns:
            A list of bytes containing the object contents.

        """
        return [self.download_object_as_bytes(uri) for uri in uris]

    def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
        """List all objects in a container recursively, starting from the given prefix.

        Directory entries (items with Size 0) are excluded from results.

        Args:
            uri (AzurePrefix): The Azure prefix to list objects from.
            limit (int): Maximum number of objects to return. Default: 0 (no limit).

        Returns:
            List[AzurePrefix]: A list of Azure prefixes for all blob files found (excluding directories).

        """
        objects = self.list_recursive(uri, limit)
        results: list[StoragePrefix] = []
        assert isinstance(uri, AzurePrefix)
        for obj in objects:
            path = f"az://{uri.container}/{obj['Name']}"
            results.append(AzurePrefix(path))
        return results

    def list_recursive(self, prefix: StoragePrefix, limit: int = 0) -> list[dict[str, Any]]:
        """List all objects in a container recursively, starting from the given prefix.

        Args:
            prefix: The Azure prefix to list objects from.
            limit (int): Maximum number of objects to return. Default: 0 (no limit).

        Returns:
            A list of dictionaries with object metadata, excluding directory entries.

        """
        assert isinstance(prefix, AzurePrefix)
        container_client = self._get_container_client(prefix.container)
        # list_blobs already returns a paged iterator
        blob_list = container_client.list_blobs(name_starts_with=prefix.blob)

        objects = []
        for blob in blob_list:
            # Skip items with Size 0 as they represent directory entries
            if blob.size == 0:
                continue

            # Convert Azure blob properties to a similar format as S3
            objects.append(
                {
                    "Name": blob.name,
                    "Size": blob.size,
                    "LastModified": blob.last_modified,
                    "ETag": blob.etag,
                    # Add any other necessary properties
                },
            )

            # Check if we've reached the limit
            if limit > 0 and len(objects) >= limit:
                logger.warning(f"Truncated list of objects in Azure prefix to {limit}.")
                break

        return objects

    def upload_file(
        self,
        local_path: str,
        remote_path: StoragePrefix,
        chunk_size: int = UPLOAD_CHUNK_SIZE_BYTES,  # noqa: ARG002
    ) -> None:
        """Upload a file to the specified Azure path.

        Args:
            local_path: The local path of the file to upload.
            remote_path: The Azure URI where the file will be uploaded.
            chunk_size: unused

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """
        if not self.can_overwrite and self.object_exists(remote_path):
            error_msg = f"Object {remote_path.path} already exists and overwriting is not allowed."
            raise ValueError(error_msg)

        logger.info(f"Uploading {local_path} to {remote_path}")
        assert isinstance(remote_path, AzurePrefix)
        blob_client = self._get_blob_client(remote_path.container, remote_path.blob)

        with pathlib.Path(local_path).open("rb") as data:
            blob_client.upload_blob(data, overwrite=self.can_overwrite)

        logger.info(f"Upload complete: {remote_path}")

    def sync_remote_to_local(
        self,
        azure_prefix: StoragePrefix,
        local_dir: pathlib.Path,
        *,
        delete: bool = False,
        chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES,  # noqa: ARG002
    ) -> None:
        """Sync contents of an Azure prefix with a local directory.

        Args:
            azure_prefix (AzurePrefix): The Azure prefix to sync from.
            local_dir (pathlib.Path): The local directory path to sync to.
            delete (bool): If True, delete local files that don't exist in the Azure prefix.
            chunk_size_bytes (int): unused

        """
        print(f"Syncing {azure_prefix} to {local_dir}")  # noqa: T201
        local_dir_path = Path(local_dir)
        local_dir_path.mkdir(parents=True, exist_ok=True)

        assert isinstance(azure_prefix, AzurePrefix)
        # List all objects in the Azure prefix
        azure_objects = self.list_recursive(azure_prefix, limit=0)

        # Download or update files
        with tqdm(
            total=len(azure_objects),
            desc="Syncing",
            unit="file",
            ncols=70,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        ) as pbar:
            for obj in azure_objects:
                relative_path = obj["Name"][len(azure_prefix.blob) :].lstrip("/")
                local_file_path = local_dir_path / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if local file exists and has the same size and last modified time
                azure_mtime = obj["LastModified"].timestamp()
                if local_file_path.exists():
                    local_mtime = local_file_path.stat().st_mtime
                    if local_mtime == azure_mtime and local_file_path.stat().st_size == obj["Size"]:
                        pbar.update(1)
                        continue

                # Download the blob
                blob_client = self._get_blob_client(azure_prefix.container, obj["Name"])
                download = blob_client.download_blob()

                with pathlib.Path(local_file_path).open("wb") as my_blob:
                    download_stream = download.readall()
                    my_blob.write(download_stream)

                # Set the local file's modification time to match the Azure blob
                os.utime(local_file_path, (azure_mtime, azure_mtime))
                pbar.update(1)

        if delete:
            # Remove local files that don't exist in Azure
            local_files = list(local_dir_path.rglob("*"))
            with tqdm(
                total=len(local_files),
                desc="Cleaning",
                unit="file",
                ncols=70,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            ) as pbar:
                for local_file in local_files:
                    if local_file.is_file():
                        relative_path = local_file.relative_to(local_dir_path)
                        azure_blob_name = f"{azure_prefix.blob}{relative_path}".rstrip("/")
                        if not any(obj["Name"] == azure_blob_name for obj in azure_objects):
                            local_file.unlink()
                    pbar.update(1)

        logger.info(f"\nSync completed: {azure_prefix} -> {local_dir}")

    def make_background_uploader(self, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> AzureBackgroundUploader:
        """Create and return a BackgroundUploader instance.

        Args:
            chunk_size_bytes: The size of chunks to use for uploading (default: UPLOAD_CHUNK_SIZE_BYTES).

        Returns:
            An initialized BackgroundUploader instance.

        """
        return AzureBackgroundUploader(self, chunk_size_bytes)

    def delete_object(self, dest: StoragePrefix) -> None:
        """Delete an object at the specified Azure URI.

        Args:
            dest (AzurePrefix): The Azure prefix of the object to delete.

        Raises:
            ValueError: If deletion is not allowed by the client configuration.

        """
        assert isinstance(dest, AzurePrefix)
        if not self.can_delete:
            error_msg = f"Deletion not allowed for object {dest.blob} in container {dest.container}."
            raise ValueError(error_msg)

        blob_client = self._get_blob_client(dest.container, dest.blob)
        blob_client.delete_blob()


def _make_azure_client_config(
    profile_path: pathlib.Path,
    profile_name: str = "default",
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> AzureClientConfig:
    """Create and return an Azure client configuration from a profile file.

    Args:
        profile_path (pathlib.Path): Path to the Azure profile file.
        profile_name (str): The name of the Azure profile to use (default: "default").
        can_overwrite (bool): Whether the client can overwrite existing objects.
        can_delete (bool): Whether the client can delete objects.

    Returns:
        AzureClientConfig: An AzureClientConfig instance.

    Raises:
        FileNotFoundError: If the profile file does not exist.
        ValueError: If the specified profile is not found in the config file.

    """
    config = configparser.ConfigParser()
    if not profile_path.exists():
        error_msg = f"Azure profile file {profile_path} does not exist."
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

    return AzureClientConfig(
        connection_string=c.get("azure_connection_string", None),
        account_url=c.get("azure_account_url", None),
        account_name=c.get("azure_account_name", None),
        account_key=c.get("azure_account_key", None),
        use_managed_identity=c.getboolean("azure_use_managed_identity", False),
        can_overwrite=can_overwrite,
        can_delete=can_delete,
    )


def get_azure_client_config(
    *,
    profile_name: str = "default",
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> AzureClientConfig:
    """Create and return an AzureClientConfig instance.

    Args:
        profile_name (str): The name of the Azure profile to use (default: "default").
        can_overwrite (bool): Whether the client is allowed to overwrite existing objects.
        can_delete (bool): Whether the client is allowed to delete objects.

    Returns:
        AzureClientConfig: An initialized AzureClientConfig instance.

    """
    if AZURE_PROFILE_PATH.exists():
        # first try azure profile
        return _make_azure_client_config(
            AZURE_PROFILE_PATH,
            profile_name,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    # then try secrets from NVCF secret store
    data = get_secrets_from_nvcf_secret_store()
    if data:
        connection_string = data.get("azure_connection_string", None)
        account_url = data.get("azure_account_url", None)
        account_name = data.get("azure_account_name", None)
        account_key = data.get("azure_account_key", None)
        str_umi = data.get("azure_use_managed_identity", "false")
        use_managed_identity = str_umi.lower() in ["true", "yes", "1"]

        return AzureClientConfig(
            connection_string=connection_string,
            account_url=account_url,
            account_name=account_name,
            account_key=account_key,
            use_managed_identity=use_managed_identity,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    error_msg = f"Not found Azure creds from {AZURE_PROFILE_PATH} or {NVCF_SECRETS_PATH}"
    raise ValueError(error_msg)


def is_azure_path(path: str | None) -> bool:
    """Check if a path string is an Azure path.

    Args:
        path: The path to check.

    Returns:
        bool: True if the path is an Azure path, False otherwise.

    """
    return is_storage_path(path, "az")


def create_azure_client(
    target_path: str | None = None,
    profile_name: str = "default",
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> AzureClient | None:
    """Create and return an AzureClient instance if the target path is an Azure path.

    Args:
        target_path: The target path to check. If it's an Azure path, an AzureClient is created.
        profile_name: The name of the Azure profile to use (default: "default").
        can_overwrite: Whether the client is allowed to overwrite existing objects.
        can_delete: Whether the client is allowed to delete objects.

    Returns:
        An initialized AzureClient instance if the target path is an Azure path, None otherwise.

    """
    if is_azure_path(target_path):
        return AzureClient(
            get_azure_client_config(
                profile_name=profile_name,
                can_overwrite=can_overwrite,
                can_delete=can_delete,
            ),
        )
    return None
