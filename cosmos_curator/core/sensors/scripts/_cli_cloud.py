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
"""Shared CLI helpers for sensor-library scripts that accept cloud sources.

Centralises ``s3://`` / ``az://`` URI detection, credential resolution, and
the actual ``smart_open`` open used by ``check_video_index``,
``camera_sensor_benchmark``, and ``cloud_io_benchmark``.

The sensor library itself is backend-agnostic and never accepts URIs (see
``cosmos_curator/core/sensors/types/types.py``). This helper module is the
single carve-out under ``cosmos_curator/core/sensors/`` that imports
``smart_open``, on behalf of the in-tree scripts only.

The helpers are split into two layers so instrumentation-heavy callers
(e.g. ``cloud_io_benchmark``) can build a boto3 client, attach event hooks
to it, and *then* hand it into :func:`open_cloud_source`:

* :func:`make_s3_client` / :func:`make_azure_client` resolve credentials
  and return the underlying SDK client.
* :func:`open_cloud_source` is a context manager that yields a seekable
  :class:`typing.BinaryIO` for an ``s3://`` / ``az://`` URI, using a
  caller-provided client when present and otherwise constructing one from
  the supplied profile-name arguments.

Credentials are loaded via ``boto3`` (S3) and ``azure.storage.blob`` /
``azure.identity`` (Azure) directly rather than through
``cosmos_curator.core.utils.storage`` to keep the sensor-package boundary
intact.
"""

import argparse
import configparser
import os
import pathlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, BinaryIO, cast

import boto3
import smart_open  # type: ignore[import-untyped]
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, NoCredentialsError, ProfileNotFound

S3_CREDENTIALS_HINT = (
    "Use --s3-profile-name to select an AWS profile, or configure standard AWS credentials "
    "with AWS_PROFILE, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or an IAM role."
)
AZURE_CREDENTIALS_HINT = (
    "Use --azure-profile-name to select an Azure profile, or populate the Azure credentials file "
    "(default: /dev/shm/azure_creds_file, override with COSMOS_AZURE_PROFILE_PATH) with one of "
    "azure_connection_string, azure_account_name+azure_account_key, or azure_use_managed_identity."
)


class CloudCliError(Exception):
    """Actionable user-facing error from cloud-source CLI helpers."""


def is_s3_uri(source: str) -> bool:
    """Return True if ``source`` is an ``s3://`` URI."""
    return source.startswith("s3://")


def is_azure_uri(source: str) -> bool:
    """Return True if ``source`` is an ``az://`` URI."""
    return source.startswith("az://")


def is_cloud_uri(source: str) -> bool:
    """Return True if ``source`` is a supported cloud URI."""
    return is_s3_uri(source) or is_azure_uri(source)


def validate_source(source: str) -> None:
    """Validate that ``source`` is a local file path or a supported cloud URI.

    Raises:
        CloudCliError: If ``source`` uses an unsupported scheme or refers to a
            local path that does not exist.

    """
    if is_cloud_uri(source):
        return
    if "://" in source:
        msg = f"unsupported source URI {source!r}; use a local file path or an s3:// or az:// URI"
        raise CloudCliError(msg)
    if not pathlib.Path(source).is_file():
        msg = f"source is not a file: {source}"
        raise CloudCliError(msg)


def make_s3_client(source: str, s3_profile_name: str | None) -> BaseClient:
    """Build a credentialled boto3 S3 client for an ``s3://`` source.

    The returned client is the place to attach botocore event hooks (e.g.
    ``before-send.s3.GetObject``) before handing it into
    :func:`open_cloud_source`.

    Raises:
        CloudCliError: When boto3 cannot construct a credentialled S3 client.

    """
    try:
        session = boto3.Session(profile_name=s3_profile_name) if s3_profile_name else boto3.Session()
        credentials = session.get_credentials()
    except (BotoCoreError, ProfileNotFound) as e:
        msg = f"could not configure S3 access for {source!r}: {e}\n{S3_CREDENTIALS_HINT}"
        raise CloudCliError(msg) from e
    except Exception as e:
        msg = f"could not configure S3 access for {source!r}: {e}"
        raise CloudCliError(msg) from e

    if credentials is None:
        msg = f"could not configure S3 access for {source!r}: {NoCredentialsError()}\n{S3_CREDENTIALS_HINT}"
        raise CloudCliError(msg)

    try:
        return cast("BaseClient", session.client("s3"))
    except (BotoCoreError, ProfileNotFound) as e:
        msg = f"could not configure S3 access for {source!r}: {e}\n{S3_CREDENTIALS_HINT}"
        raise CloudCliError(msg) from e
    except Exception as e:
        msg = f"could not configure S3 access for {source!r}: {e}"
        raise CloudCliError(msg) from e


def _azure_profile_path() -> pathlib.Path:
    """Return the on-disk Azure credentials file path.

    Mirrors the default used by ``cosmos_curator.core.utils.environment`` so
    operators with an existing profile file get the same lookup behaviour
    here as elsewhere in the codebase, without taking a hard import on the
    storage package (which the sensor library is not allowed to depend on).
    """
    return pathlib.Path(os.getenv("COSMOS_AZURE_PROFILE_PATH", "/dev/shm/azure_creds_file"))  # noqa: S108


def _load_azure_profile_section(profile_name: str) -> configparser.SectionProxy:
    """Locate ``profile_name`` in the on-disk Azure credentials file."""
    path = _azure_profile_path()
    if not path.exists():
        msg = f"Azure profile file {path} does not exist"
        raise CloudCliError(msg)

    parser = configparser.ConfigParser()
    parser.read(path)

    section_lookup_len = 2
    for section in parser.sections():
        if section == profile_name:
            return parser[section]
        if section.startswith("profile "):
            parts = section.split()
            if len(parts) == section_lookup_len and parts[1] == profile_name:
                return parser[section]

    msg = f"Azure profile {profile_name!r} not found in {path}"
    raise CloudCliError(msg)


def _build_azure_service_client(profile_name: str) -> BlobServiceClient:
    """Construct an Azure ``BlobServiceClient`` from the profile file.

    Supports the same three credential modes as the storage-package helper:
    connection string, account name + key, and managed identity.
    """
    section = _load_azure_profile_section(profile_name)

    connection_string = section.get("azure_connection_string", None)
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    if section.getboolean("azure_use_managed_identity", False):
        account_url = section.get("azure_account_url", None)
        if not account_url:
            msg = f"Azure profile {profile_name!r}: azure_use_managed_identity set but azure_account_url missing"
            raise CloudCliError(msg)
        return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())

    account_name = section.get("azure_account_name", None)
    account_key = section.get("azure_account_key", None)
    if account_name and account_key:
        account_url = section.get("azure_account_url", None) or f"https://{account_name}.blob.core.windows.net"
        return BlobServiceClient(
            account_url=account_url,
            credential={"account_name": account_name, "account_key": account_key},
        )

    msg = (
        f"Azure profile {profile_name!r} has no usable credentials "
        "(need one of azure_connection_string, azure_account_name+azure_account_key, "
        "or azure_use_managed_identity+azure_account_url)"
    )
    raise CloudCliError(msg)


def make_azure_client(source: str, azure_profile_name: str) -> BlobServiceClient:
    """Build a credentialled Azure ``BlobServiceClient`` for an ``az://`` source.

    Raises:
        CloudCliError: When the Azure profile is missing or invalid.

    """
    try:
        return _build_azure_service_client(azure_profile_name)
    except CloudCliError:
        raise
    except Exception as e:
        msg = f"could not configure Azure access for {source!r}: {e}\n{AZURE_CREDENTIALS_HINT}"
        raise CloudCliError(msg) from e


@contextmanager
def open_cloud_source(
    source: str,
    *,
    s3_client: BaseClient | None = None,
    azure_client: BlobServiceClient | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
) -> Generator[BinaryIO, None, None]:
    """Open an ``s3://`` or ``az://`` URI as a seekable :class:`BinaryIO`.

    If a pre-built SDK client is supplied via ``s3_client`` / ``azure_client``
    it is used as-is, which preserves any caller-attached event hooks (e.g.
    botocore's ``before-send.s3.GetObject``). Otherwise the corresponding
    client is constructed from the profile-name arguments.

    Args:
        source: ``s3://`` or ``az://`` URI to open. Local paths are rejected.
        s3_client: Pre-built boto3 S3 client (overrides ``s3_profile_name``).
        azure_client: Pre-built Azure ``BlobServiceClient`` (overrides
            ``azure_profile_name``).
        s3_profile_name: Optional AWS profile used when ``s3_client`` is not
            provided. ``None`` falls back to boto3's default credential chain.
        azure_profile_name: Azure profile used when ``azure_client`` is not
            provided.

    Yields:
        A seekable :class:`BinaryIO` opened in binary read mode via
        ``smart_open``. Ownership stays with this context manager; the
        caller must not close the stream.

    Raises:
        CloudCliError: If ``source`` is not a supported cloud URI.

    """
    transport_params: dict[str, Any]
    if is_s3_uri(source):
        s3 = s3_client if s3_client is not None else make_s3_client(source, s3_profile_name)
        transport_params = {"client": s3}
    elif is_azure_uri(source):
        azure = azure_client if azure_client is not None else make_azure_client(source, azure_profile_name)
        transport_params = {"client": azure}
    else:
        msg = f"open_cloud_source requires an s3:// or az:// URI, got {source!r}"
        raise CloudCliError(msg)

    with smart_open.open(source, "rb", transport_params=transport_params) as stream:
        yield cast("BinaryIO", stream)


def add_cloud_credential_args(parser: argparse.ArgumentParser) -> None:
    """Attach ``--s3-profile-name`` / ``--azure-profile-name`` flags to ``parser``."""
    parser.add_argument(
        "--s3-profile-name",
        default=None,
        help="Optional AWS profile name used for s3:// sources. If omitted, boto3's default credential chain is used.",
    )
    parser.add_argument(
        "--azure-profile-name",
        default="default",
        help="Azure profile name used for az:// sources (default: 'default').",
    )
