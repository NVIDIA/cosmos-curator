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
"""Command-line helpers for tools that accept an ``s3://`` / ``az://`` source.

The bridge between an operator-supplied URI and
:mod:`cosmos_curator.core.utils.storage`: argument parsing, credential resolution
from the *standard AWS* and Curator Azure profile namespaces, actionable errors,
and the ``smart_open`` open a sensor reader needs. It sits outside the
``storage/`` package because it is CLI policy rather than storage mechanism --
nothing in a pipeline stage should reach for an ``argparse`` flag or for the
user-facing error type here.

Credentials deliberately do *not* go through ``s3_client.get_s3_client_config``
or ``azure_client.get_azure_client_config``. Those read Curator's own
``COSMOS_S3_PROFILE_PATH`` / NVCF secret namespace, which is not where an
operator running a CLI keeps credentials, and both import ``nvcf_utils`` and so
``ray``. :class:`~cosmos_curator.core.utils.storage.s3_client.S3ClientConfig` is
therefore constructed directly (it carries ``profile_name``, which names a
section of ``~/.aws/credentials``), and the Azure side goes through
``make_azure_client_config``, which is the ray-free half of that module.

The helpers come in two layers so an instrumentation-heavy caller (see
``benchmarks/sensors/cloud_io_benchmark.py``) can build a client, attach botocore
event hooks to it, and *then* hand it in:

* :func:`make_s3_client` / :func:`make_azure_client` / :func:`make_storage_client`
  resolve credentials and return a ``StorageClient``.
* :func:`open_storage_source` and :func:`list_storage_objects` accept a
  caller-supplied client and reuse it rather than rebuilding one, so any hooks
  attached to it survive.
"""

import argparse
import os
import pathlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import BinaryIO, cast

import smart_open  # type: ignore[import-untyped]
from botocore.exceptions import BotoCoreError, NoCredentialsError, ProfileNotFound

from cosmos_curator.core.utils.environment import DEFAULT_AZURE_PROFILE_PATH
from cosmos_curator.core.utils.storage.azure_client import (
    AzureClient,
    AzureClientConfig,
    AzurePrefix,
    is_azure_path,
    make_azure_client_config,
)
from cosmos_curator.core.utils.storage.block_cache_reader import BlockCacheReader
from cosmos_curator.core.utils.storage.s3_client import (
    S3Client,
    S3ClientConfig,
    S3Prefix,
    is_s3path,
    make_s3_session,
    resolve_s3_endpoint_url,
)
from cosmos_curator.core.utils.storage.storage_client import StorageClient, StoragePrefix
from cosmos_curator.core.utils.storage.storage_utils import get_smart_open_client_params, is_remote_path

_S3_CREDENTIALS_HINT = (
    "Use --s3-profile-name to select an AWS profile, or configure standard AWS credentials "
    "with AWS_PROFILE, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or an IAM role."
)
_AZURE_CREDENTIALS_HINT = (
    "Use --azure-profile-name to select an Azure profile, or populate the Azure credentials file "
    "(default: /dev/shm/azure_creds_file, override with COSMOS_AZURE_PROFILE_PATH) with one of "
    "azure_connection_string, azure_account_name+azure_account_key, or azure_use_managed_identity."
)


class StorageCliError(Exception):
    """Actionable user-facing error from the storage CLI helpers."""


def validate_source(source: str) -> None:
    """Validate that ``source`` is a local file path or a supported storage URI.

    Raises:
        StorageCliError: If ``source`` uses an unsupported scheme or refers to a
            local path that does not exist.

    """
    if is_remote_path(source):
        return
    if "://" in source:
        msg = f"unsupported source URI {source!r}; use a local file path or an s3:// or az:// URI"
        raise StorageCliError(msg)
    if not pathlib.Path(source).is_file():
        msg = f"source is not a file: {source}"
        raise StorageCliError(msg)


def storage_prefix(uri: str) -> StoragePrefix:
    """Convert an ``s3://`` / ``az://`` URI into the prefix type its backend uses.

    A bare bucket or container addresses everything under it, with or without a
    trailing delimiter, on both backends alike.

    Raises:
        StorageCliError: If the URI names neither backend, or if it names one but
            cannot address anything -- a bare scheme, or a bucket or container name
            the backend's own rules reject.

    """
    if is_s3path(uri):
        try:
            return S3Prefix(uri)
        except ValueError as e:
            msg = f"malformed s3 URI ({_container_detail(uri, 's3', 'bucket', e)}): {uri!r}"
            raise StorageCliError(msg) from e
    if is_azure_path(uri):
        try:
            return AzurePrefix(_azure_container_root(uri))
        except ValueError as e:
            msg = f"malformed az URI ({_container_detail(uri, 'az', 'container', e)}): {uri!r}"
            raise StorageCliError(msg) from e
    msg = f"expected an s3:// or az:// URI, got {uri!r}"
    raise StorageCliError(msg)


def _azure_container_root(uri: str) -> str:
    """Give a container-only ``az://`` URI the delimiter ``AzurePrefix`` insists on.

    ``AzurePrefix`` requires a blob component, so ``az://container`` -- how an operator
    names a whole container, and what the S3 side already accepts as ``s3://bucket`` --
    would be rejected as malformed. A bare ``az://`` is left alone so that it still
    fails as the missing-container case it is.
    """
    rest = uri.removeprefix("az://")
    return f"{uri}/" if rest and "/" not in rest else uri


def _container_detail(uri: str, scheme: str, noun: str, error: ValueError) -> str:
    """Say *why* a URI is malformed, naming the empty top-level component when that is it.

    A bare ``s3://`` is the mistake worth naming in the operator's own terms -- it is
    what a shell variable that did not expand looks like. Anything else is a name the
    backend's own rules rejected, and its complaint is more specific than ours.
    """
    if not uri.removeprefix(f"{scheme}://").split("/", 1)[0]:
        return f"no {noun}"
    return str(error)


def _s3_access_error(source: str, detail: object, *, hint: bool) -> StorageCliError:
    """Build the S3 credential error every entry point here reports.

    ``hint`` is False for a failure boto3 does not model: the hint names the ways to
    supply credentials, which is the wrong advice for a chain that broke rather than
    came up empty.
    """
    message = f"could not configure S3 access for {source!r}: {detail}"
    return StorageCliError(f"{message}\n{_S3_CREDENTIALS_HINT}" if hint else message)


def _aws_profile(name: str | None) -> str | None:
    """Read an empty ``--s3-profile-name`` as no profile at all.

    ``boto3.Session`` treats an empty ``profile_name`` as a profile to go and look up,
    and raises ``ProfileNotFound`` when it is not in the config. But a shell expanding
    an unset variable into ``--s3-profile-name "$AWS_PROFILE"`` means the opposite --
    take whatever the default chain finds.
    """
    return name or None


def make_s3_client(
    source: str,
    s3_profile_name: str | None,
    endpoint_url: str | None = None,
    *,
    can_overwrite: bool = False,
) -> S3Client:
    """Build a credentialled ``S3Client`` for an ``s3://`` source.

    ``client.s3`` is the underlying boto3 client, and so the place to attach botocore
    event hooks (e.g. ``before-send.s3.GetObject``) before handing the client to
    :func:`open_storage_source`.

    The credential chain is probed once here, on the session the client keeps, because
    ``S3Client`` resolves nothing at construction time: without the probe an exhausted
    chain would surface as an opaque signing failure on the first request instead of as
    a message naming the flag that fixes it. Credentials stay refreshable -- only their
    presence is checked, not their value.

    Args:
        source: The ``s3://`` URI the client will be used for (for error messages).
        s3_profile_name: Optional AWS profile; ``None`` or ``""`` uses boto3's default
            chain.
        endpoint_url: Optional S3 endpoint override. Passed through
            :func:`~cosmos_curator.core.utils.storage.s3_client.resolve_s3_endpoint_url`,
            so ``None`` still honours the ``AWS_ENDPOINT_URL_S3`` / ``AWS_ENDPOINT_URL``
            environment variables before falling back to boto3's default AWS endpoint.
        can_overwrite: Whether the client may replace an object that already exists.
            Off by default: every caller here reads, bar the store rewriting
            ``manifest.json`` on every run (see ``storage_io.write_text``). Nothing
            deletes, so ``can_delete`` is not exposed at all.

    Raises:
        StorageCliError: When boto3 cannot construct a credentialled S3 client.

    """
    config = S3ClientConfig(
        profile_name=_aws_profile(s3_profile_name),
        endpoint_url=resolve_s3_endpoint_url(endpoint_url),
        can_overwrite=can_overwrite,
    )
    try:
        client = S3Client(config)
        credentials = client.session.get_credentials()
    except (BotoCoreError, ProfileNotFound) as e:
        raise _s3_access_error(source, e, hint=True) from e
    except Exception as e:
        raise _s3_access_error(source, e, hint=False) from e
    if credentials is None:
        raise _s3_access_error(source, NoCredentialsError(), hint=True)
    return client


def _azure_profile_path() -> pathlib.Path:
    """Return the on-disk Azure credentials file path, re-read at call time.

    ``core.utils.environment.AZURE_PROFILE_PATH`` freezes the same lookup when it is
    imported, which is right for a long-lived pipeline process but wrong for a CLI:
    ``COSMOS_AZURE_PROFILE_PATH`` is how a deployment without ``/dev/shm`` supplies
    credentials, and it may be exported after this module has been imported. The
    fallback is the bare default rather than that frozen path, so an override that goes
    away is honoured in both directions.
    """
    override = os.getenv("COSMOS_AZURE_PROFILE_PATH")
    return pathlib.Path(override) if override else DEFAULT_AZURE_PROFILE_PATH


def _validate_azure_profile(profile_name: str, config: AzureClientConfig) -> None:
    """Reject an unusable Azure profile in the vocabulary the operator wrote it in.

    ``AzureClient`` checks the same three modes in the same order, but reports them in
    terms of its own config fields rather than of profile keys, and *asserts* on the
    missing account URL -- so the one misconfiguration that most needs naming would
    arrive with no message at all.

    Raises:
        StorageCliError: If no credential mode is fully configured.

    """
    if config.connection_string:
        return
    if config.use_managed_identity:
        if config.account_url:
            return
        msg = f"Azure profile {profile_name!r}: azure_use_managed_identity set but azure_account_url missing"
        raise StorageCliError(msg)
    if config.account_name and config.account_key:
        return
    msg = (
        f"Azure profile {profile_name!r} has no usable credentials "
        "(need one of azure_connection_string, azure_account_name+azure_account_key, "
        "or azure_use_managed_identity+azure_account_url)"
    )
    raise StorageCliError(msg)


def _azure_client_config(azure_profile_name: str) -> AzureClientConfig:
    """Resolve an Azure profile into a usable config, or say what to fix.

    ``make_azure_client_config`` reports a missing profile as
    ``Profile X not found in config file Y``, which reads as though ``X`` were an AWS
    profile and gives an operator no clue that ``--azure-profile-name`` is the flag
    involved. Both lookup failures are reworded here to name the profile as it was
    typed, and both carry the credentials hint, since a wrong name and a missing file
    are exactly the two mistakes the hint addresses.

    Raises:
        StorageCliError: If the profile file or the profile is missing, if a value
            inside the profile is malformed, or if the profile configures no usable
            credential mode. Only the two lookup failures carry the hint: a malformed
            value and a missing credential mode both name the key at fault already, so
            the hint's list of alternatives would only bury it.

    """
    path = _azure_profile_path()
    try:
        config = make_azure_client_config(path, azure_profile_name)
    except FileNotFoundError as e:
        msg = f"{str(e).rstrip('.')}\n{_AZURE_CREDENTIALS_HINT}"
        raise StorageCliError(msg) from e
    except ValueError as e:
        # ``make_azure_client_config`` spends ValueError twice: on a profile it cannot find,
        # and on a profile it found whose ``azure_use_managed_identity`` is not a boolean.
        # Only the first is a naming mistake, and only it is worth the hint.
        if "not found in config file" in str(e):
            msg = f"Azure profile {azure_profile_name!r} not found in {path}\n{_AZURE_CREDENTIALS_HINT}"
        else:
            msg = f"Azure profile {azure_profile_name!r} in {path} is malformed: {e}"
        raise StorageCliError(msg) from e
    _validate_azure_profile(azure_profile_name, config)
    return config


def make_azure_client(source: str, azure_profile_name: str) -> AzureClient:
    """Build a credentialled ``AzureClient`` for an ``az://`` source.

    Credential precedence is the profile file's, resolved by
    ``make_azure_client_config``: connection string, then managed identity, then
    account name and key.

    Raises:
        StorageCliError: When the Azure profile file, the profile, or its credentials
            are missing or unusable, or when the SDK refuses what the profile said.
            Only the last of those is decorated here: a profile problem was already
            diagnosed in the profile's own vocabulary by :func:`_azure_client_config`,
            and prefixing it with the source URI would put the least actionable part
            of the message first.

    """
    try:
        return AzureClient(_azure_client_config(azure_profile_name))
    except StorageCliError:
        raise
    except Exception as e:
        msg = f"could not configure Azure access for {source!r}: {e}\n{_AZURE_CREDENTIALS_HINT}"
        raise StorageCliError(msg) from e


def make_storage_client(
    source: str,
    *,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> StorageClient:
    """Build a credentialled client for whichever backend ``source`` names.

    Raises:
        StorageCliError: If ``source`` is not an ``s3://`` or ``az://`` URI, or if
            credentials for it cannot be resolved.

    """
    if is_s3path(source):
        return make_s3_client(source, s3_profile_name, endpoint_url)
    if is_azure_path(source):
        return make_azure_client(source, azure_profile_name)
    msg = f"make_storage_client requires an s3:// or az:// URI, got {source!r}"
    raise StorageCliError(msg)


@contextmanager
def open_storage_source(
    source: str,
    *,
    client: StorageClient | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> Generator[BinaryIO]:
    """Open an ``s3://`` or ``az://`` URI as a seekable :class:`BinaryIO`.

    A supplied ``client`` is used as-is, which is what preserves any caller-attached
    botocore event hooks; otherwise one is built from the profile-name arguments.

    Args:
        source: ``s3://`` or ``az://`` URI to open. Local paths are rejected.
        client: Pre-built storage client (overrides the profile arguments).
        s3_profile_name: Optional AWS profile used when ``client`` is not provided.
            ``None`` falls back to boto3's default credential chain.
        azure_profile_name: Azure profile used when ``client`` is not provided.
        endpoint_url: Optional S3 endpoint override used only when ``client`` is not
            supplied. Ignored for Azure, whose endpoint is baked into the account URL.

    Yields:
        A seekable :class:`BinaryIO` over the ``smart_open`` stream, behind a
        :class:`~cosmos_curator.core.utils.storage.block_cache_reader.BlockCacheReader`
        so that libav's seek-heavy access pattern costs a bounded number of requests
        rather than one per seek. Ownership stays with this context manager; the
        caller must not close it.

    Raises:
        StorageCliError: If ``source`` is not a supported storage URI.
        OSError: From ``smart_open`` when the object cannot be opened -- a missing
            object on S3 or Azure raises here, before anything is yielded.

    """
    if not is_remote_path(source):
        msg = f"open_storage_source requires an s3:// or az:// URI, got {source!r}"
        raise StorageCliError(msg)
    if client is None:
        client = make_storage_client(
            source,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
        )
    with (
        smart_open.open(source, "rb", **get_smart_open_client_params(client)) as stream,
        BlockCacheReader(stream) as reader,
    ):
        yield cast("BinaryIO", reader)


def list_storage_objects(  # noqa: PLR0913 -- a prefix, two backends' credentials, and two listing controls
    prefix: str,
    *,
    client: StorageClient | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
    limit: int = 0,
    suffixes: tuple[str, ...] | None = None,
) -> list[str]:
    """List object URIs under an ``s3://`` or ``az://`` prefix.

    Returns fully-qualified URIs (``s3://bucket/key`` / ``az://container/blob``)
    suitable for handing straight to :func:`open_storage_source`. Both the suffix
    filter and ``limit`` live inside the backend's pagination loop, so a caller can
    cheaply sample a huge prefix without paging it in full.

    Args:
        prefix: ``s3://`` or ``az://`` prefix to list under.
        client: Pre-built storage client (overrides the profile arguments).
        s3_profile_name: Optional AWS profile used when ``client`` is absent.
        azure_profile_name: Azure profile used when ``client`` is absent.
        endpoint_url: Optional S3 endpoint override used when ``client`` is absent.
        limit: Maximum number of objects to return; ``0`` (default) means all. With
            ``suffixes`` it counts matches, so a request for five videos is not
            answered with five sidecar JSON files.
        suffixes: Suffixes to match, compared case-insensitively at both ends (for
            example ``(".mp4", ".mkv")``). ``None`` (default) applies no filter; an
            *empty* tuple is a caller mistake rather than a way to spell that, and the
            backend rejects it. Note that unfiltered is not quite everything on Azure:
            ``AzureClient.list_recursive`` reads a zero-byte blob as a directory entry
            and drops it, so a truncated upload disappears from an unfiltered listing.
            Passing ``suffixes`` avoids that path entirely, which is why the only
            caller in the tree does.

    Raises:
        StorageCliError: If ``prefix`` is not a supported storage URI.
        ValueError: If ``suffixes`` is empty or holds an empty string.

    """
    if not is_remote_path(prefix):
        msg = f"list_storage_objects requires an s3:// or az:// URI, got {prefix!r}"
        raise StorageCliError(msg)
    root = storage_prefix(prefix)
    if client is None:
        client = make_storage_client(
            prefix,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
        )
    listed = (
        client.list_recursive_directory(root, limit)
        if suffixes is None
        else client.list_recursive_with_suffixes(root, suffixes, limit)
    )
    # The zero-byte "directory" placeholder keys some upload tools create are not
    # objects to read. A suffix filter already excludes them -- a name ending in the
    # delimiter cannot end in ".mp4" -- so this only bites the unfiltered listing.
    return [str(entry) for entry in listed if not str(entry).endswith("/")]


def get_lance_storage_options(
    uri: str,
    *,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, str] | None:
    """Build Lance ``storage_options`` for ``uri``, or ``None`` for a local path.

    Same option names as ``storage_utils.get_lance_storage_options`` but resolved from
    the standard AWS profile namespace rather than Curator's, so a store written by
    these CLIs is readable by them again. Lance's object store cannot call back into
    boto3, so the credentials the chain produced are frozen into concrete values here;
    falsy entries are dropped rather than passed as ``None``, which Lance rejects.

    Azure raises rather than silently writing an unauthenticated store: Lance's Azure
    options are a different set of keys, and a wrong guess would surface as an opaque
    permission error at write time.

    Deliberately narrower error translation than :func:`make_s3_client`: only failures
    boto3 models are turned into a :class:`StorageCliError`, so an unmodelled one still
    reaches the caller with its own type and traceback.

    Raises:
        StorageCliError: If ``uri`` is an ``az://`` URI, or S3 credentials cannot be
            resolved.

    """
    if is_azure_path(uri):
        msg = f"the data-integrity store does not support az:// yet: {uri!r}; use a local path or an s3:// URI"
        raise StorageCliError(msg)
    if not is_s3path(uri):
        return None

    endpoint_url = resolve_s3_endpoint_url(endpoint_url)
    try:
        # A session, not an ``S3Client``: nothing here issues a request, and building the
        # service client an ``S3Client`` carries costs an order of magnitude more than
        # the session itself. The store resolves options once per operation, so that is
        # paid several times per invocation for a client no one uses.
        session = make_s3_session(S3ClientConfig(profile_name=_aws_profile(s3_profile_name)))
        credentials = session.get_credentials()
    except (BotoCoreError, ProfileNotFound) as e:
        raise _s3_access_error(uri, e, hint=True) from e
    if credentials is None:
        raise _s3_access_error(uri, NoCredentialsError(), hint=True)

    frozen = credentials.get_frozen_credentials()
    options = {
        "aws_access_key_id": frozen.access_key,
        "aws_secret_access_key": frozen.secret_key,
        "aws_session_token": frozen.token,
        "aws_region": session.region_name,
        "aws_endpoint": endpoint_url,
    }
    return {key: value for key, value in options.items() if value} or None


def add_storage_credential_args(parser: argparse.ArgumentParser) -> None:
    """Attach ``--s3-profile-name`` / ``--azure-profile-name`` / ``--endpoint-url`` flags to ``parser``."""
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
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help=(
            "Optional S3 endpoint URL for s3:// sources on S3-compatible stores. "
            "Falls back to AWS_ENDPOINT_URL_S3 / AWS_ENDPOINT_URL, "
            "then boto3's default AWS endpoint. Ignored for az:// sources."
        ),
    )
