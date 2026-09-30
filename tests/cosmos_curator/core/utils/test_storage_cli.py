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
"""Characterization tests for the storage CLI helpers, carried over from their predecessor.

Written against the private cloud-source helper module that used to sit in
``cosmos_curator/core/sensors/scripts/``, while it was still unmodified, to pin what it
did before :mod:`cosmos_curator.core.utils.storage` replaced it. That module is now
deleted and these tests have moved here with it, rewired onto
:mod:`cosmos_curator.core.utils.storage_cli` -- so they no longer describe a retirement
plan, they are the suite for the helpers that replaced it.

The predecessor had a second heir, the recipe policy layer
:mod:`cosmos_curator.next.recipes.data_integrity.storage_io`; the tests pinning that
half sit in the recipe's own test directory, as
``tests/cosmos_curator/next/recipes/data_integrity/test_storage_io.py``. The fakes both
suites drive are shared from :mod:`tests.cosmos_curator.core.utils.storage_fakes`, so
neither file carries a fake SDK the other one needs.

**How this file is organised, and why.** Everything above the ``BEHAVIOUR`` banner is
mechanism: how a subject is constructed, which SDK entry point is faked, which
environment variables are scrubbed. Everything below it is behaviour: values, exception
types, message substrings, call counts and ordering. The move rewrote the mechanism
block and left the behaviour block standing, which is what shows the replacement
preserves behaviour rather than merely compiling.

Four assertions did change, and each says so in its docstring under ``DIVERGENCE``,
naming what the storage layer does instead:

* an uppercase suffix in the filter matched nothing before, and matches now
* an empty suffix tuple matched nothing before, and is rejected now
* a missing Azure profile arrived without the credentials hint before, and carries it now
* the URI predicates raised on ``None`` before, and return ``False`` now

There is no cloud-mocking library in any pixi environment (no ``moto``, no
``botocore.stub``), so the fakes are hand-rolled and the credential tests drive real
``boto3`` against a throwaway shared-credentials file. Nothing here performs network I/O.
"""

import argparse
import io
import pathlib
import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, BinaryIO

import boto3
import pytest
from botocore.client import BaseClient

from cosmos_curator.core.utils import storage_cli
from cosmos_curator.core.utils.storage import azure_client as azure_client_module
from cosmos_curator.core.utils.storage import s3_client as s3_client_module
from cosmos_curator.core.utils.storage.azure_client import is_azure_path
from cosmos_curator.core.utils.storage.block_cache_reader import BlockCacheReader
from cosmos_curator.core.utils.storage.s3_client import is_s3path, resolve_s3_endpoint_url
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import is_remote_path
from cosmos_curator.core.utils.storage_cli import StorageCliError
from tests.cosmos_curator.core.utils.storage_fakes import (
    FakeAzureContainer,
    FakeAzureService,
    FakeBlob,
    azure_client,
    s3_client,
)

# ======================================================================================
# MECHANISM -- subject construction and hand-rolled backend fakes.
# The move to the storage layer rewrote this block. Nothing here asserts anything.
#
# Two shapes recur, and both exist so the behaviour block below could stay put:
#
# * The storage layer takes a ``StorageClient``, not a backend SDK client, so the SDK
#   fakes are wrapped by ``storage_fakes.s3_client`` / ``storage_fakes.azure_client``.
#   The fakes themselves are still SDK-shaped, because the request each Curator call
#   makes on the wire is exactly what these tests exist to pin.
# * The subject sometimes returns a wrapper where it used to return the SDK client
#   (``make_s3_client``) or a section mapping (the Azure profile lookup). The adapters
#   named ``_make_*`` / ``_load_*`` unwrap that, so an assertion still reads the value
#   an operator would care about.
# ======================================================================================

_ObjectLister = Callable[..., list[str]]
_SourceOpener = Callable[..., AbstractContextManager[BinaryIO]]
_ClientFactories = tuple["_RecordingClientFactory", "_RecordingClientFactory"]


class _FakePaginator:
    """Paginator stand-in that records how many pages were actually consumed.

    Pages are yielded lazily so a test can prove an early-exiting listing stopped
    fetching, which a list of pages could not show.
    """

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages
        self.last_paginate_kwargs: dict[str, object] | None = None
        self.pages_yielded = 0

    def paginate(self, **kwargs: object) -> Iterator[dict[str, Any]]:
        self.last_paginate_kwargs = kwargs
        return self._iter_pages()

    def _iter_pages(self) -> Iterator[dict[str, Any]]:
        for page in self._pages:
            self.pages_yielded += 1
            yield page


class _FakeListS3:
    """S3 stand-in whose only wired-up call is the ``list_objects_v2`` paginator."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._paginator = _FakePaginator(pages)

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return self._paginator

    @property
    def pages_yielded(self) -> int:
        """How many pages the listing actually pulled off the paginator."""
        return self._paginator.pages_yielded

    @property
    def paginate_kwargs(self) -> dict[str, object] | None:
        """The arguments the listing paged the bucket with."""
        return self._paginator.last_paginate_kwargs


class _FakeAzureCredential:
    """Marker standing in for ``DefaultAzureCredential``.

    Constructing the real credential inspects the ambient environment; a marker keeps the
    managed-identity branch observable while staying offline and deterministic.
    """


class _FakeSmartOpen:
    """``smart_open.open`` stand-in serving a payload and recording how it was called."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.streams: list[io.BytesIO] = []

    def __call__(self, uri: str, mode: str, transport_params: dict[str, Any]) -> io.BytesIO:
        self.calls.append((uri, mode, transport_params))
        self.streams.append(io.BytesIO(self._payload))
        return self.streams[-1]


class _RecordingClientFactory:
    """Stand-in for one of the module's own client-building functions.

    Records the positional arguments it was handed and always returns the same client,
    so a test can tell "a client was built from these arguments" apart from "the caller's
    client was reused". ``client`` is the *SDK* client inside the wrapper, since that is
    what reaches ``smart_open``.
    """

    def __init__(self, wrap: Callable[[object], StorageClient]) -> None:
        self.client = object()
        self._storage_client = wrap(self.client)
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> StorageClient:
        self.calls.append(args)
        return self._storage_client


class _FakeFrozenCredentials:
    """Credentials whose frozen view deliberately differs from its live attributes."""

    def __init__(self) -> None:
        self.access_key = "live-key"
        self.secret_key = "live-secret"  # noqa: S105 - a marker value, not a credential
        self.token = "live-token"  # noqa: S105 - a marker value, not a credential

    def get_frozen_credentials(self) -> SimpleNamespace:
        return SimpleNamespace(
            access_key="frozen-key",
            secret_key="frozen-secret",  # noqa: S106 - a marker value, not a credential
            token="frozen-token",  # noqa: S106 - a marker value, not a credential
        )


class _FakeSession:
    """boto3 session stand-in resolving to credentials with a distinct frozen view."""

    region_name = "us-east-2"

    def __init__(self, **_kwargs: object) -> None:
        pass

    def get_credentials(self) -> _FakeFrozenCredentials:
        return _FakeFrozenCredentials()

    def client(self, *_args: object, **_kwargs: object) -> object:
        """Serve the S3 client ``S3Client`` builds up front but Lance options never use."""
        return object()


_PROFILE_CREDENTIALS = (
    "[curator-test]\n"
    "aws_access_key_id = profile-key\n"
    "aws_secret_access_key = profile-secret\n"
    "region = us-west-1\n"
    "\n"
    "[tokened]\n"
    "aws_access_key_id = session-key\n"
    "aws_secret_access_key = session-secret\n"
    "aws_session_token = session-token\n"
)

_SCRUBBED_AWS_VARS = (
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
)


@pytest.fixture
def aws_profile_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point boto3 at a throwaway credentials file, with rival env credentials set.

    The environment credentials are deliberate: botocore drops its env provider once a
    profile is named explicitly, so their presence is what proves the profile was honoured
    rather than merely not contradicted. The region and endpoint variables are cleared for
    the opposite reason -- they outrank a profile's own values, so an ambient one on the
    dev node would mask what is under test. ``AWS_CONFIG_FILE`` is aimed at a path that
    does not exist so the developer's real ``~/.aws/config`` cannot contribute either.
    """
    credentials = tmp_path / "credentials"
    credentials.write_text(_PROFILE_CREDENTIALS)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-such-config"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for name in _SCRUBBED_AWS_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def aws_empty_chain_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave boto3's whole credential chain with nothing to find, and no network to try.

    ``AWS_EC2_METADATA_DISABLED`` matters: instance metadata is the last provider in the
    chain, and without it a machine off the network waits out a connect timeout before
    reporting the miss this fixture exists to produce. A region is still supplied, so a
    failure here is about credentials and cannot be a missing-region error wearing a
    disguise.
    """
    for name in (*_SCRUBBED_AWS_VARS, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-such-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-such-config"))
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


#: Never a real key. Base64 because that is the shape the Azure SDK expects to parse.
_AZURE_ACCOUNT_KEY = "bXktZmFrZS1hY2NvdW50LWtleQ=="
_AZURE_CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;AccountName=curator-test;"
    f"AccountKey={_AZURE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
)

# Section order is load-bearing for the ``[profile name]`` scan, which walks sections in
# file order and returns the first match of either spelling.
_AZURE_PROFILE_FILE = f"""\
[default]
azure_connection_string = {_AZURE_CONNECTION_STRING}

[profile prod]
azure_account_name = prod-account
azure_account_key = {_AZURE_ACCOUNT_KEY}

[with-account-url]
azure_account_name = prod-account
azure_account_key = {_AZURE_ACCOUNT_KEY}
azure_account_url = https://prod.blob.example.invalid

[managed]
azure_use_managed_identity = true
azure_account_url = https://managed.blob.example.invalid

[managed-without-url]
azure_use_managed_identity = true

[managed-and-key]
azure_use_managed_identity = true
azure_account_url = https://managed.blob.example.invalid
azure_account_name = prod-account
azure_account_key = {_AZURE_ACCOUNT_KEY}

[everything]
azure_connection_string = {_AZURE_CONNECTION_STRING}
azure_use_managed_identity = true
azure_account_url = https://managed.blob.example.invalid
azure_account_name = prod-account
azure_account_key = {_AZURE_ACCOUNT_KEY}

[unusable]
azure_account_name = prod-account

[malformed-flag]
azure_use_managed_identity = perhaps
azure_account_url = https://managed.blob.example.invalid

[profile two words]
azure_connection_string = {_AZURE_CONNECTION_STRING}
"""


@pytest.fixture
def azure_profile_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Write a real Azure profile file and point the module's path override at it."""
    path = tmp_path / "azure_creds_file"
    path.write_text(_AZURE_PROFILE_FILE)
    monkeypatch.setenv("COSMOS_AZURE_PROFILE_PATH", str(path))
    return path


def _make_s3_client(
    source: str,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> BaseClient:
    """Build the subject's client and hand back the boto3 client inside it.

    The credential and endpoint questions below are about what boto3 was told, and
    ``S3Client`` is a wrapper around exactly that, so unwrapping keeps the assertions on
    the object whose behaviour an operator sees.
    """
    return storage_cli.make_s3_client(source, s3_profile_name, endpoint_url).s3


def _s3_client_and_session(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> tuple[BaseClient, boto3.Session]:
    """Build the subject's S3 client and hand back the session it was built from.

    ``make_s3_client`` keeps its session on the client it returns, but the session is
    captured here anyway, by wrapping ``boto3.Session``: a refreshability test has to
    rotate the credential object the session resolved, and taking it off the wrapper
    would assert against the same object the subject is being asked about. Deliberately
    no botocore privates: everything asserted against the result goes through
    ``client.meta`` or ``generate_presigned_url``.
    """
    sessions: list[boto3.Session] = []
    real_session = boto3.Session

    def _record(**kwargs: object) -> boto3.Session:
        session = real_session(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(s3_client_module.boto3, "Session", _record)
    client = _make_s3_client("s3://bucket/key.mp4", profile_name, endpoint_url)
    return client, sessions[-1]


def _make_azure_client(source: str, azure_profile_name: str) -> object:
    """Build the subject's Azure client and hand back the SDK client inside it.

    Same unwrapping as :func:`_make_s3_client`, and for the same reason: the credential
    modes below are questions about what the Azure SDK was constructed with.
    """
    return storage_cli.make_azure_client(source, azure_profile_name).service_client


def _load_azure_profile_section(profile_name: str) -> dict[str, str | None]:
    """Resolve an Azure profile the way the CLI does, as a section-shaped mapping.

    The profile scan is no longer a helper of its own -- it is
    ``azure_client.make_azure_client_config``, which returns a config object rather than
    a ``configparser`` section. Projecting that back onto the profile's own key names is
    what lets these tests keep asking which section was found rather than which field
    was populated.
    """
    config = storage_cli._azure_client_config(profile_name)
    return {
        "azure_connection_string": config.connection_string,
        "azure_account_url": config.account_url,
        "azure_account_name": config.account_name,
        "azure_account_key": config.account_key,
    }


def _azure_sdk_recorder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: Exception | None = None,
) -> list[tuple[str, dict[str, object]]]:
    """Replace the Azure SDK entry points with recorders; return their shared call log.

    Recording which constructor ran, with which arguments, is what makes the credential-mode
    ordering observable at all: the real SDK would need a live account to say anything, and
    ``moto``-style doubles do not exist for Azure in this repo. ``error`` makes the
    connection-string entry point fail the way a malformed string does.
    """
    calls: list[tuple[str, dict[str, object]]] = []

    class _Double:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("constructor", kwargs))

        @staticmethod
        def from_connection_string(connection_string: str) -> object:
            calls.append(("from_connection_string", {"connection_string": connection_string}))
            if error is not None:
                raise error
            return _Double.__new__(_Double)

    monkeypatch.setattr(azure_client_module, "BlobServiceClient", _Double)
    monkeypatch.setattr(azure_client_module, "DefaultAzureCredential", _FakeAzureCredential)
    return calls


def _s3_lister(pages: list[dict[str, Any]]) -> tuple[_ObjectLister, _FakeListS3]:
    """Return a listing callable bound to a fake S3 serving ``pages``, plus that fake."""
    fake = _FakeListS3(pages)

    def _list(prefix: str, *, limit: int = 0, suffixes: tuple[str, ...] | None = None) -> list[str]:
        return storage_cli.list_storage_objects(prefix, client=s3_client(fake), limit=limit, suffixes=suffixes)

    return _list, fake


def _azure_lister(blob_names: list[str]) -> tuple[_ObjectLister, FakeAzureContainer]:
    """Return a listing callable bound to a fake container serving ``blob_names``."""
    container = FakeAzureContainer(blobs=[FakeBlob(name) for name in blob_names])
    service = FakeAzureService(container)

    def _list(prefix: str, *, limit: int = 0, suffixes: tuple[str, ...] | None = None) -> list[str]:
        return storage_cli.list_storage_objects(prefix, client=azure_client(service), limit=limit, suffixes=suffixes)

    return _list, container


def _source_opener(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes = b"object-bytes",
) -> tuple[_SourceOpener, _FakeSmartOpen]:
    """Return the open-a-storage-source subject with ``smart_open`` faked, plus the recorder."""
    fake = _FakeSmartOpen(payload)
    monkeypatch.setattr(storage_cli.smart_open, "open", fake)

    def _open(source: str, **kwargs: Any) -> AbstractContextManager[BinaryIO]:  # noqa: ANN401 - argparse-ish kwargs
        # One ``client`` argument now covers both backends, the client itself carrying
        # which one it is. The per-backend spellings survive here so the reuse
        # assertions stay about reuse rather than about the new signature.
        sdk_client = kwargs.pop("s3_client", None)
        azure_sdk_client = kwargs.pop("azure_client", None)
        client: StorageClient | None = None
        if sdk_client is not None:
            client = s3_client(sdk_client)
        elif azure_sdk_client is not None:
            client = azure_client(azure_sdk_client)
        return storage_cli.open_storage_source(source, client=client, **kwargs)

    return _open, fake


def _client_factories(monkeypatch: pytest.MonkeyPatch) -> _ClientFactories:
    """Replace both per-backend client factories with recorders; return (s3, azure)."""
    s3_factory = _RecordingClientFactory(s3_client)
    azure_factory = _RecordingClientFactory(azure_client)
    monkeypatch.setattr(storage_cli, "make_s3_client", s3_factory)
    monkeypatch.setattr(storage_cli, "make_azure_client", azure_factory)
    return s3_factory, azure_factory


# ======================================================================================
# BEHAVIOUR -- what the helpers promise their callers. This block survived the move to
# the storage layer, apart from the four assertions marked DIVERGENCE.
# ======================================================================================

# --------------------------------------------------------------------------------------
# 1. make_s3_client
# --------------------------------------------------------------------------------------


@pytest.mark.usefixtures("aws_profile_env")
def test_a_named_profile_beats_ambient_environment_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile's keys and region must reach the client and outrank the environment.

    Rival ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` are exported by the fixture, so
    a client that merely happened to work would sign with those. A presigned URL is the
    cheapest signature to inspect: it carries the access key id in ``X-Amz-Credential`` and
    is computed with no network I/O.
    """
    client, session = _s3_client_and_session(monkeypatch, "curator-test")

    assert session.profile_name == "curator-test"
    signed = client.generate_presigned_url("get_object", Params={"Bucket": "bucket", "Key": "key.mp4"})
    assert "profile-key" in signed
    assert "env-key" not in signed
    assert client.meta.region_name == "us-west-1"


@pytest.mark.usefixtures("aws_profile_env")
def test_profile_credentials_stay_refreshable_instead_of_being_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client must sign with the session's live credentials, not a snapshot.

    ``method`` is the discriminator: resolving the chain here and handing the result back as
    explicit keys would report ``explicit`` instead. Rotating the session's own credential
    object and re-signing then shows the signer reads that object per request rather than a
    construction-time copy, which is what lets an SSO or instance credential renew mid-run
    instead of expiring the job.
    """
    client, session = _s3_client_and_session(monkeypatch, "curator-test")
    credentials = session.get_credentials()

    assert credentials.method == "shared-credentials-file"

    params = {"Bucket": "bucket", "Key": "key.mp4"}
    assert "profile-key" in client.generate_presigned_url("get_object", Params=params)

    credentials.access_key = "rotated-key"
    signed_after_rotation = client.generate_presigned_url("get_object", Params=params)

    assert "rotated-key" in signed_after_rotation
    assert "profile-key" not in signed_after_rotation


@pytest.mark.usefixtures("aws_profile_env")
def test_an_explicit_endpoint_url_reaches_the_built_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """An S3-compatible store is only reachable if the override survives client creation."""
    client, _ = _s3_client_and_session(monkeypatch, "curator-test", "https://s3.example.invalid")

    assert client.meta.endpoint_url == "https://s3.example.invalid"


@pytest.mark.usefixtures("aws_profile_env")
def test_an_environment_endpoint_url_reaches_the_built_client_without_an_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the argument still consults the environment, not just boto3's default."""
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://s3-env.invalid")

    client, _ = _s3_client_and_session(monkeypatch, "curator-test")

    assert client.meta.endpoint_url == "https://s3-env.invalid"


@pytest.mark.usefixtures("aws_profile_env")
def test_a_missing_profile_becomes_a_storage_cli_error_carrying_the_credentials_hint() -> None:
    """A typo in ``--s3-profile-name`` must arrive as actionable advice, not a traceback."""
    with pytest.raises(StorageCliError) as excinfo:
        _make_s3_client("s3://bucket/key.mp4", "no-such-profile")

    message = str(excinfo.value)
    assert "could not configure S3 access for 's3://bucket/key.mp4'" in message
    assert "no-such-profile" in message
    assert storage_cli._S3_CREDENTIALS_HINT in message


@pytest.mark.usefixtures("aws_empty_chain_env")
def test_an_empty_credential_chain_becomes_a_storage_cli_error_carrying_the_credentials_hint() -> None:
    """No credentials anywhere is the other half of the translation, and needs the same hint."""
    with pytest.raises(StorageCliError) as excinfo:
        _make_s3_client("s3://bucket/key.mp4", None)

    message = str(excinfo.value)
    assert "could not configure S3 access for 's3://bucket/key.mp4'" in message
    assert "Unable to locate credentials" in message
    assert storage_cli._S3_CREDENTIALS_HINT in message


@pytest.mark.usefixtures("aws_profile_env")
def test_an_empty_profile_name_means_no_profile_rather_than_a_profile_named_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--s3-profile-name "$AWS_PROFILE"`` with the variable unset must still work.

    boto3 reads an empty ``profile_name`` as a profile to go and look up, and raises
    ``ProfileNotFound`` from ``Session.__init__`` when it is not in the config. The
    fixture's environment credentials are what a working default chain signs with, so
    finding them in the signature is the evidence the chain was consulted at all.
    """
    client, _ = _s3_client_and_session(monkeypatch, "")

    signed = client.generate_presigned_url("get_object", Params={"Bucket": "bucket", "Key": "key.mp4"})
    assert "env-key" in signed
    assert "profile-key" not in signed


# --------------------------------------------------------------------------------------
# 2. resolve_s3_endpoint_url
# --------------------------------------------------------------------------------------

#: Exactly three tiers, then ``None``. The storage-layer twin grew a fourth
#: ``profile_endpoint_url`` tier below these; this module has no such tier, and pinning the
#: three-tier shape is what will show whether adopting the twin changed anything.
_ENDPOINT_PRECEDENCE = [
    pytest.param(
        ("https://explicit.invalid", "https://s3-env.invalid", "https://env.invalid"),
        "https://explicit.invalid",
        id="an-explicit-argument-outranks-everything",
    ),
    pytest.param(
        (None, "https://s3-env.invalid", "https://env.invalid"),
        "https://s3-env.invalid",
        id="the-s3-specific-variable-outranks-the-generic-one",
    ),
    pytest.param(
        (None, None, "https://env.invalid"),
        "https://env.invalid",
        id="the-generic-variable-is-the-last-resort",
    ),
    pytest.param((None, None, None), None, id="nothing-configured-defers-to-boto3"),
]


@pytest.mark.parametrize(("candidates", "expected"), _ENDPOINT_PRECEDENCE)
def test_endpoint_resolution_follows_the_three_tier_precedence(
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[str | None, str | None, str | None],
    expected: str | None,
) -> None:
    """Reproduce the order the data-integrity CLIs already depend on."""
    explicit, env_s3, env_generic = candidates
    for name, value in (("AWS_ENDPOINT_URL_S3", env_s3), ("AWS_ENDPOINT_URL", env_generic)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert resolve_s3_endpoint_url(explicit) == expected


def test_an_empty_endpoint_argument_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiers test emptiness, not presence, so ``--endpoint-url ''`` falls through."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env.invalid")
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)

    assert resolve_s3_endpoint_url("") == "https://env.invalid"


# --------------------------------------------------------------------------------------
# 3. Azure credential resolution
# --------------------------------------------------------------------------------------


def test_the_azure_profile_path_comes_from_the_environment_override(azure_profile_file: pathlib.Path) -> None:
    """``COSMOS_AZURE_PROFILE_PATH`` is how every deployment points this at its own file."""
    assert storage_cli._azure_profile_path() == azure_profile_file


def test_the_azure_profile_path_defaults_to_the_shared_memory_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default has to match the rest of the codebase or operators get two lookups."""
    monkeypatch.delenv("COSMOS_AZURE_PROFILE_PATH", raising=False)

    assert storage_cli._azure_profile_path() == pathlib.Path("/dev/shm/azure_creds_file")  # noqa: S108


def test_the_azure_profile_path_default_is_not_the_import_time_override(
    azure_profile_file: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Withdrawing the override must restore the default, not leave the old value behind.

    The fixture exports ``COSMOS_AZURE_PROFILE_PATH`` the way a deployment does, and
    this removes it again mid-process. Falling back to ``environment.AZURE_PROFILE_PATH``
    would answer with whatever that variable held when the module was imported, which is
    the opposite of the per-call re-read this function exists to provide.
    """
    assert storage_cli._azure_profile_path() == azure_profile_file

    monkeypatch.delenv("COSMOS_AZURE_PROFILE_PATH")

    assert storage_cli._azure_profile_path() == pathlib.Path("/dev/shm/azure_creds_file")  # noqa: S108


@pytest.mark.usefixtures("azure_profile_file")
def test_a_section_named_exactly_like_the_profile_is_found() -> None:
    """The plain spelling, which is what the Azure credentials file actually uses."""
    section = _load_azure_profile_section("default")

    assert section.get("azure_connection_string") == _AZURE_CONNECTION_STRING


@pytest.mark.usefixtures("azure_profile_file")
def test_a_profile_prefixed_section_is_found_by_its_bare_name() -> None:
    """``[profile prod]`` is the AWS config spelling, and operators reuse it here."""
    section = _load_azure_profile_section("prod")

    assert section.get("azure_account_name") == "prod-account"


@pytest.mark.usefixtures("azure_profile_file")
@pytest.mark.parametrize("profile_name", ["two words", "two", "words"])
def test_a_profile_section_with_two_words_after_the_prefix_is_not_matched(profile_name: str) -> None:
    """The scan splits on whitespace and requires exactly two parts, so ``[profile two words]`` is inert.

    Neither the full name nor either word resolves it, which means a profile named with a
    space is simply unreachable rather than ambiguously matched.
    """
    with pytest.raises(StorageCliError, match="not found in"):
        _load_azure_profile_section(profile_name)


def test_a_missing_azure_profile_file_is_a_storage_cli_error_naming_the_path(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path is the actionable part: the file lives in shared memory and is easy to lose."""
    missing = tmp_path / "azure_creds_file"
    monkeypatch.setenv("COSMOS_AZURE_PROFILE_PATH", str(missing))

    with pytest.raises(StorageCliError, match=re.escape(f"Azure profile file {missing} does not exist")):
        _load_azure_profile_section("default")


def test_a_profile_absent_from_the_file_is_a_storage_cli_error_naming_the_profile(
    azure_profile_file: pathlib.Path,
) -> None:
    """Both the profile and the file it was looked for in belong in the message."""
    with pytest.raises(StorageCliError) as excinfo:
        _load_azure_profile_section("no-such-profile")

    message = str(excinfo.value)
    assert "Azure profile 'no-such-profile' not found" in message
    assert str(azure_profile_file) in message


def test_a_profile_present_but_malformed_is_not_reported_as_a_missing_profile(
    azure_profile_file: pathlib.Path,
) -> None:
    """A profile that exists but holds a non-boolean flag must not be called absent.

    ``make_azure_client_config`` raises ``ValueError`` both for a profile it cannot find
    and for one whose ``azure_use_managed_identity`` will not parse, so catching the type
    alone would send someone hunting for a section that is sitting right there.
    """
    with pytest.raises(StorageCliError) as excinfo:
        _load_azure_profile_section("malformed-flag")

    message = str(excinfo.value)
    assert "not found" not in message
    assert "malformed" in message
    assert str(azure_profile_file) in message


@pytest.mark.usefixtures("azure_profile_file")
def test_a_connection_string_profile_builds_the_client_from_that_connection_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection string is handed to the SDK verbatim; nothing is unpacked from it here."""
    calls = _azure_sdk_recorder(monkeypatch)

    _make_azure_client("az://container/blob.mp4", "default")

    assert calls == [("from_connection_string", {"connection_string": _AZURE_CONNECTION_STRING})]


@pytest.mark.usefixtures("azure_profile_file")
def test_an_account_name_and_key_profile_builds_a_shared_key_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an explicit account URL the blob endpoint is derived from the account name."""
    calls = _azure_sdk_recorder(monkeypatch)

    _make_azure_client("az://container/blob.mp4", "prod")

    assert calls == [
        (
            "constructor",
            {
                "account_url": "https://prod-account.blob.core.windows.net",
                "credential": {"account_name": "prod-account", "account_key": _AZURE_ACCOUNT_KEY},
            },
        )
    ]


@pytest.mark.usefixtures("azure_profile_file")
def test_an_explicit_account_url_overrides_the_derived_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sovereign-cloud or emulator endpoint is only reachable through this override."""
    calls = _azure_sdk_recorder(monkeypatch)

    _make_azure_client("az://container/blob.mp4", "with-account-url")

    assert calls[0][1]["account_url"] == "https://prod.blob.example.invalid"


@pytest.mark.usefixtures("azure_profile_file")
def test_a_managed_identity_profile_builds_the_client_from_a_default_azure_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed identity means no key material in the profile at all, only an account URL."""
    calls = _azure_sdk_recorder(monkeypatch)

    _make_azure_client("az://container/blob.mp4", "managed")

    assert len(calls) == 1
    kind, kwargs = calls[0]
    assert kind == "constructor"
    assert kwargs["account_url"] == "https://managed.blob.example.invalid"
    assert isinstance(kwargs["credential"], _FakeAzureCredential)


@pytest.mark.usefixtures("azure_profile_file")
def test_a_connection_string_outranks_every_other_credential_in_the_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First tier: a profile carrying all three modes is resolved by its connection string."""
    calls = _azure_sdk_recorder(monkeypatch)

    _make_azure_client("az://container/blob.mp4", "everything")

    assert calls == [("from_connection_string", {"connection_string": _AZURE_CONNECTION_STRING})]


@pytest.mark.usefixtures("azure_profile_file")
def test_managed_identity_outranks_an_account_name_and_key_in_the_same_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second tier is managed identity, *not* the account key, despite the docstring order.

    A profile carrying both is resolved by identity and the key is ignored, so an operator
    who adds ``azure_use_managed_identity = true`` to a working key-based profile silently
    changes how it authenticates. This matches ``AzureClient.__init__`` in the storage
    layer, so the ordering survives the move.
    """
    calls = _azure_sdk_recorder(monkeypatch)

    _make_azure_client("az://container/blob.mp4", "managed-and-key")

    assert len(calls) == 1
    assert isinstance(calls[0][1]["credential"], _FakeAzureCredential)


@pytest.mark.usefixtures("azure_profile_file")
def test_managed_identity_without_an_account_url_is_a_storage_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is nothing to derive the endpoint from, so this cannot fall back silently."""
    _azure_sdk_recorder(monkeypatch)

    with pytest.raises(StorageCliError, match="azure_use_managed_identity set but azure_account_url missing"):
        _make_azure_client("az://container/blob.mp4", "managed-without-url")


@pytest.mark.usefixtures("azure_profile_file")
def test_a_profile_with_no_usable_credentials_is_a_storage_cli_error_listing_the_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-filled profile (name without key) is a common mistake and must name the fix."""
    _azure_sdk_recorder(monkeypatch)

    with pytest.raises(StorageCliError) as excinfo:
        _make_azure_client("az://container/blob.mp4", "unusable")

    message = str(excinfo.value)
    assert "Azure profile 'unusable' has no usable credentials" in message
    assert "azure_connection_string" in message
    assert "azure_account_name+azure_account_key" in message
    assert "azure_use_managed_identity" in message


@pytest.mark.usefixtures("azure_profile_file")
def test_a_missing_profile_now_arrives_with_the_credentials_hint() -> None:
    """DIVERGENCE (the hint used to be missing here, and now is not).

    The predecessor re-raised its own error type untouched and decorated only exceptions
    from elsewhere, so the two most likely operator mistakes -- no profile file, wrong
    profile name -- were exactly the two that never mentioned ``--azure-profile-name``.
    The replacement translates both lookup failures itself and appends the hint, which is
    a strict improvement: the hint's whole content is how to name a profile and how to
    populate the file.

    What is deliberately still absent is the ``could not configure Azure access for <uri>``
    prefix. It belongs to the SDK-failure path, where the URI is the only clue available;
    here the profile name is the actionable part and putting the URI first would bury it.
    """
    with pytest.raises(StorageCliError) as excinfo:
        _make_azure_client("az://container/blob.mp4", "no-such-profile")

    message = str(excinfo.value)
    assert "Azure profile 'no-such-profile' not found" in message
    assert storage_cli._AZURE_CREDENTIALS_HINT in message
    assert "could not configure Azure access" not in message


@pytest.mark.usefixtures("azure_profile_file")
def test_an_sdk_failure_is_wrapped_with_the_source_and_the_credentials_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed connection string comes out of the SDK, and that is the path that gets the hint."""
    _azure_sdk_recorder(monkeypatch, error=ValueError("Connection string is either blank or malformed."))

    with pytest.raises(StorageCliError) as excinfo:
        _make_azure_client("az://container/blob.mp4", "default")

    message = str(excinfo.value)
    assert "could not configure Azure access for 'az://container/blob.mp4'" in message
    assert "blank or malformed" in message
    assert storage_cli._AZURE_CREDENTIALS_HINT in message


@pytest.mark.usefixtures("azure_profile_file")
def test_a_connection_string_profile_yields_a_real_blob_service_client() -> None:
    """The recorded-constructor tests fake the SDK; this one proves the real SDK accepts the profile.

    No network: ``BlobServiceClient`` construction only parses the connection string and
    assembles the endpoint.
    """
    client = _make_azure_client("az://container/blob.mp4", "default")

    assert client.account_name == "curator-test"
    assert client.url.startswith("https://curator-test.blob.core.windows.net")


# --------------------------------------------------------------------------------------
# 4. list_storage_objects
# --------------------------------------------------------------------------------------

_S3_PAGES = [
    {"Contents": [{"Key": "root/"}, {"Key": "root/a.mp4"}, {"Key": "root/a.json"}]},
    {"Contents": [{"Key": "root/b.MP4"}, {"Key": "root/c.mp4"}]},
    {"Contents": [{"Key": "root/d.mp4"}]},
]


def test_the_s3_limit_counts_matching_keys_rather_than_listed_objects() -> None:
    """A caller asking for two videos must get two videos, not two objects.

    The first page alone holds a placeholder and a sidecar JSON, so a limit applied before
    filtering would return one video here.
    """
    list_objects, _ = _s3_lister(_S3_PAGES)

    assert list_objects("s3://bucket/root", limit=2, suffixes=(".mp4",)) == [
        "s3://bucket/root/a.mp4",
        "s3://bucket/root/b.MP4",
    ]


def test_the_s3_listing_stops_paging_once_the_limit_is_met() -> None:
    """The cap lives inside the pagination loop, which is what makes sampling a huge bucket cheap."""
    list_objects, fake = _s3_lister(_S3_PAGES)

    list_objects("s3://bucket/root", limit=2, suffixes=(".mp4",))

    assert fake.pages_yielded == 2


def test_a_zero_s3_limit_lists_every_match_across_every_page() -> None:
    """Zero means unlimited, and unlimited must page the whole prefix."""
    list_objects, fake = _s3_lister(_S3_PAGES)

    assert list_objects("s3://bucket/root", limit=0, suffixes=(".mp4",)) == [
        "s3://bucket/root/a.mp4",
        "s3://bucket/root/b.MP4",
        "s3://bucket/root/c.mp4",
        "s3://bucket/root/d.mp4",
    ]
    assert fake.pages_yielded == len(_S3_PAGES)


def test_s3_suffix_matching_ignores_the_case_of_the_key() -> None:
    """Cameras and transcoders write ``.MP4`` as often as ``.mp4``; both are videos."""
    list_objects, _ = _s3_lister([{"Contents": [{"Key": "root/a.MP4"}, {"Key": "root/b.Mp4"}]}])

    assert list_objects("s3://bucket/root", suffixes=(".mp4",)) == [
        "s3://bucket/root/a.MP4",
        "s3://bucket/root/b.Mp4",
    ]


def test_an_uppercase_s3_suffix_in_the_filter_now_matches() -> None:
    """DIVERGENCE (matching widened): an uppercase suffix used to match nothing.

    The predecessor lowercased the key but not the filter, so ``('.MKV',)`` returned a
    clean empty listing -- a caller's mistake answered with a plausible-looking result.
    The storage layer lowercases both sides, so the filter now means what it says.

    Widening, and unreachable in production: the only suffix filter in the tree is
    ``discovery.VIDEO_SUFFIXES``, a hardcoded all-lowercase constant, so no caller can
    observe the change. If one ever passes an uppercase suffix, matching is the answer it
    wanted.
    """
    list_objects, _ = _s3_lister([{"Contents": [{"Key": "root/a.mkv"}, {"Key": "root/b.MKV"}]}])

    assert list_objects("s3://bucket/root", suffixes=(".MKV",)) == [
        "s3://bucket/root/a.mkv",
        "s3://bucket/root/b.MKV",
    ]


def test_no_s3_suffix_filter_returns_every_object() -> None:
    """``None`` is the default and means "do not filter", not "filter with nothing"."""
    list_objects, _ = _s3_lister(_S3_PAGES)

    assert list_objects("s3://bucket/root", suffixes=None) == [
        "s3://bucket/root/a.mp4",
        "s3://bucket/root/a.json",
        "s3://bucket/root/b.MP4",
        "s3://bucket/root/c.mp4",
        "s3://bucket/root/d.mp4",
    ]


def test_an_empty_s3_suffix_tuple_is_now_rejected_rather_than_matching_nothing() -> None:
    """DIVERGENCE (an empty filter is now an error): it used to match nothing, silently.

    ``str.endswith(())`` is False for every key, so the predecessor answered a caller that
    computed its suffix tuple and got an empty one with a clean empty listing -- the one
    result indistinguishable from "the prefix is empty". The storage layer rejects it.

    Rejecting is the better answer and it is unreachable in production, since the tree's
    only filter is a non-empty constant. Note the type, though: this is a bare
    ``ValueError`` from the storage layer rather than a ``StorageCliError``, so a future
    caller computing suffixes at runtime would see it as a crash, not as a CLI complaint.
    ``None`` remains the supported way to spell "no filter" -- see the test above.
    """
    list_objects, _ = _s3_lister(_S3_PAGES)

    with pytest.raises(ValueError, match="suffix"):
        list_objects("s3://bucket/root", suffixes=())


def test_s3_directory_placeholder_keys_are_skipped() -> None:
    """The zero-byte ``prefix/`` keys some tools create are not objects a caller can open."""
    list_objects, _ = _s3_lister([{"Contents": [{"Key": "root/"}, {"Key": "root/nested/"}, {"Key": "root/a.mp4"}]}])

    assert list_objects("s3://bucket/root") == ["s3://bucket/root/a.mp4"]


def test_the_s3_listing_pages_the_bucket_and_key_prefix_split_out_of_the_uri() -> None:
    """The URI is split once, at the first slash, and the remainder is the key prefix verbatim."""
    list_objects, fake = _s3_lister(_S3_PAGES)

    list_objects("s3://bucket/root")

    assert fake.paginate_kwargs == {"Bucket": "bucket", "Prefix": "root"}


def test_an_s3_uri_without_a_bucket_is_a_storage_cli_error() -> None:
    """``s3://`` alone would otherwise page a bucket named the empty string."""
    list_objects, _ = _s3_lister(_S3_PAGES)

    with pytest.raises(StorageCliError, match=r"malformed s3 URI \(no bucket\)"):
        list_objects("s3://")


def test_a_bucket_root_pages_the_whole_bucket() -> None:
    """``s3://bucket`` names everything in it, with the key prefix simply empty."""
    list_objects, fake = _s3_lister(_S3_PAGES)

    list_objects("s3://bucket")

    assert fake.paginate_kwargs == {"Bucket": "bucket", "Prefix": ""}


def test_the_azure_limit_counts_matching_blobs_rather_than_listed_blobs() -> None:
    """Same promise as S3: the limit is denominated in matches."""
    list_objects, _ = _azure_lister(["root/", "root/a.mp4", "root/a.json", "root/b.mp4", "root/c.mp4"])

    assert list_objects("az://container/root", limit=2, suffixes=(".mp4",)) == [
        "az://container/root/a.mp4",
        "az://container/root/b.mp4",
    ]


def test_the_azure_listing_stops_consuming_once_the_limit_is_met() -> None:
    """Abandoning the paged iterator early is what keeps a small limit cheap."""
    list_objects, container = _azure_lister(["root/a.mp4", "root/b.mp4", "root/c.mp4"])

    list_objects("az://container/root", limit=1, suffixes=(".mp4",))

    assert container.blobs_yielded == 1


def test_azure_suffix_matching_ignores_the_case_of_the_blob_name() -> None:
    """The blob name is lowercased before comparison, exactly as the S3 key is."""
    list_objects, _ = _azure_lister(["root/a.MP4", "root/b.txt"])

    assert list_objects("az://container/root", suffixes=(".mp4",)) == ["az://container/root/a.MP4"]


def test_no_azure_suffix_filter_returns_every_blob() -> None:
    """``None`` means everything on this backend too."""
    list_objects, _ = _azure_lister(["root/a.mp4", "root/a.json"])

    assert list_objects("az://container/root") == [
        "az://container/root/a.mp4",
        "az://container/root/a.json",
    ]


def test_azure_directory_placeholder_blobs_are_skipped() -> None:
    """Hierarchical-namespace accounts really do return ``prefix/`` entries."""
    list_objects, _ = _azure_lister(["root/", "root/nested/", "root/a.mp4"])

    assert list_objects("az://container/root") == ["az://container/root/a.mp4"]


def test_the_azure_listing_starts_from_the_blob_prefix_split_out_of_the_uri() -> None:
    """The container is the first path segment; everything after it is the blob prefix."""
    list_objects, container = _azure_lister(["root/a.mp4"])

    list_objects("az://container/root")

    assert container.name_starts_with == "root"


def test_an_azure_uri_without_a_container_is_a_storage_cli_error() -> None:
    """Mirrors the S3 guard, for the same reason."""
    list_objects, _ = _azure_lister(["root/a.mp4"])

    with pytest.raises(StorageCliError, match=r"malformed az URI \(no container\)"):
        list_objects("az://")


@pytest.mark.parametrize("prefix", ["az://container", "az://container/"])
def test_a_container_root_lists_the_whole_container(prefix: str) -> None:
    """``az://container`` has to work the way ``s3://bucket`` does.

    ``AzurePrefix`` requires a blob component where ``S3Prefix`` tolerates an empty key,
    so unnormalized the same command shape succeeds on one backend and is rejected as
    malformed on the other.
    """
    list_objects, container = _azure_lister(["top.mp4", "root/a.mp4"])

    assert list_objects(prefix) == ["az://container/top.mp4", "az://container/root/a.mp4"]
    assert container.name_starts_with == ""


def test_listing_a_local_path_is_a_storage_cli_error() -> None:
    """Only cloud prefixes are listed here; a local path is a caller mistake, not a glob."""
    list_objects, _ = _s3_lister(_S3_PAGES)

    with pytest.raises(StorageCliError, match="list_storage_objects requires an s3:// or az:// URI"):
        list_objects("/data/sessions")


# --------------------------------------------------------------------------------------
# 5. get_lance_storage_options
# --------------------------------------------------------------------------------------


@pytest.mark.usefixtures("aws_profile_env")
def test_the_lance_options_carry_the_profile_credentials_region_and_endpoint() -> None:
    """Lance reads these keys verbatim, so both the names and the values are the contract.

    A store written with these options has to be readable by the pipeline side, which builds
    the same names from the storage package.
    """
    options = storage_cli.get_lance_storage_options(
        "s3://bucket/store",
        s3_profile_name="curator-test",
        endpoint_url="https://s3.example.invalid",
    )

    assert options == {
        "aws_access_key_id": "profile-key",
        "aws_secret_access_key": "profile-secret",
        "aws_region": "us-west-1",
        "aws_endpoint": "https://s3.example.invalid",
    }


@pytest.mark.usefixtures("aws_profile_env")
def test_the_lance_options_omit_an_absent_endpoint_and_session_token() -> None:
    """Falsy values are dropped rather than passed as ``None``, which Lance would reject."""
    options = storage_cli.get_lance_storage_options("s3://bucket/store", s3_profile_name="curator-test")

    assert options == {
        "aws_access_key_id": "profile-key",
        "aws_secret_access_key": "profile-secret",
        "aws_region": "us-west-1",
    }


@pytest.mark.usefixtures("aws_profile_env")
def test_the_lance_options_resolve_credentials_without_building_a_service_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing here issues a request, so an S3 service client would be pure overhead.

    Constructing one is roughly an order of magnitude dearer than the session it hangs
    off, and the store resolves options once per operation rather than threading them
    through, so the cost is paid several times per invocation. Invisible in behaviour,
    which is why it needs pinning: building an ``S3Client`` for the session would pass
    every other test in this section.
    """
    services_built: list[object] = []

    class RecordingSession(boto3.Session):
        def client(self, *args: object, **kwargs: object) -> object:
            services_built.append(args[0] if args else kwargs.get("service_name"))
            return super().client(*args, **kwargs)

    monkeypatch.setattr(s3_client_module.boto3, "Session", RecordingSession)

    options = storage_cli.get_lance_storage_options("s3://bucket/store", s3_profile_name="curator-test")

    assert options is not None
    assert services_built == []


@pytest.mark.usefixtures("aws_profile_env")
def test_the_lance_options_for_an_empty_profile_name_come_from_the_default_chain() -> None:
    """The store side has to normalize the empty profile the same way the client side does.

    Otherwise a run that opens the store and a run that reads an object disagree about
    what ``--s3-profile-name ""`` meant.
    """
    options = storage_cli.get_lance_storage_options("s3://bucket/store", s3_profile_name="")

    assert options == {
        "aws_access_key_id": "env-key",
        "aws_secret_access_key": "env-secret",
    }


@pytest.mark.usefixtures("aws_profile_env")
def test_the_lance_options_include_a_session_token_when_the_profile_has_one() -> None:
    """An assumed-role or SSO profile only works if its token is carried through.

    This profile also declares no region, which is what shows region is dropped on the same
    falsy rule rather than defaulted.
    """
    options = storage_cli.get_lance_storage_options("s3://bucket/store", s3_profile_name="tokened")

    assert options == {
        "aws_access_key_id": "session-key",
        "aws_secret_access_key": "session-secret",
        "aws_session_token": "session-token",
    }


@pytest.mark.usefixtures("aws_profile_env")
def test_the_lance_options_take_the_endpoint_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same three-tier endpoint resolution applies here as when building a client."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env.invalid")

    options = storage_cli.get_lance_storage_options("s3://bucket/store", s3_profile_name="curator-test")

    assert options is not None
    assert options["aws_endpoint"] == "https://env.invalid"


@pytest.mark.usefixtures("aws_profile_env")
def test_the_lance_options_come_from_frozen_credentials_not_the_live_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lance's object store cannot call back into boto3, so the chain has to be resolved here.

    The fake session's live attributes differ from its frozen view, so reading the live ones
    -- which would work fine for static keys and silently break for SSO or an assumed role --
    is visible rather than accidental.
    """
    monkeypatch.setattr(s3_client_module.boto3, "Session", _FakeSession)

    options = storage_cli.get_lance_storage_options("s3://bucket/store")

    assert options == {
        "aws_access_key_id": "frozen-key",
        "aws_secret_access_key": "frozen-secret",
        "aws_session_token": "frozen-token",
        "aws_region": "us-east-2",
    }


def test_a_local_store_path_has_no_lance_storage_options() -> None:
    """A local store needs no credentials, and ``None`` is what Lance wants for that."""
    assert storage_cli.get_lance_storage_options("/data/di-store") is None


def test_an_azure_store_uri_is_refused_rather_than_written_unauthenticated() -> None:
    """Lance's Azure options are a different set of keys, so a guess would surface at write time.

    Other tests in the data-integrity suite assert this refusal reaches the CLI, so the
    ``az://`` mention in the message is part of the contract.
    """
    with pytest.raises(StorageCliError, match=r"does not support az:// yet"):
        storage_cli.get_lance_storage_options("az://container/store")


@pytest.mark.usefixtures("aws_profile_env")
def test_lance_options_for_a_missing_profile_are_a_storage_cli_error_with_the_hint() -> None:
    """Same translation as client construction: a profile typo is advice, not a traceback."""
    with pytest.raises(StorageCliError) as excinfo:
        storage_cli.get_lance_storage_options("s3://bucket/store", s3_profile_name="no-such-profile")

    message = str(excinfo.value)
    assert "could not configure S3 access for 's3://bucket/store'" in message
    assert storage_cli._S3_CREDENTIALS_HINT in message


@pytest.mark.usefixtures("aws_empty_chain_env")
def test_lance_options_with_an_empty_credential_chain_are_a_storage_cli_error_with_the_hint() -> None:
    """An unauthenticated store would fail opaquely much later, so it is refused up front."""
    with pytest.raises(StorageCliError) as excinfo:
        storage_cli.get_lance_storage_options("s3://bucket/store")

    message = str(excinfo.value)
    assert "Unable to locate credentials" in message
    assert storage_cli._S3_CREDENTIALS_HINT in message


# --------------------------------------------------------------------------------------
# 6. open_storage_source
# --------------------------------------------------------------------------------------


def test_opening_an_s3_uri_yields_a_seekable_stream_of_the_object_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sensor readers seek, so a forward-only stream would fail on real footage."""
    open_source, opened = _source_opener(monkeypatch, b"mp4-bytes")
    s3_factory, _ = _client_factories(monkeypatch)

    with open_source("s3://bucket/root/a.mp4") as stream:
        assert stream.seekable()
        assert stream.read() == b"mp4-bytes"

    assert opened.calls == [("s3://bucket/root/a.mp4", "rb", {"client": s3_factory.client})]


def test_a_caller_supplied_s3_client_is_reused_rather_than_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuilding would drop the caller's botocore event hooks.

    ``cloud_io_benchmark`` attaches ``before-send.s3.GetObject`` to its own client and then
    hands it in; a helper that built a fresh one would measure an unhooked transfer and
    report nothing.
    """
    open_source, opened = _source_opener(monkeypatch)
    s3_factory, _ = _client_factories(monkeypatch)
    hooked_client = object()

    with open_source("s3://bucket/root/a.mp4", s3_client=hooked_client):
        pass

    assert opened.calls[0][2] == {"client": hooked_client}
    assert s3_factory.calls == []


def test_a_caller_supplied_azure_client_is_reused_rather_than_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same reuse contract on the other backend."""
    open_source, opened = _source_opener(monkeypatch)
    _, azure_factory = _client_factories(monkeypatch)
    prebuilt_client = object()

    with open_source("az://container/root/a.mp4", azure_client=prebuilt_client):
        pass

    assert opened.calls[0][2] == {"client": prebuilt_client}
    assert azure_factory.calls == []


def test_without_a_client_the_s3_client_is_built_from_the_profile_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The profile and endpoint arguments are the fallback path, and both must be forwarded."""
    open_source, opened = _source_opener(monkeypatch)
    s3_factory, _ = _client_factories(monkeypatch)

    with open_source(
        "s3://bucket/root/a.mp4",
        s3_profile_name="curator-test",
        endpoint_url="https://s3.example.invalid",
    ):
        pass

    assert s3_factory.calls == [("s3://bucket/root/a.mp4", "curator-test", "https://s3.example.invalid")]
    assert opened.calls[0][2] == {"client": s3_factory.client}


def test_without_a_client_the_azure_client_is_built_from_the_default_profile_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``az://`` sources default to the ``default`` profile rather than to no profile."""
    open_source, opened = _source_opener(monkeypatch)
    _, azure_factory = _client_factories(monkeypatch)

    with open_source("az://container/root/a.mp4"):
        pass

    assert azure_factory.calls == [("az://container/root/a.mp4", "default")]
    assert opened.calls[0][2] == {"client": azure_factory.client}


def test_an_az_uri_dispatches_to_the_azure_backend_and_never_to_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch is on the scheme alone, and the two backends must not both be consulted."""
    open_source, opened = _source_opener(monkeypatch, b"blob-bytes")
    s3_factory, azure_factory = _client_factories(monkeypatch)

    with open_source("az://container/root/a.mp4") as stream:
        assert stream.read() == b"blob-bytes"

    assert len(azure_factory.calls) == 1
    assert s3_factory.calls == []
    assert opened.calls[0][0] == "az://container/root/a.mp4"


@pytest.mark.parametrize("source", ["/data/sessions/a.mp4", "gs://bucket/a.mp4", "file:///data/a.mp4"])
def test_opening_anything_other_than_a_cloud_uri_is_a_storage_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """Local paths are rejected here on purpose: the callers open those directly."""
    open_source, opened = _source_opener(monkeypatch)

    with (
        pytest.raises(StorageCliError, match="open_storage_source requires an s3:// or az:// URI"),
        open_source(source),
    ):
        pass

    assert opened.calls == []


def test_the_stream_is_closed_when_the_context_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ownership stays with the context manager, so callers must not have to close it."""
    open_source, _ = _source_opener(monkeypatch)
    _client_factories(monkeypatch)

    with open_source("s3://bucket/root/a.mp4") as stream:
        assert not stream.closed

    assert stream.closed


def test_the_smart_open_stream_is_read_through_a_block_cache_and_closed_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Libav's per-fragment size probes must stop at the cache rather than reach smart_open.

    Each seek smart_open's S3 reader sees outside its read-ahead buffer is a fresh
    ``GetObject``, which made a fragmented MP4 cost one request per probe.
    """
    open_source, opened = _source_opener(monkeypatch, b"mp4-bytes")
    _client_factories(monkeypatch)

    with open_source("s3://bucket/root/a.mp4") as stream:
        assert isinstance(stream, BlockCacheReader)
        assert stream.read(3) == b"mp4"
        assert stream.seek(-1, io.SEEK_END) == len(b"mp4-bytes") - 1
        assert opened.streams[0].tell() == len(b"mp4-bytes")

    assert opened.streams[0].closed


def test_an_object_that_cannot_be_opened_raises_smart_opens_error_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers tell a 404 from a 403 by walking ``backend_error`` / ``__cause__`` on this very error."""
    error = OSError("unable to access bucket: 'bucket' key: 'root/missing.mp4'")
    error.backend_error = RuntimeError("NoSuchKey")  # type: ignore[attr-defined]

    def _fail(*_args: object, **_kwargs: object) -> io.BytesIO:
        raise error

    monkeypatch.setattr(storage_cli.smart_open, "open", _fail)
    _client_factories(monkeypatch)

    with (
        pytest.raises(OSError, match="unable to access bucket") as excinfo,
        storage_cli.open_storage_source("s3://bucket/root/missing.mp4"),
    ):
        pass

    assert excinfo.value is error


# --------------------------------------------------------------------------------------
# 7. is_s3path / is_azure_path / is_remote_path
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected_s3", "expected_azure"),
    [
        pytest.param("s3://bucket/key.mp4", True, False, id="an-s3-uri"),
        pytest.param("s3://", True, False, id="a-bare-s3-scheme-still-counts"),
        pytest.param("az://container/blob.mp4", False, True, id="an-az-uri"),
        pytest.param("S3://bucket/key.mp4", False, False, id="the-prefix-match-is-case-sensitive"),
        pytest.param(" s3://bucket/key.mp4", False, False, id="a-leading-space-is-not-stripped"),
        pytest.param("s3:/bucket/key.mp4", False, False, id="a-single-slash-is-not-an-s3-uri"),
        pytest.param("azure://container/blob.mp4", False, False, id="the-long-azure-scheme-is-not-recognised"),
        pytest.param("gs://bucket/key.mp4", False, False, id="google-cloud-storage-is-not-supported"),
        pytest.param("/data/sessions/a.mp4", False, False, id="a-local-path"),
        pytest.param("", False, False, id="the-empty-string"),
    ],
)
def test_the_uri_predicates_use_exact_prefix_semantics(
    source: str,
    expected_s3: bool,  # noqa: FBT001
    expected_azure: bool,  # noqa: FBT001
) -> None:
    """Scheme detection is a literal prefix test, with no normalisation of any kind."""
    assert is_s3path(source) is expected_s3
    assert is_azure_path(source) is expected_azure
    assert is_remote_path(source) is (expected_s3 or expected_azure)


def test_the_uri_predicates_now_answer_false_for_none_rather_than_raising() -> None:
    """DIVERGENCE (``None`` is now answered, not rejected): it used to raise ``AttributeError``.

    The predecessor's predicates called ``str`` methods on their argument unguarded, so a
    ``None`` that reached one crashed. The storage-layer predicates take ``str | None`` and
    answer ``False``, which was the change this assertion existed to catch.

    Safe because no in-tree caller relied on the exception: every one of them is an
    ``if``, and a ``None`` source reaching one is already rejected by ``validate_source``
    or by the argument parser upstream. The widening removes a crash rather than hiding a
    check.
    """
    assert is_remote_path(None) is False
    assert is_s3path(None) is False
    assert is_azure_path(None) is False


# --------------------------------------------------------------------------------------
# 8. validate_source
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["s3://bucket/key.mp4", "az://container/blob.mp4"])
def test_a_cloud_uri_validates_without_touching_the_filesystem(source: str) -> None:
    """Existence is not checked for cloud sources: that would cost a request per argument."""
    assert storage_cli.validate_source(source) is None


def test_an_existing_local_file_validates(tmp_path: pathlib.Path) -> None:
    """The local path is the common case and must stay allowed."""
    local = tmp_path / "a.mp4"
    local.write_bytes(b"bytes")

    assert storage_cli.validate_source(str(local)) is None


def test_an_unsupported_scheme_is_rejected_by_name() -> None:
    """Anything with a scheme that is not S3 or Azure is a mistake worth naming.

    Without this branch a ``gs://`` argument would be treated as a relative local path and
    reported as a missing file, which sends the operator looking in the wrong place.
    """
    with pytest.raises(StorageCliError, match=r"unsupported source URI 'gs://bucket/key\.mp4'"):
        storage_cli.validate_source("gs://bucket/key.mp4")


def test_a_missing_local_file_is_rejected(tmp_path: pathlib.Path) -> None:
    """Failing here beats failing inside a reader with a container-level error."""
    missing = tmp_path / "absent.mp4"

    with pytest.raises(StorageCliError, match=re.escape(f"source is not a file: {missing}")):
        storage_cli.validate_source(str(missing))


def test_a_local_directory_is_rejected_because_it_is_not_a_file(tmp_path: pathlib.Path) -> None:
    """A directory exists but cannot be opened as a video, so existence alone is not enough."""
    with pytest.raises(StorageCliError, match="source is not a file"):
        storage_cli.validate_source(str(tmp_path))


# --------------------------------------------------------------------------------------
# 9. add_storage_credential_args
# --------------------------------------------------------------------------------------


def test_the_credential_flags_default_to_the_documented_values() -> None:
    """These defaults are the difference between "boto3's chain" and "a named profile".

    ``--azure-profile-name`` defaults to a name while ``--s3-profile-name`` defaults to
    ``None``, because only the Azure path requires a profile to resolve at all.
    """
    parser = argparse.ArgumentParser()
    storage_cli.add_storage_credential_args(parser)

    args = parser.parse_args([])

    assert args.s3_profile_name is None
    assert args.azure_profile_name == "default"
    assert args.endpoint_url is None


def test_the_credential_flags_parse_supplied_values() -> None:
    """All three flags take a value, and the destinations are what the call sites read."""
    parser = argparse.ArgumentParser()
    storage_cli.add_storage_credential_args(parser)

    args = parser.parse_args(
        [
            "--s3-profile-name",
            "curator-test",
            "--azure-profile-name",
            "prod",
            "--endpoint-url",
            "https://s3.example.invalid",
        ]
    )

    assert args.s3_profile_name == "curator-test"
    assert args.azure_profile_name == "prod"
    assert args.endpoint_url == "https://s3.example.invalid"
