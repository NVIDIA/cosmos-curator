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

"""Arrow-level helpers shared by Curator Next recipes.

One layer below ``lance_utils``: a helper belongs here when its answer depends on
PyArrow alone, and would be the same whether the bytes came from Lance, Parquet,
or neither. ``is_text_type`` is asked about a Lance schema by one recipe's
preflight and about a Parquet schema by another's discovery, so it cannot be
scoped to Lance without making one of those callers import a storage format it
never touches.

Distinct from ``cosmos_curator.core.utils.arrow_utils``, which owns
consumer-agnostic Arrow table manipulation for any caller in the repository.
This module is the Curator Next seam, so it may carry a run's storage-profile
convention - which a core utility, serving callers that have no such notion,
must not.
"""

import os
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.fs as pafs

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options

# The Arrow encodings a scan hands back as ``str``. Every one of them also
# survives a parquet round trip as itself.
_TEXT_TYPE_PREDICATES = (pa.types.is_string, pa.types.is_large_string, pa.types.is_string_view)
_SUPPORTED_FILESYSTEM_SCHEMES = frozenset({"file", "s3"})
_LOCAL_FILE_AUTHORITIES = frozenset({"", "localhost"})
_GCS_S3_ENDPOINT_HOSTNAME = "storage.googleapis.com"
_AWS_RESPONSE_CHECKSUM_VALIDATION = "AWS_RESPONSE_CHECKSUM_VALIDATION"
_WHEN_REQUIRED = "when_required"


def is_text_type(field_type: pa.DataType) -> bool:
    """Return whether a scanned value of this Arrow type reaches Python as ``str``.

    Accepted: ``string``, ``large_string``, ``string_view``, and a dictionary of
    any of those. Two facts the Arrow type alone does not give away:

    - ``binary`` is refused. Its values arrive as ``bytes``, which stringify to
      ``"b'text'"`` rather than to the text, so a check that only asked whether a
      value can be rendered at all would let it through.
    - A dictionary of an accepted type decodes to ``str`` on the way out of the
      scanner, so it is accepted rather than refused - measured against a
      Lance-stored ``dictionary<values=string>`` column, not assumed from the
      Arrow type alone.

    """
    if pa.types.is_dictionary(field_type):
        return is_text_type(field_type.value_type)
    return any(predicate(field_type) for predicate in _TEXT_TYPE_PREDICATES)


def _claimed_uri_scheme(location: str) -> str:
    """Return the URI scheme ``location`` claims, or ``""`` if it claims none."""
    try:
        return urlsplit(location).scheme.lower()
    except ValueError:
        # The parser only rejects an authority it has already started reading, so
        # a failure here means a ``//`` came first and no scheme preceded it - a
        # bare path. A location that does claim a scheme is parsed again in the
        # canonical branch below, which reports its own parse failures.
        return ""


def _validated_filesystem_scheme(location: str) -> str:
    """Return the supported canonical filesystem URI scheme from ``location``."""
    # Filesystem URIs in this API use ``scheme://``. Splitting that delimiter
    # before parsing preserves parser-hostile POSIX names such as
    # ``//host[0]/share`` while urlsplit remains the authority on URI syntax.
    raw_scheme, separator, _ = location.partition("://")
    if not separator:
        # Text without the delimiter is a bare local path only if it claims no
        # scheme at all. The parser answers that, and prefix-matching the schemes
        # this module serves must not: refusing only ``s3:/`` and ``file:/`` would
        # leave ``gs:/bucket`` - and every scheme added later - silently selecting
        # local disk. A scheme has to precede the first ``/``, so ``/tmp/a:b``
        # remains a path while ``s3:/bucket`` is refused.
        claimed = _claimed_uri_scheme(location)
        if not claimed:
            return ""
        msg = (
            f"Arrow filesystem URI scheme {claimed!r} requires a {claimed}:// prefix; "
            f"supported forms are bare local paths, file://, and s3://"
        )
        raise ValueError(msg)

    scheme = urlsplit(f"{raw_scheme}:").scheme.lower()
    if not scheme or scheme != raw_scheme.lower():
        msg = "malformed Arrow filesystem URI; supported forms are bare local paths, file://, and s3://"
        raise ValueError(msg)
    if scheme not in _SUPPORTED_FILESYSTEM_SCHEMES:
        supported = "bare local paths, file://, and s3://"
        msg = f"unsupported Arrow filesystem URI scheme {scheme!r}; supported forms are {supported}"
        raise ValueError(msg)
    try:
        parsed = urlsplit(location)
    except ValueError as error:
        msg = f"malformed Arrow filesystem URI for scheme {scheme!r}"
        raise ValueError(msg) from error
    if scheme == "file" and parsed.netloc.lower() not in _LOCAL_FILE_AUTHORITIES:
        msg = "file URI authority must be empty or localhost"
        raise ValueError(msg)
    if scheme == "file" and (parsed.query or parsed.fragment):
        # ``?`` and ``#`` are URI delimiters, so everything from the first one
        # onwards is split out of ``path`` - and ``filesystem_path`` returns that
        # path. Both are legal in a POSIX name, so an unencoded one does not
        # fail: it names a SHORTER directory that may well exist, publishing a
        # run's output beside the location that was written. Scoped to ``file``
        # because an S3 key is literal bytes this module slices rather than
        # parses, so a ``?`` there is part of the key and not a delimiter.
        msg = "file URI must percent-encode '?' as %3F and '#' as %23"
        raise ValueError(msg)
    if scheme == "s3" and not parsed.netloc:
        msg = "S3 URI must include a bucket authority"
        raise ValueError(msg)
    return scheme


def _s3_storage_options(location: str, storage_profile: str) -> dict[str, str]:
    """Return one S3 location's normalized storage-profile options."""
    # The storage-profile resolver recognizes the canonical lowercase prefix.
    # Normalize only that prefix so case-variant S3 URIs cannot fall back to
    # PyArrow's ambient credential chain.
    profile_location = f"s3://{location[len('s3://') :]}"
    return get_lance_storage_options(profile_location, profile_name=storage_profile) or {}


def _s3_endpoint_compatibility_env_vars(endpoint: str | None) -> dict[str, str]:
    """Return the AWS SDK environment required by one S3-compatible endpoint."""
    if not endpoint:
        return {}
    parsed = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    hostname = parsed.hostname
    if hostname is None or hostname.lower() != _GCS_S3_ENDPOINT_HOSTNAME:
        return {}
    return {_AWS_RESPONSE_CHECKSUM_VALIDATION: _WHEN_REQUIRED}


def s3_compatibility_env_vars(
    *locations: str,
    storage_profile: str = "default",
) -> dict[str, str]:
    """Return the AWS SDK environment required by a set of filesystem locations.

    Takes every location the run will open rather than one, because the value it
    selects is set per PROCESS while a worker typically reads and writes several
    locations: passing only the one the caller happens to own leaves a worker
    reading a second location under the SDK default that location cannot serve.
    A location needing no compatibility mode contributes nothing, and the union
    cannot disagree with itself because the whole range of a per-location result
    is one fixed name/value pair, so no location can ask for a different value
    for a name another location set.

    The list establishes only that the run reaches S3 at all, since the endpoint
    is resolved from the profile rather than from any path. It is therefore not a
    completeness claim: per-row media URIs are not knowable here, so a run whose
    configured roots are all local can still open S3 objects.

    Args:
        locations: Bare local paths or canonical ``scheme://`` filesystem URIs.
        storage_profile: Credential profile to resolve S3 endpoints from.

    Returns:
        The environment every worker must be started with, empty when no
        location needs a compatibility mode.

    Raises:
        ValueError: If any location is malformed or names an unsupported scheme.

    """
    env_vars: dict[str, str] = {}
    for location in locations:
        if _validated_filesystem_scheme(location) != "s3":
            continue
        options = _s3_storage_options(location, storage_profile)
        env_vars.update(_s3_endpoint_compatibility_env_vars(options.get("aws_endpoint")))
    return env_vars


def arrow_filesystem(location: str, *, storage_profile: str = "default") -> pafs.FileSystem:
    """Return the Arrow filesystem a URI is written and read through.

    Built from the run's storage profile rather than left to Arrow's own default
    credential chain: a profile that is not the ambient one would otherwise write
    to a location the same run cannot read back, and the failure would surface as
    an empty read rather than as an access error. GCS S3 endpoints also set their
    required response-checksum compatibility mode in the process environment
    before PyArrow constructs its embedded AWS client.

    Args:
        location: A bare local path or canonical ``scheme://`` filesystem URI.
            The scheme selects the filesystem; ``file://`` authorities are also
            validated as empty or ``localhost``, and ``s3://`` requires a bucket
            authority.
        storage_profile: Credential profile to resolve S3 options from.

    Returns:
        A local or S3 filesystem, matching ``location``'s scheme.

    Raises:
        ValueError: If ``location`` is malformed, claims a scheme without the
            ``scheme://`` delimiter, has an unsupported scheme, names a
            non-local ``file://`` authority or an S3 URI without a bucket, or
            is a ``file://`` URI carrying an unencoded ``?`` or ``#``.

    """
    scheme = _validated_filesystem_scheme(location)
    if scheme != "s3":
        return pafs.LocalFileSystem()
    options = _s3_storage_options(location, storage_profile)
    endpoint = options.get("aws_endpoint")
    # Written into the process environment and deliberately left there. PyArrow
    # exposes no per-client checksum option and carries none through the pickle
    # reduction a worker rebuilds the filesystem from, while the embedded AWS SDK
    # resolves the value once per client construction - so the environment is the
    # only lever there is. A run resolves exactly one storage profile, so no later
    # client in this process reaches a different endpoint under the value set here.
    os.environ.update(_s3_endpoint_compatibility_env_vars(endpoint))
    endpoint_override = endpoint
    # The transport scheme is the endpoint's own, never ``scheme`` above: that one
    # named the filesystem to build, and handing "s3" to Arrow as a transport
    # would name a protocol no client speaks. Left unset, Arrow applies https.
    endpoint_scheme: str | None = None
    if endpoint and "://" in endpoint:
        parsed = urlsplit(endpoint)
        endpoint_scheme, endpoint_override = parsed.scheme, parsed.netloc + parsed.path
    return pafs.S3FileSystem(
        access_key=options.get("aws_access_key_id"),
        secret_key=options.get("aws_secret_access_key"),
        session_token=options.get("aws_session_token"),
        region=options.get("aws_region"),
        scheme=endpoint_scheme,
        endpoint_override=endpoint_override,
    )


def filesystem_path(location: str) -> str:
    """Return ``location`` without its scheme, as an Arrow filesystem expects.

    The counterpart to ``arrow_filesystem``: a caller reaching one URI needs the
    filesystem from that function and the path from this one, because an Arrow
    filesystem rejects the scheme it was itself selected by.

    A ``file://`` URI is percent-decoded, matching
    ``storage_utils._local_path_from_uri``; bare local paths are returned as
    written. Unsupported schemes, a scheme written without ``://``, non-local
    ``file://`` authorities, and a ``file://`` URI with an unencoded ``?`` or
    ``#`` are rejected before Arrow can interpret them as local paths.
    """
    scheme = _validated_filesystem_scheme(location)
    if scheme == "s3":
        # An S3 key is literal bytes rather than a URI path, and every other
        # reader of one in this repository slices the string. Decoding here would
        # rewrite the key instead of revealing it.
        return location[len("s3://") :]
    if scheme == "file":
        # Scoped to the parsed-URI branch on purpose. A bare path is not a URI,
        # so hoisting this decode over the passthrough below would rewrite a
        # directory literally named "my%20dir" to "my dir" - reading nothing, or
        # worse, a different directory that happens to exist.
        return unquote(urlsplit(location).path)
    return location


def normalize_location_root(location: str) -> str:
    """Return ``location`` in the one spelling that segments may be appended to.

    Surrounding whitespace and redundant trailing separators are removed, so two
    spellings of one directory join to one URI. Callers that derive a durable
    identity from the joined path need that: without it a configured
    ``"/staging/"`` and ``"/staging"`` name two different places, and work meant
    for one lands in both.

    A filesystem ROOT keeps its final slash, because there the slash is not
    redundant - it is what makes the path absolute. Stripping it would turn
    ``"/"`` into ``""`` (which resolves against the process working directory,
    and a Ray worker's is not the driver's) and ``"file:///"`` into
    ``"file://"`` (an authority, not a path).

    Lives beside :func:`filesystem_path` rather than with its callers because
    the two must agree on what a ``file://`` URI's path is; a private copy could
    be "simplified" to a bare ``rstrip`` and silently disagree.

    Returns:
        The normalized location, unvalidated - an unsupported scheme is
        :func:`filesystem_path`'s or :func:`arrow_filesystem`'s to reject, and a
        blank result is the caller's, since only the caller knows whether a
        location was required.

    """
    normalized = location.strip()
    if normalized == "/":
        return normalized
    try:
        scheme = _validated_filesystem_scheme(normalized)
    except ValueError:
        return normalized.rstrip("/")
    if scheme == "file":
        path = filesystem_path(normalized)
        if path == "/":
            return "file:///"
        canonical = f"file://{path}"
        return canonical.rstrip("/")
    if scheme == "s3":
        _, _, rest = normalized.partition("://")
        return f"s3://{rest}".rstrip("/")
    return normalized.rstrip("/")
