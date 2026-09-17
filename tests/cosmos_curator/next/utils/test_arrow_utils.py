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

"""Answers the shared Arrow helpers give, pinned at the helper rather than a caller.

``is_text_type`` gates two recipes' schema checks - one against a Lance schema,
one against a Parquet one - so its answers are pinned here, at the shared helper,
instead of through whichever caller happens to exercise it. Neither caller's own
gate can stand in for the cases below: a struct, a list or an int is refused by
any str-only predicate, so only the bytes-versus-str pairs distinguish this
predicate from one that merely asked whether a value renders at all.

The types are declared directly rather than written to a table, because the
predicate reads an Arrow type and nothing else - including types no producer in
this project would write.

``filesystem_path`` is pinned here for the same reason and one more: its two
branches disagree about percent-encoding, and only a test that spans both can
tell a correctly scoped decode from one hoisted over the passthrough.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.fs as pafs
import pytest

from cosmos_curator.next.utils import arrow_utils


def _s3_constructor_options(filesystem: pafs.S3FileSystem) -> dict[str, Any]:
    """Return the options an ``S3FileSystem`` was built with.

    Read back through the pickle reduction because PyArrow exposes no accessor
    for the endpoint or its transport scheme, and asserting on the constructed
    object rather than on a recorded call keeps Arrow's own defaulting in the
    answer - which is exactly what an unset scheme relies on.
    """
    reduction = cast("tuple[Any, ...]", filesystem.__reduce__())
    return cast("dict[str, Any]", reduction[1][0])


@pytest.mark.parametrize(
    "field_type",
    [pa.string(), pa.large_string(), pa.string_view(), pa.dictionary(pa.int32(), pa.string())],
    ids=["string", "large_string", "string_view", "dictionary_of_string"],
)
def test_is_text_type_accepts_every_encoding_a_scan_hands_back_as_str(field_type: pa.DataType) -> None:
    """A caller that renders a value needs it to arrive as ``str``, however it is encoded.

    The dictionary case is the recursive branch: it decodes to ``str`` on the way
    out of the scanner, so refusing it would reject a perfectly renderable column
    for its encoding.
    """
    assert arrow_utils.is_text_type(field_type) is True


@pytest.mark.parametrize(
    "field_type",
    [pa.binary(), pa.large_binary(), pa.dictionary(pa.int32(), pa.binary())],
    ids=["binary", "large_binary", "dictionary_of_binary"],
)
def test_is_text_type_refuses_bytes_however_renderable_they_look(field_type: pa.DataType) -> None:
    """Bytes stringify to ``"b'text'"``, so "can it be rendered" is the wrong question.

    This is the distinction the predicate exists for, and the only one a
    non-text refusal for some other reason (a struct, a list, an int) cannot
    stand in for: a binary column renders, and renders wrongly. The dictionary
    case pins the same answer through the recursive branch, where an encoding
    check that stopped at ``is_dictionary`` would accept it.
    """
    assert arrow_utils.is_text_type(field_type) is False


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("s3://bucket/a/b", "bucket/a/b"),
        ("S3://bucket/a/b", "bucket/a/b"),
        ("file:///var/lib/caption/a", "/var/lib/caption/a"),
        ("FILE:///var/lib/caption/a", "/var/lib/caption/a"),
        ("file://localhost/var/lib/caption/a", "/var/lib/caption/a"),
        ("/var/lib/caption/a", "/var/lib/caption/a"),
        ("relative/a", "relative/a"),
        ("//host[0]/share", "//host[0]/share"),
        ("/var/lib/caption/a:b", "/var/lib/caption/a:b"),
    ],
    ids=[
        "s3",
        "case_variant_s3",
        "file_uri",
        "case_variant_file_uri",
        "localhost_file_uri",
        "absolute_path",
        "relative_path",
        "parser_hostile_bare_path",
        "colon_after_first_slash_bare_path",
    ],
)
def test_filesystem_path_strips_only_the_scheme_that_selected_the_filesystem(
    location: str,
    expected: str,
) -> None:
    """An Arrow filesystem rejects the scheme it was itself selected by.

    The two bare-path cases are not filler: they pin that a location carrying no
    scheme is returned untouched, so the helper cannot be reimplemented as an
    unconditional prefix strip.
    """
    assert arrow_utils.filesystem_path(location) == expected


@pytest.mark.parametrize(
    "location",
    ["//host[0]/share", "/var/lib/caption/a:b", "file:///tmp/x", "file://localhost/tmp/x"],
    ids=[
        "bracket_bare_path",
        "colon_after_first_slash_bare_path",
        "empty_file_authority",
        "localhost_file_authority",
    ],
)
def test_arrow_filesystem_accepts_local_path_forms(location: str) -> None:
    """A bare path or explicitly local file URI selects the local filesystem."""
    assert isinstance(arrow_utils.arrow_filesystem(location), pafs.LocalFileSystem)


def test_arrow_filesystem_normalizes_case_variant_s3_before_profile_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A case-variant S3 URI must still resolve credentials from the requested profile."""
    lookups: list[tuple[str, str]] = []

    def record_profile_lookup(location: str, *, profile_name: str) -> dict[str, str]:
        lookups.append((location, profile_name))
        return {"aws_region": "us-east-1"}

    monkeypatch.setattr(arrow_utils, "get_lance_storage_options", record_profile_lookup)

    filesystem = arrow_utils.arrow_filesystem("S3://bucket/key", storage_profile="curation")

    assert isinstance(filesystem, pafs.S3FileSystem)
    assert lookups == [("s3://bucket/key", "curation")]


def test_gcs_s3_endpoint_requires_response_checksum_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GCS S3 endpoint requires validation only for required checksums."""
    monkeypatch.setattr(
        arrow_utils,
        "get_lance_storage_options",
        lambda *_args, **_kwargs: {"aws_endpoint": "https://storage.googleapis.com"},
        raising=True,
    )

    result = arrow_utils.s3_compatibility_env_vars("s3://bucket/checkpoints")

    assert result == {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"}


def test_one_location_needing_compatibility_settles_the_whole_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A location needing no mode cannot cancel one that does.

    The environment is per process and a worker opens all of the locations it is
    given, so the result is their union: a caller passing a local staging root
    beside a GCS table must still get the table's requirement.
    """
    monkeypatch.setattr(
        arrow_utils,
        "get_lance_storage_options",
        lambda *_args, **_kwargs: {"aws_endpoint": "https://storage.googleapis.com"},
        raising=True,
    )

    result = arrow_utils.s3_compatibility_env_vars(str(tmp_path / "staging"), "s3://bucket/clips.lance")

    assert result == {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"}


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://s3.us-east-1.amazonaws.com",
        "https://minio.example.invalid",
        "https://storage.googleapis.com.attacker.invalid",
        None,
    ],
    ids=["aws", "other_compatible_service", "hostname_suffix_attack", "no_endpoint"],
)
def test_non_gcs_s3_endpoints_keep_default_response_checksum_validation(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str | None,
) -> None:
    """Only the exact GCS service hostname selects the compatibility mode."""
    monkeypatch.setattr(
        arrow_utils,
        "get_lance_storage_options",
        lambda *_args, **_kwargs: {"aws_endpoint": endpoint} if endpoint else {},
        raising=True,
    )

    assert arrow_utils.s3_compatibility_env_vars("s3://bucket/checkpoints") == {}


def test_non_s3_location_needs_no_s3_profile_for_checksum_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A local location must not resolve an irrelevant S3 profile."""
    profile_lookup = Mock(side_effect=AssertionError("local paths need no S3 profile"))
    monkeypatch.setattr(arrow_utils, "get_lance_storage_options", profile_lookup, raising=True)

    assert arrow_utils.s3_compatibility_env_vars(str(tmp_path / "checkpoints")) == {}
    profile_lookup.assert_not_called()


def test_arrow_filesystem_sets_gcs_response_checksum_compatibility_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the GCS filesystem configures its process-local AWS client.

    The write is process-global and is not undone, which this pins as accepted
    rather than incidental: the SDK resolves the value once per client
    construction and offers no per-client option, and a run resolves a single
    storage profile, so no later client in the process reaches a different
    endpoint under it.
    """
    monkeypatch.delenv("AWS_RESPONSE_CHECKSUM_VALIDATION", raising=False)
    monkeypatch.setattr(
        arrow_utils,
        "get_lance_storage_options",
        lambda *_args, **_kwargs: {
            "aws_endpoint": "https://storage.googleapis.com",
            "aws_region": "auto",
        },
        raising=True,
    )

    arrow_utils.arrow_filesystem("s3://bucket/checkpoints")

    assert os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] == "when_required"


def test_arrow_filesystem_leaves_checksum_environment_untouched_for_non_gcs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An AWS endpoint keeps the caller's checksum-validation setting."""
    monkeypatch.setenv("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_supported")
    monkeypatch.setattr(
        arrow_utils,
        "get_lance_storage_options",
        lambda *_args, **_kwargs: {
            "aws_endpoint": "https://s3.us-east-1.amazonaws.com",
            "aws_region": "us-east-1",
        },
        raising=True,
    )

    arrow_utils.arrow_filesystem("s3://bucket/checkpoints")

    assert os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] == "when_supported"


@pytest.mark.parametrize(
    ("endpoint", "expected_transport_scheme"),
    [("http://127.0.0.1:9000", "http"), ("127.0.0.1:9000", "https")],
    ids=["endpoint_carries_a_transport_scheme", "endpoint_omits_one"],
)
def test_arrow_filesystem_takes_the_s3_transport_scheme_only_from_the_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    expected_transport_scheme: str,
) -> None:
    """The endpoint's own scheme selects the transport; the URI scheme never does.

    The second case is what separates the two. A scheme-less endpoint must leave
    Arrow's ``https`` default standing, so reusing ``s3`` - the scheme that
    selected this filesystem in the first place - would name a transport no
    client speaks. The host is reached either way, so ``endpoint_override``
    alone cannot tell the two apart.
    """
    monkeypatch.setattr(
        arrow_utils,
        "get_lance_storage_options",
        lambda *_args, **_kwargs: {"aws_endpoint": endpoint, "aws_region": "us-east-1"},
    )

    options = _s3_constructor_options(arrow_utils.arrow_filesystem("s3://bucket/key"))

    assert options["scheme"] == expected_transport_scheme
    assert options["endpoint_override"] == "127.0.0.1:9000"


@pytest.mark.parametrize(
    ("scheme", "location"),
    [
        ("gs", "gs://bucket/path"),
        ("abfs", "abfs://bucket/path"),
        ("s3a", "s3a://bucket/path"),
        ("gs", "gs://bucket[0]/path"),
    ],
    ids=["gs", "abfs", "s3a", "parser_hostile_gs"],
)
def test_arrow_filesystem_refuses_unsupported_uri_schemes(scheme: str, location: str) -> None:
    """An unsupported URI must not select the local filesystem."""
    with pytest.raises(ValueError, match=rf"unsupported Arrow filesystem URI scheme {scheme!r}"):
        arrow_utils.arrow_filesystem(location)


@pytest.mark.parametrize(
    ("scheme", "location"),
    [
        ("gs", "gs://bucket/path"),
        ("abfs", "abfs://bucket/path"),
        ("s3a", "s3a://bucket/path"),
        ("gs", "gs://bucket[0]/path"),
    ],
    ids=["gs", "abfs", "s3a", "parser_hostile_gs"],
)
def test_filesystem_path_refuses_unsupported_uri_schemes(scheme: str, location: str) -> None:
    """An unsupported URI must not pass through as a local filesystem path."""
    with pytest.raises(ValueError, match=rf"unsupported Arrow filesystem URI scheme {scheme!r}"):
        arrow_utils.filesystem_path(location)


@pytest.mark.parametrize(
    "operation",
    [arrow_utils.arrow_filesystem, arrow_utils.filesystem_path],
    ids=["filesystem_selection", "path_conversion"],
)
def test_unsupported_scheme_error_does_not_expose_the_location(operation: Callable[[str], object]) -> None:
    """An unsupported URI error identifies its scheme without exposing location details."""
    location = "gs://private-bucket[0]/clip.mp4?signature=secret"

    with pytest.raises(ValueError, match="unsupported Arrow filesystem URI scheme 'gs'") as error_info:
        operation(location)

    assert "private-bucket" not in str(error_info.value)
    assert "signature" not in str(error_info.value)


@pytest.mark.parametrize(
    "operation",
    [arrow_utils.arrow_filesystem, arrow_utils.filesystem_path],
    ids=["filesystem_selection", "path_conversion"],
)
@pytest.mark.parametrize("authority", ["server", "SERVER"], ids=["lowercase", "case_variant"])
def test_filesystem_helpers_refuse_nonlocal_file_uri_authorities(
    operation: Callable[[str], object],
    authority: str,
) -> None:
    """A file URI naming another host must not degrade into a local path."""
    with pytest.raises(ValueError, match="file URI authority must be empty or localhost"):
        operation(f"file://{authority}/path")


@pytest.mark.parametrize(
    "operation",
    [arrow_utils.arrow_filesystem, arrow_utils.filesystem_path],
    ids=["filesystem_selection", "path_conversion"],
)
@pytest.mark.parametrize(
    "location",
    ["file:///data/runs?copy=1", "file:///data/runs#2", "file://localhost/data/runs#2"],
    ids=["query", "fragment", "fragment_under_localhost_authority"],
)
def test_filesystem_helpers_refuse_a_file_uri_with_an_unencoded_delimiter(
    operation: Callable[[str], object],
    location: str,
) -> None:
    """An unencoded ``?`` or ``#`` truncates the path instead of failing.

    Every one of these names ``/data/runs`` once parsed, a directory that may
    well exist and hold another run's output, so accepting them publishes to a
    location nobody wrote. Refusing costs no expressible name: %3F and %23 both
    survive into the path, and a bare local path carries either character
    literally.
    """
    with pytest.raises(ValueError, match=r"percent-encode '\?' as %3F"):
        operation(location)


def test_filesystem_path_keeps_a_question_mark_inside_an_s3_key() -> None:
    """The delimiter refusal is scoped to ``file://``, because an S3 key is not a path.

    Negative control for the rule above: this module slices an S3 key rather than
    parsing it, so ``?`` there is an ordinary key byte. A gate that rejected the
    character everywhere would refuse a key S3 stores happily.
    """
    assert arrow_utils.filesystem_path("s3://bucket/run?copy=1/part.parquet") == "bucket/run?copy=1/part.parquet"


@pytest.mark.parametrize(
    "operation",
    [arrow_utils.arrow_filesystem, arrow_utils.filesystem_path],
    ids=["filesystem_selection", "path_conversion"],
)
@pytest.mark.parametrize("location", ["s3://", "s3:///key"], ids=["empty", "path_without_bucket"])
def test_filesystem_helpers_refuse_s3_uris_without_a_bucket(
    operation: Callable[[str], object],
    location: str,
) -> None:
    """An S3 URI without authority cannot name a bucket-backed filesystem path."""
    with pytest.raises(ValueError, match="S3 URI must include a bucket authority"):
        operation(location)


@pytest.mark.parametrize(
    "operation",
    [arrow_utils.arrow_filesystem, arrow_utils.filesystem_path],
    ids=["filesystem_selection", "path_conversion"],
)
@pytest.mark.parametrize(
    ("scheme", "location"),
    [("s3", "s3:/bucket/run"), ("file", "file:/tmp/run"), ("gs", "gs:/bucket/x")],
    ids=["s3", "file", "scheme_this_module_cannot_serve"],
)
def test_filesystem_helpers_refuse_a_scheme_written_without_its_double_slash(
    operation: Callable[[str], object],
    scheme: str,
    location: str,
) -> None:
    """A one-slash URI claims a scheme, so it cannot be honoured as a bare local path.

    This is the refusal both helpers exist for: read as a path, ``s3:/bucket/run``
    selects the local filesystem and the path passes through untouched, so a run
    aimed at object storage writes to the machine it happens to run on and reports
    success. The third case is what separates a general rule from a pair of
    hardcoded prefixes - ``gs`` must be refused without this module recognising
    the name.
    """
    with pytest.raises(ValueError, match=rf"scheme {scheme!r} requires a {scheme}:// prefix"):
        operation(location)


@pytest.mark.parametrize(
    "operation",
    [arrow_utils.arrow_filesystem, arrow_utils.filesystem_path],
    ids=["filesystem_selection", "path_conversion"],
)
def test_filesystem_helpers_refuse_a_relative_path_whose_first_segment_claims_a_scheme(
    operation: Callable[[str], object],
) -> None:
    """A colon before the first ``/`` is a claimed scheme, whatever the author meant.

    ``my:dir/file`` is a legal relative path and an indistinguishable malformed
    URI, so it is refused rather than guessed at: honouring the path reading is
    exactly what let a mistyped ``s3:/bucket`` reach local disk. The accepted
    counterpart is a colon after the first ``/``, which no scheme can occupy.
    """
    with pytest.raises(ValueError, match="scheme 'my' requires a my:// prefix"):
        operation("my:dir/file")


@pytest.mark.parametrize(
    "operation",
    [arrow_utils.arrow_filesystem, arrow_utils.filesystem_path],
    ids=["filesystem_selection", "path_conversion"],
)
@pytest.mark.parametrize(
    "location",
    ["s3://bucket[0]/key", "not a scheme://value"],
    ids=["invalid_authority", "invalid_scheme_token"],
)
def test_filesystem_helpers_refuse_malformed_canonical_uris(
    operation: Callable[[str], object],
    location: str,
) -> None:
    """Canonical URI-looking text must fail instead of degrading to a local path."""
    with pytest.raises(ValueError, match="malformed Arrow filesystem URI"):
        operation(location)


def test_filesystem_path_decodes_a_file_uri_onto_the_directory_it_names(tmp_path: Path) -> None:
    """``%20`` in a ``file://`` URI is an escaped space, not part of the name.

    Asserted against a directory that really exists, because the whole point is
    that an Arrow filesystem must be able to open what comes back; a string
    comparison alone would not notice a path nothing can reach.
    """
    spaced = tmp_path / "my staging root"
    spaced.mkdir()
    assert "%20" in spaced.as_uri()

    assert Path(arrow_utils.filesystem_path(spaced.as_uri())) == spaced


def test_filesystem_path_leaves_a_percent_escape_in_a_bare_path_alone(tmp_path: Path) -> None:
    """A bare path is not a URI, so ``%20`` in one is two literal characters.

    Both directories are created so the hazard is the real one: a decode reaching
    this branch would not merely mangle a name, it would name the sibling - a
    different directory that exists and holds someone else's staged rows.
    """
    literal = tmp_path / "my%20dir"
    literal.mkdir()
    (tmp_path / "my dir").mkdir()

    assert Path(arrow_utils.filesystem_path(str(literal))) == literal


def test_filesystem_path_decodes_a_file_uri_exactly_once() -> None:
    """``%2520`` is an escaped ``%20``, so one decode is the whole contract.

    A second pass would resolve it to a space, so this separates a correct decode
    from a doubled one - which the escaped-space case above cannot, since both
    give the same answer there.
    """
    assert arrow_utils.filesystem_path("file:///staging/my%2520dir") == "/staging/my%20dir"


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        pytest.param("/staging/", "/staging", id="trailing separator"),
        pytest.param("/staging///", "/staging", id="repeated separators"),
        pytest.param("  /staging  ", "/staging", id="surrounding whitespace"),
        pytest.param("/staging", "/staging", id="already normal"),
        pytest.param("file:///tmp///", "file:///tmp", id="file uri"),
        pytest.param("FILE:///tmp/", "file:///tmp", id="case variant file uri"),
        pytest.param("file://localhost/tmp/", "file:///tmp", id="localhost file authority"),
        pytest.param("s3://bucket///", "s3://bucket", id="s3 uri"),
    ],
)
def test_normalize_location_root_collapses_spellings_of_one_directory(location: str, expected: str) -> None:
    """Two spellings of one root must normalize to one, or derived work lands in two places."""
    assert arrow_utils.normalize_location_root(location) == expected


@pytest.mark.parametrize("root", ["/", "file:///", "  file:///  "])
def test_normalize_location_root_keeps_a_filesystem_roots_final_slash(root: str) -> None:
    """At a root the final slash is not redundant - it is what makes the path absolute.

    The case a bare ``rstrip`` gets wrong, and gets wrong silently: ``"/"`` would
    become ``""``, which resolves against the process working directory (a Ray
    worker's is not the driver's), and ``"file:///"`` would become
    ``"file://"``, an authority rather than a path.
    """
    assert arrow_utils.normalize_location_root(root) == root.strip()
