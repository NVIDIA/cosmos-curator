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

"""What ``local_media_path`` accepts as a local location, and what it refuses.

Every case here is a way a URI can *look* resolvable while naming something
else: a scheme the helper cannot honour, an authority on another host, a
delimiter that truncates the path. The refusals matter more than the
acceptances, because each one is silent when it is missing - the caller gets a
plausible ``Path`` and reads or writes the wrong object.

The same rules reach storage through ``storage.write_if_absent`` and
``storage.artifact_uri``, whose own tests in ``test_storage`` pin that those
entry points apply them too.
"""

import pathlib

import pytest

from cosmos_curator.next.utils import local_uri


@pytest.mark.parametrize("uri", ["http://x/y", "s33://x/y", "s3://bucket/key"])
def test_local_media_path_rejects_a_non_file_scheme(uri: str) -> None:
    """A non-empty scheme other than file:// is refused, never coerced to a local path.

    ``s3://`` is included on purpose: this local helper does not resolve object
    stores (a caller routes those to a storage client first), so a scheme it
    cannot honour must fail loudly rather than become a bogus relative path. A
    Windows drive-letter path such as ``C:/x`` parses as the single-char scheme
    ``c``, since one letter is a legal scheme, and is refused for the same
    reason, consistent with the sibling ``storage.write_if_absent`` /
    ``LOCAL_SCHEMES`` convention on this linux-64 target.
    """
    with pytest.raises(ValueError, match="unsupported local media URI scheme"):
        local_uri.local_media_path(uri)


def test_local_media_path_accepts_an_absolute_bare_path() -> None:
    """A bare absolute path (empty scheme) resolves to that exact Path."""
    assert local_uri.local_media_path("/abs/path.mp4") == pathlib.Path("/abs/path.mp4")


def test_local_media_path_accepts_a_relative_bare_path() -> None:
    """A bare relative path (empty scheme) resolves to that exact Path."""
    assert local_uri.local_media_path("clips/a.mp4") == pathlib.Path("clips/a.mp4")


def test_local_media_path_accepts_a_bare_path_whose_first_segment_holds_a_bracket() -> None:
    """A bare path is honoured even where URL parsing of its authority would fail.

    ``urlsplit`` reads the leading ``//`` as an authority and rejects a bracket
    there as an invalid IPv6 host, so deciding the scheme with it would refuse a
    path this helper's contract accepts.
    """
    assert local_uri.local_media_path("//host[0]/share/a.mp4") == pathlib.Path("//host[0]/share/a.mp4")


def test_local_media_path_still_rejects_a_bracketed_remote_uri() -> None:
    """Accepting an unparseable authority must not widen into accepting any scheme.

    The negative control for the bare-path case above: ``s3://host[0]/key`` fails
    URL parsing in exactly the same way, so treating that failure as "not a URI"
    would coerce an object-store key into a local path and read the wrong bytes.
    """
    with pytest.raises(ValueError, match="unsupported local media URI scheme 's3'"):
        local_uri.local_media_path("s3://host[0]/key.mp4")


@pytest.mark.parametrize("uri", [" s3://bucket/key", "s\t3://bucket/key"])
def test_local_media_path_rejects_a_scheme_url_parsing_would_repair(uri: str) -> None:
    """A scheme only a parser's own cleanup makes legal is refused, not read locally.

    ``urlsplit`` strips leading and embedded whitespace before parsing, so it
    reports both of these as scheme ``s3``. Deciding the scheme by matching the
    raw text instead would see no scheme and route an object-store key to local
    disk, so the claimed token is compared against what the parser accepted.
    """
    with pytest.raises(ValueError, match="malformed URI"):
        local_uri.local_media_path(uri)


def test_local_media_path_accepts_a_file_uri() -> None:
    """A file:// URI routes through the file-URI decoder to its filesystem path."""
    assert local_uri.local_media_path("file:///abs/path.mp4") == pathlib.Path("/abs/path.mp4")


def test_local_media_path_rejects_a_one_slash_file_uri() -> None:
    """A malformed file URI cannot be accepted as if it used the canonical delimiter."""
    with pytest.raises(ValueError, match="Unsupported local file URI"):
        local_uri.local_media_path("file:/abs/path.mp4")


def test_local_media_path_accepts_a_case_variant_localhost_authority() -> None:
    """A file:// authority is a host name, so its case cannot change which file is meant.

    The shared Arrow gate case-folds before comparing, so a URI it resolves to a
    local path must not be refused here: one run would write through the Arrow
    helper what the media reader then declines to open.
    """
    assert local_uri.local_media_path("file://LOCALHOST/abs/path.mp4") == pathlib.Path("/abs/path.mp4")


def test_local_media_path_rejects_a_remote_file_authority() -> None:
    """A file:// URI naming another host is refused rather than read off this one.

    The negative control for case-folding the authority: folding must not widen
    the gate into accepting any authority at all, which a check that only
    compared lowercase would still pass.
    """
    with pytest.raises(ValueError, match="Unsupported local file URI"):
        local_uri.local_media_path("file://SERVER/abs/path.mp4")


@pytest.mark.parametrize(
    "uri",
    ["file:///data/runs?copy=1", "file:///data/runs#2", "file://localhost/data/runs#2"],
    ids=["query", "fragment", "fragment_under_localhost_authority"],
)
def test_local_media_path_rejects_a_file_uri_with_an_unencoded_delimiter(uri: str) -> None:
    """An unencoded ``?`` or ``#`` truncates the path rather than failing to parse.

    Each of these resolves to ``/data/runs`` - a shorter location than was
    written, and one that may exist - so accepting them reads or writes someone
    else's object while reporting success. Both characters are legal in a POSIX
    name, which is why the truncation is silent and why refusing is the only
    answer that cannot lose data.
    """
    with pytest.raises(ValueError, match="Unsupported local file URI"):
        local_uri.local_media_path(uri)


def test_local_media_path_accepts_a_percent_encoded_delimiter_as_a_literal_name() -> None:
    """The refusal above costs no expressible name, because %23 survives into the path.

    The positive control: a directory really called ``runs#2`` is still
    addressable, so the gate rejects a malformed URI rather than a legal name.
    """
    assert local_uri.local_media_path("file:///data/runs%232/a.mp4") == pathlib.Path("/data/runs#2/a.mp4")
