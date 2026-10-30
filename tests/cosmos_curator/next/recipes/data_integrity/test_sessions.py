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

"""Unit tests for expanding the ``input`` block into session paths."""

import json
import pathlib

import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.next.recipes.data_integrity import sessions
from cosmos_curator.next.recipes.data_integrity.config import (
    DataIntegrityExecutionConfig,
    DataIntegrityInputConfig,
)

EXECUTION = DataIntegrityExecutionConfig()


def _expand(**input_fields: object) -> tuple[str, ...]:
    """Expand one input block with default execution settings."""
    return sessions.expand_sessions(DataIntegrityInputConfig(**input_fields), execution=EXECUTION)


def _stub_s3_listing(monkeypatch: pytest.MonkeyPatch, children_by_prefix: dict[str, list[str]]) -> list[str]:
    """Answer S3 listings from a fake client, recording which prefixes were enumerated."""
    seen: list[str] = []

    class _FakeS3Client:
        """Stands in for the ``S3Client`` the expansion builds, listing only."""

        @staticmethod
        def list_child_prefixes(uri: S3Prefix) -> list[str]:
            seen.append(f"s3://{uri.bucket}/{uri.prefix}")
            return children_by_prefix[uri.prefix]

    monkeypatch.setattr(sessions, "make_s3_client", lambda *_a, **_k: _FakeS3Client())
    return seen


def test_a_single_session_comes_back_canonicalized() -> None:
    """Same normalization as the expanding path, so one session has one ``session_id``.

    Without this, a session measured by the ``di`` tool and the same session measured by
    a pipeline run would land in the store as two, differing only by a trailing slash.
    """
    assert sessions.single_session(DataIntegrityInputConfig(sessions=["s3://bucket/clips/one/"])) == (
        "s3://bucket/clips/one"
    )


@pytest.mark.parametrize(
    ("input_fields", "expected"),
    [
        ({"sessions": ["/data/one"], "session_roots": ["/data"]}, "session_roots"),
        ({"session_list_uri": "/data/list.txt"}, "session_list_uri"),
        ({"session_roots": ["/data"], "session_list_uri": "/data/list.txt"}, "session_list_uri and session_roots"),
    ],
)
def test_the_expanding_forms_are_refused_without_being_expanded(
    monkeypatch: pytest.MonkeyPatch, input_fields: dict[str, object], expected: str
) -> None:
    """Judged on the config, not on the expansion, so it costs no listing and cannot flip.

    Checking the result instead would *pass* for a root that happens to hold exactly one
    session today, and start failing the day a second one lands -- and would pay for a
    bucket listing to reach either answer.
    """
    monkeypatch.setattr(sessions, "make_s3_client", lambda *_a, **_k: pytest.fail("listed a root anyway"))
    monkeypatch.setattr(sessions, "list_sessions_under_root", lambda *_a, **_k: pytest.fail("listed a root anyway"))

    with pytest.raises(ValueError, match=expected) as excinfo:
        sessions.single_session(DataIntegrityInputConfig(**input_fields))
    assert "run-pipeline" in str(excinfo.value), "the message has to name the tool that does cover a dataset"


def test_more_than_one_explicit_session_is_refused() -> None:
    """One invocation measures one session; the count is the whole of the rule."""
    with pytest.raises(ValueError, match="names 2"):
        sessions.single_session(DataIntegrityInputConfig(sessions=["/data/one", "/data/two"]))


def test_explicit_sessions_come_back_canonicalized() -> None:
    """The same normalization the store uses, so a session path means one session."""
    assert _expand(sessions=["s3://bucket/clips/one/", "s3://bucket/clips/two"]) == (
        "s3://bucket/clips/one",
        "s3://bucket/clips/two",
    )


def test_a_relative_local_session_becomes_absolute(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local paths are resolved against the working directory, as any CLI would."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "clips").mkdir()

    assert _expand(sessions=["clips"]) == (str(tmp_path / "clips"),)


def test_a_newline_session_list_is_read(tmp_path: pathlib.Path) -> None:
    """The shell-pipeline form: one path per line, with comments and blanks ignored."""
    listing = tmp_path / "sessions.txt"
    listing.write_text("# generated\n\ns3://bucket/clips/one\ns3://bucket/clips/two\n")

    assert _expand(session_list_uri=str(listing)) == ("s3://bucket/clips/one", "s3://bucket/clips/two")


def test_a_json_session_list_is_read(tmp_path: pathlib.Path) -> None:
    """The query-output form, which ``multimodal-split``'s reader does not accept."""
    listing = tmp_path / "sessions.json"
    listing.write_text(json.dumps(["s3://bucket/clips/two", "s3://bucket/clips/one"]))

    assert _expand(session_list_uri=str(listing)) == ("s3://bucket/clips/one", "s3://bucket/clips/two")


def test_a_json_session_list_of_the_wrong_shape_is_rejected(tmp_path: pathlib.Path) -> None:
    """A mapping or a list of objects is a different file, not a session list."""
    listing = tmp_path / "sessions.json"
    listing.write_text(json.dumps([{"session": "s3://bucket/clips/one"}]))

    with pytest.raises(ValueError, match="array of strings"):
        _expand(session_list_uri=str(listing))


def test_a_session_list_on_an_unsupported_scheme_is_rejected() -> None:
    """Only local files and s3:// objects can be read; az:// lists are not supported."""
    with pytest.raises(ValueError, match="local path or an s3:// object"):
        _expand(session_list_uri="az://container/sessions.txt")


def test_a_local_root_expands_to_its_child_directories(tmp_path: pathlib.Path) -> None:
    """A root holds sessions as immediate children, the layout ``di`` documents."""
    root = tmp_path / "clips"
    for name in ("b-session", "a-session"):
        (root / name).mkdir(parents=True)
    (root / "notes.txt").write_text("not a session")

    assert _expand(session_roots=[str(root)]) == (str(root / "a-session"), str(root / "b-session"))


def test_a_missing_local_root_is_reported_as_missing(tmp_path: pathlib.Path) -> None:
    """A typo'd root must not look like a dataset that happens to be empty."""
    with pytest.raises(FileNotFoundError):
        _expand(session_roots=[str(tmp_path / "absent")])


def test_an_s3_root_expands_through_the_shared_lister(monkeypatch: pytest.MonkeyPatch) -> None:
    """Child prefixes come from the same lister ``multimodal-split`` uses."""
    seen = _stub_s3_listing(monkeypatch, {"clips/": ["one", "two"]})

    expanded = _expand(session_roots=["s3://bucket/clips/"])

    assert expanded == ("s3://bucket/clips/one", "s3://bucket/clips/two")
    assert seen == ["s3://bucket/clips/"]


def test_an_azure_root_is_rejected_with_a_workaround() -> None:
    """Sessions may live on az://; only expanding a root needs a delimited listing."""
    with pytest.raises(ValueError, match="list the sessions explicitly"):
        _expand(session_roots=["az://container/clips/"])


def test_an_unsupported_root_scheme_is_rejected() -> None:
    """Anything else would be taken for a local directory named after the URI."""
    with pytest.raises(ValueError, match="unsupported session root"):
        _expand(session_roots=["gs://bucket/clips/"])


def test_the_three_forms_are_unioned_and_deduplicated(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A root plus a few named sessions is normal, and the overlap must collapse.

    Two spellings of one session would otherwise be measured twice and then collide on
    ``stream_id`` with nothing to break the tie.
    """
    _stub_s3_listing(monkeypatch, {"clips/": ["one", "two"]})
    listing = tmp_path / "sessions.txt"
    listing.write_text("s3://bucket/clips/two/\ns3://bucket/clips/three\n")

    expanded = _expand(
        sessions=["s3://bucket/clips/one/", "s3://bucket/clips/one"],
        session_list_uri=str(listing),
        session_roots=["s3://bucket/clips/"],
    )

    assert expanded == (
        "s3://bucket/clips/one",
        "s3://bucket/clips/three",
        "s3://bucket/clips/two",
    )


def test_a_session_named_inside_another_is_dropped() -> None:
    """Listing recurses, so the nested session's streams are already covered.

    Measured on its own as well it would produce a second row for each of those
    streams, carrying the same ``stream_id`` under the same ``run_id``.
    """
    assert _expand(sessions=["s3://bucket/clips/a", "s3://bucket/clips/a/inner"]) == ("s3://bucket/clips/a",)


def test_a_root_named_alongside_its_children_swallows_them() -> None:
    """The enclosing path is the one kept, because dropping it would lose coverage."""
    assert _expand(sessions=["s3://bucket/clips", "s3://bucket/clips/a", "s3://bucket/clips/b"]) == (
        "s3://bucket/clips",
    )


def test_a_session_sharing_a_name_prefix_is_not_nested() -> None:
    """Only whole path segments nest: ``clips/a`` does not contain ``clips/ab``."""
    assert _expand(sessions=["s3://bucket/clips/a", "s3://bucket/clips/ab"]) == (
        "s3://bucket/clips/a",
        "s3://bucket/clips/ab",
    )


def test_nesting_is_found_past_a_sibling_that_sorts_between() -> None:
    """``-`` sorts before ``/``, so ``clips/a-b`` lands between ``clips/a`` and its children."""
    assert _expand(sessions=["s3://bucket/clips/a", "s3://bucket/clips/a-b", "s3://bucket/clips/a/inner"]) == (
        "s3://bucket/clips/a",
        "s3://bucket/clips/a-b",
    )


def test_an_input_that_expands_to_nothing_is_an_error(tmp_path: pathlib.Path) -> None:
    """A config naming only an empty list measured nothing; that is a mistake, not a pass."""
    listing = tmp_path / "sessions.txt"
    listing.write_text("\n# nothing here\n")

    with pytest.raises(ValueError, match="zero sessions"):
        _expand(session_list_uri=str(listing))
