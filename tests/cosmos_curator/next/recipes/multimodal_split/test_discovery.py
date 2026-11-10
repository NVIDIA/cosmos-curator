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

"""Tests for deterministic local candidate session discovery."""

import os
import pathlib
from pathlib import Path

import pyarrow as pa
import pytest

from cosmos_curator.core.utils.storage.storage_utils import path_exists, read_bytes
from cosmos_curator.next.recipes.multimodal_split.config import MultimodalSplitInputConfig
from cosmos_curator.next.recipes.multimodal_split.discovery import (
    CANDIDATE_SESSION_SCHEMA,
    _file_uri_to_path,
    _validate_session_id,
    discover_candidate_sessions,
)


def _make_sessions(root: Path, names: list[str]) -> None:
    """Create session directories, each holding artifacts discovery must not read."""
    for name in names:
        camera = root / name / "camera"
        camera.mkdir(parents=True)
        (camera / "front.mp4").write_bytes(b"media")
        (root / name / "imu.mcap").write_bytes(b"sensor")


def test_local_prefix_lists_only_immediate_child_directories(tmp_path: Path) -> None:
    """Loose files beside the sessions never become candidate sessions."""
    _make_sessions(tmp_path, ["session-b", "session-a"])
    (tmp_path / "README.txt").write_text("not a session", encoding="utf-8")

    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a", "session-b"]


def test_local_session_uri_is_an_absolute_filesystem_path(tmp_path: Path) -> None:
    """Local session URIs stay filesystem paths beneath the configured prefix."""
    _make_sessions(tmp_path, ["session-a"])

    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    table = discover_candidate_sessions(config)

    assert table.column("session_uri").to_pylist() == [str(tmp_path / "session-a")]


def test_local_session_uri_is_directly_consumable_by_the_storage_helpers(tmp_path: Path) -> None:
    """The splitting stage must be able to join artifact patterns on without converting.

    The sensor library parses no URI scheme, so a ``file://`` session_uri would
    reject every local session as missing its cameras.
    """
    _make_sessions(tmp_path, ["session-a"])
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    table = discover_candidate_sessions(config)
    session_uri = table.column("session_uri").to_pylist()[0]

    assert path_exists(f"{session_uri}/camera/front.mp4")
    assert read_bytes(f"{session_uri}/camera/front.mp4") == b"media"


def test_local_prefix_discovery_never_enumerates_inside_a_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery stops at the session boundary and leaves artifact enumeration to later work.

    Every directory-enumeration entry point is recorded, so descending into a
    session by any of them fails this test rather than only the one currently in
    use. Recorded paths are filtered to the fixture tree because pytest itself
    scans directories while the patch is installed.
    """
    _make_sessions(tmp_path, ["session-a", "session-b"])
    listed: list[Path] = []
    real_scandir = os.scandir
    real_iterdir = pathlib.Path.iterdir
    real_walk = os.walk

    def _record(target: object) -> None:
        candidate = Path(target if isinstance(target, str | Path) else str(target))
        if candidate == tmp_path or tmp_path in candidate.parents:
            listed.append(candidate)

    def _recording_scandir(path: object = ".") -> object:
        _record(path)
        return real_scandir(path)  # type: ignore[arg-type]

    def _recording_iterdir(self: Path) -> object:
        _record(self)
        return real_iterdir(self)

    def _recording_walk(top: object, *args: object, **kwargs: object) -> object:
        _record(top)
        return real_walk(top, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "scandir", _recording_scandir)
    monkeypatch.setattr(os, "walk", _recording_walk)
    monkeypatch.setattr(pathlib.Path, "iterdir", _recording_iterdir)
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    discover_candidate_sessions(config)

    assert listed == [tmp_path]


def test_an_empty_prefix_yields_an_empty_table_rather_than_an_error(tmp_path: Path) -> None:
    """A prefix with no sessions is an empty result, not a failure."""
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    table = discover_candidate_sessions(config)

    assert table.num_rows == 0
    assert table.schema == CANDIDATE_SESSION_SCHEMA


def test_missing_local_prefix_fails_before_ray_starts(tmp_path: Path) -> None:
    """A typo in the input prefix is reported on the driver instead of yielding nothing."""
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="Input path prefix does not exist"):
        discover_candidate_sessions(config)


def test_local_prefix_pointing_at_a_file_is_rejected(tmp_path: Path) -> None:
    """A prefix must name a directory of sessions, not a single object."""
    target = tmp_path / "sessions.txt"
    target.write_text("session-a\n", encoding="utf-8")
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(target))

    with pytest.raises(NotADirectoryError, match="is not a directory"):
        discover_candidate_sessions(config)


def test_session_id_list_canonicalizes_whitespace_blanks_and_duplicates(tmp_path: Path) -> None:
    """List entries are stripped, blank lines dropped, duplicates collapsed, order stabilized."""
    listing = tmp_path / "sessions.txt"
    listing.write_text(
        "  session-c  \n\nsession-a\n\t\nsession-c\nsession-b\n\n",
        encoding="utf-8",
    )
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path=str(listing),
    )

    table = discover_candidate_sessions(config)

    assert table.to_pylist() == [
        {"source_session_id": "session-a", "session_uri": "s3://example-bucket/recordings/session-a"},
        {"source_session_id": "session-b", "session_uri": "s3://example-bucket/recordings/session-b"},
        {"source_session_id": "session-c", "session_uri": "s3://example-bucket/recordings/session-c"},
    ]


def test_session_id_list_does_not_require_the_sessions_to_exist(tmp_path: Path) -> None:
    """Discovery emits candidates; artifact existence is checked by the next stage."""
    listing = tmp_path / "sessions.txt"
    listing.write_text("session-a\n", encoding="utf-8")
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix=str(tmp_path / "recordings"),
        session_id_list_path=str(listing),
    )

    table = discover_candidate_sessions(config)

    assert table.column("session_uri").to_pylist() == [str(tmp_path / "recordings" / "session-a")]


def test_missing_session_id_list_fails_before_ray_starts(tmp_path: Path) -> None:
    """A missing list file is a configuration error, not an empty selection."""
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix=str(tmp_path),
        session_id_list_path=str(tmp_path / "missing.txt"),
    )

    with pytest.raises(FileNotFoundError, match="Session ID list does not exist"):
        discover_candidate_sessions(config)


@pytest.mark.parametrize("session_id", ["../escape", "nested/child", ".", ".."])
def test_session_ids_that_escape_the_prefix_are_rejected(tmp_path: Path, session_id: str) -> None:
    """A session ID names one immediate child, so it can never traverse the prefix."""
    listing = tmp_path / "sessions.txt"
    listing.write_text(f"session-a\n{session_id}\n", encoding="utf-8")
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path=str(listing),
    )

    with pytest.raises(ValueError, match="session ID"):
        discover_candidate_sessions(config)


def test_limit_is_applied_after_deduplication_and_sorting(tmp_path: Path) -> None:
    """The limit takes a stable prefix of the canonical order, not of the file order."""
    listing = tmp_path / "sessions.txt"
    listing.write_text("session-c\nsession-c\nsession-a\nsession-b\n", encoding="utf-8")
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path=str(listing),
        limit=2,
    )

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a", "session-b"]


def test_output_table_uses_the_declared_non_nullable_large_string_schema(tmp_path: Path) -> None:
    """Large runs must not be constrained by 32-bit string offsets."""
    _make_sessions(tmp_path, ["session-a"])
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    table = discover_candidate_sessions(config)

    assert isinstance(table, pa.Table)
    assert table.schema == CANDIDATE_SESSION_SCHEMA
    assert table.schema.names == ["source_session_id", "session_uri"]
    for name in table.schema.names:
        field = table.schema.field(name)
        assert field.type == pa.large_string()
        assert not field.nullable


def test_discovery_is_stable_across_repeated_runs(tmp_path: Path) -> None:
    """Two runs over the same prefix produce byte-identical tables."""
    _make_sessions(tmp_path, ["session-c", "session-a", "session-b"])
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    assert discover_candidate_sessions(config).equals(discover_candidate_sessions(config))


def test_session_id_list_order_does_not_affect_the_result(tmp_path: Path) -> None:
    """Reordering the list file cannot reorder the candidate sessions."""
    forward = tmp_path / "forward.txt"
    reverse = tmp_path / "reverse.txt"
    forward.write_text("session-a\nsession-b\nsession-c\n", encoding="utf-8")
    reverse.write_text("session-c\nsession-b\nsession-a\n", encoding="utf-8")
    base = {"input_path_prefix": "s3://example-bucket/recordings"}

    forward_table = discover_candidate_sessions(
        MultimodalSplitInputConfig(mp4_index_uri="/index.lance", **base, session_id_list_path=str(forward))
    )
    reverse_table = discover_candidate_sessions(
        MultimodalSplitInputConfig(mp4_index_uri="/index.lance", **base, session_id_list_path=str(reverse))
    )

    assert forward_table.equals(reverse_table)


def test_file_uri_prefix_is_discovered_like_a_local_path(tmp_path: Path) -> None:
    """A file:// prefix and the equivalent bare path give the same candidates."""
    _make_sessions(tmp_path, ["session-a"])

    from_uri = discover_candidate_sessions(
        MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=tmp_path.as_uri())
    )
    from_path = discover_candidate_sessions(
        MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))
    )

    assert from_uri.equals(from_path)


def test_session_id_list_tolerates_crlf_line_endings(tmp_path: Path) -> None:
    """List files exported from Windows tools are read without stray carriage returns."""
    listing = tmp_path / "sessions.txt"
    listing.write_bytes(b"session-a\r\nsession-b\r\n")
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path=str(listing),
    )

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a", "session-b"]


def test_session_id_list_tolerates_a_utf8_byte_order_mark(tmp_path: Path) -> None:
    """A BOM must not silently become part of the first session ID."""
    listing = tmp_path / "sessions.txt"
    listing.write_bytes("\ufeffsession-a\nsession-b\n".encode())
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path=str(listing),
    )

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a", "session-b"]


def test_a_list_of_only_blank_lines_yields_an_empty_table(tmp_path: Path) -> None:
    """A list file with no IDs is an empty selection rather than a malformed table."""
    listing = tmp_path / "sessions.txt"
    listing.write_text("\n   \n\t\n", encoding="utf-8")
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path=str(listing),
    )

    table = discover_candidate_sessions(config)

    assert table.num_rows == 0
    assert table.schema == CANDIDATE_SESSION_SCHEMA


def test_limit_larger_than_the_selection_returns_everything(tmp_path: Path) -> None:
    """An oversized limit is a cap, not a requirement."""
    _make_sessions(tmp_path, ["session-a", "session-b"])
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path), limit=100)

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a", "session-b"]


def test_filesystem_root_prefix_joins_without_a_doubled_separator(tmp_path: Path) -> None:
    """A root prefix must not produce a doubled leading separator."""
    listing = tmp_path / "sessions.txt"
    listing.write_text("session-a\n", encoding="utf-8")
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="/",
        session_id_list_path=str(listing),
    )

    table = discover_candidate_sessions(config)

    assert table.column("session_uri").to_pylist() == ["/session-a"]


def test_a_relative_prefix_is_absolutized_for_ray_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ray workers do not share the driver's cwd, so a relative prefix cannot survive."""
    _make_sessions(tmp_path, ["session-a"])
    monkeypatch.chdir(tmp_path.parent)
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=tmp_path.name)

    table = discover_candidate_sessions(config)

    assert table.column("session_uri").to_pylist() == [str(tmp_path / "session-a")]


def test_a_symlinked_prefix_keeps_the_name_the_workers_know(tmp_path: Path) -> None:
    """A recording root is often a symlink or autofs mount.

    Resolving it would emit the link target, which the Ray workers may not have
    mounted at all; they know the recording root by its link name.
    """
    real_root = tmp_path / "real"
    real_root.mkdir()
    _make_sessions(real_root, ["session-a"])
    link_root = tmp_path / "current"
    link_root.symlink_to(real_root)
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(link_root))

    table = discover_candidate_sessions(config)

    assert table.column("session_uri").to_pylist() == [str(link_root / "session-a")]


def test_a_home_relative_prefix_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ~-prefixed path is expanded rather than treated as a literal directory name."""
    _make_sessions(tmp_path, ["session-a"])
    monkeypatch.setenv("HOME", str(tmp_path))
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix="~")

    table = discover_candidate_sessions(config)

    assert table.column("session_uri").to_pylist() == [str(tmp_path / "session-a")]


def test_a_percent_encoded_file_uri_prefix_is_decoded(tmp_path: Path) -> None:
    """A file:// URI escapes spaces, so the escape must be undone before listing."""
    spaced = tmp_path / "my recordings"
    spaced.mkdir()
    _make_sessions(spaced, ["session-a"])
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=spaced.as_uri())

    table = discover_candidate_sessions(config)

    assert "%20" in spaced.as_uri()
    assert table.column("session_uri").to_pylist() == [str(spaced / "session-a")]


def test_a_non_utf8_directory_name_is_reported_with_its_name(tmp_path: Path) -> None:
    """POSIX names are bytes, and Arrow's own surrogate error names neither entry nor prefix."""
    _make_sessions(tmp_path, ["session-a"])
    try:
        (tmp_path / b"caf\xe9".decode("utf-8", "surrogateescape")).mkdir()
    except (OSError, UnicodeError) as exc:  # pragma: no cover - filesystem dependent
        pytest.skip(f"filesystem rejects non-UTF-8 names: {exc}")

    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    with pytest.raises(ValueError, match="not valid UTF-8"):
        discover_candidate_sessions(config)


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x0b", "\x0c", "\x85"])
def test_only_newlines_split_the_session_id_list(tmp_path: Path, separator: str) -> None:
    """str.splitlines would break these, splitting one real session ID into two.

    All of them are legal in POSIX filenames and S3 keys.
    """
    listing = tmp_path / "sessions.txt"
    listing.write_text(f"sess{separator}ion-a\n", encoding="utf-8")
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path=str(listing),
    )

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == [f"sess{separator}ion-a"]


def test_a_symlinked_session_is_a_second_candidate_for_the_same_recording(tmp_path: Path) -> None:
    """Pins the documented aliasing behavior rather than asserting it is desirable.

    Listing follows symlinks, so the common ``latest -> session-a`` drop-directory
    convention yields two candidate IDs for one recording. Deduplication is by
    session ID, not by target, so a caller that must not process a recording twice
    has to deduplicate by resolved target itself.
    """
    _make_sessions(tmp_path, ["session-a"])
    (tmp_path / "latest").symlink_to(tmp_path / "session-a")
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["latest", "session-a"]


def test_a_dangling_symlink_is_excluded_rather_than_reported(tmp_path: Path) -> None:
    """A session whose mount is not ready yet is silently absent, not a failure."""
    _make_sessions(tmp_path, ["session-a"])
    (tmp_path / "not-mounted-yet").symlink_to(tmp_path / "nowhere")
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a"]


@pytest.mark.parametrize("prefix", ["file:///", "file://localhost/"])
def test_both_spellings_of_the_file_uri_root_are_usable(tmp_path: Path, prefix: str) -> None:
    """A root file:// URI must survive canonicalization and still be joinable."""
    listing = tmp_path / "sessions.txt"
    listing.write_text("session-a\n", encoding="utf-8")
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix=prefix,
        session_id_list_path=str(listing),
    )

    table = discover_candidate_sessions(config)

    assert table.column("session_uri").to_pylist() == ["/session-a"]


@pytest.mark.parametrize("session_id", ["session 1", "session#1", "%2F", "100%"])
def test_awkward_but_legal_session_names_stay_reachable(tmp_path: Path, session_id: str) -> None:
    """session_uri is a path, not a URI, so no percent-encoding may be applied to it.

    Encoding these would produce a path that does not exist on disk, and decoding
    them would let a name like ``%2F`` change what the path means.
    """
    _make_sessions(tmp_path, [session_id])
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix=str(tmp_path))

    table = discover_candidate_sessions(config)
    session_uri = table.column("session_uri").to_pylist()[0]

    assert table.column("source_session_id").to_pylist() == [session_id]
    assert session_uri == str(tmp_path / session_id)
    assert path_exists(f"{session_uri}/camera/front.mp4")


def test_an_empty_session_id_is_rejected_rather_than_naming_the_prefix() -> None:
    """An empty ID would join to a bare separator that names the prefix, not a session."""
    with pytest.raises(ValueError, match="immediate child"):
        _validate_session_id("")


@pytest.mark.parametrize("uri", ["file:///data/my#dir", "file:///data/my?dir"])
def test_a_file_uri_with_an_unencoded_delimiter_fails_instead_of_truncating(uri: str) -> None:
    """The config rejects these, but the conversion itself must not truncate silently.

    ``urlsplit`` drops everything from the first '#' or '?' out of ``path``, so
    without this guard ``file:///data/my#dir`` would resolve to ``/data/my`` for
    any caller that reaches discovery by another route.
    """
    with pytest.raises(ValueError, match="unencoded"):
        _file_uri_to_path(uri)
