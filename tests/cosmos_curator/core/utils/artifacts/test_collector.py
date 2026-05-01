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

"""Tests for the artifact collector (``collector.py``).

Exercises the driver-side chunk reassembly logic and the
``CollectResult`` outcome reporting without a live Ray cluster.

::

    What we test                          How we test it
    +----------------------------------+  +--------------------------------------+
    | CollectResult properties         |  | Construct with various ok/failed     |
    |   all_succeeded / partial        |  |   tuples, assert computed booleans   |
    | _process_chunk single chunk      |  | One _FileChunk(is_last=True)         |
    |   -> complete file on disk       |  |   -> assert file bytes match         |
    | _process_chunk multi-chunk       |  | Two chunks, same arcname             |
    |   -> concatenated file on disk   |  |   -> assert concatenated bytes       |
    | Sequential files                 |  | Two different arcnames               |
    |   -> both files + count=2        |  |   -> assert both exist               |
    | _process_chunk premature arcname |  | New arcname before is_last=True      |
    |   -> truncates + opens new file  |  |   -> partial data + new file exists  |
    | _process_chunk empty file        |  | _FileChunk(data=b"", is_last=True)   |
    |   -> zero-byte file on disk      |  |   -> file exists, size 0, count=1    |
    | CollectResult all_failed         |  | nodes_ok=(), nodes_failed=((...),)   |
    |   -> not partial, not succeeded  |  |   -> all_succeeded=F, partial=F      |
    | _finalize_node                   |  | Open handle -> finalize -> closed    |
    +----------------------------------+  +--------------------------------------+

Test setup:
    All filesystem tests use pytest's ``tmp_path`` fixture as the
    output directory.  ``_NodeState`` is constructed directly with
    ``gen=None`` (Ray generator not needed for static method tests).
    ``_FileChunk`` instances are built in-process -- no serialization.

    No Ray cluster, no network, no GPU required.
"""

import pathlib

from cosmos_curator.core.utils.artifacts.collector import (
    CollectResult,
    RayFileTransport,
    _FileChunk,
    _NodeState,
)


class TestCollectResult:
    """Verify CollectResult computed properties."""

    def test_all_succeeded(self) -> None:
        """When no nodes failed, all_succeeded is True and partial is False."""
        result = CollectResult(
            total_files=5,
            nodes_ok=("node-a", "node-b"),
            nodes_failed=(),
        )
        assert result.all_succeeded is True
        assert result.partial is False

    def test_partial(self) -> None:
        """When some nodes succeeded and some failed, partial is True."""
        result = CollectResult(
            total_files=3,
            nodes_ok=("node-a",),
            nodes_failed=(("node-b", "TimeoutError"),),
        )
        assert result.all_succeeded is False
        assert result.partial is True

    def test_all_failed(self) -> None:
        """When all nodes failed, all_succeeded is False and partial is False."""
        result = CollectResult(
            total_files=0,
            nodes_ok=(),
            nodes_failed=(("node-a", "TimeoutError"), ("node-b", "RayActorError")),
        )
        assert result.all_succeeded is False
        assert result.partial is False

    def test_empty_result(self) -> None:
        """When no nodes are reported, all_succeeded is True and partial is False."""
        result = CollectResult(
            total_files=0,
            nodes_ok=(),
            nodes_failed=(),
        )
        assert result.all_succeeded is True
        assert result.partial is False


class TestProcessChunk:
    """Verify _process_chunk writes files correctly from chunk data."""

    def test_single_chunk_creates_complete_file(self, tmp_path: pathlib.Path) -> None:
        """A single chunk with is_last=True creates a complete file on disk."""
        state = _NodeState(node_name="test-node", gen=None)
        chunk = _FileChunk(
            arcname="cpu/stage.html",
            data=b"<html>test content</html>",
            is_last=True,
        )

        RayFileTransport._process_chunk(state, chunk, tmp_path)

        output = tmp_path / "cpu" / "stage.html"
        assert output.exists()
        assert output.read_bytes() == b"<html>test content</html>"
        assert state.files_collected == 1
        assert state.current_handle is None

    def test_multi_chunk_reassembly(self, tmp_path: pathlib.Path) -> None:
        """Two chunks with the same arcname are concatenated into one file."""
        state = _NodeState(node_name="test-node", gen=None)
        chunk1 = _FileChunk(arcname="memory/data.bin", data=b"AAAA", is_last=False)
        chunk2 = _FileChunk(arcname="memory/data.bin", data=b"BBBB", is_last=True)

        RayFileTransport._process_chunk(state, chunk1, tmp_path)
        RayFileTransport._process_chunk(state, chunk2, tmp_path)

        output = tmp_path / "memory" / "data.bin"
        assert output.exists()
        assert output.read_bytes() == b"AAAABBBB"
        assert state.files_collected == 1

    def test_sequential_files_closes_previous_handle(self, tmp_path: pathlib.Path) -> None:
        """Processing chunks from different files creates both files correctly."""
        state = _NodeState(node_name="test-node", gen=None)
        chunk_a = _FileChunk(arcname="cpu/a.html", data=b"<a>", is_last=True)
        chunk_b = _FileChunk(arcname="cpu/b.html", data=b"<b>", is_last=True)

        RayFileTransport._process_chunk(state, chunk_a, tmp_path)
        RayFileTransport._process_chunk(state, chunk_b, tmp_path)

        assert (tmp_path / "cpu" / "a.html").read_bytes() == b"<a>"
        assert (tmp_path / "cpu" / "b.html").read_bytes() == b"<b>"
        assert state.files_collected == 2

    def test_premature_arcname_change_truncates_previous(self, tmp_path: pathlib.Path) -> None:
        """When a new arcname arrives before is_last=True, the old file is truncated."""
        state = _NodeState(node_name="test-node", gen=None)
        chunk_partial = _FileChunk(arcname="data/big.bin", data=b"PARTIAL", is_last=False)
        chunk_new = _FileChunk(arcname="data/other.bin", data=b"NEW", is_last=True)

        RayFileTransport._process_chunk(state, chunk_partial, tmp_path)
        RayFileTransport._process_chunk(state, chunk_new, tmp_path)

        assert (tmp_path / "data" / "big.bin").read_bytes() == b"PARTIAL"
        assert (tmp_path / "data" / "other.bin").read_bytes() == b"NEW"
        # Only the completed file (other.bin) counts; big.bin was truncated.
        assert state.files_collected == 1
        assert state.current_handle is None

    def test_empty_file_created(self, tmp_path: pathlib.Path) -> None:
        """A chunk with empty data and is_last=True creates a zero-byte file."""
        state = _NodeState(node_name="test-node", gen=None)
        chunk = _FileChunk(arcname="empty.txt", data=b"", is_last=True)

        RayFileTransport._process_chunk(state, chunk, tmp_path)

        output = tmp_path / "empty.txt"
        assert output.exists()
        assert output.stat().st_size == 0
        assert state.files_collected == 1
        assert state.current_handle is None


class TestFinalizeNode:
    """Verify _finalize_node closes open handles."""

    def test_closes_open_handle_and_nulls_it(self, tmp_path: pathlib.Path) -> None:
        """After _finalize_node, the file handle is closed and set to None."""
        file_path = tmp_path / "test.bin"
        handle = file_path.open("wb")
        handle.write(b"data")

        state = _NodeState(
            node_name="test-node",
            gen=None,
            current_handle=handle,
            current_arcname="test.bin",
        )

        RayFileTransport._finalize_node(state)

        assert state.current_handle is None
        assert handle.closed
