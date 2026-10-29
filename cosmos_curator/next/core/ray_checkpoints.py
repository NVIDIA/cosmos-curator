# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the Ray Data checkpoint configuration for one complete processing plan, and the rows it stages.

Staged rows are Arrow IPC files. They keep every canonical Arrow type, including
null fixed-size lists, list child names, and metadata, so publication reads back
exactly the rows processing validated. Ray's two-phase checkpointed write covers
file sinks, so a staged file and its checkpointed IDs stay consistent.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pyarrow as pa
import pyarrow.fs as pafs
import ray
from loguru import logger
from ray.data.block import BlockAccessor
from ray.data.checkpoint import CheckpointConfig
from ray.data.datasource import BlockBasedFileDatasink, FileBasedDatasource

_CHECKPOINT_CONFIG_LOCK = threading.Lock()
# Terminal-value validation reads staged or published results in Arrow batches of at most this many rows.
VALIDATION_BATCH_SIZE = 1_024
# Staged files are zstd-compressed Arrow IPC files.
STAGED_FILE_EXTENSION = "arrow"
_STAGED_WRITE_OPTIONS = pa.ipc.IpcWriteOptions(compression="zstd")


@contextmanager
def installed_checkpoint(
    *,
    id_column: str,
    checkpoint_path: str,
    filesystem: pafs.FileSystem | None = None,
) -> Iterator[None]:
    """Install one Ray Data job checkpoint for the enclosed plan and always clear it.

    Callers must serialize construction and execution of plans sharing the driver's
    ``DataContext``. The lock makes competing installations fail without replacing
    an active owner; it does not support concurrent plans within one driver.
    Ray workers within a single plan may still execute in parallel.

    Two properties of the id column decide whether a resume works at all, and both
    are the CALLER's responsibility - this function only names the column::

        Read --> ... --> terminal write
          ^                    ^
          |                    +-- the id column must SURVIVE into the written
          |                        output; Ray marks a row complete by reading
          |                        the column back off the sink
          +-- the id column must be present on the FIRST operator; the checkpoint
              filter reads it off the Read output, so a column added later in the
              plan is invisible to the skip

    A plan that drops the column anywhere between those two points records no
    progress at all while appearing to work. Ray also plans job checkpointing only
    for a logical DAG that ends in a write, so a ``materialize()`` terminal
    disables the mechanism with nothing but a Python warning.

    Args:
        id_column: Column holding one completed row's checkpoint id. Values are
            the caller's; only its own datasource knows the snapshot they are
            drawn from.
        checkpoint_path: Cluster-accessible directory - object storage or a
            network-mounted path. A node-local path gives each worker a
            checkpoint no other worker can read.
        filesystem: Filesystem the checkpoint is read and written through. Pass
            the same one the sink uses, or a run under a non-ambient credential
            profile writes results it cannot read back.

    Yields:
        Nothing; the context is the installation.

    Raises:
        RuntimeError: If a configuration is already installed, or if the active
            one is not the one installed here.

    """
    installed = CheckpointConfig(
        id_column=id_column,
        checkpoint_path=checkpoint_path,
        # The most dangerous default in this design. Ray's default is True, and it
        # deletes the checkpoint when the DATASET EXECUTION succeeds - which is the
        # end of the inference phase, before anything has been published. Accepting
        # the default would destroy the progress record inside exactly the window
        # this configuration exists to protect.
        delete_checkpoint_on_success=False,
        override_filesystem=filesystem,
    )
    with _CHECKPOINT_CONFIG_LOCK:
        context = ray.data.DataContext.get_current()
        # Process-global ambient state, so the guard and installation are one
        # operation: otherwise two threads can both observe None and replace
        # each other's checkpoint before either plan starts.
        if context.checkpoint_config is not None:
            msg = (
                f"a Ray Data checkpoint configuration is already installed "
                f"(id column {context.checkpoint_config.id_column!r}); refusing to replace it"
            )
            raise RuntimeError(msg)
        context.checkpoint_config = installed
        # ASSERTED rather than assumed: setting the config on a context the
        # execution does not use is a silent no-op that costs a full re-run of a
        # GPU phase to discover.
        active = ray.data.DataContext.get_current().checkpoint_config
        if (
            active is None
            or active.id_column != id_column
            or active.checkpoint_path != checkpoint_path
            or active.filesystem is not installed.filesystem
        ):
            context.checkpoint_config = None
            msg = (
                f"the active Ray Data checkpoint configuration is not the one just installed "
                f"(expected id column {id_column!r} at {checkpoint_path!r}, found "
                f"{active and active.id_column!r} at {active and active.checkpoint_path!r}; "
                f"expected filesystem type {type(installed.filesystem).__name__!r}, found "
                f"{active and type(active.filesystem).__name__!r}); "
                f"refusing to run a phase that would record no progress"
            )
            raise RuntimeError(msg)
    try:
        logger.info(f"Installed Ray Data job checkpoint on {id_column!r} at {checkpoint_path}")
        yield
    finally:
        with _CHECKPOINT_CONFIG_LOCK:
            context.checkpoint_config = None


def write_staged_table(table: pa.Table, sink: str | pa.NativeFile) -> None:
    """Write staged rows as one file that reads back with the same schema and values."""
    # An IPC file holds one dictionary per field, so every chunk must share it. Each
    # chunk would also become its own record batch, which readers fetch with a
    # separate request; a checkpoint group joins one chunk per processing block.
    table = table.unify_dictionaries().combine_chunks()
    with pa.ipc.new_file(sink, table.schema, options=_STAGED_WRITE_OPTIONS) as writer:
        writer.write_table(table)


def read_staged_table(source: str | pa.NativeFile) -> pa.Table:
    """Read one file written by ``write_staged_table``.

    An open file must be random access; it is fetched whole with one read. A
    staged file holds one checkpoint group, and object storage charges a round
    trip for each of the footer, metadata, and body reads the IPC reader issues.
    """
    if isinstance(source, pa.NativeFile):
        source = pa.BufferReader(source.read_buffer())
    return pa.ipc.open_file(source).read_all()


class StagedResultsDatasink(BlockBasedFileDatasink):
    """Write each write task's checkpoint group to one staged file.

    Ray applies its two-phase checkpoint to file sinks: a pending checkpoint names
    the file before it is written, and recovery deletes files whose checkpoints
    never committed.
    """

    def __init__(self, path: str, *, filesystem: pafs.FileSystem, rows_per_file: int) -> None:
        """Stage at least ``rows_per_file`` rows per file; a strict upstream repartition makes it exact."""
        super().__init__(
            path, filesystem=filesystem, min_rows_per_file=rows_per_file, file_format=STAGED_FILE_EXTENSION
        )

    def write_block_to_file(self, block: BlockAccessor, file: pa.NativeFile) -> None:
        """Write the task's combined block."""
        write_staged_table(block.to_arrow(), file)


class StagedResultsDatasource(FileBasedDatasource):
    """Read staged files with Ray Data, one block per file."""

    def _open_input_source(self, filesystem: pafs.FileSystem, path: str, **open_args: object) -> pa.NativeFile:
        # A random-access file knows its size, so it is read in one request; Ray opens sequential streams by default.
        del open_args
        return filesystem.open_input_file(path)

    def _read_stream(self, f: pa.NativeFile, path: str) -> Iterator[pa.Table]:
        del path
        yield read_staged_table(f)


def read_staged_results(paths: str | list[str], *, filesystem: pafs.FileSystem) -> ray.data.Dataset:
    """Read staged files, or a directory of them, with Ray Data."""
    return cast(
        "ray.data.Dataset",
        ray.data.read_datasource(
            StagedResultsDatasource(paths, filesystem=filesystem, file_extensions=[STAGED_FILE_EXTENSION])
        ),
    )
