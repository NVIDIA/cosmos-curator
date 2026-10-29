# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable result workspaces retained until canonical publication is verified.

Recipes define compatibility through a manifest containing input identity and
result-defining settings. Credentials and execution tuning stay outside it.
``checkpoint_writer`` owns one plan's Ray checkpoint and writes its validated
results into the workspace. The processing marker only skips expensive setup;
publishers must still prove exact coverage and validate staged results before
exposing them in Lance.
"""

import hashlib
import json
from collections.abc import Callable, Collection, Hashable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, Self

import attrs
import pyarrow as pa
import pyarrow.fs as pafs
import ray
from loguru import logger

from cosmos_curator.next.core.config import CheckpointedExecutionConfig
from cosmos_curator.next.core.ray_checkpoints import StagedResultsDatasink, installed_checkpoint
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, filesystem_path
from cosmos_curator.next.utils.identity import json_snapshot
from cosmos_curator.next.utils.storage import download_bytes, remove_prefix, write_if_absent

_REMEDY = "Remove the workspace to discard its staged results, or use a different staging root."


class RecoveryStateError(ValueError):
    """Durable recovery state conflicts with the current input, contract, or history.

    Recovery retains every named workspace. An operator must resolve the conflict,
    for example by removing an abandoned workspace; rerunning cannot.
    """


@attrs.frozen
class RecoveryWorkspace:
    """One compatible result set and its native Ray checkpoints.

    The manifest is snapshotted so its digest cannot change after choosing a path.
    """

    root: str
    _manifest_json: str = attrs.field(alias="manifest", converter=json_snapshot)

    @property
    def manifest(self) -> dict[str, Any]:
        """Return an independent JSON snapshot of the recorded contract."""
        return dict(json.loads(self._manifest_json))

    @classmethod
    def load(cls, root: str, *, storage_profile: str) -> Self | None:
        """Read one fixed workspace's manifest, or return ``None`` when it has none."""
        uri = f"{root}/workspace.json"
        filesystem = arrow_filesystem(root, storage_profile=storage_profile)
        entry = filesystem.get_file_info(filesystem_path(uri))
        if entry.type == pafs.FileType.NotFound:
            return None
        manifest = json.loads(download_bytes(uri, storage_profile=storage_profile))
        if not isinstance(manifest, dict):
            msg = f"Recovery workspace manifest is not an object: {uri}. {_REMEDY}"
            raise RecoveryStateError(msg)
        return cls(root, manifest)

    @property
    def results(self) -> str:
        """Return the durable result directory."""
        return f"{self.root}/results"

    @property
    def checkpoints(self) -> str:
        """Return the native Ray row-checkpoint directory."""
        return f"{self.root}/checkpoints"

    @property
    def completion(self) -> str:
        """Return the optional processing-completion marker."""
        return f"{self.root}/processing-complete.json"

    @property
    def manifest_digest(self) -> str:
        """Identify compatible durable work by hashing the frozen canonical manifest."""
        return hashlib.sha256(self._manifest_json.encode("utf-8")).hexdigest()

    def _exists(self, uri: str, *, storage_profile: str) -> bool:
        filesystem = arrow_filesystem(self.root, storage_profile=storage_profile)
        return bool(filesystem.get_file_info(filesystem_path(uri)).type == pafs.FileType.File)

    def ensure(self, *, storage_profile: str) -> None:
        """Atomically create the manifest or require equivalent JSON content."""
        uri = f"{self.root}/workspace.json"
        data = self._manifest_json.encode("utf-8")
        if not write_if_absent(uri, data, storage_profile=storage_profile):
            actual = json.loads(download_bytes(uri, storage_profile=storage_profile))
            if actual != json.loads(data):
                msg = f"Recovery workspace does not match this input and result contract: {self.root}. {_REMEDY}"
                raise RecoveryStateError(msg)

    def resume_or_clear(
        self, selected: Collection[Hashable], published: Collection[Hashable], *, storage_profile: str
    ) -> bool:
        """Return whether this saved selection resumes, or remove it once it is fully published.

        ``published`` holds the selected members that current rows or values show
        as published. A selection with none resumes. Selections publish
        atomically, so partial publication means another writer or invalid state.
        A fully published selection must be cleaned up before a new one reuses
        this path.
        """
        if not published:
            return True
        if set(published) != set(selected):
            msg = (
                f"Recovery selection is partially published; overlapping writers are unsupported: {self.root}. "
                "Confirm no other writer owns this output, then remove the workspace."
            )
            raise RecoveryStateError(msg)
        if not self.cleanup(storage_profile=storage_profile):
            msg = f"Cannot start a new selection until workspace cleanup succeeds: {self.root}. Retry cleanup."
            raise OSError(msg)
        return False

    def start(self, *, storage_profile: str) -> None:
        """Begin a new selection: clear files left by interrupted cleanup, then create the manifest.

        Cleanup may remove the manifest before the results or checkpoints, so a
        new selection never inherits them. Saved selections resume without this.
        """
        remove_prefix(self.root, storage_profile=storage_profile)
        self.ensure(storage_profile=storage_profile)

    def processing_complete(self, *, storage_profile: str) -> bool:
        """Treat absent, unreadable, or incompatible optimization markers as a miss."""
        try:
            if not self._exists(self.completion, storage_profile=storage_profile):
                return False
            payload = json.loads(download_bytes(self.completion, storage_profile=storage_profile))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return bool(payload == {"manifest_digest": self.manifest_digest})

    def record_completion(self, *, storage_profile: str) -> None:
        """Record successful processing without making marker writes a failure boundary."""
        try:
            write_if_absent(
                self.completion,
                json.dumps({"manifest_digest": self.manifest_digest}).encode(),
                storage_profile=storage_profile,
            )
        except Exception as exc:  # noqa: BLE001 -- the marker is only a recovery optimization
            logger.warning(
                "Results are checkpointed, but the processing completion marker could not be written: {!r}", exc
            )

    def cleanup(self, *, storage_profile: str) -> bool:
        """Remove this validated workspace only after the caller verifies publication."""
        try:
            remove_prefix(self.root, storage_profile=storage_profile)
        except Exception as exc:  # noqa: BLE001 -- cleanup cannot invalidate canonical results
            logger.warning("Results are canonical, but workspace cleanup failed for {}: {!r}", self.root, exc)
            return False
        return True


def _stage_results(
    results: pa.Table,
    *,
    result_schema: pa.Schema,
    validate: Callable[[pa.Table], pa.Table],
) -> pa.Table:
    """Prove that publication can accept canonical recipe results before they are staged."""
    if not results.schema.equals(result_schema, check_metadata=True):
        msg = "Processing results do not match the canonical result schema"
        raise ValueError(msg)
    if results.num_rows:
        validate(results)
    return results


@attrs.frozen
class CheckpointWriter:
    """The write end of one checkpointed plan over a workspace, created by ``checkpoint_writer``.

    Processing scopes expose ``result_schema`` and delegate ``write`` here.
    """

    workspace: RecoveryWorkspace
    result_schema: pa.Schema
    validate: Callable[[pa.Table], pa.Table] = attrs.field(repr=False)
    execution: CheckpointedExecutionConfig = attrs.field(repr=False)
    filesystem: pafs.FileSystem = attrs.field(repr=False, eq=False)

    def write(self, results: ray.data.Dataset) -> None:
        """Group canonical results strictly, validate each group, stage it, then record completion.

        Rows must match ``result_schema`` and stage in it unchanged. Ray checkpoints
        a result's ID once its staged file is written, so every row is validated
        first: a checkpointed result can always be published. Publication must
        still prove exact coverage before committing. Validating after the
        repartition, with default remote args, lets Ray fuse it into each group's
        write task instead of adding a task per plan block.
        """
        rows_per_checkpoint = self.execution.rows_per_checkpoint
        stage = partial(_stage_results, result_schema=self.result_schema, validate=self.validate)
        groups = results.repartition(target_num_rows_per_block=rows_per_checkpoint, strict=True)
        publishable = groups.map_batches(stage, batch_format="pyarrow", batch_size=None, zero_copy_batch=True)
        publishable.write_datasink(
            StagedResultsDatasink(self.workspace.results, filesystem=self.filesystem, rows_per_file=rows_per_checkpoint)
        )
        self.workspace.record_completion(storage_profile=self.execution.storage_profile)


@contextmanager
def checkpoint_writer(
    workspace: RecoveryWorkspace,
    *,
    id_column: str,
    result_schema: pa.Schema,
    validate: Callable[[pa.Table], pa.Table],
    execution: CheckpointedExecutionConfig,
) -> Iterator[CheckpointWriter]:
    """Install one plan's Ray checkpoint on ``id_column`` and yield a writer sharing its filesystem.

    The writer stages ``result_schema`` rows that ``validate`` accepts. Construct
    and execute the plan inside this context. The checkpoint stays in the
    workspace until publication is verified; the ambient configuration is
    cleared on every exit.
    """
    filesystem = arrow_filesystem(workspace.root, storage_profile=execution.storage_profile)
    with installed_checkpoint(id_column=id_column, checkpoint_path=workspace.checkpoints, filesystem=filesystem):
        yield CheckpointWriter(workspace, result_schema, validate, execution, filesystem)
