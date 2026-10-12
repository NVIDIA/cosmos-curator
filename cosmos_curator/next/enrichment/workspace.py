# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable result workspaces retained until canonical publication is verified.

Recipes define compatibility through a manifest containing input identity and
result-defining settings. Credentials and execution tuning stay outside it.
The processing marker only skips expensive setup; publishers must still prove
exact coverage and validate staged results before exposing them in Lance.
"""

import json
from typing import Any, Self

import attrs
import pyarrow.fs as pafs
from loguru import logger

from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, filesystem_path
from cosmos_curator.next.utils.identity import canonical_digest, json_snapshot
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
    def discover(cls, prefix: str, *, storage_profile: str) -> tuple[Self, ...]:
        """Read workspace manifests directly beneath one compatible namespace.

        Never walk result/checkpoint contents. Unrecognized directories remain
        untouched; unreadable or inconsistent manifests fail without cleanup.
        """
        filesystem = arrow_filesystem(prefix, storage_profile=storage_profile)
        entries = filesystem.get_file_info(
            pafs.FileSelector(filesystem_path(prefix), recursive=False, allow_not_found=True)
        )
        workspaces = []
        for entry in sorted(entries, key=lambda item: item.path):
            if entry.type != pafs.FileType.Directory:
                continue
            name = entry.path.rstrip("/").rsplit("/", 1)[-1]
            root = f"{prefix.rstrip('/')}/{name}"
            uri = f"{root}/workspace.json"
            if filesystem.get_file_info(filesystem_path(uri)).type != pafs.FileType.File:
                continue
            manifest = json.loads(download_bytes(uri, storage_profile=storage_profile))
            if not isinstance(manifest, dict):
                msg = f"Recovery workspace manifest is not an object: {uri}. {_REMEDY}"
                raise RecoveryStateError(msg)
            workspaces.append(cls(root, manifest))
        return tuple(workspaces)

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
        """Identify compatible durable work independently of its storage location."""
        return canonical_digest(self.manifest)

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
