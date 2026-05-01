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
"""Shared fixtures and helpers for storage tests."""

from pathlib import Path
from typing import Any

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_client import (
    BackgroundUploader,
    StorageClient,
    StoragePrefix,
)


class FakeStorageClient(StorageClient):
    """In-memory storage client for exercising storage_utils and StorageWriter."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        """Initialize fake storage with optional object payloads."""
        self.objects = dict(objects or {})
        self.last_list_limit: int | None = None

    def object_exists(self, dest: StoragePrefix) -> bool:
        """Return True if the destination was registered."""
        return str(dest) in self.objects

    def upload_bytes(self, dest: StoragePrefix, data: bytes) -> None:
        """Store bytes for later retrieval."""
        self.objects[str(dest)] = data

    def upload_bytes_uri(
        self,
        uri: str,
        data: bytes,
        _chunk_size_bytes: int = 100,
    ) -> None:
        """Store bytes addressed by a simple URI."""
        self.objects[uri] = data

    def download_object_as_bytes(self, uri: StoragePrefix, _chunk_size_bytes: int = 10) -> bytes:
        """Return bytes for the stored URI."""
        try:
            return self.objects[str(uri)]
        except KeyError as exc:
            raise FileNotFoundError(str(uri)) from exc

    def download_objects_as_bytes(self, uris: list[StoragePrefix]) -> list[bytes]:
        """Return bytes for a sequence of URIs."""
        return [self.download_object_as_bytes(uri) for uri in uris]

    def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
        """List fake objects under a prefix, respecting the optional limit."""
        self.last_list_limit = limit
        prefix = str(uri).rstrip("/") + "/"
        results: list[StoragePrefix] = []
        for path in sorted(self.objects):
            if not path.startswith(prefix):
                continue
            results.append(storage_utils.path_to_prefix(path))
            if limit and len(results) >= limit:
                break
        return results

    def list_recursive(
        self,
        prefix: StoragePrefix,
        limit: int = 0,
    ) -> list[dict[str, Any]]:  # pragma: no cover - unused API surface
        """Unused StorageClient API surface."""
        raise NotImplementedError

    def upload_file(
        self,
        local_path: str,
        remote_path: StoragePrefix,
        _chunk_size: int = 100,
    ) -> None:
        """Read a local file and store its contents under the remote key."""
        data = Path(local_path).read_bytes()
        self.objects[str(remote_path)] = data

    def sync_remote_to_local(
        self,
        remote_prefix: StoragePrefix,
        local_dir: Path,
        *,
        delete: bool = False,
        chunk_size_bytes: int = 10,
    ) -> None:  # pragma: no cover - unused API surface
        """Unused StorageClient API surface."""
        raise NotImplementedError

    def make_background_uploader(
        self,
        chunk_size_bytes: int = 100,
    ) -> BackgroundUploader:  # pragma: no cover - unused API surface
        """Unused StorageClient API surface."""
        raise NotImplementedError

    def delete_object(self, dest: StoragePrefix) -> None:
        """Delete an object from the fake storage."""
        self.objects.pop(str(dest), None)


def remote_path(*components: str) -> str:
    """Build a fake S3 path for testing (e.g. ``s3://test-bucket/a/b``)."""
    cleaned = [part.strip("/") for part in components if part]
    base = "s3://test-bucket"
    if not cleaned:
        return base
    return "/".join([base, *cleaned])
