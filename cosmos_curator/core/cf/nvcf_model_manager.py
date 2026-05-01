"""Manage model downloads and tracking from the NVCF."""

import fcntl
import json

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pathlib
import time
import typing

from loguru import logger
from ngcsdk import Client

if typing.TYPE_CHECKING:
    from registry.api.models import ModelAPI


class NvcfModelManager:
    """A class to download and track models from the NVCF."""

    def __init__(self, api_key: str, org: str, team: str, cache_dir: str | None = None) -> None:
        """Initialize the NVCF model manager.

        Args:
            api_key: NVCF API key
            org: NVCF organization name
            team: NVCF team name within the organization
            cache_dir: Directory where models are cached. If None, will use the current directory.

        """
        self.api_key: str = api_key
        self.org: str = org
        self.team: str = team
        self.logger = logger
        self.cache_dir = pathlib.Path(cache_dir or pathlib.Path.cwd())
        self.tracking_file = self.cache_dir / ".downloaded_models.json"
        self._ensure_tracking_file_exists()

    def _ensure_tracking_file_exists(self) -> None:
        """Ensure the tracking file exists and is properly initialized."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.tracking_file.exists():
            with self.tracking_file.open("w") as f:
                json.dump({"downloaded_models": []}, f)

    def _read_tracking_file(self) -> set[str]:
        """Read the tracking file with proper locking.

        Returns:
            Set of downloaded model names

        """
        with self.tracking_file.open() as f:
            # Get an exclusive lock for reading
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
                return set(data.get("downloaded_models", []))
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_tracking_file(self, models: set[str]) -> None:
        """Write to the tracking file with proper locking.

        Args:
            models: Set of model names to write

        """
        with self.tracking_file.open("w") as f:
            # Get an exclusive lock for writing
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump({"downloaded_models": sorted(models)}, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get_new_models(self, requested_models: set[str]) -> set[str]:
        """Get the set of models that need to be downloaded.

        Args:
            requested_models: Set of model names being requested

        Returns:
            Set of model names that need to be downloaded

        """
        downloaded = self._read_tracking_file()
        # Also verify the files actually exist
        actually_downloaded = {model for model in downloaded if (self.cache_dir / model).exists()}

        # If we found models in tracking file but not on disk,
        # update the tracking file
        if actually_downloaded != downloaded:
            logger.warning(
                f"Found {len(downloaded - actually_downloaded)} models in tracking file "
                "that don't exist on disk. Updating tracking file.",
            )
            self._write_tracking_file(actually_downloaded)

        return requested_models - actually_downloaded

    def mark_models_downloaded(self, models: set[str]) -> None:
        """Mark models as successfully downloaded.

        Args:
            models: Set of model names that were downloaded

        """
        current = self._read_tracking_file()
        updated = current | models
        self._write_tracking_file(updated)

    def download_model(self, mname: str, dest: str) -> None:
        """Download a model from the NVCF.

        Args:
            mname: Name of the model to download
            dest: Destination directory for the model

        """
        clt: Client = Client()
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            try:
                clt.configure(api_key=self.api_key, org_name=self.org, team_name=self.team)
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error configuring client: {e}")
                time.sleep(5)
                attempts += 1

        if attempts == max_attempts:
            error_msg = f"Failed to configure client after {max_attempts} attempts"
            raise RuntimeError(error_msg)

        model: ModelAPI = clt.registry.model
        count = 0
        max_attempts = 5
        while count < max_attempts:
            try:
                target = f"{self.org}/{mname}" if self.team == "no-team" else f"{self.org}/{self.team}/{mname}"
                logger.info(f"Downloading model {mname} from {target} to {dest}; retry-counter={count}")
                model.download_version(target=target, destination=dest)
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error downloading model {mname}: {e}")
                time.sleep(5)
                count += 1

        if count == max_attempts:
            error_msg = f"Failed to download model {mname} after {max_attempts} attempts"
            raise RuntimeError(error_msg)
