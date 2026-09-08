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
import os
import pathlib
import time
import typing

from loguru import logger
from ngcbase.errors import ResourceNotFoundException
from ngcsdk import Client

from cosmos_curator.core.cf.nvcf_secrets import get_secrets_from_nvcf_secret_store

if typing.TYPE_CHECKING:
    from registry.api.models import ModelAPI

# ngcsdk 4.34.10's download_version docstring documents lowercase statuses
# ("completed"/"failed") while ThreadTransferProgress actually assigns capitalized ones
# ("Completed"/"Failed"). Compare case-insensitively so this works either way.
_TRANSFER_STATUS_COMPLETED = "completed"


def _raise_if_transfer_incomplete(result: object, mname: str, dest: str) -> None:
    """Raise unless the NGC SDK reported the transfer as completed.

    ngcsdk reports a failed download by *returning* a status rather than raising: its
    download workers catch their own exceptions, log them at DEBUG, and mark the file
    failed, so a download that never transferred a single byte still returns normally.
    The output file is present and exactly the right size because the SDK pre-allocates
    it (``f.truncate(size)``) before the workers fill it in, so a swallowed failure
    leaves a plausible-looking file full of zeros rather than a missing or truncated one.
    Checking the returned status is the only cheap way to tell the difference here.

    Args:
        result: whatever ``ModelAPI.download_version`` returned (a ``DotDict``).
        mname: NVCF model name, for the error message.
        dest: download destination, for the error message.

    Raises:
        RuntimeError: If the transfer status is anything other than completed.

    """
    status = getattr(result, "status", None)
    if status is None and isinstance(result, dict):
        status = result.get("status")

    if status is None:
        # Unknown return shape (e.g. an SDK version that changed it). Don't hard-fail on
        # something we can't interpret; pinned sha256 checksums still catch bad content.
        logger.warning(f"Could not read transfer status for {mname}; SDK returned {result!r}")
        return

    if str(status).lower() == _TRANSFER_STATUS_COMPLETED:
        return

    completed_count = getattr(result, "completed_count", "unknown")
    completed_bytes = getattr(result, "completed_bytes", "unknown")
    error_msg = (
        f"NGC SDK reported status={status!r} downloading {mname} to {dest} "
        f"(completed_count={completed_count}, completed_bytes={completed_bytes}). "
        "The SDK pre-allocates output files to their full size before filling them, so "
        "treating this as success would leave correctly-sized zero-filled weights on disk. "
        "The underlying cause is only logged by ngcsdk at DEBUG -- enable "
        "logging.getLogger('ngcbase').setLevel(logging.DEBUG) to see it."
    )
    raise RuntimeError(error_msg)


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

    def download_model(self, mname: str, dest: str, file_patterns: list[str] | None = None) -> None:
        """Download a model from the NVCF.

        Args:
            mname: Name of the model to download
            dest: Destination directory for the model
            file_patterns: Optional glob patterns limiting which files are fetched. Used by
                standalone checksum verification to pull only the file being checked rather
                than an entire multi-hundred-GB model.

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
                kwargs: dict[str, typing.Any] = {"target": target, "destination": dest}
                if file_patterns:
                    kwargs["file_patterns"] = file_patterns
                result = model.download_version(**kwargs)
                # Raised inside the try on purpose: a reported-but-not-raised failure is
                # retried by this same loop, exactly like a thrown one, and escalates to
                # the RuntimeError below once the attempts are exhausted.
                _raise_if_transfer_incomplete(result, mname, dest)
                break
            except ResourceNotFoundException:
                # A 404 is definitive: the model or version is not in this org's registry.
                # Retrying cannot change that, and five attempts with a 5s sleep costs 25s
                # per absent model. Let the caller decide whether that is fatal -- not every
                # model in all_models.json is guaranteed to be published to NVCF.
                logger.warning(f"Model {mname} not found in org {self.org}; not retrying")
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error downloading model {mname}: {e}")
                time.sleep(5)
                count += 1

        if count == max_attempts:
            error_msg = f"Failed to download model {mname} after {max_attempts} attempts"
            raise RuntimeError(error_msg)


def get_nvcf_download_manager() -> NvcfModelManager:
    """Build an NvcfModelManager from the NVCF secret store, falling back to the environment.

    Lives here rather than in ``nvcf_utils`` so that callers can construct a downloader
    without pulling in ``ray``, which ``nvcf_utils`` imports at module scope.

    Returns:
        A configured NvcfModelManager.

    Raises:
        ValueError: If no API key or org can be found.

    """
    data = get_secrets_from_nvcf_secret_store()
    # `or` rather than dict.get's default: a secrets file that contains the key with an
    # empty or null value would otherwise shadow the environment variable, and an empty
    # team would build an "org//model" target instead of taking the no-team path.
    api_key = data.get("NGC_NVCF_API_KEY") or os.getenv("NGC_NVCF_API_KEY", "")
    if not api_key:
        error_msg = "NGC_NVCF_API_KEY not found in NVCF secrets or environment variable"
        raise ValueError(error_msg)

    org = data.get("NGC_NVCF_ORG") or os.getenv("NGC_NVCF_ORG", "")
    if not org:
        error_msg = "NGC_NVCF_ORG not found in NVCF secrets or environment variable"
        raise ValueError(error_msg)

    team = data.get("NGC_NVCF_TEAM") or os.getenv("NGC_NVCF_TEAM") or "no-team"
    cache_dir = os.getenv("NVCF_MODEL_CACHE_DIR")
    return NvcfModelManager(api_key=api_key, org=org, team=team, cache_dir=cache_dir)
