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

"""Provide utilities for interacting with the NVCF."""

import base64
import json
import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import ray
from loguru import logger

from cosmos_curator.core.cf.nvcf_model_manager import NvcfModelManager
from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.infra import ray_cluster_utils
from cosmos_curator.models.all_models import get_all_models_by_id

NVCF_SECRETS_PATH = "/var/secrets/secrets.json"
_NVCF_MISC_RUNNER_CPU_REQUEST = 1


def is_nvcf_container_deployment() -> bool:
    """Check if the NVCF is a container deployment.

    Returns:
        True if the NVCF is a container deployment, False otherwise.

    """
    return os.getenv("NVCF_SINGLE_NODE", "false").lower() == "true"


def is_nvcf_helm_deployment() -> bool:
    """Check if the NVCF is a helm deployment.

    Returns:
        True if the NVCF is a helm deployment, False otherwise.

    """
    return os.getenv("NVCF_MULTI_NODE", "false").lower() == "true"


def get_secrets_from_nvcf_secret_store() -> dict[str, str]:
    """Return the NVCF secrets as a dictionary."""
    data: dict[str, str] = {}
    nsp = pathlib.Path(NVCF_SECRETS_PATH)
    if nsp.exists():
        with nsp.open(encoding="utf-8") as f:
            data = json.load(f)
    return data


def get_nvcf_download_manager() -> NvcfModelManager:
    """Return a NvcfModelManager instance."""
    data = get_secrets_from_nvcf_secret_store()
    api_key = data.get("NGC_NVCF_API_KEY", os.getenv("NGC_NVCF_API_KEY", ""))
    if not api_key:
        error_msg = "NGC_NVCF_API_KEY not found in NVCF secrets or environment variable"
        raise ValueError(error_msg)

    org = data.get("NGC_NVCF_ORG", os.getenv("NGC_NVCF_ORG", ""))
    if not org:
        error_msg = "NGC_NVCF_ORG not found in NVCF secrets or environment variable"
        raise ValueError(error_msg)

    team = data.get("NGC_NVCF_TEAM", os.getenv("NGC_NVCF_TEAM", "no-team"))
    cache_dir = os.getenv("NVCF_MODEL_CACHE_DIR")
    return NvcfModelManager(
        api_key=api_key,
        org=org,
        team=team,
        cache_dir=cache_dir,
    )


def download_model_weight_from_nvcf(
    model_weights_name: str,
    download_dir: pathlib.Path,
) -> None:
    """Download model weights from the NVCF model registry to the local workspace.

    Args:
        model_weights_name: A string representing the name of the model weights to download.
        download_dir: target download dir

    """
    nvcf_download_manager = get_nvcf_download_manager()

    model_details = get_all_models_by_id().get(model_weights_name, None)
    if model_details is None:
        error_msg = f"Model {model_weights_name} not found in NVCF model registry"
        raise ValueError(error_msg)

    nvcf_model_name = model_details.get("nvcf_model_id", None)
    nvcf_model_version = model_details.get("version", None)
    assert isinstance(nvcf_model_name, str)
    assert isinstance(nvcf_model_version, str)

    local_download_dir = pathlib.Path(download_dir)
    local_download_dir.mkdir(parents=True, exist_ok=True)

    nvcf_download_manager.download_model(
        f"{nvcf_model_name}:{nvcf_model_version}",
        str(local_download_dir),
    )

    renamed_model_path = local_download_dir / model_weights_name
    downloaded_model_path = local_download_dir / f"{nvcf_model_name}_v{nvcf_model_version}"

    if renamed_model_path.exists():
        if renamed_model_path.is_symlink():
            renamed_model_path.unlink()
        else:
            shutil.rmtree(renamed_model_path)

    parent_dir = renamed_model_path.parent
    parent_dir.mkdir(exist_ok=True)
    shutil.move(str(downloaded_model_path), str(renamed_model_path))
    # Update the tracking file with the renamed model name instead of the NVCF name
    nvcf_download_manager.mark_models_downloaded({model_weights_name})
    logger.info(f"Downloaded model weights to {renamed_model_path.as_posix()}")


def download_model_weights_from_nvcf_to_workspace(
    model_weights_names: list[str],
    download_dir: pathlib.Path,
) -> None:
    """Download model weights for a list of models from NVCF to the local workspace.

    Uses model tracking to avoid redundant downloads.

    Args:
        model_weights_names: A list of strings representing the names of the models to download weights for.
        download_dir: Directory where the model weights will be downloaded.

    """
    model_manager = get_nvcf_download_manager()
    model_manager.cache_dir = download_dir

    # Convert to set for efficient lookup
    requested_models = set(model_weights_names)
    new_models = model_manager.get_new_models(requested_models)

    if new_models:
        logger.info(
            f"Downloading {len(new_models)} new model weights from NVCF model store",
        )
        for model_name in new_models:
            download_model_weight_from_nvcf(model_name, download_dir)
    else:
        logger.info("All required models already downloaded from NVCF model store")


# If we want to run something on all nodes
@ray.remote(num_cpus=_NVCF_MISC_RUNNER_CPU_REQUEST)
class _NvcfMiscRunner:
    def __init__(self) -> None:
        self._node_name = ray_cluster_utils.get_node_name()

    def create_s3_profile(self, contents: str) -> bool:
        try:
            for profile_path in [environment.S3_PROFILE_PATH, environment.AZURE_PROFILE_PATH]:
                _profile_path = pathlib.Path(profile_path).expanduser()
                # Create the directory if it does not exist. Lock it down if we're making it.
                if not _profile_path.exists():
                    pathlib.Path(_profile_path.parent).mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                with _profile_path.open("wb") as config_file:
                    config_file.write(base64.b64decode(contents))
                _profile_path.chmod(0o600)
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self._node_name} failed to create s3 profile: {e}")
            return False
        else:
            return True

    def remove_s3_profile(self) -> bool:
        try:
            pathlib.Path(environment.S3_PROFILE_PATH).expanduser().unlink()
            pathlib.Path(environment.AZURE_PROFILE_PATH).expanduser().unlink()
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self._node_name} failed to remove s3 profile: {e}")
            return False
        else:
            return True


def _setup_executors_on_all_nodes(msg: str) -> list[Any]:
    ray.init(address="auto", ignore_reinit_error=True)
    num_nodes = len(ray_cluster_utils.get_live_nodes())
    logger.info(f"Running {msg} on {num_nodes} nodes")
    bundles = [{"CPU": _NVCF_MISC_RUNNER_CPU_REQUEST} for _ in range(num_nodes)]
    pg = ray.util.placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    return [
        _NvcfMiscRunner.options(placement_group=pg).remote()  # type: ignore[attr-defined]
        for _ in range(num_nodes)
    ]


# This will take a base64 encoded profile, and write it to the expected destination
def _create_s3_profile(contents: str) -> None:
    executors = _setup_executors_on_all_nodes("create_s3_profile")
    results = ray.get([x.create_s3_profile.remote(contents) for x in executors])
    ray.shutdown()
    if not all(results):
        error_msg = "Failed to create S3 profile on all nodes"
        raise RuntimeError(error_msg)


def create_s3_profile(contents: str) -> bool:
    """Create a S3 profile.

    Args:
        contents: The contents of the S3 profile.

    Returns:
        True if the S3 profile was created successfully, False otherwise.

    """
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_create_s3_profile, contents)
        try:
            _ = future.result()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in create_s3_profile: {e}")
            return False
    return True


def _remove_s3_profile() -> None:
    executors = _setup_executors_on_all_nodes("remove_s3_profile")
    results = ray.get([x.remove_s3_profile.remote() for x in executors])
    ray.shutdown()
    if not all(results):
        error_msg = "Failed to remove S3 profile on all nodes"
        raise RuntimeError(error_msg)


def remove_s3_profile() -> None:
    """Remove a S3 profile."""
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_remove_s3_profile)
        try:
            _ = future.result()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in remove_s3_profile: {e}")
