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

"""Image pipeline input extraction."""

import pathlib

from loguru import logger

from cosmos_curator.core.utils.storage.storage_client import StorageClient, StoragePrefix
from cosmos_curator.core.utils.storage.storage_utils import (
    get_files_relative,
    get_full_path,
    get_storage_client,
    path_exists,
    read_json_file,
)
from cosmos_curator.pipelines.image.read_write.image_writer_stage import get_image_output_id
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _is_image_file(relative_path: str) -> bool:
    """Return True if path has an image extension (case-insensitive)."""
    p = relative_path.lower()
    return any(p.endswith(ext) for ext in IMAGE_EXTENSIONS)


def get_image_relative_paths(
    input_path: str,
    input_s3_profile_name: str,
) -> list[str]:
    """Return sorted image paths relative to ``input_path``."""
    client = get_storage_client(input_path, profile_name=input_s3_profile_name)
    all_files = get_files_relative(input_path, client, limit=0)
    return sorted(f for f in all_files if _is_image_file(f))


def _read_processed_ids_from_summary(
    output_path: str,
    summary_path: str | StoragePrefix | pathlib.Path,
    client: StorageClient | None,
    *,
    verbose: bool = False,
) -> set[str] | None:
    """Read summary.json and return set of processed image IDs, or None on failure/empty."""
    try:
        data = read_json_file(summary_path, client)
        images_list = data.get("images") or []
        filtered_list = data.get("filtered_images") or []
        processed = data.get("processed_images") or (images_list + filtered_list) or data.get("captioned_images") or []
        ids = {str(x) for x in processed}
    except Exception as e:  # noqa: BLE001
        if verbose:
            logger.debug(f"Could not read summary for already-processed check: {e}")
        return None
    else:
        if ids:
            logger.info(f"Found {len(ids)} already-processed image(s) in {output_path}/summary.json")
        return ids


def _read_processed_ids_from_metas(
    output_path: str,
    metas_prefix: str | StoragePrefix | pathlib.Path,
    client: StorageClient | None,
) -> set[str]:
    """Scan metas/ and return output IDs for any image that has a metadata file."""
    try:
        meta_files = get_files_relative(str(metas_prefix), client, limit=0)
    except Exception:  # noqa: BLE001
        return set()
    meta_ids: set[str] = set()
    for rel in meta_files:
        if not rel.endswith(".json"):
            continue
        meta_ids.add(rel.removesuffix(".json"))
    if meta_ids:
        logger.info(f"Found {len(meta_ids)} already-processed image(s) in {output_path}/metas/")
    return meta_ids


def _get_already_processed_output_ids(
    output_path: str,
    output_s3_profile_name: str | None,
    *,
    verbose: bool = False,
) -> set[str]:
    """Return set of output IDs that already have a metadata file in the output."""
    client = get_storage_client(output_path, profile_name=output_s3_profile_name or "default")
    summary_path = get_full_path(output_path, "summary.json")
    if path_exists(summary_path, client):
        ids = _read_processed_ids_from_summary(output_path, summary_path, client, verbose=verbose)
        if ids is not None and ids:
            return ids
    metas_prefix = get_full_path(output_path, "metas")
    return _read_processed_ids_from_metas(output_path, metas_prefix, client)


def extract_image_tasks(
    input_path: str,
    input_s3_profile_name: str,
    limit: int = 0,
    *,
    output_path_and_profile: tuple[str, str | None] | None = None,
    verbose: bool = False,
) -> list[ImagePipeTask]:
    """Discover image files under input_path and build one ImagePipeTask per image.

    Uses get_files_relative (no extension filter), then filters by IMAGE_EXTENSIONS.
    When output_path_and_profile is set, skips images that already have a metadata
    file in the output, matching video pipeline behavior.

    Args:
        input_path: Local or S3 path to list (directory/prefix).
        input_s3_profile_name: S3 profile for input_path when remote.
        limit: If > 0, process at most this many images (first after sort).
        output_path_and_profile: Optional (output_path, output_s3_profile_name) to skip
            already-processed images; None to process all.
        verbose: Log each discovered path.

    Returns:
        List of ImagePipeTask, one per image file.

    """
    image_files = get_image_relative_paths(input_path, input_s3_profile_name)

    already_processed: set[str] = set()
    if output_path_and_profile is not None:
        out_path, out_profile = output_path_and_profile
        already_processed = _get_already_processed_output_ids(out_path, out_profile, verbose=verbose)

    to_process: list[str] = []
    skipped = 0
    for rel in image_files:
        full = get_full_path(input_path, rel)
        session_id = str(full)
        out_id = get_image_output_id(session_id)
        if out_id in already_processed:
            skipped += 1
            continue
        to_process.append(rel)
    if skipped:
        logger.info(f"Skipping {skipped} already-processed image(s)")

    if limit > 0:
        to_process = to_process[:limit]
    logger.info(f"Found {len(image_files)} input images in {input_path}, {len(to_process)} to process")
    if verbose:
        for p in to_process:
            logger.debug(p)
    tasks = []
    for rel in to_process:
        full = get_full_path(input_path, rel)
        image = Image(input_image=full, relative_path=rel)
        session_id = str(full)
        tasks.append(ImagePipeTask(session_id=session_id, image=image))
    return tasks
