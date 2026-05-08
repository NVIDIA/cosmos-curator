# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper utilities for working with presigned S3 URLs that reference zip archives.

This module provides a minimal set of helpers to:

1. Download a zip archive from a presigned HTTPS URL and extract it to a
   temporary location so the pipeline can treat the contents like a normal
   *input_video_path* directory.
2. Create a zip archive from a local directory and upload it to a presigned
   HTTPS URL so the caller can fetch the results without direct access to the
   backing object store.

The implementation intentionally avoids pulling in any extra heavy-weight
dependencies so that importing it has negligible impact on start-up time.

Note: This module was previously called ``presigned_zip_utils``. It was renamed
in favour of a more explicit name indicating that it handles presigned **S3**
URLs specifically. Imports using the old name should be updated accordingly.
"""

import argparse
import contextlib
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO

import ray
import requests
from loguru import logger
from requests_toolbelt.streaming_iterator import StreamingIterator  # type: ignore[import-untyped]

from cosmos_curator.core.utils.infra import ray_cluster_utils
from cosmos_curator.core.utils.storage.storage_utils import (
    get_files_relative,
    get_storage_client,
)
from cosmos_curator.core.utils.storage.zip_utils import safe_extract_zip
from cosmos_curator.pipelines.video.read_write.summary_writers import (
    _write_all_window_captions,
)

__all__ = [
    "download_and_extract_zip",
    "gather_and_upload_outputs",
    "gather_outputs_from_all_nodes",
    "handle_presigned_urls",
    "zip_and_upload_directory",
    "zip_and_upload_directory_multipart",
]


def _download_file(url: str, dst_path: Path) -> None:
    """Download *url* to *dst_path* in a streaming fashion.

    Args:
        url: A presigned HTTPS URL pointing to the remote zip archive.
        dst_path: Local filesystem path where the downloaded file will be
            written. All missing parent directories will be created.

    Raises:
        requests.RequestException: If the remote server responds with a non-2xx
            HTTP status code.
        OSError: If the destination file cannot be written.

    """
    logger.info(f"Downloading file from presigned URL to {dst_path} …")

    # Ensure the destination directory exists before we start writing data.
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream the response in manageable chunks so very large archives do not
    # need to fit entirely in memory.
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with dst_path.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=8 * 1024):
                if chunk:  # Filter out keep-alive chunks.
                    fp.write(chunk)

    logger.info("Download completed.")


def _download_and_extract_zip_single_node(
    presigned_url: str,
    tmp_dir: str | None = None,
) -> str:
    """Download a presigned zip archive and extract its contents.

    The downloaded archive is saved into a temporary directory (or *tmp_dir* if
    provided) before extraction.  The function returns the directory that
    contains the extracted files so downstream pipeline stages can treat the
    returned path like a normal ``input_video_path``.

    Args:
        presigned_url: Presigned HTTPS URL that grants temporary access to the
            zip archive stored in S3.
        tmp_dir: Optional path to an existing directory that should be used as
            the base for all temporary files.  If *None*, a fresh directory is
            created via :pyfunc:`tempfile.mkdtemp`.

    Returns:
        Path (as ``str``) to the directory containing the extracted archive
        contents.

    Raises:
        requests.RequestException: If the download fails.
        zipfile.BadZipFile: If the downloaded file is not a valid zip archive.
        OSError: If the archive cannot be written or extracted.

    """
    base_tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="input_videos_"))
    base_tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download the archive.
    zip_path = base_tmp_dir / "archive.zip"
    _download_file(presigned_url, zip_path)

    # 2. Extract the archive into its own sub-directory so that the *.zip* file
    # itself will never be mistaken for an input video by downstream code.
    extract_dir = base_tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path) as zf:
        safe_extract_zip(zf, extract_dir)
    logger.info("Extraction completed.")

    # Heuristic: if the archive contains a single top-level directory, return
    # that directory directly; otherwise return *extract_dir*.
    top_level_items = list(extract_dir.iterdir())
    if len(top_level_items) == 1 and top_level_items[0].is_dir():
        return str(top_level_items[0])

    return str(extract_dir)


def zip_and_upload_directory(directory: str, presigned_url: str) -> None:
    """Create a zip archive from *directory* and upload it to *presigned_url*.

    Args:
        directory: Local directory whose contents should be archived.
        presigned_url: Presigned HTTPS URL (``PUT``) that grants write access to
            the destination object in S3.

    Raises:
        ValueError: If *directory* does not exist or is not a directory.
        requests.RequestException: If the upload fails or S3 returns a non-2xx
            response.
        OSError: If the temporary zip archive cannot be created or read.

    """
    # Create the zip archive
    archive_path = _create_zip_archive(directory)

    try:
        logger.info(f"Uploading zipped output ({archive_path}) to presigned URL …")

        with archive_path.open("rb") as fp:
            response = requests.put(presigned_url, data=fp, timeout=60)

        response.raise_for_status()
        logger.info("Upload completed successfully.")

    except Exception as exc:
        logger.error(f"Failed to upload archive: {exc}\n{response.text}")
        raise
    finally:
        # Always attempt to clean-up the temporary archive
        with contextlib.suppress(OSError):
            archive_path.unlink(missing_ok=True)


def _validate_archive_size(archive_size: int, part_urls: list[str], chunk_size_bytes: int) -> None:
    """Validate that archive size fits within the provided parts."""
    expected_max_size = len(part_urls) * chunk_size_bytes
    if archive_size > expected_max_size:
        msg = (
            f"Archive size ({archive_size:,} bytes) exceeds maximum expected size "
            f"({expected_max_size:,} bytes) for {len(part_urls)} parts"
        )
        raise ValueError(msg)


def _create_zip_archive(directory: str) -> Path:
    """Create a zip archive from directory and return the archive path.

    Args:
        directory: Local directory whose contents should be archived.

    Returns:
        Path to the created zip archive.

    Raises:
        ValueError: If directory does not exist or is not a directory.
        OSError: If the zip archive cannot be created.

    """
    src_dir = Path(directory).expanduser().resolve()
    if not src_dir.is_dir():
        msg = f"Directory to zip does not exist: {src_dir}"
        raise ValueError(msg)

    # Create the zip archive in the same filesystem to avoid cross-device issues
    fd, tmp_path = tempfile.mkstemp(prefix="output_archive_", suffix=".zip")
    os.close(fd)
    tmp_path_path = Path(tmp_path)

    # shutil.make_archive expects the base_name without the extension
    base_name = tmp_path_path.with_suffix("")
    shutil.make_archive(str(base_name), "zip", root_dir=str(src_dir))
    archive_path = base_name.with_suffix(".zip")

    # Validate ZIP integrity immediately after creation
    corrupt_file = None
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            # Test the ZIP file by reading its central directory
            file_count = len(zf.namelist())
            logger.info(f"ZIP validation: {file_count} files in archive")

            # Test ZIP integrity by verifying all file data
            corrupt_file = zf.testzip()

    except zipfile.BadZipFile as e:
        logger.error(f"ZIP validation failed: {e}")
        archive_path.unlink(missing_ok=True)
        msg = f"Created ZIP file is corrupt: {e}"
        raise OSError(msg) from e
    except Exception as e:
        logger.error(f"ZIP validation error: {e}")
        archive_path.unlink(missing_ok=True)
        raise

    # Check for corruption outside the try block
    if corrupt_file:
        logger.error(f"ZIP validation failed: corrupt file detected: {corrupt_file}")
        archive_path.unlink(missing_ok=True)
        msg = f"Created ZIP file is corrupt: file '{corrupt_file}' failed CRC check"
        raise OSError(msg)

    return archive_path


def _validate_upload_completion(bytes_uploaded: int, archive_size: int) -> None:
    """Validate that the entire archive was uploaded."""
    if bytes_uploaded != archive_size:
        missing_bytes = archive_size - bytes_uploaded
        msg = (
            f"Upload size mismatch: uploaded {bytes_uploaded:,} bytes, expected {archive_size:,} bytes. "
            f"Difference: {missing_bytes:,} bytes ({missing_bytes / 1024 / 1024:.1f} MB)."
        )
        logger.error(msg)
        raise ValueError(msg)


def _upload_part_streaming(fp: BinaryIO, part_url: str, chunk_size_bytes: int) -> int:
    # Get current position and calculate remaining bytes without seeking
    start_pos = fp.tell()

    # Seek to end to get file size, then immediately return to start position
    fp.seek(0, os.SEEK_END)
    file_size = fp.tell()
    fp.seek(start_pos, os.SEEK_SET)

    remaining = file_size - start_pos
    part_size = min(chunk_size_bytes, remaining)

    if part_size == 0:
        return 0

    # Upload with retry on connection issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Reset file position for retry
            fp.seek(start_pos, os.SEEK_SET)

            # Use StreamingIterator with a generator for proper streaming
            def chunk_generator() -> Iterator[bytes]:
                """Yield file data in chunks."""
                bytes_read = 0
                chunk_size = 8192  # 8KB chunks for streaming
                while bytes_read < part_size:
                    remaining = part_size - bytes_read
                    read_size = min(chunk_size, remaining)
                    chunk = fp.read(read_size)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    yield chunk

            streaming_iterator = StreamingIterator(part_size, chunk_generator())
            response = requests.put(part_url, data=streaming_iterator, timeout=3600)
            response.raise_for_status()
            break  # Success, exit retry loop

        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
            else:
                logger.error(f"Failed to upload part after {max_retries} attempts: {e}")
                raise

    # Verify StreamingIterator consumed all expected bytes
    final_pos = fp.tell()
    expected_pos = start_pos + part_size
    if final_pos != expected_pos:
        actual_read = final_pos - start_pos
        logger.error(f"StreamingIterator read mismatch: expected {part_size:,}, actually read {actual_read:,}")
        msg = f"StreamingIterator read mismatch: expected {part_size:,}, actually read {actual_read:,}"
        raise ValueError(msg)

    return part_size


def zip_and_upload_directory_multipart(
    directory: str,
    multipart_config: dict[str, Any],
    chunk_size_bytes: int = 4_500_000_000,  # 4.5GB chunks
) -> None:
    """Create a zip archive from *directory* and upload it via S3 multipart upload.

    Args:
        directory: Local directory whose contents should be archived.
        multipart_config: Dictionary containing:
            - uploadId: The S3 multipart upload ID
            - key: The S3 object key
            - parts: List of presigned URLs for each part
        chunk_size_bytes: Size of each chunk in bytes (default: 4.5GB)

    Returns:
        None

    Raises:
        ValueError: If *directory* does not exist or is not a directory, or if archive size
            doesn't match expected size based on number of parts.
        requests.RequestException: If any part upload fails or S3 returns a non-2xx response.
        OSError: If the temporary zip archive cannot be created or read.

    Note:
        This function assumes multipart upload dicts will always have multiple urls.
        This behavior is enforced by the managed service.

        This function only uploads the parts using presigned URLs. The caller must complete
        the multipart upload using S3 API credentials and list_parts to get ETags, we cannot
        with only the pre-signed URLs.

    """
    key = multipart_config["key"]
    part_urls = multipart_config["parts"]

    logger.info(f"Starting multipart upload for {key} with {len(part_urls)} parts")

    archive_path = _create_zip_archive(directory)

    try:
        # Get the total size of the zip file
        archive_size = archive_path.stat().st_size
        logger.info(f"Created ZIP archive: {archive_size:,} bytes ({archive_size / 1024 / 1024:.1f} MB)")

        # Validate that we don't exceed the maximum possible size for the given parts
        _validate_archive_size(archive_size, part_urls, chunk_size_bytes)

        # Multipart upload
        logger.info(f"Uploading {len(part_urls)} parts with chunk size {chunk_size_bytes:,} bytes")

        bytes_uploaded = 0

        with archive_path.open("rb") as fp:
            for part_num, part_url in enumerate(part_urls, 1):
                logger.info(f"Uploading part {part_num}/{len(part_urls)}")
                try:
                    part_bytes = _upload_part_streaming(fp, part_url, chunk_size_bytes)
                    if part_bytes == 0:
                        break

                    bytes_uploaded += part_bytes
                except Exception as exc:
                    logger.error(f"Failed to upload part {part_num}: {exc}")
                    raise

            # Validate that we uploaded the entire archive
            _validate_upload_completion(bytes_uploaded, archive_size)

        logger.info(f"Multipart upload completed successfully ({bytes_uploaded:,} bytes total)")

    except Exception as exc:
        logger.error(f"Failed to upload multipart archive: {exc}")
        raise
    finally:
        # Always attempt to clean-up the temporary archive
        with contextlib.suppress(OSError):
            archive_path.unlink(missing_ok=True)


# Reserve CPU resources for Ray actors
_ZIP_DOWNLOADER_CPU_REQUEST: float = 1.0  # full core per node for download
_OUTPUT_GATHERER_CPU_REQUEST: float = 0.1  # fractional core for lightweight gather tasks


def _download_and_extract_zip_impl(presigned_url: str, base_tmp_dir: str) -> str:
    """Download *presigned_url* to *base_tmp_dir* and extract its contents."""
    tmp_dir = Path(base_tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    zip_path = tmp_dir / "archive.zip"
    _download_file(presigned_url, zip_path)

    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting downloaded archive …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        safe_members = [m for m in zf.namelist() if "__MACOSX" not in m]
        safe_extract_zip(zf, extract_dir, members=safe_members)

    # If the archive contains a single top-level directory, return that; else the extraction dir
    top_items = list(extract_dir.iterdir())
    if len(top_items) == 1 and top_items[0].is_dir():
        return str(top_items[0])
    return str(extract_dir)


@ray.remote(num_cpus=_ZIP_DOWNLOADER_CPU_REQUEST)
class _ZipDownloader:
    """Ray actor that performs a single download/extract on its node."""

    def __init__(self) -> None:
        self._node = ray_cluster_utils.get_node_name()

    def run(self, url: str, base_tmp_dir: str) -> tuple[str, str]:
        path = _download_and_extract_zip_impl(url, base_tmp_dir)
        return self._node, path


def _download_and_extract_on_all_nodes(url: str, base_tmp_dir: str) -> str:
    """Ensure the archive is downloaded & extracted once per Ray node."""
    bundles = [{"CPU": _ZIP_DOWNLOADER_CPU_REQUEST} for _ in ray_cluster_utils.get_live_nodes()]
    pg = ray.util.placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    actors = [_ZipDownloader.options(placement_group=pg).remote() for _ in bundles]  # type: ignore[attr-defined]
    results = ray.get([a.run.remote(url, base_tmp_dir) for a in actors])

    for node, path in results:
        logger.info(f"Archive extracted on node {node} at {path}")

    # Path is identical across nodes
    return str(Path(base_tmp_dir) / "extracted")


def _worker_download_and_extract(url: str, base_tmp_dir: str) -> str:
    """Download and extract an archive on all Ray nodes in a subprocess."""
    ray_cluster_utils.init_or_connect_to_cluster()
    extracted = _download_and_extract_on_all_nodes(url, base_tmp_dir)
    if ray.is_initialized():
        ray.shutdown()
    return extracted


def download_and_extract_zip(presigned_url: str) -> str:
    """Download & extract input videos on **all** Ray nodes, returning driver path."""
    base_tmp_dir = str(Path(tempfile.gettempdir()) / f"input_videos_{uuid.uuid4().hex}")
    with ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(_worker_download_and_extract, presigned_url, base_tmp_dir)
        return fut.result()


def handle_presigned_urls(pipeline_type: str, pipeline_args: argparse.Namespace) -> argparse.Namespace:
    """Update *pipeline_args* in-place based on any presigned URLs present."""
    if getattr(pipeline_args, "input_presigned_s3_url", None):
        logger.info("Input presigned URL detected - downloading …")
        extracted_path = download_and_extract_zip(pipeline_args.input_presigned_s3_url)
        logger.info(f"Extracted to temporary directory: {extracted_path}")
        if pipeline_type == "split":
            pipeline_args.input_video_path = extracted_path
        elif pipeline_type == "semantic-dedup":
            pipeline_args.input_embeddings_path = extracted_path
        else:
            logger.warning(f"Unsupported pipeline type '{pipeline_type}' for presigned input URL.")

    # Handle both single presigned URL and multipart upload
    has_single_output = getattr(pipeline_args, "output_presigned_s3_url", None)
    has_multipart_output = getattr(pipeline_args, "output_presigned_multipart", None)

    if has_single_output or has_multipart_output:
        if pipeline_type == "split":
            if not getattr(pipeline_args, "output_clip_path", None):
                pipeline_args.output_clip_path = tempfile.mkdtemp(prefix="output_split_")
                logger.warning(
                    f"No output_clip_path provided; using temporary directory {pipeline_args.output_clip_path}",
                )
        elif pipeline_type == "semantic-dedup":
            if not getattr(pipeline_args, "output_path", None):
                pipeline_args.output_path = tempfile.mkdtemp(prefix="output_dedup_")
                logger.warning(
                    f"No output_path provided; using temporary directory {pipeline_args.output_path}",
                )
        else:
            logger.warning(f"Unsupported pipeline type '{pipeline_type}' for presigned output URL.")

    return pipeline_args


@ray.remote(num_cpus=_OUTPUT_GATHERER_CPU_REQUEST)
class _OutputGatherer:
    """Actor that zips local output directory and returns bytes."""

    def run(self, output_dir: str) -> tuple[str, Any | None]:
        node = ray_cluster_utils.get_node_name()
        out_path = Path(output_dir)
        if not out_path.exists():
            return node, None
        if sum(len(files) for _, _, files in os.walk(out_path)) == 0:
            return node, None

        fd, zip_path_str = tempfile.mkstemp(prefix="node_output_", suffix=".zip")
        os.close(fd)
        zip_path = Path(zip_path_str)
        shutil.make_archive(zip_path.with_suffix("").as_posix(), "zip", output_dir)
        zip_file = zip_path.with_suffix(".zip")
        with zip_file.open("rb") as fh:
            data = fh.read()
        zip_file.unlink(missing_ok=True)
        return node, ray.put(data)


def _extract_zip_bytes(buf: bytes, dest_dir: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(buf)
        tmp_path = Path(tmp.name)
    with zipfile.ZipFile(tmp_path, "r") as zf:
        safe_extract_zip(zf, dest_dir)
    tmp_path.unlink(missing_ok=True)


def _gather_outputs_on_all_nodes(output_dir: str) -> None:
    bundles = [{"CPU": _OUTPUT_GATHERER_CPU_REQUEST} for _ in ray_cluster_utils.get_live_nodes()]
    pg = ray.util.placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    actors = [_OutputGatherer.options(placement_group=pg).remote() for _ in bundles]  # type: ignore[attr-defined]
    results = ray.get([a.run.remote(output_dir) for a in actors])

    for node, obj in results:
        if obj is None:
            logger.info(f"No output data on node {node}")
            continue
        data = ray.get(obj)
        _extract_zip_bytes(data, output_dir)
        logger.info(f"Merged output from node {node}")


def _worker_gather_outputs(output_dir: str) -> None:
    """Gather outputs from all Ray nodes in a subprocess."""
    ray_cluster_utils.init_or_connect_to_cluster()
    _gather_outputs_on_all_nodes(output_dir)
    if ray.is_initialized():
        ray.shutdown()


def gather_outputs_from_all_nodes(output_directory: str) -> None:
    """Collect outputs from all Ray nodes into *output_directory*."""
    with ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(_worker_gather_outputs, output_directory)
        try:
            fut.result(timeout=300)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Gather outputs failed: {exc}")


def _get_output_path(pipeline_type: str, args: argparse.Namespace) -> str | None:
    """Get the output path for the given pipeline type and args."""
    if pipeline_type == "split":
        if getattr(args, "output_clip_path", None) is None:
            logger.warning("output_clip_path for split pipeline is not set?")
            return None
        return str(args.output_clip_path)
    if pipeline_type == "semantic-dedup":
        if getattr(args, "output_path", None) is None:
            logger.warning("output_path for semantic-dedup pipeline is not set?")
            return None
        return str(args.output_path)
    logger.warning(f"Unsupported pipeline type '{pipeline_type}' for presigned output URL.")
    return None


def _write_split_metadata(args: argparse.Namespace, output_path: str) -> None:
    """Write consolidated window-captions metadata for split pipeline."""
    logger.info("Re-writing consolidated window-captions metadata …")
    input_client = get_storage_client(args.input_video_path)
    output_client = get_storage_client(output_path, can_overwrite=True)
    _write_all_window_captions(
        output_path=output_path,
        client_output=output_client,
        output_s3_profile_name=None,
        input_videos_relative=get_files_relative(args.input_video_path, input_client),
        limit=0,
    )


def gather_and_upload_outputs(pipeline_type: str, args: argparse.Namespace) -> None:
    """Gather outputs, write metadata, zip, and upload via presigned URL."""
    # Check for either single URL or multipart config
    single_url = getattr(args, "output_presigned_s3_url", None)
    multipart_config = getattr(args, "output_presigned_multipart", None)

    if single_url is None and multipart_config is None:
        return

    output_path = _get_output_path(pipeline_type, args)
    if output_path is None:
        return

    try:
        logger.info("Gathering per-node outputs …")
        gather_outputs_from_all_nodes(output_path)

        if pipeline_type == "split":
            _write_split_metadata(args, output_path)

        # Choose upload method based on available configuration
        if multipart_config is not None:
            logger.info("Uploading zipped outputs via multipart upload …")
            zip_and_upload_directory_multipart(output_path, multipart_config)
        elif single_url is not None:
            logger.info("Uploading zipped outputs via single URL …")
            zip_and_upload_directory(output_path, single_url)

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to gather/upload outputs: {exc}")
    finally:
        if "output_split_" in output_path or "output_dedup_" in output_path:
            with contextlib.suppress(OSError):
                shutil.rmtree(output_path)
