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
#
# Portions of this file were derived from:
# https://github.com/rapidsai/ray-rapids/blob/main/rapids-experiments/ray-kmeans.py
"""Semantic de-duplication pipeline for video clip embeddings.

This module orchestrates semantic de-duplication (sem-dedup) of video clip
embeddings on a Ray GPU cluster. It discovers parquet embedding files from
local or remote storage, creates a RAFT-backed actor pool sized by available
GPUs, executes K-means and dedup steps in parallel across actors, and writes a
compact CSV summary of the results.

High-level flow:
- Validate inputs (e.g., number of clusters > 0)
- Initialize or connect to a Ray cluster
- Discover input parquet files to process
- Create an actor pool sized by min(num_gpus, num_input_files)
- For each actor:
  - Run K-means on its shard of data
  - Run dedup within assigned clusters, returning per-actor stats
- Reduce per-actor stats and write a summary CSV

Key functions:
- ``dedup``: main pipeline routine
- ``add_dedup_command``: CLI integration for the ``dedup`` subcommand
- ``nvcf_run_semdedup``: entrypoint for NVCF executions

Inputs (from ``args``):
- ``input_embeddings_path``: directory/prefix with parquet embeddings
- ``input_s3_profile_name``: optional storage profile for reading
- ``output_path`` / ``output_s3_profile_name``: where to write results
- ``n_clusters``, ``max_iter``, ``random_state``: K-means parameters
- ``eps_to_extract``: epsilon threshold used by the dedup stage
- Common flags (via ``add_common_args``): e.g., ``--limit``, ``--verbose``

Outputs and side effects:
- Writes ``extraction/dedup_summary_<eps>.csv`` under ``output_path``
- May emit profiling artifacts when enabled

Requirements:
- A running Ray cluster with at least one GPU
"""

import argparse
import pathlib

import numpy as np
import ray
from loguru import logger

from cosmos_curator.core.utils.config import args_utils
from cosmos_curator.core.utils.infra import ray_cluster_utils
from cosmos_curator.core.utils.infra.profiling import profiling_scope
from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.core.utils.storage.storage_utils import extract_parquet_files
from cosmos_curator.core.utils.storage.writer_utils import write_bytes
from cosmos_curator.pipelines.pipeline_args import add_common_args
from cosmos_curator.pipelines.video.dedup.dedup_actor import (
    SemDedupActor,
    SemDedupConfig,
)
from cosmos_curator.pipelines.video.dedup.raft_actor import initialize_raft_actor_pool
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage

EPS_TO_EXTRACT_MIN = 1e-4


def _validate_args(args: argparse.Namespace) -> None:
    """Validate input arguments for the dedup pipeline.

    Checks:
    - ``n_clusters`` must be > 0
    - ``eps_to_extract`` must be >= 1e-4 (to avoid numerical-noise thresholds)

    Raises:
        ValueError: If any validation fails.

    """
    if args.n_clusters <= 0:
        error_message = "n_clusters must be > 0"
        raise ValueError(error_message)
    if args.eps_to_extract < EPS_TO_EXTRACT_MIN:
        error_message = f"eps_to_extract must be >= {EPS_TO_EXTRACT_MIN}"
        raise ValueError(error_message)


def _get_parquet_files_from_summary(input_path: str, input_s3_profile_name: str) -> list[StoragePrefix | pathlib.Path]:
    parquet_files: list[StoragePrefix | pathlib.Path] = []
    try:
        summary_file_path = storage_utils.get_full_path(input_path, "summary.json")
        client = storage_utils.get_storage_client(str(summary_file_path), profile_name=input_s3_profile_name)
        logger.info(f"Reading split-annotate pipeline summary from {summary_file_path}")
        summary_data = storage_utils.read_json_file(summary_file_path, client=client, max_attempts=2)
        embedding_parquet_path = ClipWriterStage.get_output_path_embd_parquets(
            input_path, summary_data["embedding_algorithm"]
        )
        for item in summary_data.values():
            if isinstance(item, dict) and "video_uuid" in item:
                parquet_files.extend(
                    [
                        ClipWriterStage.get_grouped_clips_uri(
                            item["video_uuid"], idx, embedding_parquet_path, "parquet"
                        )
                        for idx in range(item["num_clip_chunks"])
                    ]
                )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not derive parquet file list from split-pipeline summary: {e}")
        return []
    else:
        return parquet_files


def _get_parquet_files(args: argparse.Namespace) -> list[StoragePrefix | pathlib.Path]:
    """Resolve and validate the list of input parquet files.

    Args:
        args: Parsed command-line arguments providing
            `input_embeddings_path`, `input_s3_profile_name`, `limit`, and `verbose`.

    Returns:
        A non-empty list of parquet file locations to process. Each element is either
        a `StoragePrefix` (for remote/object storage) or a `pathlib.Path` (for local files).

    Raises:
        RuntimeError: If no parquet files are found for the given input path/filters.

    """
    parquet_files = _get_parquet_files_from_summary(args.input_embeddings_path, args.input_s3_profile_name)
    if not parquet_files:
        logger.info(f"Falling back to scanning for parquet files under {args.input_embeddings_path}.")
        parquet_files = extract_parquet_files(
            input_path=args.input_embeddings_path,
            profile_name=args.input_s3_profile_name,
            limit=args.limit,
            verbose=args.verbose,
        )
    num_parquet_files = len(parquet_files)
    if num_parquet_files <= 0:
        error_message = "No parquet files found. Nothing to do."
        raise RuntimeError(error_message)
    logger.info(f"Found {num_parquet_files} parquet files to process.")
    return parquet_files


def _get_num_gpus() -> int:
    """Return the number of GPUs in the Ray cluster and validate it's > 0.

    Raises:
        RuntimeError: If the Ray cluster has 0 GPUs.

    """
    num_gpus = int(ray.cluster_resources().get("GPU", 0))  # type: ignore[no-untyped-call]
    if num_gpus <= 0:
        error_message = "The Ray cluster has 0 GPUs; semantic dedup requires at least 1 GPU."
        raise RuntimeError(error_message)
    logger.info(f"The ray cluster has {num_gpus} GPUs.")
    return num_gpus


def _create_actor_pool(num_tasks: int, *, verbose: bool) -> tuple[list[SemDedupActor], int]:
    """Create the RAFT actor pool for semantic dedup.

    This function queries the Ray cluster for available GPUs and chooses a pool size
    based on the minimum of the GPU count and the number of tasks.

    Args:
        num_tasks: Number of parallel tasks to distribute (typically number of parquet files).
        verbose: Verbosity flag passed through to actor initialization.

    Returns:
        A tuple of (pool, pool_size) where pool is the initialized actor pool and
        pool_size is the number of actors created.

    """
    num_gpus = _get_num_gpus()
    pool_size = min(num_gpus, num_tasks)
    logger.info(f"Using a pool size of {pool_size} for semantic dedup.")
    pool = initialize_raft_actor_pool(
        pool_size=pool_size,
        actor_class=SemDedupActor,
        verbose=verbose,
    )
    return pool, pool_size


def _reduce_and_write_summary(stats: list[dict[str, int]], args: argparse.Namespace) -> None:
    """Aggregate stats and write the dedup summary CSV.

    Args:
        stats: Per-actor statistics dictionaries containing keys "total" and "kept".
        args: Parsed command-line arguments; uses `eps_to_extract` for the tag,
            and `output_path`/`output_s3_profile_name` for writing the CSV.

    """
    if not stats:
        error_message = "No per-actor stats returned; dedup stage produced no results."
        raise RuntimeError(error_message)
    total = sum(p["total"] for p in stats)
    kept = sum(p["kept"] for p in stats)
    removed = total - kept

    tag = f"{args.eps_to_extract:.6g}".rstrip("0").rstrip(".")
    client = storage_utils.get_storage_client(
        target_path=args.output_path,
        profile_name=args.output_s3_profile_name,
        can_overwrite=True,
    )
    dest = storage_utils.get_full_path(args.output_path, "extraction", f"dedup_summary_{tag}.csv")
    csv_text = f"eps,kept,removed,total\n{tag},{kept},{removed},{total}\n"
    write_bytes(
        csv_text.encode("utf-8"),
        dest,
        "sem-dedup summary",
        "driver",
        verbose=args.verbose,
        client=client,
        overwrite=True,
    )
    logger.info(f"Wrote dedup summary to {dest}")


def dedup(args: argparse.Namespace) -> None:
    """Run the semantic dedup pipeline with profiling and tracing.

    Public entry point that wraps ``_dedup()`` with ``profiling_scope``
    so that every execution path (CLI, Slurm, NVCF, local launch)
    automatically gets profiling and distributed tracing.

    Args:
        args: Command line arguments.

    """
    with profiling_scope(args):
        _dedup(args)


def _dedup(args: argparse.Namespace) -> None:
    """Run the semantic dedup pipeline.

    Args:
        args: Command line arguments.

    """
    _validate_args(args)

    ray_cluster_utils.init_or_connect_to_cluster()

    parquet_files = _get_parquet_files(args)
    pool, pool_size = _create_actor_pool(num_tasks=len(parquet_files), verbose=args.verbose)
    logger.info(f"Created actor pool with {pool_size} actors for semantic dedup.")

    params = SemDedupConfig(
        input_s3_profile_name=args.input_s3_profile_name,
        output_path=args.output_path,
        output_s3_profile_name=args.output_s3_profile_name,
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        random_state=args.random_state,
        eps=args.eps_to_extract,
        enable_profiling=args.perf_profile,
    )

    # Configure each actor in the pool with the runtime parameters
    ray.get([pool[i].config.remote(params) for i in range(pool_size)])  # type: ignore[attr-defined]
    logger.info("Configured all actors with dedup parameters.")

    # Split parquet files evenly across actors and launch k-means on each actor
    sublists = [list(chunk) for chunk in np.array_split(parquet_files, pool_size)]  # type: ignore[arg-type, var-annotated]
    ray.get([pool[i].kmeans.remote(sublists[i]) for i in range(pool_size)])  # type: ignore[attr-defined]
    logger.info("K-means clustering completed on all actors.")

    # Split cluster ids across actors and run dedup on each actor's assigned clusters
    sublists = [list(chunk) for chunk in np.array_split(list(range(args.n_clusters)), pool_size)]
    stats = ray.get([pool[i].dedup.remote(sublists[i]) for i in range(pool_size)])  # type: ignore[attr-defined]
    logger.info("Deduplication completed on all actors.")

    _reduce_and_write_summary(stats=stats, args=args)

    ray_cluster_utils.shutdown_cluster()


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-embeddings-path",
        type=str,
        required=True,
        help=(
            "Path to input embeddings; could be either output_clip_path of split pipeline for best performance "
            "(the path containing a summary.json and an iv2_embd_parquet, ce1_embd_<variant>_parquet, "
            "or openai_embd_parquet directory) "
            "or a path with embedding parquet files which includes id & embedding columns."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output location",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of clusters for K-means clustering",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations for clustering",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state used for reproducibility",
    )
    parser.add_argument(
        "--eps-to-extract",
        type=float,
        default=0.01,
        help="Epsilon value to extract deduplicated data.",
    )

    # add common args applicable to all pipelines
    add_common_args(parser)


def nvcf_run_semdedup(args: argparse.Namespace) -> None:
    """Run the semantic dedup pipeline.

    Args:
        args: Command line arguments.

    """
    args_utils.fill_default_args(args, _setup_parser)
    dedup(args)


def add_dedup_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add the dedup command to the parser.

    Args:
        subparsers: The subparsers action to add the parser to.

    """
    parser = subparsers.add_parser(
        "dedup",
        help="Semantic dedup of video embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=dedup)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
