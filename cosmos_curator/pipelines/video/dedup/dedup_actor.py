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

"""Ray actor for semantic dedup.

Defines `SemDedupActor`, a GPU Ray actor built on `RAFTActor` that uses RAPIDS
cuML multi-GPU K-Means (KMeansMG) and RAFT handles to support semantic
dedup workflows.
"""

import importlib
import io
import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import ray
from loguru import logger

from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_client import StorageClient, StoragePrefix
from cosmos_curator.core.utils.storage.storage_utils import extract_parquet_files
from cosmos_curator.core.utils.storage.writer_utils import write_bytes, write_parquet

from .raft_actor import RAFTActor


@dataclass
class SemDedupConfig:
    """Configuration for a single sem-dedup run on an actor.

    Encapsulates IO/infra options and algorithmic parameters that are constant
    for the lifetime of one `dedup` execution.

    Attributes:
        input_s3_profile_name (str): Source S3 profile name for reading inputs.
        output_path (str): Destination base path for all outputs.
        output_s3_profile_name (str): Target S3 profile name for writing outputs.
        n_clusters (int): Number of clusters for K-Means. Defaults to 4.
        max_iter (int): Maximum iterations for K-Means. Defaults to 100.
        random_state (int): Random seed for K-Means. Defaults to 42.
        eps (float): Pruning tolerance parameter (epsilon). Defaults to 0.01.
        enable_profiling (bool): Whether to enable optional profiling. Defaults to False.

    """

    input_s3_profile_name: str
    output_path: str
    output_s3_profile_name: str
    n_clusters: int = 4
    max_iter: int = 100
    random_state: int = 42
    eps: float = 0.01
    enable_profiling: bool = False


@ray.remote(num_gpus=1, runtime_env=PixiRuntimeEnv("cuml"))
class SemDedupActor(RAFTActor):
    """A Ray actor class for performing semantic dedup."""

    def __init__(
        self,
        index: int,
        pool_size: int,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize the SemDedupActor.

        Args:
            index (int): Index of the actor in the pool.
            pool_size (int): Size of the actor pool.
            verbose (bool, optional): Enable verbose logging. Defaults to False.

        Returns:
            None

        """
        super().__init__(
            index=index,
            pool_size=pool_size,
            verbose=verbose,
        )

    def config(self, config: SemDedupConfig) -> None:
        """Configure the actor with parameters for semantic dedup.

        Args:
            config (SemDedupConfig): Configuration parameters for the semantic dedup run.

        Returns:
            None

        """
        self._config = config

    def _read_parquet_bytes(
        self,
        filepath: StoragePrefix | pathlib.Path,
        client: StorageClient | None,
    ) -> bytes:
        """Read a single parquet file as bytes.

        Args:
            filepath (StoragePrefix | pathlib.Path): Parquet file path to read.
            client (StorageClient | None): Optional preconfigured storage client.

        Returns:
            bytes: Raw parquet bytes.

        Raises:
            RuntimeError: If the read operation fails.

        """
        try:
            data = storage_utils.read_bytes(filepath, client)
        except Exception as e:
            err_msg = f"{self.display_name}: failed to read {filepath}"
            logger.error(f"{err_msg}: {e}")
            raise RuntimeError(err_msg) from e
        else:
            if self._verbose:
                logger.debug(f"{self.display_name}: read {len(data):,}B from {filepath}")
            return data

    def _download(self, files: list[StoragePrefix | pathlib.Path], *, profile_name: str | None = None) -> list[bytes]:
        """Download assigned parquet files into memory as bytes.

        Args:
            files (list[StoragePrefix | pathlib.Path]): Parquet file paths assigned to this actor.
            profile_name (str | None): Optional storage profile to use. Defaults to
                the actor's configured input profile if not provided.

        Returns:
            list[bytes]: File contents for successfully downloaded parquet files.

        """
        if not files:
            return []

        client = storage_utils.get_storage_client(
            target_path=str(files[0]),
            profile_name=profile_name or self._config.input_s3_profile_name,
        )

        # Use assigned CPU cores to parallelize downloads; cap to number of files
        cpu_workers = self.get_assigned_cpu_count()
        max_workers = max(1, min(cpu_workers, len(files)))

        if max_workers == 1:
            buffers: list[bytes] = [self._read_parquet_bytes(pf, client) for pf in files]
        else:
            buffers = [b"" for _ in range(len(files))]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(self._read_parquet_bytes, pf, client): i for i, pf in enumerate(files)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    buffers[idx] = future.result()

        if self._verbose:
            logger.debug(f"{self.display_name}: downloaded {len(buffers)}/{len(files)} parquet files")
        return buffers

    def kmeans(self, parquet_files: list[StoragePrefix | pathlib.Path]) -> None:
        """Perform semantic dedup on the assigned parquet files.

        Args:
            parquet_files (list[StoragePrefix | pathlib.Path]): Parquet files assigned to this actor for processing.

        Returns:
            None

        Raises:
            ValueError: If there are no rows to cluster or embeddings are ragged (inconsistent lengths).

        """
        cudf = importlib.import_module("cudf")
        cp = importlib.import_module("cupy")
        KMeansMG = importlib.import_module("cuml.cluster.kmeans_mg").KMeansMG

        parquet_buffers = self._download(parquet_files)

        # Load only what we need
        embeddings_df = cudf.read_parquet(parquet_buffers, columns=["id", "embedding"])
        if self._verbose:
            logger.debug(f"{self.display_name}: loaded {len(embeddings_df):,} rows from parquet files")

        n = len(embeddings_df)
        if n == 0:
            error_message = f"{self.display_name}: no rows to cluster"
            raise ValueError(error_message)

        # ---- Validate fixed length & build (n, d) CuPy array ----
        lens = embeddings_df["embedding"].list.len()
        dmin, dmax = int(lens.min()), int(lens.max())
        if dmin != dmax:
            error_message = (
                f"{self.display_name}: ragged embeddings (min={dmin}, max={dmax}); "
                f"SemDeDup requires fixed-length vectors"
            )
            raise ValueError(error_message)
        d = dmin

        leaves = embeddings_df["embedding"].list.leaves
        X = leaves.values.reshape(n, d).astype("float32")
        X = cp.ascontiguousarray(X)

        # ---- Spherical k-means approximation: unit-norm rows ----
        norms = cp.linalg.norm(X, axis=1, keepdims=True)
        X /= cp.maximum(norms, 1e-12)  # avoid div-by-zero

        # ---- Fit MG KMeans ----
        # handle= is required for RAFT comms but triggers a spurious deprecation
        # warning in cuml >=26.02 (KMeansMG inherits the KMeans warning even though
        # MG classes still need the handle). Suppress it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*\bhandle\b.*deprecated", category=FutureWarning)
            km = KMeansMG(
                handle=self._raft_handle,
                n_clusters=self._config.n_clusters,
                max_iter=self._config.max_iter,
                random_state=self._config.random_state,
                verbose=self._verbose,
            )
            labels = km.fit_predict(X).astype("int32", copy=False)
        if self._verbose:
            logger.debug(f"{self.display_name}: KMeans complete with {len(labels)} labels")

        # Cosine distance: 1 - cosine_similarity; X is unit-norm already
        centroids = cp.asarray(km.cluster_centers_, dtype=cp.float32)
        centroid_norms = cp.linalg.norm(centroids, axis=1, keepdims=True)
        centroids_unit = centroids / cp.maximum(centroid_norms, 1e-12)
        cosine_sim = cp.sum(X * centroids_unit[labels], axis=1)
        cosine_dist = 1.0 - cp.clip(cosine_sim, -1.0, 1.0)

        # ---- Prepare client for writing ----
        write_client = storage_utils.get_storage_client(
            target_path=self._config.output_path,
            profile_name=self._config.output_s3_profile_name,
            can_overwrite=True,
        )

        # ---- On actor 0, persist centroids to npy ----
        if self._index == 0:
            centroids_np = cp.asnumpy(centroids)
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, centroids_np)
            centroids_dest = storage_utils.get_full_path(
                self._config.output_path, "clustering_results", "kmeans_centroids.npy"
            )
            write_bytes(
                npy_buffer.getvalue(),
                centroids_dest,
                "kmeans centroids",
                self.display_name,
                verbose=self._verbose,
                client=write_client,
                overwrite=True,
            )

        # ---- Assemble cudf DataFrame with requested columns ----
        out_df = cudf.DataFrame(
            {
                "id": embeddings_df["id"],
                "embedding": embeddings_df["embedding"],
                "cosine_dist_to_cent": cudf.Series(cosine_dist),
                "_nearest_cent": cudf.Series(labels),
            }
        )

        # ---- Write one parquet per cluster ----
        for cluster_id in range(self._config.n_clusters):
            cluster_rows = out_df[out_df["_nearest_cent"] == cluster_id][["id", "embedding", "cosine_dist_to_cent"]]
            if len(cluster_rows) == 0:
                continue

            dest = storage_utils.get_full_path(
                self._config.output_path,
                "clustering_results",
                "embs_by_nearest_center",
                f"nearest_cent_{cluster_id}",
                f"actor_{self._index}.parquet",
            )

            # Convert to pandas for writer utility
            pdf = cluster_rows.to_pandas()
            write_parquet(
                pdf,
                dest,
                "sem-dedup clustered embeddings",
                self.display_name,
                verbose=self._verbose,
                client=write_client,
                overwrite=True,
            )

    def dedup(self, clusters: list[int]) -> dict[str, int] | None:  # noqa: PLR0915
        """Compute per-row maximum cosine similarity to any earlier row and emit pruning tables for the given clusters.

        Overview:
        - Reads per-cluster shards produced by k-means from
          `{output_path}/clustering_results/embs_by_nearest_center/nearest_cent_{cid}/actor_*.parquet`
          with columns `id`, `embedding`, and `cosine_dist_to_cent`.
        - Sorts rows by `cosine_dist_to_cent` descending (farthest from centroid first).
        - L2-normalizes embeddings and computes, in GPU tiles, the strict upper-triangular
          cosine similarity matrix so that for each position j only earlier rows i<j are
          considered. Tracks per-column maximum M_j and the argmax index i*.
        - Converts i* to `max_id` (using the sorted `id` column) and writes, per cluster,
          a parquet file at
          `{output_path}/extraction/semdedup_pruning_tables/cluster_{cid}.parquet`
          with columns: `id`, `max_id`, `cosine_sim_score` (== M_j).
        - Uses pruning threshold `1 - eps` from the actor config; counts a row as "kept"
          if `cosine_sim_score <= 1 - eps`. Returns partial counts aggregated on this actor
          as `{ "kept": int, "total": int }`.

        Reference: SemDeDup, Table A7 (https://arxiv.org/abs/2303.09540).

        Args:
            clusters (list[int]): Cluster IDs to process.

        Returns:
            dict[str, int] | None: Partial stats for this actor with keys "kept" and "total".

        Raises:
            ValueError: If a cluster shard contains ragged embeddings (inconsistent lengths).

        """
        cudf = importlib.import_module("cudf")
        cp = importlib.import_module("cupy")

        threshold = float(1 - self._config.eps)
        kept = 0
        total = 0

        # I/O client
        io_client = storage_utils.get_storage_client(
            target_path=self._config.output_path,
            profile_name=self._config.output_s3_profile_name,
            can_overwrite=True,
        )

        # Tiling controls: adjust if you hit memory pressure on giant clusters
        TILE = 4096  # rows per block: 2048-4096 is a safe default

        for cid in clusters:
            # Gather all actor shards for this cluster
            cluster_dir = storage_utils.get_full_path(
                self._config.output_path,
                "clustering_results",
                "embs_by_nearest_center",
                f"nearest_cent_{cid}",
            )
            parquet_files = extract_parquet_files(
                input_path=str(cluster_dir),
                profile_name=self._config.output_s3_profile_name,
                verbose=self._verbose,
            )
            buffers = self._download(parquet_files, profile_name=self._config.output_s3_profile_name)
            if not buffers:
                if self._verbose:
                    logger.warning(f"{self.display_name}: cluster {cid} has no shards to dedup")
                continue

            # Load minimal columns; we'll reorder by farthest-from-centroid
            cols = ["id", "embedding", "cosine_dist_to_cent"]
            dist_df = cudf.read_parquet(buffers, columns=cols)
            m = len(dist_df)
            if m == 0:
                continue

            # Sort by distance-to-centroid DESC (== lowest centroid cosine first)
            dist_df = dist_df.sort_values("cosine_dist_to_cent", ascending=False, ignore_index=True)
            m = len(dist_df)

            # Build (m, d) CuPy and L2-normalize rows (cosine = dot on unit vectors)
            lens = dist_df["embedding"].list.len()
            dmin, dmax = int(lens.min()), int(lens.max())
            if dmin != dmax:
                error_message = (
                    f"{self.display_name}: cluster {cid} has ragged embeddings "
                    f"(min={dmin}, max={dmax}); expected fixed length"
                )
                raise ValueError(error_message)
            d = dmin
            leaves = dist_df["embedding"].list.leaves
            E = leaves.values.reshape(m, d).astype(cp.float32)
            E = cp.ascontiguousarray(E)
            norms = cp.linalg.norm(E, axis=1, keepdims=True)
            E /= cp.maximum(norms, 1e-12)

            # Strict upper-triangle max and argmax for each column j
            maxv = cp.full(m, -1.0, dtype=cp.float32)  # M_j
            argi = cp.full(m, -1, dtype=cp.int32)  # index i* attaining M_j

            for j0 in range(0, m, TILE):
                j1 = min(m, j0 + TILE)
                Bj = j1 - j0
                block_max = cp.full(Bj, -1.0, dtype=cp.float32)
                block_arg = cp.full(Bj, -1, dtype=cp.int32)

                for i0 in range(0, j0, TILE):  # only earlier rows i<j
                    i1 = min(j0, i0 + TILE)
                    Bi = i1 - i0
                    if Bi == 0:
                        break

                    # (Bi, Bj) cosine block; clip for numerical safety
                    S = E[i0:i1] @ E[j0:j1].T
                    S = cp.clip(S, -1.0, 1.0)

                    # Column-wise argmax within this tile
                    local_arg = cp.argmax(S, axis=0)  # (Bj,)
                    local_max = S[local_arg, cp.arange(Bj)]

                    # Keep better candidates and their global indices
                    better = local_max > block_max
                    block_max = cp.where(better, local_max, block_max)
                    block_arg = cp.where(better, (i0 + local_arg).astype(cp.int32), block_arg)

                # --- Intra-tile strict upper triangle: i < j within [j0, j1) ---
                Ej = E[j0:j1]  # (Bj, d)
                Sjj = Ej @ Ej.T  # (Bj, Bj), cosine

                # mask out diagonal and lower triangle so only i<j remain
                ninf = cp.asarray(-cp.inf, dtype=cp.float32)
                idx = cp.tril_indices(Sjj.shape[0], k=0)  # diagonal + lower
                Sjj[idx] = ninf

                tile_intra_max = Sjj.max(axis=0)  # (Bj,)
                tile_intra_arg = Sjj.argmax(axis=0)  # (Bj,)

                better = tile_intra_max > block_max
                block_max = cp.where(better, tile_intra_max, block_max)
                block_arg = cp.where(better, (j0 + tile_intra_arg).astype(cp.int32), block_arg)

                maxv[j0:j1] = block_max
                argi[j0:j1] = block_arg

            # Match legacy: first item has no earlier neighbor → point at itself with score 0.0
            maxv[0] = 0.0
            argi[0] = 0

            # --- summary update for this cluster (GPU → Python int) ---
            kept += int(cp.count_nonzero(maxv <= threshold).item())
            total += len(maxv)

            # Map arg indices -> IDs (positions in *sorted* dist_df)
            # (Guard against any -1 that could slip in)
            argi = cp.where(argi < 0, 0, argi).astype(cp.int32)
            max_id_series = dist_df["id"].take(cudf.Series(argi)).reset_index(drop=True)

            # Emit the legacy schema
            out = cudf.DataFrame(
                {
                    "id": dist_df["id"],
                    "max_id": max_id_series,
                    "cosine_sim_score": cudf.Series(maxv),
                }
            )

            # Write one file for this cluster
            dest = storage_utils.get_full_path(
                self._config.output_path,
                "extraction",
                "semdedup_pruning_tables",
                f"cluster_{cid}.parquet",
            )
            write_parquet(
                out.to_pandas(),  # writer expects pandas
                dest,
                "sem-dedup stats",
                self.display_name,
                verbose=self._verbose,
                client=io_client,
                overwrite=True,
            )

            if self._verbose:
                logger.debug(f"{self.display_name}: cluster {cid}: wrote {len(out):,} rows to {dest}")

        # Return partial stats for driver reduction
        return {"kept": int(kept), "total": int(total)}
