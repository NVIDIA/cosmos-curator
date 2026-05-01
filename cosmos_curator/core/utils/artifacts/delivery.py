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

r"""Generic artifact delivery: staging, cross-node collection, and upload.

Provides :class:`ArtifactDelivery` -- a single, consumer-agnostic
orchestrator for the full artifact lifecycle on a distributed
compute cluster.  It sits one layer above :class:`RayFileTransport`
(which handles raw file transport between nodes) and
:class:`StorageWriter` (which handles upload to local / S3 / Azure
destinations)::

    Package layers
    ==============

    +-------------------------------+
    | Consumer code                 |   profiling.py, tracing_hook.py,
    | (profiling_scope, etc.)       |   or any future subsystem
    +-------------------------------+
                 |
                 v
    +-------------------------------+
    | ArtifactDelivery              |   <-- THIS MODULE
    | (staging + collect + upload   |       Generic orchestrator
    |  lifecycle)                   |
    +-------------------------------+
                 |
          +------+------+
          |             |
          v             v
    +-----------+  +-----------+
    | RayFile   |  | Storage   |
    | Transport |  | Writer    |
    +-----------+  +-----------+
    (collector.py) (storage_utils.py)

Multiple subsystems (profiling backends, OTel tracing, and
potentially future ones) need the same three-phase lifecycle:

1. **Before** the pipeline: ensure a shared staging environment
   variable (``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR``) is set so all
   worker processes write artifacts to a known local directory.
2. **Before** ``ray.shutdown()``: gather staged files from every
   node and upload them to the final output directory.
3. **Upload** uses :class:`StorageWriter` -- local paths, S3, and
   Azure are handled transparently.

Collection strategy is chosen automatically based on the
destination:

- **Remote** (S3 / Azure): each worker node uploads directly via
  its own ``StorageWriter``, eliminating the driver-hop that would
  otherwise double network transfer.
- **Local**: files are gathered to the driver via
  ``RayFileTransport``, then written locally via ``StorageWriter``.

::

    Remote destination (S3 / Azure)
    ===============================

    Worker A: staging/ --+--> StorageWriter --> S3
    Worker B: staging/ --+--> StorageWriter --> S3
    Worker C: staging/ --+--> StorageWriter --> S3
        (each node uploads directly; driver only coordinates)

    Local destination
    =================

    Worker A: staging/ --+
    Worker B: staging/ --+--> RayFileTransport --> driver --> StorageWriter --> local dir
    Worker C: staging/ --+
        (files gathered to driver first, then written locally)

Before this module existed, each subsystem duplicated this logic
independently.  ``ArtifactDelivery`` replaces those duplicates with
a single parameterised implementation -- behaviour is controlled
entirely by constructor arguments (``kind``, ``upload_subdir``,
``collect_on_shutdown``), with zero knowledge of any specific
subsystem.

Example::

    from cosmos_curator.core.utils.artifacts.delivery import ArtifactDelivery

    # Auto-collect at shutdown (default):
    ArtifactDelivery.create(
        kind="profiling",
        output_dir="/output/profiles",
    )

    # On-demand collection (consumer controls timing):
    delivery = ArtifactDelivery.create(
        kind="traces",
        output_dir="s3://bucket/profiles",
        upload_subdir="traces",
        collect_on_shutdown=False,
    )
    # ... run pipeline ...
    delivery.collect()  # explicit call
"""

import concurrent.futures
import contextlib
import os
import pathlib
import shutil
import tempfile

import attrs
import ray
from loguru import logger

from cosmos_curator.core.utils.artifacts.collector import RayFileTransport
from cosmos_curator.core.utils.infra import ray_cluster_utils
from cosmos_curator.core.utils.infra.tracing import StatusCode, TracedSpan, traced_span
from cosmos_curator.core.utils.storage import storage_utils

_REMOTE_COLLECT_TIMEOUT_S: float = 600.0
"""Maximum cumulative stall seconds before declaring upload timeout.

Acts as a stall guard for the adaptive ``ray.wait()`` loop in
``_process_upload_results``: if no node produces a result for this
many cumulative seconds, remaining nodes are marked as failed
(lenient) or ``ArtifactDeliveryError`` is raised (strict).
"""

_UPLOAD_WAIT_BATCH_TIMEOUT_S: float = 0.1
"""Short timeout for the adaptive ``ray.wait()`` batching strategy.

Uses ``ray.wait(remaining, num_returns=len(remaining), timeout=0.1)``
to drain all ready upload results within 100 ms per call.  On large
clusters (1000+ nodes), multiple actors finish within the same 100 ms
window, reducing gRPC round-trips from O(N) to O(N/batch_size).

The ``_REMOTE_COLLECT_TIMEOUT_S`` stall guard tracks cumulative time
with zero results and triggers the timeout path when exceeded.
"""


@attrs.define(frozen=True)
class _NodeUploadResult:
    """Result from a single ``_NodeUploader`` actor.

    Immutable data class returned by each per-node upload actor so
    the driver can aggregate results without exposing mutable state.

    Attributes:
        node_name: Human-readable node identifier.
        files_uploaded: Number of files successfully uploaded.
        errors: Error messages for files that failed to upload.
            Empty when all files succeeded.

    """

    node_name: str
    files_uploaded: int
    errors: tuple[str, ...]


class ArtifactDeliveryError(Exception):
    """Raised when artifact delivery fails in strict mode.

    When ``strict=True`` is passed to :meth:`ArtifactDelivery.create`,
    any file upload failure or node collection failure raises this
    exception instead of being silently logged and swallowed.

    The exception chain (``__cause__``) preserves the original error
    so callers can inspect the root cause via standard Python chaining::

        try:
            delivery.collect()
        except ArtifactDeliveryError as e:
            original = e.__cause__   # the underlying IOError, etc.

    This exception is raised from three locations:

    - ``_NodeUploader.upload()``: per-file upload failure on a worker
      node (remote collection path).
    - ``_collect_local()``: node collection failure or per-file upload
      failure on the driver (local collection path).
    - ``_collect_remote()``: any failure during remote collection
      coordination (wraps ``RayTaskError`` from failed actors).

    """


@ray.remote(num_cpus=0)
class _NodeUploader:
    """Upload staged files directly to a remote destination from this node.

    Deployed on each worker node via ``NodeAffinitySchedulingStrategy``
    so it reads files from the local staging directory and uploads
    them directly to the remote destination via ``StorageWriter``.

    Design decisions:
        - ``num_cpus=0``: upload is I/O-bound (network to S3/Azure),
          not CPU-bound.  Requesting zero CPUs avoids displacing
          pipeline actors during post-pipeline collection.
        - One actor per node: mirrors the ``_NodeCollector`` pattern
          from ``RayFileTransport``.  Each actor handles all files
          on its node to amortize actor creation overhead.
        - ``StorageWriter`` constructed on the worker: each node
          creates its own ``StorageWriter`` instance, which resolves
          the S3/Azure client locally.  The driver never transfers
          file bytes -- only the destination path and credentials
          profile cross the actor boundary.
        - Concurrent intra-node uploads: files are uploaded via a
          ``ThreadPoolExecutor`` so multiple network-I/O-bound
          uploads run in parallel within the actor.  The thread
          pool wraps ``StorageWriter.upload_file_to()`` which
          includes built-in retry logic (``do_with_retries``).

    ::

        _NodeUploader.upload()
        |
        +-- staging_dir exists?
        |     NO  -> return(files_uploaded=0, errors=())
        |     YES -> continue
        |
        +-- StorageWriter(upload_dest, profile_name=...)
        |
        +-- enumerate files -> submit to ThreadPoolExecutor
        |     +-- thread 1: writer.upload_file_to(sub_path_1, file_1)
        |     +-- thread 2: writer.upload_file_to(sub_path_2, file_2)
        |     +-- ...
        |     +-- thread N: writer.upload_file_to(sub_path_N, file_N)
        |
        +-- as_completed() loop:
        |     SUCCESS -> uploaded += 1
        |     FAIL (strict=True)  -> cancel remaining, raise
        |     FAIL (strict=False) -> errors.append(sub_path + error)
        |
        +-- return _NodeUploadResult(node_name, uploaded, errors)

    Edge cases:
        - Empty staging directory: returns immediately with
          ``files_uploaded=0`` and no errors.
        - Individual file upload failure (``strict=False``): caught
          per-file, recorded in ``errors``, and remaining files
          continue uploading.
        - Individual file upload failure (``strict=True``): raises
          ``ArtifactDeliveryError`` immediately, cancelling pending
          uploads for this node.

    """

    _UPLOAD_MAX_WORKERS: int = 10

    def upload(
        self,
        *,
        node_name: str,
        staging_dir: str,
        upload_dest: str,
        s3_profile_name: str,
        strict: bool = False,
    ) -> _NodeUploadResult:
        """Read local staged files and upload concurrently via ``StorageWriter``.

        Enumerates all files under ``staging_dir`` recursively,
        submits each to a ``ThreadPoolExecutor`` for concurrent
        upload to ``upload_dest/<relative_path>`` via a
        locally-constructed ``StorageWriter``, and returns a result
        with per-node counts.

        Concurrency model::

            ThreadPoolExecutor (max_workers=10)
            +---------------------------------------------------+
            |  thread 1: writer.upload_file_to(sub_1, file_1)   |
            |  thread 2: writer.upload_file_to(sub_2, file_2)   |
            |  ...                                               |
            |  thread N: writer.upload_file_to(sub_N, file_N)   |
            +-------------------------+-------------------------+
                                      |
                                      v
                      +-------------------------------+
                      | StorageWriter.upload_file_to  |
                      |   -> _upload_file             |
                      |     -> do_with_retries(3x)    |
                      |       -> client.upload_file   |
                      +-------------------------------+
                                      |
                                      v
                      +-------------------------------+
                      | as_completed() collection     |
                      |   OK   -> uploaded += 1       |
                      |   FAIL -> strict ? raise      |
                      |           lenient ? record err |
                      +-------------------------------+

            Both ``boto3`` and ``azure-storage-blob`` clients are
            thread-safe for independent upload operations.

        Error handling:
            Controlled by the ``strict`` parameter:

            - ``strict=False`` (default): individual file failures
              are caught per-file and recorded in the ``errors``
              tuple.  Remaining files continue uploading -- a single
              poisoned file does NOT abort the entire node's upload.
              Partial delivery is preferable to zero delivery for
              diagnostic artifacts.

            - ``strict=True``: the first file failure cancels all
              pending futures and raises ``ArtifactDeliveryError``
              immediately.  The caller (typically ``_collect_remote``
              via ``ray.get()``) receives the exception and can
              implement fail-fast semantics.

        Edge cases:
            - Non-existent staging directory: returns immediately
              with ``files_uploaded=0`` and empty errors (both modes).
            - Empty staging directory (exists but no files):
              no futures are submitted, returning
              ``files_uploaded=0`` with empty errors (both modes).
            - Subdirectories in staging: only regular files are
              uploaded (``is_file()`` check), directories are
              skipped.
            - Symlinks: skipped entirely (``is_symlink()`` check)
              to prevent traversal outside the staging directory.

        Args:
            node_name: Human-readable name for this node (for
                logging and result reporting).
            staging_dir: Absolute local path to the staging
                directory on this node.
            upload_dest: Remote destination path (``s3://...`` or
                ``az://...``).
            s3_profile_name: Named credential profile for
                ``StorageWriter``.
            strict: When ``True``, the first file upload failure
                raises ``ArtifactDeliveryError`` instead of recording
                the error and continuing.  When ``False`` (default),
                errors are collected and returned in the result.

        Returns:
            ``_NodeUploadResult`` with per-node upload counts and
            any error messages.

        Raises:
            ArtifactDeliveryError: When ``strict=True`` and a file
                upload fails.  The ``__cause__`` attribute holds
                the original exception.

        """
        logger.debug(
            f"[_NodeUploader] Starting upload on node {node_name} "
            f"(staging={staging_dir}, dest={upload_dest}, pid={os.getpid()})"
        )

        staging = pathlib.Path(staging_dir)
        if not staging.exists():
            logger.debug(f"[_NodeUploader] No staging dir on node {node_name} (pid={os.getpid()})")
            return _NodeUploadResult(node_name=node_name, files_uploaded=0, errors=())

        writer = storage_utils.StorageWriter(
            upload_dest,
            profile_name=s3_profile_name,
        )

        file_paths = [fp for fp in staging.rglob("*") if not fp.is_symlink() and fp.is_file()]
        if not file_paths:
            logger.debug(f"[_NodeUploader] No files to upload on node {node_name} (pid={os.getpid()})")
            return _NodeUploadResult(node_name=node_name, files_uploaded=0, errors=())

        logger.debug(
            f"[_NodeUploader] Uploading {len(file_paths)} files concurrently "
            f"(max_workers={self._UPLOAD_MAX_WORKERS}) on node {node_name} (pid={os.getpid()})"
        )

        uploaded = 0
        errors: list[str] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._UPLOAD_MAX_WORKERS) as pool:
            future_to_sub_path: dict[concurrent.futures.Future[None], str] = {}
            for file_path in file_paths:
                sub_path = str(file_path.relative_to(staging))
                future = pool.submit(writer.upload_file_to, sub_path, file_path)
                future_to_sub_path[future] = sub_path

            for future in concurrent.futures.as_completed(future_to_sub_path):
                sub_path = future_to_sub_path[future]
                try:
                    future.result()
                    uploaded += 1
                except Exception as exc:
                    if strict:
                        for f in future_to_sub_path:
                            f.cancel()
                        msg = f"Failed to upload {sub_path} on node {node_name}: {exc}"
                        raise ArtifactDeliveryError(msg) from exc
                    errors.append(f"{sub_path}: {exc}")

        logger.debug(
            f"[_NodeUploader] Completed upload on node {node_name}: "
            f"{uploaded} files, {len(errors)} errors (pid={os.getpid()})"
        )
        return _NodeUploadResult(
            node_name=node_name,
            files_uploaded=uploaded,
            errors=tuple(errors),
        )


class ArtifactDelivery:
    r"""Collect staged files from distributed worker nodes and upload to a destination.

    Generic orchestrator for the three-phase artifact lifecycle:
    staging, cross-node collection, and upload.  Fully consumer-
    agnostic -- knows nothing about the nature of the files it
    moves or which subsystem produced them.  All behaviour is
    controlled exclusively by constructor parameters.

    Why this class exists
    ~~~~~~~~~~~~~~~~~~~~~
    Multiple independent subsystems write artifacts to local
    staging directories on worker nodes during a distributed
    pipeline run.  After the pipeline finishes (but before the
    compute cluster shuts down), those files must be gathered
    from every node and uploaded to a final output directory
    (local path, S3, or Azure).

    Collection and upload are deliberately **deferred to the end
    of execution** rather than performed inline during the
    pipeline run.  Moving artifacts while stages are actively
    processing would compete for network bandwidth, disk I/O,
    and CPU cycles -- directly impacting pipeline throughput and
    stage latency.  By staging files locally during the run and
    collecting them in a single batch after the pipeline
    completes, the hot path remains uncontested.

    The three-phase lifecycle is identical across subsystems -- only
    the staging subdirectory name and the upload destination path
    differ.  Rather than duplicating the same env-var management,
    collection, upload loop, idempotency guard, and crash-safe
    cleanup in every subsystem, this class provides a single
    implementation parameterised by ``kind`` and ``upload_subdir``.

    Design decisions
    ~~~~~~~~~~~~~~~~
    - **Factory pattern** (:meth:`create`): consumers call
      ``create()`` instead of ``__init__`` because the factory
      handles env-var setup and optional pre-shutdown hook
      registration -- side effects that a constructor should not
      perform.
    - **``collect_on_shutdown`` parameter**: controls whether
      :meth:`collect` is registered as a pre-shutdown hook
      (default) or left to the consumer to call explicitly.
      This enables on-demand collection when the consumer needs
      to control timing (e.g. collect between pipeline phases).
    - **Local vs remote routing**: :meth:`collect` constructs a
      ``StorageWriter`` and checks ``writer.is_remote`` to pick
      the optimal collection strategy.  For remote destinations,
      each worker uploads directly (``_NodeUploader`` actors),
      avoiding a driver-hop that would double network transfer.
      For local destinations, ``RayFileTransport`` gathers files
      to the driver first.  The ``StorageWriter`` abstraction is
      the single decision point -- ``ArtifactDelivery`` never
      inspects path prefixes directly.

    ::

        ArtifactDelivery lifecycle
        ==========================

        create(kind, output_dir, ...)
              |
              +-- 1. Set COSMOS_CURATOR_ARTIFACTS_STAGING_DIR env var (idempotent)
              +-- 2. Derive staging_dir = <base>/<kind>
              +-- 3. If collect_on_shutdown: register collect() as pre-shutdown hook
              |
              v
        [pipeline runs -- workers write to <staging>/<kind>/]
              |
              v
        collect()  [fired by hook or called explicitly]
              |
              +-- _collected guard (idempotent, no-op on 2nd call)
              +-- StorageWriter(upload_dest).is_remote?
              |       |                       |
              |       v (yes)                 v (no)
              |   _collect_remote()      _collect_local(writer)
              |   Deploy _NodeUploader   RayFileTransport.collect()
              |   on each node           directly into dest dir
              |   (direct S3/Azure)      (no temp dir, no rewrite)
              |       |                       |
              +-------+-----------------------+
              |
              +-- return file count

    Parameters
    ----------
    Behaviour is controlled without any subsystem knowledge:

    - ``kind`` -- names the staging subdirectory under the shared
      base (e.g. ``"profiling"``, ``"traces"``).  Each kind is
      isolated so different subsystems never mix files.
    - ``upload_subdir`` -- optional path appended to ``output_dir``
      when uploading.  Empty string means upload directly to
      ``output_dir``; ``"traces"`` means upload to
      ``<output_dir>/traces/``.
    - ``collect_on_shutdown`` -- when ``True`` (default),
      :meth:`collect` is registered as a pre-shutdown hook and
      fires automatically.  When ``False``, the consumer calls
      :meth:`collect` explicitly.
    - ``strict`` -- when ``True``, any failure (per-file upload,
      node collection, or transport) raises
      ``ArtifactDeliveryError`` instead of logging and continuing.
      When ``False`` (default), failures are logged as warnings
      and partial delivery proceeds.

    Guarantees
    ~~~~~~~~~~
    - **Idempotent**: the ``_collected`` flag prevents double-
      execution if both the pre-shutdown hook and an explicit call
      fire.
    - **Crash-safe**: on failure, the local collection directory
      is preserved on disk and its path is logged so operators can
      manually retry the upload.  On success, only the kind-specific
      subdirectory (``<base>/<kind>/``) is removed -- the shared
      base staging directory is left intact so sibling deliveries
      can still collect their artifacts.
    - **Partial-failure tolerant** (``strict=False``): nodes that
      fail to deliver files are logged as warnings; collection
      continues for the remaining nodes.
    - **Fail-fast** (``strict=True``): the first failure raises
      ``ArtifactDeliveryError`` so the caller can abort early
      and handle the error (e.g. retry, alert, fail the job).

    Edge cases
    ~~~~~~~~~~
    - **Empty output_dir**: :meth:`collect` returns ``0``
      immediately -- no actors are deployed.
    - **Empty staging on all nodes**: returns ``0`` with an
      informational log message.
    - **Mixed success/failure across nodes**: successful nodes
      have their files uploaded; failed nodes are logged as
      warnings.  The caller receives the total uploaded count.
    - **Double call**: second call is a no-op returning ``0``.

    Attributes
    ----------
        _kind: Staging subdirectory name (e.g. ``"profiling"``).
        _output_dir: Final upload base (local / S3 / Azure path).
        _s3_profile_name: Named credential profile for
            ``StorageWriter``.
        _staging_dir: Resolved local staging path
            (``<base>/<kind>``).
        _upload_subdir: Path appended to ``_output_dir`` for the
            ``StorageWriter`` destination.  Empty string means
            upload directly to ``_output_dir``.
        _strict: When ``True``, failures raise
            ``ArtifactDeliveryError``.  When ``False``, they are
            logged and swallowed.
        _collected: ``True`` after :meth:`collect` has successfully
            run.

    """

    _ENV_STAGING_DIR = "COSMOS_CURATOR_ARTIFACTS_STAGING_DIR"

    def __init__(  # noqa: PLR0913
        self,
        *,
        kind: str,
        output_dir: str,
        s3_profile_name: str,
        staging_dir: str,
        upload_subdir: str = "",
        strict: bool = False,
    ) -> None:
        """Initialize an ``ArtifactDelivery`` instance.

        Prefer :meth:`create` over direct construction.  The factory
        handles env-var setup and optional pre-shutdown hook
        registration that ``__init__`` intentionally does not perform
        (constructor should be side-effect-free).

        Args:
            kind: Staging subdirectory name (e.g. ``"profiling"``,
                ``"traces"``).  Used only for log messages after
                construction (the actual path is ``staging_dir``).
            output_dir: Final upload base directory.  Supports local
                paths, ``s3://...``, and ``az://...``.
            s3_profile_name: Named credential profile for
                ``StorageWriter`` when ``output_dir`` is a remote
                path.
            staging_dir: Absolute local path to the staging
                directory (``<base>/<kind>``).  Workers write
                artifacts here during the pipeline run.
            upload_subdir: Optional path appended to ``output_dir``
                when constructing the ``StorageWriter`` destination.
                Empty string means upload directly to
                ``output_dir``.
            strict: When ``True``, any file upload or node
                collection failure raises ``ArtifactDeliveryError``
                instead of logging a warning and continuing.
                When ``False`` (default), failures are logged and
                partial delivery proceeds.

        """
        self._kind = kind
        self._output_dir = output_dir
        self._s3_profile_name = s3_profile_name
        self._staging_dir = staging_dir
        self._upload_subdir = upload_subdir
        self._strict = strict
        self._collected = False

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        *,
        kind: str,
        output_dir: str,
        s3_profile_name: str = "default",
        upload_subdir: str = "",
        collect_on_shutdown: bool = True,
        strict: bool = False,
    ) -> "ArtifactDelivery":
        """Create an instance, set the staging env var, and optionally register the collection hook.

        Must be called on the **driver** process **before** the
        pipeline starts so that worker processes inherit the
        ``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR`` environment variable.

        Performs up to three actions:

        1. Reads or creates the base staging directory from the
           ``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR`` env var.  If the
           env var is already set (e.g. by a prior ``create()`` call
           from another subsystem), it is read idempotently -- the
           value is never overwritten.
        2. Derives the staging subdirectory as ``<base>/<kind>``
           so that different subsystems are isolated.
        3. When ``collect_on_shutdown`` is ``True`` (default),
           registers :meth:`collect` as a pre-shutdown hook via
           :func:`ray_cluster_utils.register_pre_shutdown_hook` so
           artifact collection runs while the cluster is still alive
           (before ``ray.shutdown()``).

        ::

            create()
            |
            +-- read/create COSMOS_CURATOR_ARTIFACTS_STAGING_DIR
            +-- staging_dir = <base>/<kind>
            +-- collect_on_shutdown? register_pre_shutdown_hook(collect)
            +-- return instance

        Args:
            kind: Staging subdirectory name (e.g. ``"profiling"``,
                ``"traces"``).
            output_dir: Final upload base (local / S3 / Azure).
            s3_profile_name: Named credential profile for
                ``StorageWriter`` (default ``"default"``).
            upload_subdir: Optional path appended to ``output_dir``
                for the upload destination.  Empty string means
                upload directly to ``output_dir``.
            collect_on_shutdown: When ``True`` (default),
                :meth:`collect` is registered as a pre-shutdown hook
                and fires automatically before cluster shutdown.
                When ``False``, the consumer is responsible for
                calling :meth:`collect` explicitly at the desired
                time.
            strict: When ``True``, any file upload or node
                collection failure raises ``ArtifactDeliveryError``
                instead of logging and continuing.  When ``False``
                (default), failures are logged as warnings and
                partial delivery proceeds -- appropriate for
                diagnostic artifacts where partial data is better
                than none.

        Returns:
            A configured ``ArtifactDelivery`` instance.

        """
        base_staging = os.environ.get(cls._ENV_STAGING_DIR)
        if base_staging is None or not pathlib.Path(base_staging).is_dir():
            if base_staging is not None:
                logger.warning(
                    f"[ArtifactDelivery:{kind}] Staging directory no longer exists: "
                    f"{base_staging}; re-creating (pid={os.getpid()})"
                )
            base_staging = tempfile.mkdtemp(prefix="cosmos_curator_staging_")
            os.environ[cls._ENV_STAGING_DIR] = base_staging

        # Each kind gets its own subdirectory so different subsystems
        # never mix files (e.g. <base>/profiling/ vs <base>/traces/).
        staging_dir = str(pathlib.Path(base_staging) / kind)
        logger.debug(f"[ArtifactDelivery:{kind}] Staging directory: {staging_dir} (pid={os.getpid()})")

        instance = cls(
            kind=kind,
            output_dir=output_dir,
            s3_profile_name=s3_profile_name,
            staging_dir=staging_dir,
            upload_subdir=upload_subdir,
            strict=strict,
        )

        if collect_on_shutdown:
            # Register collection as a pre-shutdown hook so it runs
            # while the cluster is still alive (before shutdown).
            ray_cluster_utils.register_pre_shutdown_hook(instance.collect)

        return instance

    def collect(self) -> int:
        """Gather staged files from all worker nodes and upload.

        Safe to call even if the pipeline failed -- collects
        whatever files were staged before the failure.

        The collection strategy is chosen automatically based on
        the destination:

        - **Remote** (``writer.is_remote``): deploys a
          ``_NodeUploader`` actor on each node that reads local
          files and uploads directly to S3/Azure.  No file data
          crosses the cluster network -- only metadata returns to
          the driver.
        - **Local** (``not writer.is_remote``): uses
          ``RayFileTransport`` to gather files to a temp dir on the
          driver, then uploads via ``StorageWriter``.

        ::

            collect()
            |
            +-- _collected guard --> return 0 if already collected
            |
            +-- try: _collect_with_tracing(tag, upload_dest)
            |       |
            |       +-- traced_span("ArtifactDelivery.collect")
            |       |       |
            |       |       +-- StorageWriter(upload_dest).is_remote?
            |       |       |       |                        |
            |       |       |       v (yes)                  v (no)
            |       |       |   _collect_remote()       _collect_local(writer)
            |       |       |       |                        |
            |       |       +-------+------------------------+
            |       |       |
            |       |       +-- span.set_attribute("artifact.files_uploaded", count)
            |       |
            |       +-- return count
            |
            +-- except ArtifactDeliveryError: re-raise (strict path)
            |
            +-- except Exception:
            |     strict=True  -> raise ArtifactDeliveryError from exc
            |     strict=False -> log warning, return 0

        Design decisions:
            - ``StorageWriter.is_remote`` is the single routing
              decision point.  ``ArtifactDelivery`` never inspects
              path prefixes (``s3://``, ``az://``) directly -- it
              delegates that knowledge to ``StorageWriter``.
            - For remote destinations, uploading from each node
              avoids doubling network transfer (worker->driver +
              driver->S3).  The driver only coordinates actors and
              aggregates results.
            - For local destinations, ``RayFileTransport`` is
              required because worker nodes may not share a
              filesystem with the driver.

        **Idempotent**: the ``_collected`` flag prevents double-
        execution if both the pre-shutdown hook and an explicit call
        fire.

        **Crash-safe**: on failure, the local collection directory
        is preserved on disk and its path is logged so operators can
        manually retry the upload.

        **Partial-failure tolerant** (``strict=False``): nodes that
        fail to deliver files are logged as warnings; collection
        proceeds for the remaining nodes so no data is unnecessarily
        lost.

        **Fail-fast** (``strict=True``): the first failure raises
        ``ArtifactDeliveryError`` so the caller can abort early.
        The exception chain preserves the original error for
        inspection.

        Tracing (graceful degradation):
            This method is instrumented with ``traced_span``.  When
            ``collect()`` fires as a pre-shutdown hook, the OTel
            TracerProvider on the driver is still alive, so spans
            are created normally.  However, the span file is written
            to ``<staging>/traces/`` **after** the trace-collection
            hook has already gathered those files (LIFO ordering).
            As a result, spans from shutdown-triggered collection
            are best-effort -- they are fully captured when
            ``collect_on_shutdown=False`` and the consumer calls
            ``collect()`` explicitly during normal execution.

        Returns:
            Total number of files collected and uploaded.
            Returns ``0`` if already collected, no output_dir is
            configured, or no files were staged.

        Raises:
            ArtifactDeliveryError: When ``strict=True`` and any
                file upload, node collection, or transport failure
                occurs.  The ``__cause__`` attribute holds the
                original exception.

        """
        if self._collected:
            return 0

        tag = f"[ArtifactDelivery:{self._kind}]"

        if not self._output_dir:
            logger.debug(f"{tag} No output_dir configured; skipping collection")
            return 0

        # Build the upload destination.  When upload_subdir is set,
        # files go to <output_dir>/<upload_subdir>/ (e.g.
        # "profiles/traces/").  Otherwise they go directly to
        # output_dir.
        upload_dest = (
            f"{self._output_dir.rstrip('/')}/{self._upload_subdir}" if self._upload_subdir else self._output_dir
        )

        # Top-level guard: catches pre-routing failures (e.g.
        # StorageWriter construction: invalid profile, malformed
        # path, client init failure) that would otherwise bypass
        # _collect_local/_collect_remote's per-strategy try/except.
        # In lenient mode the API contract requires returning 0 on
        # any failure -- not raising.
        try:
            return self._collect_with_tracing(tag, upload_dest)
        except ArtifactDeliveryError:
            raise
        except Exception as exc:
            if self._strict:
                msg = f"{tag} Collection failed: {exc}"
                raise ArtifactDeliveryError(msg) from exc
            logger.warning(f"{tag} Collection failed (pid={os.getpid()}): {exc}", exc_info=True)
            return 0

    def _collect_with_tracing(self, tag: str, upload_dest: str) -> int:
        """Execute collection inside a traced span.

        Separated from :meth:`collect` so the top-level exception
        guard can catch failures from ``StorageWriter`` construction
        without duplicating the idempotency and guard-clause logic.

        Args:
            tag: Log prefix (e.g. ``"[ArtifactDelivery:profiling]"``).
            upload_dest: Fully-resolved upload destination path.

        Returns:
            Total number of files collected and uploaded.

        """
        # Tracing: best-effort during shutdown -- spans created here
        # may not be collected when this fires as a pre-shutdown hook
        # because the trace-collection hook runs in LIFO order and
        # may have already gathered files.  See collect() docstring.
        with traced_span(
            "ArtifactDelivery.collect",
            attributes={
                "artifact.kind": self._kind,
                "artifact.destination": upload_dest,
            },
        ) as span:
            # StorageWriter.is_remote is the single routing decision
            # point -- we never inspect path prefixes directly.
            writer = storage_utils.StorageWriter(
                upload_dest,
                profile_name=self._s3_profile_name,
            )

            strategy = "remote" if writer.is_remote else "local"
            span.set_attribute("artifact.strategy", strategy)

            count = self._collect_remote(tag, upload_dest) if writer.is_remote else self._collect_local(tag, writer)

            span.set_attribute("artifact.files_uploaded", count)
            return count

    def _cleanup_kind_staging(self, tag: str) -> None:
        """Remove only this instance's kind-specific staging subdirectory.

        Called after successful collection to reclaim disk space.
        Only removes ``<base>/<kind>/`` (e.g. ``<base>/profiling/``),
        **not** the shared base directory.  This is critical when
        multiple ``ArtifactDelivery`` instances share the same base
        staging directory (e.g. profiling + traces): the first to
        finish must not destroy the sibling's subdirectory.

        The shared base temp directory is left in place and cleaned
        up by the OS on reboot (it lives under ``/tmp/``).

        ``shutil.rmtree`` is wrapped in a broad ``except`` so
        cleanup failures are logged but never propagate -- cleanup
        must not break the success path.

        Args:
            tag: Log prefix for diagnostic messages.

        """
        kind_dir = pathlib.Path(self._staging_dir)
        if not kind_dir.exists():
            return
        try:
            shutil.rmtree(kind_dir)
            logger.debug(f"{tag} Removed staging subdirectory: {kind_dir}")
        except Exception as e:  # noqa: BLE001 -- cleanup must never crash the pipeline
            logger.warning(
                f"{tag} Failed to remove staging subdirectory {kind_dir}: {e}",
                exc_info=True,
            )

    def _collect_local(self, tag: str, writer: storage_utils.StorageWriter) -> int:
        """Gather staged files via ``RayFileTransport`` directly into the destination.

        Used when the destination is a local path.  Worker nodes may
        not share a filesystem with the driver, so
        ``RayFileTransport`` streams file data through the Ray object
        store to the driver.

        ::

            _collect_local(writer)
            |
            +-- dest_dir = Path(writer.base_path)
            |
            +-- RayFileTransport.collect(staging_dir, dest_dir)
            |     Stream files from all nodes directly into dest_dir
            |     (streaming, backpressure-bounded, interleaved)
            |
            +-- Report partial node failures
            |     strict=True  -> raise ArtifactDeliveryError
            |     strict=False -> log warnings, continue
            |
            +-- total_files == 0?
            |     YES -> return 0 (nothing to deliver)
            |
            +-- _collected = True
            |
            +-- except ArtifactDeliveryError: re-raise (strict path)
            |
            +-- except Exception:
            |     strict=True  -> raise ArtifactDeliveryError from exc
            |     strict=False -> record in traced_span, log, return 0

        Design decisions:
            - Direct-to-destination: the transport's
              ``_process_chunk`` already creates parent directories
              and writes file data to ``output_dir``.  For a local
              destination this is the same work that
              ``StorageWriter._resolve_local`` would do, so an
              intermediate temp dir followed by a read-back +
              rewrite loop would double the disk I/O for no benefit.
            - No cleanup step: since files land at the final
              destination, there is no temp directory to remove on
              success or preserve on failure.  On partial failure
              the destination itself contains whatever files were
              delivered, which is inspectable by operators.

        Error handling:
            Controlled by ``self._strict``:

            - ``strict=False`` (default): the entire try block is
              wrapped in a broad ``except Exception`` because
              collection runs during cluster shutdown and must
              never crash the shutdown sequence.  On failure:

              1. The exception is recorded on the current
                 ``TracedSpan`` (if tracing is active).
              2. Partially delivered files remain at the
                 destination for operator inspection.
              3. The destination path is logged at WARNING level.

            - ``strict=True``: any failure (node collection failure,
              transport error) raises ``ArtifactDeliveryError``
              with the original exception chained as ``__cause__``.

        Edge cases:
            - ``RayFileTransport.collect()`` raises: partially
              delivered files remain at the destination.  The
              exception is handled per the strict/lenient policy.
            - Zero files across all nodes: returns ``0`` early
              via the ``total_files == 0`` guard.
            - Strict + node failure: ``ArtifactDeliveryError``
              raised before the count is returned.

        Args:
            tag: Log prefix (e.g. ``"[ArtifactDelivery:profiling]"``).
            writer: Pre-constructed ``StorageWriter`` whose
                ``base_path`` is used as the output directory.

        Returns:
            Number of files delivered.  Returns ``0`` on failure
            (lenient mode only).

        Raises:
            ArtifactDeliveryError: When ``strict=True`` and any
                failure occurs (node collection or transport error).

        """
        dest_dir = pathlib.Path(writer.base_path)
        try:
            transport = RayFileTransport()
            # NOTE: strict is intentionally NOT passed to the transport.
            # Local collection always runs in lenient mode to maximise
            # data recovery.  The strict check happens post-collection
            # at the delivery layer (see nodes_failed check below).
            # This differs from _collect_remote where strict is passed
            # to each _NodeUploader so per-file failures abort the node.
            result = transport.collect(
                staging_dir=self._staging_dir,
                output_dir=dest_dir,
            )

            # Annotate the parent span with transport outcome so
            # trace viewers show node/file counts without drilling
            # into logs.
            span = TracedSpan.current()
            span.set_attributes(
                {
                    "artifact.nodes_ok": len(result.nodes_ok),
                    "artifact.nodes_failed": len(result.nodes_failed),
                    "artifact.total_files": result.total_files,
                }
            )
            span.add_event(
                "transport_completed",
                attributes={
                    "total_files": result.total_files,
                    "nodes_ok": len(result.nodes_ok),
                    "nodes_failed": len(result.nodes_failed),
                },
            )

            # Report partial failures so operators know which nodes
            # lost files.  In strict mode, any node failure aborts;
            # in lenient mode, the consumer decides severity.
            for node_name, err_msg in result.nodes_failed:
                logger.warning(f"{tag} Lost files from node {node_name}: {err_msg}")

            if self._strict and result.nodes_failed:
                failed_names = ", ".join(name for name, _ in result.nodes_failed)
                msg = f"{tag} Lost files from node(s): {failed_names}"
                raise ArtifactDeliveryError(msg)  # noqa: TRY301

            if result.total_files == 0:
                logger.info(f"{tag} No files to deliver")

            self._collected = True
            self._cleanup_kind_staging(tag)
        except ArtifactDeliveryError:
            # Strict-mode errors raised inside the try block
            # (e.g. from the node-failure check above).  Let them
            # propagate directly.
            raise
        except Exception as exc:
            # Annotate the parent traced_span with the failure.
            current = TracedSpan.current()
            current.record_exception(exc)
            current.set_status(StatusCode.ERROR, f"Local collection failed: {exc}")
            if self._strict:
                msg = f"{tag} Local collection failed: {exc}"
                raise ArtifactDeliveryError(msg) from exc
            logger.warning(
                f"{tag} Local collection failed: {exc}; partially delivered files may exist at: {dest_dir}",
                exc_info=True,
            )
            return 0
        else:
            if result.total_files > 0:
                logger.info(f"{tag} Delivered {result.total_files} files to {dest_dir}")
            return result.total_files

    def _process_upload_results(
        self,
        pending: dict[ray.ObjectRef, str],  # type: ignore[type-arg]
        tag: str,
    ) -> tuple[int, int, list[tuple[str, str]]]:
        """Collect per-node upload results via adaptive ``ray.wait`` loop.

        Uses adaptive short-timeout batching: drains all ready
        results within 100 ms per ``ray.wait()`` call, reducing
        gRPC round-trips from O(N) to O(N/batch_size) on large
        clusters.  A cumulative stall guard
        (``_REMOTE_COLLECT_TIMEOUT_S``) detects genuinely stalled
        nodes.  Per-node error isolation is preserved: a single
        node failure does not discard results from successful
        nodes (lenient mode).

        Args:
            pending: Mapping from upload future to node name.
            tag: Log prefix for diagnostic messages.

        Returns:
            Tuple of ``(total_uploaded, total_errors, nodes_failed)``
            where ``nodes_failed`` is a list of ``(node_name, error_msg)``
            tuples for nodes that failed.

        Raises:
            ArtifactDeliveryError: In strict mode, on the first node
                failure or timeout.

        """
        total_uploaded = 0
        total_errors = 0
        nodes_failed: list[tuple[str, str]] = []

        # Adaptive short-timeout batching: drain all ready results
        # within 100 ms per call, reducing gRPC round-trips from
        # O(N) to O(N/batch_size) on large clusters.  The stall
        # guard (stall_s >= _REMOTE_COLLECT_TIMEOUT_S) triggers the
        # timeout path when no node produces a result for too long.
        remaining = list(pending)
        stall_s = 0.0
        while remaining:
            ready, remaining = ray.wait(
                remaining,
                num_returns=len(remaining),
                timeout=_UPLOAD_WAIT_BATCH_TIMEOUT_S,
            )

            if not ready:
                stall_s += _UPLOAD_WAIT_BATCH_TIMEOUT_S
                if stall_s >= _REMOTE_COLLECT_TIMEOUT_S:
                    timeout_msg = f"Timeout ({_REMOTE_COLLECT_TIMEOUT_S}s) waiting for uploads"
                    for ref in remaining:
                        ref_node = pending[ref]
                        if self._strict:
                            msg = f"{tag} {timeout_msg} from node {ref_node}"
                            raise ArtifactDeliveryError(msg)
                        nodes_failed.append((ref_node, timeout_msg))
                        logger.warning(f"{tag} {timeout_msg} from node {ref_node}")
                    break
                continue

            stall_s = 0.0
            for ref in ready:
                ref_node = pending[ref]
                try:
                    result: _NodeUploadResult = ray.get(ref)
                    total_uploaded += result.files_uploaded
                    total_errors += len(result.errors)
                    logger.debug(
                        f"{tag} Node {result.node_name}: {result.files_uploaded} files, {len(result.errors)} errors"
                    )
                    for err in result.errors:
                        logger.warning(f"{tag} Upload error on {result.node_name}: {err}")
                except Exception as exc:
                    if self._strict:
                        msg = f"{tag} Remote upload failed on node {ref_node}: {exc}"
                        raise ArtifactDeliveryError(msg) from exc
                    nodes_failed.append((ref_node, str(exc)))
                    logger.warning(
                        f"{tag} Failed to collect from node {ref_node}: {exc}",
                        exc_info=True,
                    )

        return total_uploaded, total_errors, nodes_failed

    def _collect_remote(self, tag: str, upload_dest: str) -> int:
        """Upload staged files directly from each worker node to a remote destination.

        Used when the destination is remote (S3 / Azure).  Deploys
        a ``_NodeUploader`` actor on each node via
        ``NodeAffinitySchedulingStrategy``.  Each actor reads files
        from the local staging directory and uploads directly to the
        remote destination via its own ``StorageWriter`` instance.

        No file data crosses the cluster network -- only metadata
        (upload counts, error messages) returns to the driver.

        Design decisions:
            - Direct upload avoids the driver-hop that
              ``_collect_local`` performs, eliminating the doubled
              network transfer (worker->driver + driver->S3).
            - Each actor creates its own ``StorageWriter`` on the
              worker node so credentials and client state are
              local to each process.
            - ``num_cpus=0`` on actors: upload is I/O-bound, so
              zero CPU reservation avoids displacing pipeline
              actors.

        Concurrency model:
            All actors upload concurrently (true parallelism).
            The driver uses adaptive ``ray.wait()`` with a short
            100 ms timeout to drain results in batches, providing
            **per-node error isolation**: a single node failure
            does not discard results from successful nodes
            (lenient mode).

            1. Each actor uploads directly to S3/Azure -- there is
               no driver bottleneck to interleave around.
            2. The driver only receives lightweight
               ``_NodeUploadResult`` metadata, not file bytes.
            3. ``ray.wait()`` returns results in completion order;
               total time = slowest successful node.

        ::

            _collect_remote(upload_dest)
            |
            +-- get_live_nodes()
            |     no nodes -> return 0
            |
            +-- for each node:
            |     Deploy _NodeUploader (NodeAffinity, num_cpus=0)
            |     actor.upload.remote(..., strict=self._strict) -> future
            |     pending[future] = node_name
            |
            |     Node A: staging/ --> StorageWriter --> S3
            |     Node B: staging/ --> StorageWriter --> S3
            |     Node C: staging/ --> StorageWriter --> S3
            |       (all uploading concurrently)
            |
            +-- adaptive ray.wait() loop:  <-- per-node error isolation
            |     stall_s = 0.0
            |     while remaining:
            |       ready, remaining = ray.wait(remaining,
            |           num_returns=len(remaining),
            |           timeout=_UPLOAD_WAIT_BATCH_TIMEOUT_S)
            |       |
            |       +-- empty (stall):
            |       |     stall_s += _UPLOAD_WAIT_BATCH_TIMEOUT_S
            |       |     stall_s >= _REMOTE_COLLECT_TIMEOUT_S?
            |       |       YES -> mark remaining as failed, break
            |       |       NO  -> continue
            |       |
            |       +-- ready refs (batch):
            |             stall_s = 0.0
            |             for ref in ready:
            |               ray.get(ref)
            |               SUCCESS -> accumulate result
            |               FAIL    -> strict ? raise : log + record
            |
            +-- Aggregate: sum files_uploaded, log errors
            |
            +-- _collected = True
            |
            +-- except ArtifactDeliveryError: re-raise
            |
            +-- except Exception:
            |     strict=True  -> raise ArtifactDeliveryError from exc
            |     strict=False -> record in traced_span, log, return 0
            |
            +-- finally: ray.kill() all actors

        Cleanup guarantees:
            The ``finally`` block kills all deployed actors
            regardless of success or failure, freeing Ray
            resources.  Unlike ``_collect_local``, there is no
            temp directory to preserve -- files either made it to
            S3/Azure or they didn't, and the staging directory on
            each node still contains the originals.  This holds
            in both strict and lenient modes.

        Edge cases:
            - No live nodes: returns ``0`` immediately without
              deploying any actors (both modes).
            - Node with empty staging: actor returns immediately
              with ``files_uploaded=0`` (both modes).
            - Individual file failure on a node (``strict=False``):
              recorded in the node's error list; remaining files on
              that node still upload.
            - Individual file failure on a node (``strict=True``):
              actor raises ``ArtifactDeliveryError``, which
              propagates through ``ray.get()`` as a
              ``RayTaskError``.  The driver re-raises as
              ``ArtifactDeliveryError`` immediately.
            - Single node actor crash (``strict=False``): the
              failure is recorded per-node via ``nodes_failed``;
              results from other successful nodes are preserved
              and counted toward ``total_uploaded``.
            - Single node actor crash (``strict=True``):
              ``ArtifactDeliveryError`` raised immediately;
              results from already-completed nodes are discarded.
            - Timeout: the adaptive ``ray.wait()`` loop uses a short
              100 ms timeout per call.  If no node produces a
              result for ``_REMOTE_COLLECT_TIMEOUT_S`` cumulative
              seconds (default 600 s), all remaining nodes are
              marked as failed (lenient) or
              ``ArtifactDeliveryError`` is raised (strict).
              Staging directories on nodes are preserved for
              manual retry in both modes.

        Args:
            tag: Log prefix (e.g. ``"[ArtifactDelivery:profiling]"``).
            upload_dest: Remote destination path (``s3://...`` or
                ``az://...``).

        Returns:
            Total number of files uploaded across all successful
            nodes.  In lenient mode, partial results are returned
            even if some nodes failed.  Returns ``0`` only when no
            node succeeded or on unexpected failure.

        Raises:
            ArtifactDeliveryError: When ``strict=True`` and any
                failure occurs (per-file upload or actor crash).

        """
        nodes = ray_cluster_utils.get_live_nodes(dump_info=False)
        if not nodes:
            logger.warning(f"{tag} No live nodes found; skipping remote collection (pid={os.getpid()})")
            return 0

        # Deploy one uploader actor per node with hard affinity.
        # ``pending`` maps each ObjectRef to its node_name so we can
        # attribute results (and failures) after ``ray.wait`` returns
        # refs in arbitrary order.
        actors: list[ray.actor.ActorHandle] = []  # type: ignore[type-arg]
        pending: dict[ray.ObjectRef, str] = {}  # type: ignore[type-arg]

        span = TracedSpan.current()

        try:
            for node_info in nodes:
                node_id = node_info["NodeID"]
                node_name = node_info["NodeName"]
                actor = _NodeUploader.options(  # type: ignore[attr-defined]
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    ),
                ).remote()
                actors.append(actor)
                future = actor.upload.remote(
                    node_name=node_name,
                    staging_dir=self._staging_dir,
                    upload_dest=upload_dest,
                    s3_profile_name=self._s3_profile_name,
                    strict=self._strict,
                )
                pending[future] = node_name

            span.set_attribute("artifact.nodes_deployed", len(actors))
            span.add_event(
                "uploaders_deployed",
                attributes={"nodes_deployed": len(actors)},
            )

            total_uploaded, total_errors, nodes_failed = self._process_upload_results(pending, tag)

            # Annotate span with aggregated upload outcome.
            span.set_attributes(
                {
                    "artifact.total_files": total_uploaded,
                    "artifact.upload_errors": total_errors,
                    "artifact.nodes_failed": len(nodes_failed),
                }
            )
            span.add_event(
                "upload_completed",
                attributes={
                    "total_files": total_uploaded,
                    "upload_errors": total_errors,
                    "nodes_deployed": len(actors),
                    "nodes_failed": len(nodes_failed),
                },
            )

            if nodes_failed:
                logger.warning(f"{tag} {len(nodes_failed)} node(s) failed during remote upload (pid={os.getpid()})")

            if total_uploaded == 0:
                logger.info(f"{tag} No files to upload (remote)")
            else:
                logger.info(f"{tag} Uploaded {total_uploaded} files to {upload_dest}")

            self._collected = True
            self._cleanup_kind_staging(tag)
        except ArtifactDeliveryError:
            raise
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, f"Remote collection failed: {exc}")
            if self._strict:
                msg = f"{tag} Remote collection failed: {exc}"
                raise ArtifactDeliveryError(msg) from exc
            logger.warning(
                f"{tag} Remote collection failed (pid={os.getpid()}): {exc}",
                exc_info=True,
            )
            return 0
        else:
            return total_uploaded
        finally:
            # Clean up actors to free resources.
            for actor in actors:
                with contextlib.suppress(Exception):
                    ray.kill(actor)
