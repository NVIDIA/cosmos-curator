# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latest Lance spans -> checkpointed clip metadata -> one Lance publication with video-caption's receipt."""

import json
from functools import partial
from typing import Any, cast

import lance
import ray
from loguru import logger
from ray.data.context import ShuffleStrategy

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.core.ray_autoscaling import configure_download_actor_autoscaling
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.enrichment.attempts import EnrichmentAttempt
from cosmos_curator.next.enrichment.contracts import EnrichmentBindings
from cosmos_curator.next.enrichment.execution import ProcessingScope, run_enrichment
from cosmos_curator.next.recipes.video_transcode.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import (
    PUBLICATION_KEY,
    PUBLICATION_VERSION,
    READ_COLUMNS,
    completed_publication,
)
from cosmos_curator.next.recipes.video_transcode.processing import (
    SourceDownloader,
    group_source_spans,
    transcode_source,
    upload_clips,
)


def start_ray(bindings: EnrichmentBindings, config: ResolvedVideoTranscodeConfig) -> None:
    """Start Ray with IO slots and the storage environment for media, spans, and staging."""
    ensure_ray_initialized(
        local_resources=curator_io_resources(),
        runtime_env_vars=bindings.storage_env_vars(config.input.media_root),
    )
    configure_ray_data_progress(progress=config.execution.progress)
    # Whole-source payloads retain logical memory reservations and backpressure.
    configure_ray_data_stability(disable_high_memory_detector=True)


def transcode_spans(processing: ProcessingScope, config: ResolvedVideoTranscodeConfig) -> None:
    """Execute one read-to-Parquet plan with span filtering before source grouping."""
    execution = config.execution
    # Sort shuffle uses the read block count to size its partitions.
    spans = processing.read(READ_COLUMNS, override_num_blocks=execution.source_partitions)
    # Hash-shuffle actors retain CPU/memory reservations until the whole
    # execution ends. Sort tasks release theirs before downloads begin.
    spans.context.shuffle_strategy = ShuffleStrategy.SORT_SHUFFLE_PULL_BASED
    configure_download_actor_autoscaling(spans.context)
    sources = spans.groupby("source_id").map_groups(
        group_source_spans,
        batch_format="pyarrow",
        num_cpus=0.25,
    )
    # Break size ties so equal-sized sources can occupy separate sort partitions.
    sources = sources.sort(["source_size_bytes", "source_id"], descending=[True, False])
    # Ray Data's task submission budget does not account for curator_io.
    # Actors acquire a slot before accepting work, bounding submission too.
    # Leave the pool uncapped so late Slurm workers can contribute.
    downloaded = sources.map(
        cast("Any", SourceDownloader),
        fn_constructor_kwargs={"config": config},
        compute=ray.data.ActorPoolStrategy(min_size=1, max_tasks_in_flight_per_actor=1),
        num_cpus=0.25,
        max_concurrency=1,
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1.0},
        max_restarts=-1,
        max_task_retries=-1,
    )
    encoded = downloaded.flat_map(
        cast("Any", transcode_source),
        fn_kwargs={"config": config},
        num_cpus=execution.transcode_cpus,
        memory=int(execution.transcode_memory_gib * 1024**3),
    )
    # Clip production and Ray backpressure pace uploads without download slots.
    uploaded = encoded.map_batches(
        cast("Any", upload_clips),
        batch_format="pyarrow",
        batch_size=None,
        zero_copy_batch=True,
        fn_kwargs={"config": config, "staged_schema": processing.staged_schema},
        num_cpus=0.25,
    )
    processing.write(uploaded)


def _publication_record(
    bindings: EnrichmentBindings, *, source_version: int, spans: int, fragment_ids: list[int]
) -> dict[str, str]:
    result_contract = bindings.spec.result_contract
    record = {
        "publication_version": PUBLICATION_VERSION,
        "contract_id": result_contract["transcode_contract_id"],
        "media_root": result_contract["media_root"],
        "source_version": source_version,
        "spans": spans,
        "fragment_ids": fragment_ids,
    }
    return {PUBLICATION_KEY: json.dumps(record, sort_keys=True)}


def publication_properties(attempt: EnrichmentAttempt) -> dict[str, str]:
    """Record the transcode receipt video-caption reads; shared publication records provenance and evidence."""
    counts = attempt.counts()
    return _publication_record(
        attempt.input.bindings,
        source_version=int(attempt.input.dataset.version),
        spans=counts["input_rows"] - counts["remaining_rows"],
        fragment_ids=list(attempt.selected),
    )


def record_empty_receipt(dataset: lance.LanceDataset, bindings: EnrichmentBindings) -> lance.LanceDataset:
    """Return the table, first recording a zero-span receipt if it is empty and has none.

    Temporary: video-caption requires a whole-table transcode receipt. Remove this
    once caption checks clip readiness from data, as video-embed does. Recording
    the receipt again is harmless, so it needs no recovery workspace: each commit
    attempt rereads the latest table and stops once any receipt exists.
    """
    if dataset.count_rows() or completed_publication(dataset) is not None:
        return dataset
    properties = _publication_record(bindings, source_version=int(dataset.version), spans=0, fragment_ids=[])
    storage_options = bindings.storage_options

    def commit() -> lance.LanceDataset:
        latest = lance.dataset(dataset.uri, storage_options=storage_options)
        if completed_publication(latest) is not None:
            return latest
        transaction = lance.Transaction(
            read_version=int(latest.version),
            operation=lance.LanceOperation.Update(updated_fragments=[], fields_modified=[]),
            transaction_properties=properties,
        )
        return lance.LanceDataset.commit(dataset.uri, transaction, storage_options=storage_options)

    return do_with_retries(
        commit, exceptions_to_retry=(OSError, RuntimeError), max_attempts=bindings.execution.commit_attempts
    )


def run_config(config: ResolvedVideoTranscodeConfig) -> dict[str, object]:
    """Resume compatible clip work, publish once, and clean verified recovery state."""
    bindings = resolve_bindings(config)
    outcome = run_enrichment(
        bindings,
        start_ray=partial(start_ray, config=config),
        process=partial(transcode_spans, config=config),
        transaction_properties=publication_properties,
    )
    dataset = record_empty_receipt(outcome.dataset, bindings)
    summary: dict[str, object] = {
        **outcome.summary(),
        "clips_lance_uri": config.input.spans_lance_uri,
        "clips_lance_version": int(dataset.version),
    }
    logger.info("video-transcode: {}", json.dumps(summary, sort_keys=True))
    return summary
