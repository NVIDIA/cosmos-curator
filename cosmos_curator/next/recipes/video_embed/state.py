# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite input selection and durable recovery independent of caption commits."""

import json
from dataclasses import dataclass
from typing import Any

import lance
from lance.fragment import LanceFragment

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec, validate_terminal_row
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA, validate_clip_table
from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.storage import download_bytes, write_if_absent


def storage_options(config: ResolvedVideoEmbedConfig) -> dict[str, str] | None:
    """Resolve credentials at the existing storage seam, never into durable metadata."""
    return get_lance_storage_options(config.input.clips_lance_uri, profile_name=config.execution.storage_profile)


def validate_fields(dataset: lance.LanceDataset, spec: ModelSpec) -> None:
    """Require exact ownership, types, and nullability for the complete field set."""
    for field in spec.fields:
        if field.name not in dataset.schema.names or not field.equals(
            dataset.schema.field(field.name), check_metadata=True
        ):
            msg = f"Missing or incompatible video-embed field: {field.name}"
            raise ValueError(msg)


def open_input(config: ResolvedVideoEmbedConfig, spec: ModelSpec) -> lance.LanceDataset:
    """Validate completed clips and atomically register both enrichment columns."""
    uri = config.input.clips_lance_uri

    def register() -> lance.LanceDataset:
        dataset = lance.dataset(uri, storage_options=storage_options(config))
        validate_clip_table(dataset, uri=uri)
        if not set(spec.fields.names) & set(dataset.schema.names):
            dataset.add_columns(spec.fields)
            dataset = lance.dataset(uri, storage_options=storage_options(config))
        validate_fields(dataset, spec)
        return dataset

    dataset = do_with_retries(
        register, exceptions_to_retry=(OSError, RuntimeError), max_attempts=config.execution.commit_attempts
    )
    if any(fragment.metadata.num_deletions for fragment in dataset.get_fragments()):
        msg = "video-embed requires undeleted video-transcode rows"
        raise ValueError(msg)
    query = (
        "SELECT COUNT(*) AS n, COUNT(DISTINCT clip_id) AS ids, "
        "COUNT(clip_uri) AS uris, COUNT(clip_size_bytes) AS sizes, MIN(clip_size_bytes) AS smallest FROM dataset"
    )
    stats = next(iter(dataset.sql(query).build().to_batch_records())).to_pylist()[0]
    if any(stats[name] != stats["n"] for name in ("ids", "uris", "sizes")) or (stats["n"] and stats["smallest"] <= 0):
        msg = "Clips require globally unique non-null IDs, non-null URIs, and positive byte sizes"
        raise ValueError(msg)
    return dataset


def fragment_complete(fragment: LanceFragment, spec: ModelSpec) -> bool:
    """Require a wholly pending fragment or valid terminal results on every row."""
    states: set[bool] = set()
    for batch in fragment.scanner(columns=spec.fields.names, batch_size=1_024).to_batches():
        for row in batch.to_pylist():
            complete = any(row[name] is not None for name in spec.fields.names)
            if complete:
                validate_terminal_row(row, spec)
            states.add(complete)
            if len(states) > 1:
                msg = f"Fragment {fragment.fragment_id} mixes pending and terminal embeddings"
                raise ValueError(msg)
    return False not in states


def pending_fragment_ids(dataset: lance.LanceDataset, spec: ModelSpec) -> tuple[int, ...]:
    """Select complete physical fragments from one immutable read snapshot."""
    validate_fields(dataset, spec)
    return tuple(
        int(fragment.fragment_id) for fragment in dataset.get_fragments() if not fragment_complete(fragment, spec)
    )


def field_ids(dataset: lance.LanceDataset, names: list[str]) -> dict[str, int]:
    """Map named fields' leaf paths to Lance IDs, including nested metadata leaves."""
    result: dict[str, int] = {}

    def visit(field: Any, path: str) -> None:  # noqa: ANN401 -- LanceField has no public type stub
        children = field.children()
        if children:
            for child in children:
                visit(child, f"{path}.{child.name()}")
        else:
            result[path] = int(field.id())

    for field in dataset.lance_schema.fields():
        if field.name() in names:
            visit(field, field.name())
    return result


def binding(fragment: LanceFragment, protected_ids: set[int]) -> str:
    """Fingerprint row layout and only the physical columns this computation reads."""
    fragment_metadata = fragment.metadata.to_json()
    selected_files = []
    for data_file in fragment_metadata["files"]:
        fields = data_file["fields"]
        positions = data_file.get("column_indices", list(range(len(fields))))
        selected = [(field, pos) for field, pos in zip(fields, positions, strict=True) if field in protected_ids]
        if selected:
            selected_files.append({**data_file, "fields": selected, "column_indices": None})
    return canonical_digest(
        {key: value for key, value in fragment_metadata.items() if key != "files"} | {"files": selected_files}
    )


@dataclass(frozen=True)
class Workspace:
    """One immutable input layout and result contract; execution tuning is irrelevant."""

    root: str
    manifest: dict[str, Any]

    @property
    def results(self) -> str:
        """Return compact, checkpointed embedding rows."""
        return f"{self.root}/results"

    @property
    def checkpoints(self) -> str:
        """Return Ray's durable clip-ID index over the result files."""
        return f"{self.root}/checkpoints"

    @property
    def completion(self) -> str:
        """Name the optional fast path that skips already-completed inference."""
        return f"{self.root}/phase-a-complete.json"

    @property
    def scope(self) -> str:
        """Identify the entire publication for ambiguous-commit reconciliation."""
        return canonical_digest(self.manifest)


def resolve_workspace(config: ResolvedVideoEmbedConfig, dataset: lance.LanceDataset, spec: ModelSpec) -> Workspace:
    """Keep recovery compatible across disjoint caption/schema commits.

    Pin input physical bindings rather than the Lance version: a caption commit
    must not discard previously checkpointed GPU work. Appends, compaction, and
    transcode changes select another workspace.
    """
    input_ids = field_ids(dataset, CLIP_SCHEMA.names)
    manifest = {
        "workspace_version": 1,
        "input_uri": config.input.clips_lance_uri,
        "input_fields": input_ids,
        "input_fragments": [binding(fragment, set(input_ids.values())) for fragment in dataset.get_fragments()],
        "contract": spec.contract,
        "result_schema": spec.result_schema.serialize().to_pybytes().hex(),
    }
    return Workspace(f"{config.output.staging_root_uri}/{spec.vector_field}/{canonical_digest(manifest)}", manifest)


def ensure_workspace(workspace: Workspace, config: ResolvedVideoEmbedConfig) -> None:
    """Atomically create a recovery manifest or verify exact compatibility."""
    uri = f"{workspace.root}/workspace.json"
    data = json.dumps(workspace.manifest, sort_keys=True).encode()
    if not write_if_absent(uri, data, storage_profile=config.execution.storage_profile):
        actual = json.loads(download_bytes(uri, storage_profile=config.execution.storage_profile))
        if actual != workspace.manifest:
            msg = "Embedding workspace does not match this input and result contract"
            raise ValueError(msg)
