# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding provenance, clip readiness, vector validation, and caption coexistence.

Shared recovery behavior is covered by ``test_conformance.py``.
"""

import json
from dataclasses import replace

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import inputs
from cosmos_curator.next.recipes.video_caption.contracts import caption_contract_digest, terminal_metadata
from cosmos_curator.next.recipes.video_caption.contracts import resolve_model_spec as caption_spec
from cosmos_curator.next.recipes.video_embed.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec
from cosmos_curator.next.recipes.video_transcode.contracts import (
    CLIP_FIELDS,
    CLIP_SCHEMA,
    PUBLICATION_KEY,
    PUBLICATION_VERSION,
)
from cosmos_curator.next.utils.identity import canonical_digest
from tests.cosmos_curator.next.enrichment.helpers import publish_results


def test_publication_records_the_embedding_contract(
    enrichment_input: inputs.EnrichmentInput, results: pa.Table, spec: ModelSpec
) -> None:
    """Shared publication names the variant's field set and model contract next to the receipt."""
    published = publish_results(results, enrichment_input)
    properties = published.read_transaction(published.version).transaction_properties
    assert properties["kind"] == "video-embed"
    assert properties["field"] == spec.vector_field
    assert properties["contract_digest"] == canonical_digest(spec.contract)
    assert published.to_table(columns=[spec.vector_field])[spec.vector_field].null_count == 1


def _write_caption(uri: str) -> None:
    current = lance.dataset(uri)
    spec = caption_spec("qwen3_8_27b_fp8")
    current.add_columns(spec.fields)
    current = lance.dataset(uri)
    fragment = current.get_fragment(0)
    rows = fragment.to_table(columns=["clip_id"]).to_pylist()
    metadata = terminal_metadata(
        spec,
        caption_contract_digest(spec),
        status="success",
        prompt_token_count=10,
        generated_token_count=5,
        error_type=None,
        error_message=None,
    )
    values = pa.Table.from_pylist(
        [row | {spec.caption_field_name: "keep this caption", spec.metadata_field_name: metadata} for row in rows],
        schema=pa.schema([("clip_id", pa.string()), *spec.fields]),
    )
    prepared, fields = fragment.update_columns(values, left_on="clip_id", right_on="clip_id")
    lance.LanceDataset.commit(
        uri,
        lance.LanceOperation.Update(updated_fragments=[prepared], fields_modified=fields),
        read_version=current.version,
    )


def test_captions_and_embeddings_coexist(
    enrichment_input: inputs.EnrichmentInput, results: pa.Table, spec: ModelSpec
) -> None:
    """A caption writer and the embedding publisher each keep their own column files."""
    _write_caption(enrichment_input.dataset.uri)
    published = publish_results(results, enrichment_input)
    captions = published.to_table(columns=[caption_spec("qwen3_8_27b_fp8").caption_field_name]).column(0)
    assert captions.to_pylist() == ["keep this caption", "keep this caption", None, None]
    assert published.to_table(columns=spec.fields.names).equals(results.select(spec.fields.names).cast(spec.fields))


@pytest.mark.parametrize("change", ["null_uri", "pruned_receipt"])
def test_input_validation_is_repeated_when_evidence_changes(
    enrichment_input: inputs.EnrichmentInput,
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    dataset: lance.LanceDataset,
    change: str,
) -> None:
    """A publication proves only its validated inputs while its receipt is retained."""
    published = publish_results(results, enrichment_input)
    if change == "null_uri":
        fragment = published.get_fragment(0)
        values = fragment.to_table(columns=["clip_id"]).append_column(
            "clip_uri", pa.array([None] * fragment.count_rows(), type=pa.large_string())
        )
        metadata, modified = fragment.update_columns(values, left_on="clip_id", right_on="clip_id")
        current = lance.LanceDataset.commit(
            dataset.uri,
            lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
            read_version=published.version,
        )
    else:
        publication_version = published.version
        published.add_columns({"unrelated": "'keep'"})
        current = lance.dataset(dataset.uri)
        current.cleanup_old_versions(versions=[publication_version])
    bindings = resolve_bindings(config)
    assert not inputs.input_is_validated(current, fingerprint=inputs.input_fingerprint(current, bindings))

    if change == "pruned_receipt":
        assert inputs.open_enrichment_input(bindings).dataset.count_rows() == 4
    else:
        with pytest.raises(ValueError, match="Clips require non-null URIs and positive byte sizes"):
            inputs.open_enrichment_input(bindings)


@pytest.mark.parametrize("fault", ["wrong_dimension", "nan"])
def test_invalid_vectors_leave_all_fragments_pending(
    enrichment_input: inputs.EnrichmentInput,
    results: pa.Table,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
    fault: str,
) -> None:
    """Vector width and values are validated before the one canonical commit."""
    rows = results.to_pylist()
    if fault == "wrong_dimension":
        rows[0][spec.vector_field] = [1.0]
    else:
        rows[0][spec.vector_field][0] = float("nan")
    with pytest.raises(ValueError, match=r"canonical output types|finite unit"):
        publish_results(pa.Table.from_pylist(rows, schema=results.schema), enrichment_input)
    assert lance.dataset(dataset.uri).version == dataset.version
    assert inputs.pending_fragment_ids(lance.dataset(dataset.uri), enrichment_input.bindings) == (0, 1)


def test_changed_model_contract_does_not_overwrite_existing_vectors(
    enrichment_input: inputs.EnrichmentInput,
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different model revision requires a new field contract, not a silent replacement."""
    published = publish_results(results, enrichment_input)
    monkeypatch.setattr(
        "cosmos_curator.next.recipes.video_embed.bindings.resolve_model_spec",
        lambda _variant: replace(spec, revision="a" * 40),
    )
    with pytest.raises(ValueError, match="Incompatible or missing embedding provenance"):
        inputs.pending_fragment_ids(published, resolve_bindings(config))


@pytest.mark.parametrize("receipt", ["absent", "pruned"])
def test_ready_clips_do_not_require_a_retained_receipt(
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
    receipt: str,
) -> None:
    """Directly written clips and clips whose history was pruned remain valid inputs."""
    if receipt == "absent":
        uri = f"{config.input.clips_lance_uri}-direct"
        lance.write_dataset(dataset.to_table(columns=CLIP_SCHEMA.names), uri, data_storage_version="2.2")
        config = config.model_copy(update={"input": config.input.model_copy(update={"clips_lance_uri": uri})})
    else:
        old_versions = [ref["version"] for ref in dataset.version_refs() if ref["version"] < dataset.version]
        dataset.cleanup_old_versions(versions=old_versions)
        assert {ref["version"] for ref in dataset.version_refs()} == {dataset.version}

    bindings = resolve_bindings(config)
    opened = inputs.open_enrichment_input(bindings).dataset

    assert opened.count_rows() == 4
    assert set(spec.fields.names).issubset(opened.schema.names)
    expected = (0, 1) if receipt == "pruned" else (0,)
    assert inputs.pending_fragment_ids(opened, bindings) == expected


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("pending", "clip_id values must be non-null and globally unique"),
        ("null_id", "clip_id values must be non-null and globally unique"),
        ("duplicate_id", "clip_id values must be non-null and globally unique"),
        ("null_uri", "Clips require non-null URIs"),
        ("null_size", "Clips require non-null URIs and positive byte sizes"),
        ("zero_size", "Clips require non-null URIs and positive byte sizes"),
        ("negative_size", "Clips require non-null URIs and positive byte sizes"),
    ],
)
def test_publication_receipt_does_not_make_invalid_clip_inputs_ready(
    config: ResolvedVideoEmbedConfig,
    dataset: lance.LanceDataset,
    fault: str,
    message: str,
) -> None:
    """A matching receipt count cannot hide missing or malformed inputs consumed by embedding."""
    rows = dataset.to_table(columns=CLIP_SCHEMA.names).to_pylist()
    if fault == "pending":
        rows[0].update(dict.fromkeys(CLIP_FIELDS.names))
    elif fault == "duplicate_id":
        rows[0]["clip_id"] = rows[1]["clip_id"]
    else:
        field, value = {
            "null_id": ("clip_id", None),
            "null_uri": ("clip_uri", None),
            "null_size": ("clip_size_bytes", None),
            "zero_size": ("clip_size_bytes", 0),
            "negative_size": ("clip_size_bytes", -1),
        }[fault]
        rows[0][field] = value
    lance.write_dataset(
        pa.Table.from_pylist(rows, schema=CLIP_SCHEMA),
        dataset.uri,
        mode="overwrite",
        data_storage_version="2.2",
        transaction_properties={
            PUBLICATION_KEY: json.dumps({"publication_version": PUBLICATION_VERSION, "spans": len(rows)})
        },
    )
    with pytest.raises(ValueError, match=message):
        inputs.open_enrichment_input(resolve_bindings(config))


def test_empty_clip_table_is_ready_without_a_receipt(config: ResolvedVideoEmbedConfig) -> None:
    """An empty canonical schema has no missing clip inputs and needs no historical marker."""
    lance.write_dataset(
        pa.Table.from_pylist([], schema=CLIP_SCHEMA), config.input.clips_lance_uri, data_storage_version="2.2"
    )

    bindings = resolve_bindings(config)
    opened = inputs.open_enrichment_input(bindings).dataset

    assert opened.count_rows() == 0
    assert inputs.pending_fragment_ids(opened, bindings) == ()
