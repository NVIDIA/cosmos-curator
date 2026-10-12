# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector validation for staged batches and canonical slices."""

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec, validate_terminal_batch


@pytest.mark.parametrize("canonical", [False, True])
def test_terminal_batch_preserves_slices_and_error_rows(results: pa.Table, spec: ModelSpec, *, canonical: bool) -> None:
    """Offsets and a null error vector do not shift or invalidate neighboring vectors."""
    table = results.select(spec.fields.names)
    if canonical:
        table = table.cast(spec.fields)
    batch = table.combine_chunks().to_batches()[0]
    for offset, count in ((0, len(batch)), (1, 2), (2, 2), (3, 1), (2, 0)):
        validate_terminal_batch(batch.slice(offset, count), spec)


@pytest.mark.parametrize(
    "fault",
    [
        "dimension",
        "nan",
        "infinity",
        "nonunit",
        "zero",
        "null_element",
        "null_vector",
        "null_metadata",
        "wrong_contract",
        "wrong_status",
        "success_error",
        "error_vector",
        "empty_error",
    ],
)
def test_terminal_batch_rejects_invalid_results(results: pa.Table, spec: ModelSpec, fault: str) -> None:
    """Processing and publication share this check for provenance and vector validity."""
    row = results.to_pylist()[0]
    vector = row[spec.vector_field]
    metadata = row[spec.metadata_field]
    if fault == "dimension":
        row[spec.vector_field] = vector[:-1]
    elif fault == "zero":
        row[spec.vector_field] = [0.0] * spec.dimension
    elif fault in {"nan", "infinity", "nonunit", "null_element"}:
        vector[0] = {"nan": float("nan"), "infinity": float("inf"), "nonunit": 2.0, "null_element": None}[fault]
    elif fault == "null_vector":
        row[spec.vector_field] = None
    elif fault == "null_metadata":
        row[spec.metadata_field] = None
    elif fault == "wrong_contract":
        metadata["contract_digest"] = "different"
    elif fault == "wrong_status":
        metadata["status"] = "pending"
    elif fault == "success_error":
        metadata["error_type"] = "unexpected"
    elif fault == "error_vector":
        metadata.update(status="error", error_type="MediaError", error_message="bad media")
    else:
        row[spec.vector_field] = None
        metadata.update(status="error", error_type="MediaError", error_message="")
    table = pa.Table.from_pylist([*results.slice(0, 1).to_pylist(), row], schema=results.schema)
    batch = table.to_batches()[0].slice(1)
    with pytest.raises(ValueError, match=r"embedding provenance|embeddings require"):
        validate_terminal_batch(batch, spec)
