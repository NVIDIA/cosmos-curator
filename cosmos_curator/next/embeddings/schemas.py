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

"""Arrow contracts for the embedding column groups on ``clips.lance`` (pure; no Ray/GPU/torch).

The single source of truth for the columns each modality owns on the shared clips
table, the columns each compute batch carries, and the source projection every leg
reads. There is no side table: a modality is a nullable column group added
directly onto ``clips.lance``::

    embedding_text   -> embedding_text_subtask, embedding_text_task
                        (fsl<f32,384>, unit-norm), embedding_text_model_id
    embedding_image  -> embedding_image (fsl<f32,384>, unit-norm),
                        embedding_image_model_id
    embedding_action -> embedding_action (fsl<f32,97>, raw PCA coords - NOT
                        unit-norm), embedding_action_descriptor_version,
                        embedding_action_pca_fingerprint

``EmbeddingColumnGroup`` wraps each group and ``EMBEDDING_COLUMN_GROUPS`` collects
them; any "all embedding columns" view is derived from that tuple, never
hand-maintained. A group is the unit of schema evolution (one ``add_columns``
metadata commit), of selection (its ``primary_vector`` defines "pending"), and of
publication (all its fields are written by one ``update_columns`` call).

Batches carry NO identity column. The group builders (``text_columns_batch`` /
``image_columns_batch`` / ``action_columns_batch``) return exactly one group's
fields, one row per input row, in input order. The caller that owns storage
attaches ``clip_id`` positionally - keeping the embedders pure compute that never
sees a key. Image and action carry a per-row ``valid`` mask so a decode or
geometry failure persists as an all-NULL (pending / retryable) group instead of
dropping the row; that cardinality-preserving contract is what lets the
positional key attachment be correct. Text has no mask (it never fails per row).

Stored-vector SCALE differs by modality and is part of the contract even though it
is not recorded in Arrow metadata: text and image vectors are **L2-normalized**
(cosine == dot product), while action vectors are **raw PCA coordinates** (NOT
unit-norm - magnitude encodes gesture size). Normalization is a property of the
model spec, so a future cross-modality fusion resolves scale from the spec rather
than from a schema tag; it must not mix the two.

Embedding and provenance columns are nullable and start NULL: NULL means "not
computed", and the companion provenance column (``*_model_id`` for the model
modalities, ``*_descriptor_version`` / ``*_pca_fingerprint`` for action) is NULL
alongside it. Vectors use ``fixed_size_list`` (not variable ``list``): it fixes
the width as a schema guarantee, lets a mixed-width model swap fail loud, and is
the type a Lance vector index requires. The list child is left nullable
(pyarrow's default) so an all-NULL column is a legal column value.
"""

from collections.abc import Sequence

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from cosmos_curator.next.embeddings.model_specs import DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL

# Stable per-row join key on clips.lance (the clip_id the source table supplies;
# in the current implementation robot_action_split mints it per (span, view)
# clip). It is the column ``update_columns`` joins a computed group on, and it
# must never be null - a null key cannot align a row's per-modality vectors, so
# the contract is hardened at the schema level rather than trusted per producer.
KEY_COLUMN = "clip_id"

# Fixed embedding widths. Text and image widths come from the model specs (the
# checkpoint decides them), so a spec swap and the stored schema cannot drift.
# ACTION_DIM is a storage decision (PCA target width, ~90% explained variance),
# not a model property, so it is owned here.
TEXT_DIM = DEFAULT_TEXT_MODEL.dim
IMAGE_DIM = DEFAULT_IMAGE_MODEL.dim
ACTION_DIM = 97

_KEY_FIELD = pa.field(KEY_COLUMN, pa.string(), nullable=False)


def _embedding_type(dim: int) -> pa.DataType:
    """Return the ``fixed_size_list<float32, dim>`` type stored for one vector column.

    The list child is left nullable (pyarrow's default) so an all-NULL column is a
    legal value; a non-nullable child makes Lance reject a fully-NULL column.
    """
    return pa.list_(pa.float32(), dim)


def _group_field(name: str, arrow_type: pa.DataType) -> pa.Field:
    """Build one nullable embedding-group column field.

    Every group column is nullable: a row that has not been computed yet reads
    NULL, which is exactly the incremental selection's "pending" state.
    """
    return pa.field(name, arrow_type, nullable=True)


# Per-modality embedding column groups added directly onto clips.lance. Each group
# is nullable end-to-end and shares an ``embedding_<name>`` prefix so it owns a
# private namespace on the shared table. Text carries two vectors (subtask +
# task) that always transition together; image and action carry one vector plus
# their producer-identity provenance.
TEXT_GROUP_SCHEMA: pa.Schema = pa.schema(
    [
        _group_field("embedding_text_subtask", _embedding_type(TEXT_DIM)),
        _group_field("embedding_text_task", _embedding_type(TEXT_DIM)),
        _group_field("embedding_text_model_id", pa.string()),
    ]
)
IMAGE_GROUP_SCHEMA: pa.Schema = pa.schema(
    [
        _group_field("embedding_image", _embedding_type(IMAGE_DIM)),
        _group_field("embedding_image_model_id", pa.string()),
    ]
)
ACTION_GROUP_SCHEMA: pa.Schema = pa.schema(
    [
        _group_field("embedding_action", _embedding_type(ACTION_DIM)),
        _group_field("embedding_action_descriptor_version", pa.string()),
        _group_field("embedding_action_pca_fingerprint", pa.string()),
    ]
)


@attrs.frozen
class EmbeddingColumnGroup:
    """One modality's atomic embedding + provenance column group on clips.lance.

    A group is the unit of schema evolution, selection, and atomic publication:
    all of its fields are added together (one ``add_columns`` metadata commit),
    filled together (one ``update_columns`` call per fragment, published by one
    ``LanceOperation.Update``), and validated together (each group is
    independently absent-as-a-whole or present-and-exactly-matching). All fields
    are nullable and share the ``embedding_<name>`` prefix.

    Attributes:
        name: Modality name; also the shared column-name prefix
            (``embedding_<name>``).
        schema: The group's Arrow columns (vectors + provenance), each nullable.
            Excluded from equality/hash so the frozen group stays hashable
            (``pa.Schema`` is unhashable) and ``name`` is identity.
        primary_vector: The vector whose NULL state marks a row unfilled. Owed
            work is "applicable AND (this vector IS NULL OR the provenance
            columns name another computation)", so an unfilled row is one of the
            two ways a row can be owed rather than the definition of it.
            Completeness, by contrast, spans ALL fields, so a companion vector
            such as ``embedding_text_task`` is still required for a row to count
            as complete even though it is not the primary.
        provenance_columns: The columns naming the computation that produced a
            row (model result fingerprint for text and image; descriptor version
            + PCA fingerprint for action). They carry the row's RESULT IDENTITY, which
            is what makes a row stale when it names a computation other than the
            one a run performs, and what lets a reader establish that a whole
            group was produced by one computation. Companion vectors are not
            identity; they are covered by the all-fields completeness rule
            instead.

    """

    name: str
    schema: pa.Schema = attrs.field(eq=False)
    primary_vector: str
    provenance_columns: tuple[str, ...]

    def __attrs_post_init__(self) -> None:
        """Validate the group is internally consistent (prefix, presence, nullability)."""
        prefix = f"embedding_{self.name}"
        names = set(self.schema.names)
        for field in self.schema:
            if not field.name.startswith(prefix):
                msg = f"embedding column group {self.name!r}: field {field.name!r} must start with {prefix!r}"
                raise ValueError(msg)
            if not field.nullable:
                msg = f"embedding column group {self.name!r}: field {field.name!r} must be nullable"
                raise ValueError(msg)
        if self.primary_vector not in names:
            msg = (
                f"embedding column group {self.name!r}: primary_vector {self.primary_vector!r} "
                f"is not one of its fields {sorted(names)}"
            )
            raise ValueError(msg)
        missing = [column for column in self.provenance_columns if column not in names]
        if missing:
            msg = (
                f"embedding column group {self.name!r}: provenance_columns {missing} "
                f"are not group fields {sorted(names)}"
            )
            raise ValueError(msg)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return every column name in the group (vectors + provenance), in schema order."""
        return tuple(self.schema.names)


TEXT_COLUMN_GROUP = EmbeddingColumnGroup(
    name="text",
    schema=TEXT_GROUP_SCHEMA,
    primary_vector="embedding_text_subtask",
    provenance_columns=("embedding_text_model_id",),
)
IMAGE_COLUMN_GROUP = EmbeddingColumnGroup(
    name="image",
    schema=IMAGE_GROUP_SCHEMA,
    primary_vector="embedding_image",
    provenance_columns=("embedding_image_model_id",),
)
ACTION_COLUMN_GROUP = EmbeddingColumnGroup(
    name="action",
    schema=ACTION_GROUP_SCHEMA,
    primary_vector="embedding_action",
    provenance_columns=("embedding_action_descriptor_version", "embedding_action_pca_fingerprint"),
)


def _assert_disjoint_groups(groups: tuple[EmbeddingColumnGroup, ...]) -> None:
    """Fail if two registered groups claim the same column name (namespace overlap)."""
    owner_by_column: dict[str, str] = {}
    for group in groups:
        for column in group.field_names:
            previous = owner_by_column.get(column)
            if previous is not None:
                msg = f"embedding column {column!r} is claimed by both group {previous!r} and group {group.name!r}"
                raise ValueError(msg)
            owner_by_column[column] = group.name


# The single source of truth for all embedding-owned columns. Any "all embedding
# columns" view is derived from this tuple, never hand-maintained separately.
EMBEDDING_COLUMN_GROUPS: tuple[EmbeddingColumnGroup, ...] = (
    TEXT_COLUMN_GROUP,
    IMAGE_COLUMN_GROUP,
    ACTION_COLUMN_GROUP,
)
_assert_disjoint_groups(EMBEDDING_COLUMN_GROUPS)


# Internal transport row between the action leg's two compute phases (extract then
# project), both of which run inside one worker call. It carries no key: rows stay
# in input order, so the projected group aligns positionally. ``action_data_uri``
# survives because the PCA fit sample de-duplicates on it (a multi-view span whose
# views share one action artifact must not get N votes in the fitted basis) and
# the per-batch descriptor memo keys on it. ``descriptor`` is nullable: a per-row
# extract failure carries a NULL descriptor that projects to a NULL action group,
# keeping one output row per input row.
DESCRIPTOR_ROW: pa.Schema = pa.schema(
    [
        pa.field("action_data_uri", pa.string()),
        pa.field("descriptor", pa.list_(pa.float32())),
    ]
)

# The canonical typed source contract projected from the clips CLIP_SCHEMA.
# clip_id is the join key every leg's write needs; task/subtask feed text;
# clip_uri feeds image; action_data_uri feeds action; source_dataset remains a
# required upstream/base-contract field even though no current embedding worker
# projects or consumes it. task_name / subtask_name / source_dataset are non-null
# because the design doc's no-drop argument depends on them. clip_uri /
# action_data_uri are large_string in the clips table. This
# schema is the single source of truth for what embed may read: EMBED_SOURCE_COLUMNS
# is derived from its field names, so the two cannot drift.
EMBED_SOURCE_ROW: pa.Schema = pa.schema(
    [
        _KEY_FIELD,
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("subtask_name", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string()),
        pa.field("action_data_uri", pa.large_string()),
        pa.field("source_dataset", pa.string(), nullable=False),
    ]
)

# Derived from EMBED_SOURCE_ROW's field names (not free-form) so the tuple and the
# schema cannot diverge; a column added to the schema is projected automatically.
EMBED_SOURCE_COLUMNS: tuple[str, ...] = tuple(EMBED_SOURCE_ROW.names)


# A 2-D input to _as_matrix must already be exactly (rows, dim); a 1-D input is a
# single vector reshaped by element count.
_MATRIX_NDIM = 2


def _as_matrix(vectors: npt.NDArray[np.floating], rows: int, dim: int) -> npt.NDArray[np.float32]:
    """Coerce ``vectors`` to a contiguous ``(rows, dim)`` float32 matrix.

    Checks rank AND shape, not merely element count: a transposed ``(dim, rows)``
    matrix or a ``(rows, 1, dim)`` stack has the right value count and would
    reshape without complaint, silently pairing every row with the wrong vector.

    Raises:
        ValueError: If the input is neither a ``(rows, dim)`` matrix nor a flat
            ``dim``-value vector for a single row.

    """
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim == _MATRIX_NDIM:
        if matrix.shape != (rows, dim):
            msg = f"expected embedding matrix of shape ({rows}, {dim}), got {matrix.shape}"
            raise ValueError(msg)
    elif matrix.ndim != 1:
        msg = f"expected a ({rows}, {dim}) matrix or a flat {dim}-value vector, got shape {matrix.shape}"
        raise ValueError(msg)
    elif rows != 1:
        msg = f"a flat vector is only accepted for a single row, got {rows} rows and shape {matrix.shape}"
        raise ValueError(msg)
    if matrix.size != rows * dim:
        msg = f"expected {rows}x{dim} embedding matrix, got {matrix.size} value(s)"
        raise ValueError(msg)
    return matrix.reshape(rows, dim)


def _checked_mask(valid: npt.NDArray[np.bool_] | None, rows: int) -> npt.NDArray[np.bool_] | None:
    """Return ``valid`` after checking it is a per-row mask of length ``rows``.

    Without a key column the NULL mask is what tells the storage layer which rows
    stayed pending, so a mask of the wrong length would misalign every NULL by an
    unknown offset - silently marking the wrong clips complete. pyarrow would also
    reject it, but with a message that names neither the row count nor the caller.

    Raises:
        ValueError: If ``valid`` is not a 1-D mask of exactly ``rows`` entries.

    """
    if valid is None:
        return None
    if valid.shape != (rows,):
        msg = f"validity mask must be shape ({rows},) to align with the batch, got {valid.shape}"
        raise ValueError(msg)
    return valid


def _fixed_size_list_array(matrix: npt.NDArray[np.float32]) -> pa.Array:
    """Build a ``fixed_size_list<float32, dim>`` column from an ``(n, d)`` matrix.

    ``FixedSizeListArray.from_arrays`` over the flattened contiguous buffer needs
    no offsets (the width is fixed) and boxes no Python lists. A zero-row matrix
    yields a valid length-0 array (``0 % dim == 0``).
    """
    _rows, dim = matrix.shape
    values = pa.array(matrix.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, dim)


def _masked_fixed_size_list_array(matrix: npt.NDArray[np.float32], valid: npt.NDArray[np.bool_] | None) -> pa.Array:
    """Build a nullable ``fixed_size_list<float32, dim>`` column with per-row NULLs.

    A row marked invalid in ``valid`` becomes a NULL list (not a zero vector), so
    a per-row failure persists as the "pending / retryable" NULL state, keeping the
    fill cardinality-preserving. ``valid is None`` means every row is valid (the
    text leg, which never fails per row). Values for invalid rows are still read
    from ``matrix`` (typically zeros) but masked out, so ``matrix`` must always be
    the full ``(rows, dim)`` shape aligned to ``valid``.
    """
    rows, dim = matrix.shape
    mask = _checked_mask(valid, rows)
    if mask is None or bool(mask.all()):
        return _fixed_size_list_array(matrix)
    values = pa.array(matrix.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, dim, mask=pa.array(np.logical_not(mask)))


def _masked_float32_list_array(matrix: npt.NDArray[np.float32], valid: npt.NDArray[np.bool_] | None) -> pa.ListArray:
    """Build a nullable variable ``list<float32>`` column with per-row NULLs.

    The descriptor-transport counterpart of ``_masked_fixed_size_list_array``: a
    row marked invalid becomes a NULL list so a per-row extract failure survives to
    the projector as a NULL descriptor (which projects to a NULL action group),
    preserving one output row per input row. All rows share the fixed ``matrix``
    stride; invalid-row values are present in the buffer but masked out.
    """
    rows, dim = matrix.shape
    mask = _checked_mask(valid, rows)
    values = pa.array(matrix.reshape(-1), type=pa.float32())
    # arange(0, rows*dim + 1, dim) always yields at least [0] - the single offset a
    # valid empty ListArray requires - so zero rows needs no special case. dim is
    # always positive here (fixed descriptor width); a zero dim would fail loudly.
    offsets = pa.array(np.arange(0, rows * dim + 1, dim, dtype=np.int32))
    if mask is None or bool(mask.all()):
        return pa.ListArray.from_arrays(offsets, values)
    return pa.ListArray.from_arrays(offsets, values, mask=pa.array(np.logical_not(mask)))


def _provenance_string_array(value: str, valid: npt.NDArray[np.bool_] | None, rows: int) -> pa.Array:
    """Build a ``string`` provenance column: ``value`` on valid rows, NULL on invalid ones.

    The provenance identity (model result fingerprint, descriptor version, PCA
    fingerprint) is the same for every successful row of a run, so it is a scalar
    broadcast to the valid rows and NULL elsewhere - a failed row's whole group
    (vector + provenance) stays NULL together. ``valid is None`` broadcasts
    ``value`` to all ``rows``.
    """
    if valid is None:
        return pa.array([value] * rows, type=pa.string())
    return pa.array([value if bool(flag) else None for flag in valid], type=pa.string())


def text_columns_batch(
    rows: int,
    subtask_vectors: npt.NDArray[np.floating],
    task_vectors: npt.NDArray[np.floating],
    result_fingerprint: str,
) -> pa.Table:
    """Build the text group batch: both vectors + the result fingerprint, one row per input row.

    Carries no key: the storage layer attaches ``clip_id`` positionally, so the
    output must stay in input order and hold exactly ``rows`` rows. The text leg
    never fails per row (task / subtask are non-null on the source contract and an
    empty instruction embeds to a valid content-free vector), so every row is
    complete and there is no validity mask. Both matrices are ``(rows, TEXT_DIM)``.
    The columns are exactly ``TEXT_COLUMN_GROUP``'s fields, so the storage layer's
    schema cast accepts the batch unchanged.

    Args:
        rows: Input row count the output must match.
        subtask_vectors: ``(rows, TEXT_DIM)`` subtask embeddings.
        task_vectors: ``(rows, TEXT_DIM)`` task embeddings.
        result_fingerprint: Identity of the computation that produced these
            vectors, stored in ``embedding_text_model_id``. It identifies the whole
            computation, not merely the checkpoint, so two runs differing only in
            how the text was formatted store different values.

    Returns:
        A table of exactly ``TEXT_COLUMN_GROUP``'s fields.

    Raises:
        ValueError: If either matrix is not ``(rows, TEXT_DIM)``.

    """
    subtask = _as_matrix(subtask_vectors, rows, TEXT_DIM)
    task = _as_matrix(task_vectors, rows, TEXT_DIM)
    return pa.table(
        {
            "embedding_text_subtask": _masked_fixed_size_list_array(subtask, None),
            "embedding_text_task": _masked_fixed_size_list_array(task, None),
            "embedding_text_model_id": _provenance_string_array(result_fingerprint, None, rows),
        }
    )


def image_columns_batch(
    rows: int,
    vectors: npt.NDArray[np.floating],
    result_fingerprint: str,
    valid: npt.NDArray[np.bool_],
) -> pa.Table:
    """Build the image group batch: one vector + the result fingerprint, per-row nullable.

    Carries no key: the storage layer attaches ``clip_id`` positionally, so the
    output holds exactly ``rows`` rows in input order. ``valid[i]`` is False for a
    clip whose media was missing or undecodable; that row's vector and fingerprint
    are both NULL, so its group stays pending / retryable and the batch still
    carries exactly one row per input row. ``vectors`` is the full
    ``(rows, IMAGE_DIM)`` matrix aligned to ``valid`` (failed rows may hold
    placeholder zeros; they are masked out). The columns are exactly
    ``IMAGE_COLUMN_GROUP``'s fields.

    Args:
        rows: Input row count the output must match.
        vectors: ``(rows, IMAGE_DIM)`` embeddings aligned to ``valid``.
        result_fingerprint: Identity of the computation that produced these
            vectors, stored in ``embedding_image_model_id``. It identifies the
            whole computation, not merely the checkpoint, so two runs differing
            only in pooling store different values.
        valid: Per-row validity mask; False yields an all-NULL group row.

    Returns:
        A table of exactly ``IMAGE_COLUMN_GROUP``'s fields.

    Raises:
        ValueError: If ``vectors`` is not ``(rows, IMAGE_DIM)`` or ``valid`` is not
            a length-``rows`` mask.

    """
    matrix = _as_matrix(vectors, rows, IMAGE_DIM)
    return pa.table(
        {
            "embedding_image": _masked_fixed_size_list_array(matrix, valid),
            "embedding_image_model_id": _provenance_string_array(result_fingerprint, valid, rows),
        }
    )


def action_columns_batch(
    rows: int,
    vectors: npt.NDArray[np.floating],
    descriptor_version: str,
    pca_fingerprint: str,
    valid: npt.NDArray[np.bool_],
) -> pa.Table:
    """Build the action group batch: one raw-PCA vector + its provenance, per-row nullable.

    Carries no key: the storage layer attaches ``clip_id`` positionally, so the
    output holds exactly ``rows`` rows in input order. ``valid[i]`` is False for a
    row whose artifact failed to decode or was geometrically rejected, so its
    vector, descriptor version, and PCA fingerprint are all NULL together (the
    group stays pending / retryable). Unlike text / image the vector is raw PCA
    coordinates, NOT unit-norm (magnitude encodes gesture size).
    ``descriptor_version`` fingerprints the descriptor semantics;
    ``pca_fingerprint`` identifies the exact basis. The matrix is the full
    ``(rows, ACTION_DIM)`` aligned to ``valid``; the columns are exactly
    ``ACTION_COLUMN_GROUP``'s fields.

    Raises:
        ValueError: If ``vectors`` is not ``(rows, ACTION_DIM)`` or ``valid`` is
            not a length-``rows`` mask.

    """
    matrix = _as_matrix(vectors, rows, ACTION_DIM)
    return pa.table(
        {
            "embedding_action": _masked_fixed_size_list_array(matrix, valid),
            "embedding_action_descriptor_version": _provenance_string_array(descriptor_version, valid, rows),
            "embedding_action_pca_fingerprint": _provenance_string_array(pca_fingerprint, valid, rows),
        }
    )


def descriptor_batch(
    action_uris: Sequence[str | None],
    descriptors: npt.NDArray[np.floating],
    descriptor_dim: int,
    valid: npt.NDArray[np.bool_],
) -> pa.Table:
    """Build the internal descriptor batch (``DESCRIPTOR_ROW``) handed to the projector.

    Carries no key: rows stay in input order so the projected action group aligns
    positionally with the rows the worker scanned. ``valid[i]`` is False for a row
    whose artifact failed to decode or was geometrically rejected; its
    ``descriptor`` list is NULL (the projector then emits a NULL action group for
    it), keeping one descriptor row per input row.

    Args:
        action_uris: The ``action_data_uri`` each row came from (``None`` for a row
            that carried no artifact, preserved rather than collapsed to ``""``); its
            length defines the batch's row count. The PCA fit sample de-duplicates on
            this so a multi-view span contributes its descriptor once, not once per view.
        descriptors: ``(len(action_uris), descriptor_dim)`` raw descriptors;
            invalid-row values are masked out.
        descriptor_dim: Descriptor width.
        valid: Per-row validity mask (False -> NULL descriptor).

    Returns:
        A table matching ``DESCRIPTOR_ROW``.

    Raises:
        ValueError: If ``descriptors`` or ``valid`` does not align with
            ``action_uris``.

    """
    rows = len(action_uris)
    matrix = _as_matrix(descriptors, rows, descriptor_dim)
    return pa.table(
        {
            "action_data_uri": pa.array(list(action_uris), type=pa.string()),
            "descriptor": _masked_float32_list_array(matrix, valid),
        }
    )
