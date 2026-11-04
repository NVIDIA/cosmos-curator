# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe bindings for the two-phase caption state machine."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import lance
import pyarrow as pa
from lance.fragment import LanceFragment

from cosmos_curator.next.recipes.video_caption.contracts import CaptionInferenceBindings
from cosmos_curator.next.recipes.video_caption.inference import ContextStage, MediaColumns, TerminalRowBuilder

BatchProjector = Callable[[pa.Table], pa.Table]
BaseSchemaValidator = Callable[[lance.LanceDataset], None]
TerminalRowValidator = Callable[[dict[str, Any]], None]
TerminalValueValidator = Callable[[object, dict[str, Any] | None], None]
TerminalBatchValidator = Callable[[pa.Table], None]
StagedRowReconciler = Callable[[pa.Table, pa.Table, int, int], pa.Table]
ResultValueFingerprint = Callable[[pa.Table], str]
CompleteFragmentValidator = Callable[[LanceFragment], None]


@dataclass(frozen=True)
class CaptionInputBindings:
    """One recipe's physical read, projected media, and staged identity columns.

    ``skip_null_ids`` excludes upstream source outcomes from media processing
    and terminal validation while retaining their physical row offsets.
    """

    source_columns: tuple[str, ...]
    source_schema: pa.Schema
    mapped_schema: pa.Schema
    project_batch: BatchProjector
    row_id_column: str
    media_uri_column: str
    media_size_column: str
    media_sha256_column: str | None
    staged_binding_columns: tuple[str, ...]
    skip_null_ids: bool = False

    @property
    def media(self) -> MediaColumns:
        """Return the projected columns naming each row's media for the shared caption plan."""
        return MediaColumns(uri=self.media_uri_column, size=self.media_size_column, sha256=self.media_sha256_column)


@dataclass(frozen=True)
class CaptionResultBindings:
    """One recipe's owned fields and terminal result contract."""

    caption_field_name: str
    metadata_field_name: str
    fields: pa.Schema
    staged_schema: pa.Schema
    digest: str
    build_terminal_row: TerminalRowBuilder
    validate_terminal_value: TerminalValueValidator
    validate_staged_row: TerminalRowValidator
    carried_result_columns: tuple[str, ...] = ()
    validate_terminal_batch: TerminalBatchValidator | None = None
    reconcile_staged_rows: StagedRowReconciler | None = None
    fingerprint_result_values: ResultValueFingerprint | None = None
    fragment_has_pending: Callable[[LanceFragment], bool] | None = None
    staged_storage_schema: pa.Schema | None = None
    prepare_staged_rows: Callable[[pa.Table, LanceFragment], pa.Table] | None = None


@dataclass(frozen=True)
class CaptionPublicationBindings:
    """One recipe's base-table protection and static transaction identity."""

    validate_base_schema: BaseSchemaValidator
    protected_field_names: tuple[str, ...]
    base_field_owner: str
    transaction_properties: Mapping[str, str]
    validate_complete_fragment: CompleteFragmentValidator | None = None


@dataclass(frozen=True)
class CaptionRunBindings:
    """Shared input, result, and publication bindings for caption and embedding recipes.

    ``prompt_template``, ``inference``, and ``context_stage`` are consumed only by
    caption execution. Embedding recipes use an empty prompt and leave the
    caption inference defaults unused.
    """

    input: CaptionInputBindings
    result: CaptionResultBindings
    publication: CaptionPublicationBindings
    prompt_template: str
    inference: CaptionInferenceBindings = field(default_factory=CaptionInferenceBindings)
    context_stage: ContextStage | None = None
