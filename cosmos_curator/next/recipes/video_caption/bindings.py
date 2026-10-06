# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private recipe bindings for the complete-MP4 caption state machine."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import lance
import pyarrow as pa
from lance.fragment import LanceFragment

BatchProjector = Callable[[pa.Table], pa.Table]
BaseSchemaValidator = Callable[[lance.LanceDataset], None]
TerminalRowBuilder = Callable[..., dict[str, Any]]
TerminalRowValidator = Callable[[dict[str, Any]], None]
TerminalValueValidator = Callable[[object, dict[str, Any] | None], None]
TerminalBatchValidator = Callable[[pa.Table], None]
StagedRowReconciler = Callable[[pa.Table, pa.Table, int, int], pa.Table]
ResultValueFingerprint = Callable[[pa.Table], str]
CompleteFragmentValidator = Callable[[LanceFragment], None]


class ContextStage(Protocol):
    """Post-fetch context enrichment used by the shared caption engine."""

    def __call__(
        self,
        batch: pa.Table,
        *,
        storage_profile: str,
        attempts: int,
    ) -> pa.Table:
        """Append transient prompt context and declared small result values."""
        ...


@dataclass(frozen=True)
class CaptionInputBindings:
    """One recipe's physical read, projected media, and staged identity columns."""

    source_columns: tuple[str, ...]
    source_schema: pa.Schema
    mapped_schema: pa.Schema
    project_batch: BatchProjector
    row_id_column: str
    media_uri_column: str
    media_size_column: str
    media_sha256_column: str | None
    staged_binding_columns: tuple[str, ...]


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
class CaptionInferenceBindings:
    """Model-specific settings consumed by the shared Ray vLLM processor.

    Public recipes select one of these static bindings. They are deliberately
    not user configuration: per-run execution sizing belongs in
    ``CaptionExecutionConfig`` and model behavior belongs in the result
    contract.
    """

    assistant_prefill: str | None = None
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    video_fps: int | float = 2
    prepare_model_config_kwargs: Mapping[str, Any] = field(default_factory=dict)
    request_mm_processor_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"do_sample_frames": False})
    sampling_params: Mapping[str, float | int] | None = None
    engine_kwargs: Mapping[str, Any] = field(default_factory=dict)


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
