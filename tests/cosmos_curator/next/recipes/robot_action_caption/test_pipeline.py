# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What ``run_config`` does BEFORE it acquires anything, and what it declines to do.

Five orderings in the driver each prevent a specific failure, and none is
observable from a full attempt - a run that acquires a GPU and then discovers it
owes nothing still finishes correctly, and a run that widens somebody's table
before refusing it still refuses it. So all four are pinned here by proving the
expensive steps were never reached::

    require_source_columns   refuse a bad table BEFORE mutating it
          |
          v
    require_selection_column refuse a missing default scope BEFORE mutating it
          |
          v
    ensure_caption_fields    widen (metadata-only); validates context on
          |                  each reopen before committing
          v
    resolve_context_fields   optional columns for the digest, from the same
          |                  handle whose version is pinned next
          v
    read_version = version   pin AFTER the widening
          |
          v
    plan -> no fragments?    return, having touched no model validation,
          |                  workspace, Ray cluster, or initialization
          v
    workspace, Ray, Phase A, Phase B

Each expensive step is replaced by a callable that FAILS if invoked. Asserting a
recorded call count would pass a run that reached the step and then decided not
to use it, which is a different property from never reaching it.

The one acquisition that is NOT stubbed out is model-directory validation, whose
own test proves it runs before the workspace and Ray - the cheapest refusal a
misconfigured deployment can get.

One group looks the other way, at the post-commit cleanup. It is the only step
whose failure must NOT change the outcome - the commit is the completion signal -
so it is pinned here beside the refusals: same driver work, opposite verdict.

No Ray cluster or model initialization is needed here, which is why these tests
run in a second rather than a minute. The end-to-end behaviour they bound lives
in ``test_recovery_integration.py``.
"""

import pathlib
from collections.abc import Sequence
from typing import Any, Never

import lance
import pyarrow as pa
import pyarrow.fs as pafs
import pytest

from cosmos_curator.core.utils import environment
from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.next.recipes.robot_action_caption import pipeline
from cosmos_curator.next.recipes.robot_action_caption.config import ResolvedRobotActionCaptionConfig
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMN,
    CAPTION_COLUMNS,
    CAPTION_DIGEST_COLUMN,
    CAPTION_MODEL_COLUMN,
    CAPTION_STATUS_COLUMN,
    CaptionStatus,
    caption_contract_digest,
)
from cosmos_curator.next.recipes.robot_action_caption.pipeline import run_config
from cosmos_curator.next.recipes.robot_action_caption.publication import PublicationResult
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CLIP_ID_COLUMN,
    CLIP_URI_COLUMN,
    CURATE_SELECTED_VALUE,
    CURATE_SELECTION_REASON,
    ensure_caption_fields,
    resolve_context_fields,
)
from cosmos_curator.next.recipes.robot_action_caption.workspace import CaptionWorkspace
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION
from tests.cosmos_curator.next.recipes.robot_action_caption.caption_harness import CaptionConfig

CLIPS = 4

# A supported context column, written at a type the prompt cannot render. Any
# non-text type serves; the point is that the column is PRESENT, which is what
# separates a source-contract fault from the supported case of an absent column.
_CONTEXT_COLUMN = "task_name"

# How a storage backend's refusal to delete the published namespace reads, so the
# warning can be told apart from any other line the attempt emits.
_CLEANUP_FAULT = "the staging bucket rejected the delete"

# The publication outcome the cleanup tests run behind, so every assertion there
# is about a wave that certainly committed. The version is arbitrary; what matters
# is that the counts reach the run's own result.
_COMMITTED_WAVE = PublicationResult(ending_version=99, published_rows=CLIPS, error_rows=0, truncated_rows=0)

# Every step a run must NOT reach once it knows it owes no caption. Named by the
# attribute ``pipeline`` calls, because that is the seam a test can close.
_ACQUISITIONS = (
    "validate_model_directory",
    "resolve_workspace",
    "ensure_ray_initialized",
    "run_inference_phase",
    "publish_captions",
)

# The steps that must not be reached once model weights are missing. Validation
# itself is absent from this tuple: it is the step under test.
_POST_VALIDATION_ACQUISITIONS = (
    "resolve_workspace",
    "ensure_ray_initialized",
    "run_inference_phase",
    "publish_captions",
)


def _refuse(name: str) -> object:
    """Return a stand-in for one acquiring step that fails the test if invoked.

    Failing rather than recording is deliberate. A counter asserted to be zero
    also passes when the step ran and returned nothing useful; raising makes
    "was this reached" the only question, and answers it at the call site.
    """

    def acquire(*_args: object, **_kwargs: object) -> Never:
        msg = f"{name} was reached by a run that should have refused first"
        raise AssertionError(msg)

    return acquire


@pytest.fixture
def refuse_every_acquisition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace each acquiring step with a callable that fails the test if reached."""
    for name in _ACQUISITIONS:
        monkeypatch.setattr(pipeline, name, _refuse(name))


@pytest.fixture
def refuse_acquisitions_after_model_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave model-directory validation live and close every step behind it."""
    for name in _POST_VALIDATION_ACQUISITIONS:
        monkeypatch.setattr(pipeline, name, _refuse(name))


@pytest.fixture
def unstaged_weights(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the weight cache root at an empty directory this test owns.

    Test-owned rather than relying on the real container path being absent: on a
    developer machine that path may exist, and a test whose premise is "the
    weights are missing" must not depend on the host for it.
    """
    root = tmp_path / "no-model-weights"
    root.mkdir()
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", root)


def test_a_no_work_run_needs_no_model_validation_workspace_or_ray(
    fully_captioned_clips: ResolvedRobotActionCaptionConfig,
    refuse_every_acquisition: None,
) -> None:
    """A table that owes nothing is answered from the manifest alone.

    This is the case a monitoring loop re-runs constantly, so it must cost
    nothing: no model-directory validation, workspace directory, Ray cluster,
    or model initialization. It is also the case that must not need a GPU node
    to be scheduled at all.
    """
    del refuse_every_acquisition
    uri = fully_captioned_clips.input.clips_lance_uri

    result = run_config(fully_captioned_clips)

    assert result.published_rows == 0
    assert result.ending_version == result.source_version == int(lance.dataset(uri).version)


def test_a_no_work_run_still_reports_where_the_table_is_canonical(
    fully_captioned_clips: ResolvedRobotActionCaptionConfig,
    refuse_every_acquisition: None,
) -> None:
    """``ending_version`` names a real version even when no commit happened.

    Reporting null would make a caller unable to distinguish "nowhere" from "here
    already", and the pinned version is already correct: it includes the caption
    column widening, which is itself a commit.
    """
    del refuse_every_acquisition
    uri = fully_captioned_clips.input.clips_lance_uri

    result = run_config(fully_captioned_clips)

    assert lance.dataset(uri, version=result.ending_version).count_rows() == CLIPS


def test_a_no_work_run_reports_owing_nothing_rather_than_a_row_count(
    fully_captioned_clips: ResolvedRobotActionCaptionConfig,
    refuse_every_acquisition: None,
) -> None:
    """Owing nothing is its own outcome, so the line says so instead of counting zeros.

    The version is still named, because the table is canonical somewhere either
    way; only the row count cannot discriminate.
    """
    del refuse_every_acquisition

    result = run_config(fully_captioned_clips)

    assert result.as_message() == f"robot-action-caption: no caption owed at v{result.ending_version}"


def test_selected_only_no_work_warns_when_unselected_rows_are_pending(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
    refuse_every_acquisition: None,
    loguru_records: list[dict[str, Any]],
) -> None:
    """Name the hidden pending population and the two operator remedies.

    The count is asserted with the phrase that scopes it, because a bare row
    count would also pass against a table-wide total - the number a sharded job
    must not print, since rows another shard owns are not this job's to explain.
    """
    del refuse_every_acquisition
    config = caption_config(_write_clips(tmp_path, selection="duplicate"))

    result = run_config(config)

    assert result.published_rows == 0
    (warning,) = _warnings(loguru_records)
    assert f"{CLIPS} row(s) in the fragments this job owns are caption-pending" in warning
    assert "Run the curate leg first" in warning
    assert "input.selected_only=false" in warning


def test_malformed_current_rows_cannot_report_no_work(
    malformed_current_clips: ResolvedRobotActionCaptionConfig,
    unstaged_weights: None,
    refuse_acquisitions_after_model_validation: None,
) -> None:
    """A current digest with an incomplete sibling group remains owed."""
    del unstaged_weights, refuse_acquisitions_after_model_validation

    with pytest.raises(FileNotFoundError, match="not staged at"):
        run_config(malformed_current_clips)


def test_an_invalid_source_fails_before_the_schema_commit(
    clips_without_a_uri_column: str,
    caption_config: CaptionConfig,
) -> None:
    """A table the recipe cannot caption is refused without having been mutated.

    Validation and widening are both cheap, so the ordering is easy to get wrong
    and invisible when it is: the run fails either way. What differs is whether
    somebody else's table is left carrying five columns from a run that never
    captioned a single row.
    """
    before = int(lance.dataset(clips_without_a_uri_column).version)

    with pytest.raises(ValueError, match="missing required source column"):
        run_config(caption_config(clips_without_a_uri_column))

    after = lance.dataset(clips_without_a_uri_column)
    assert int(after.version) == before
    assert not set(CAPTION_COLUMNS) & set(after.schema.names)


def test_selected_only_requires_the_curation_column_before_schema_widening(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
) -> None:
    """Refuse an unavailable default scope without mutating a pre-curation table."""
    uri = _write_clips(tmp_path, omit_selection=True)
    before = int(lance.dataset(uri).version)

    with pytest.raises(ValueError, match=r"selected-only captioning requires column.*selected_only=false"):
        run_config(caption_config(uri))

    after = lance.dataset(uri)
    assert int(after.version) == before
    assert not set(CAPTION_COLUMNS) & set(after.schema.names)


def test_disabling_selection_accepts_a_pre_curation_table(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
    unstaged_weights: None,
    refuse_acquisitions_after_model_validation: None,
) -> None:
    """Reach model validation when the operator explicitly requests all pending rows."""
    del unstaged_weights, refuse_acquisitions_after_model_validation
    uri = _write_clips(tmp_path, omit_selection=True)

    with pytest.raises(FileNotFoundError, match="not staged at"):
        run_config(caption_config(uri, selected_only=False))


def test_a_wrong_typed_context_column_fails_before_the_schema_commit(
    clips_with_a_non_text_context_column: str,
    caption_config: CaptionConfig,
) -> None:
    """A present context column at the wrong type is refused with the table untouched.

    The structural columns are all valid here, so only the context gate can
    produce this refusal - and it has to produce it before the widening. A run
    refused afterwards would leave five caption fields on a table whose prompt
    contract the recipe just rejected, and the next reader cannot tell that group
    apart from one a real attempt registered.
    """
    before = int(lance.dataset(clips_with_a_non_text_context_column).version)

    with pytest.raises(ValueError, match="has non-text type"):
        run_config(caption_config(clips_with_a_non_text_context_column))

    after = lance.dataset(clips_with_a_non_text_context_column)
    assert int(after.version) == before
    assert not set(CAPTION_COLUMNS) & set(after.schema.names)


def test_unstaged_model_weights_are_refused_before_the_workspace_or_ray(
    pending_clips: str,
    caption_config: CaptionConfig,
    unstaged_weights: None,
    refuse_acquisitions_after_model_validation: None,
) -> None:
    """A deployment missing its weights is told so without claiming anything.

    Every clip is pending here, so the run genuinely intends to caption: it is
    past the no-op return and would otherwise resolve a workspace and start Ray.
    Validating the weights first is what turns a misconfigured node into an
    immediate, self-describing refusal rather than a Ray cluster that comes up
    and then fails inside an engine actor.
    """
    del unstaged_weights, refuse_acquisitions_after_model_validation

    with pytest.raises(FileNotFoundError, match="not staged at"):
        run_config(caption_config(pending_clips))


def test_an_absent_clip_table_raises_lances_native_error(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
) -> None:
    """A missing table keeps Lance's native exception type and diagnostic."""
    uri = str(tmp_path / "absent.lance")

    with pytest.raises(ValueError) as raised:  # noqa: PT011 - native Lance wording is intentionally not a contract
        run_config(caption_config(uri))

    assert type(raised.value) is ValueError


def test_a_permission_error_opening_the_clip_table_propagates_unchanged(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An access denial remains distinguishable from a mistyped table path."""
    failure = PermissionError("clip-table credentials were rejected")

    def refuse_access(*_args: object, **_kwargs: object) -> Never:
        raise failure

    monkeypatch.setattr(pipeline.lance, "dataset", refuse_access)

    with pytest.raises(PermissionError) as raised:
        run_config(caption_config(str(tmp_path / "clips.lance")))

    assert raised.value is failure


def test_a_credential_transport_error_opening_the_clip_table_propagates_unchanged(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untyped credential-provider failure is not relabeled as a missing table."""
    failure = OSError("credential provider is unavailable")

    def fail_credentials(*_args: object, **_kwargs: object) -> Never:
        raise failure

    monkeypatch.setattr(pipeline.lance, "dataset", fail_credentials)

    with pytest.raises(OSError, match="credential provider is unavailable") as raised:
        run_config(caption_config(str(tmp_path / "clips.lance")))

    assert raised.value is failure


def test_an_unrelated_lance_value_error_propagates_unchanged(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Lance validation fault remains available to the caller for diagnosis."""
    failure = ValueError("read parameters are internally inconsistent")

    def reject_arguments(*_args: object, **_kwargs: object) -> Never:
        raise failure

    monkeypatch.setattr(pipeline.lance, "dataset", reject_arguments)

    with pytest.raises(ValueError, match="read parameters are internally inconsistent") as raised:
        run_config(caption_config(str(tmp_path / "clips.lance")))

    assert raised.value is failure


def test_pending_attempt_propagates_storage_compatibility_for_every_worker_location(
    pending_clips: str,
    caption_config: CaptionConfig,
    staged_weights: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray workers receive the AWS client settings selected by every location they open.

    The clip table is configured independently of the staging root and a worker
    scans that table's fragments itself, so a selection made from the staging
    root alone would leave the scan running under the SDK default.
    """
    del staged_weights
    workspace = CaptionWorkspace(
        root_uri="s3://caption-checkpoints/contract",
        filesystem=pafs.LocalFileSystem(),
    )
    config = caption_config(pending_clips)
    compatibility_calls: list[tuple[tuple[str, ...], str]] = []
    ray_init_calls: list[dict[str, object]] = []

    def classify_endpoints(*locations: str, storage_profile: str) -> dict[str, str]:
        compatibility_calls.append((locations, storage_profile))
        return {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"}

    monkeypatch.setattr(pipeline, "resolve_workspace", lambda **_kwargs: workspace)
    monkeypatch.setattr(pipeline, "s3_compatibility_env_vars", classify_endpoints)
    monkeypatch.setattr(pipeline, "curator_io_resources", lambda: {"curator_io": 13.0})
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **kwargs: ray_init_calls.append(kwargs))
    monkeypatch.setattr(pipeline, "run_inference_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "publish_captions", lambda *_args, **_kwargs: _COMMITTED_WAVE)
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda _workspace: None)

    run_config(config)

    assert compatibility_calls == [
        ((config.input.clips_lance_uri, workspace.root_uri), config.execution.storage_profile)
    ]
    assert ray_init_calls == [
        {
            "local_resources": {"curator_io": 13.0},
            "runtime_env_vars": {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"},
        }
    ]


def test_a_cleanup_failure_after_the_commit_is_reported_as_one_warning(
    pending_clips: str,
    caption_config: CaptionConfig,
    staged_weights: pathlib.Path,
    a_committed_attempt: None,
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """Not failing the run is not the same as saying nothing about it.

    The leftover namespace is only reclaimable by hand, so the line has to carry
    both what happened and where it happened; a silent swallow would leave the
    directory with no record that it exists. That the run still reports its
    committed wave is pinned end-to-end by
    ``test_recovery_integration.test_a_cleanup_failure_still_reports_the_canonical_commit``,
    and a regression to raising fails here too, since the call would not return.
    """
    del staged_weights, a_committed_attempt
    monkeypatch.setattr(pipeline, "cleanup_workspace", _refuse_cleanup)
    config = caption_config(pending_clips)

    run_config(config)

    warned = _warnings(loguru_records)
    assert len(warned) == 1
    assert config.staging_root in warned[0]
    assert _CLEANUP_FAULT in warned[0]


def test_a_successful_cleanup_warns_about_nothing(
    pending_clips: str,
    caption_config: CaptionConfig,
    staged_weights: pathlib.Path,
    a_committed_attempt: None,
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The counterfactual, without which the warning above is not attributable.

    A cleanup that warned on every run would satisfy the test above while telling
    an operator nothing, so the ordinary path has to be silent.
    """
    del staged_weights, a_committed_attempt
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda _workspace: None)

    run_config(caption_config(pending_clips))

    assert _warnings(loguru_records) == []


@pytest.fixture
def a_committed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Carry a run from its plan to its cleanup without a cluster, a GPU, or a staged row.

    Every step between is replaced, because what the cleanup tests bound is the
    step AFTER the commit and reaching it for real costs an engine pool.
    Publication returns a wave rather than raising, since the cleanup runs only
    once the captions are canonical.
    """
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "run_inference_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "publish_captions", lambda *_args, **_kwargs: _COMMITTED_WAVE)


def _refuse_cleanup(_workspace: CaptionWorkspace) -> Never:
    """Fail the way a storage backend does: after the commit, with nothing to retry."""
    raise OSError(_CLEANUP_FAULT)


def _warnings(records: Sequence[dict[str, Any]]) -> list[str]:
    """Return the WARNING messages captured so far."""
    return [record["message"] for record in records if record["level"].name == "WARNING"]


@pytest.fixture
def pending_clips(tmp_path: pathlib.Path) -> str:
    """Return a table whose every row is owed a caption."""
    return _write_clips(tmp_path)


@pytest.fixture
def fully_captioned_clips(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
) -> ResolvedRobotActionCaptionConfig:
    """Return the config for a table whose every row is already captioned.

    Returns the config rather than the URI because the digest is derived from the
    config's own semantic blocks: the fixture must stamp rows under exactly the
    contract the run under test will compute, and building the config twice would
    leave two chances for them to diverge.

    The digest is derived through the same functions the pipeline uses rather
    than written as a literal, so the fixture stays correct when the contract
    changes - which is the whole point, since a stale literal here would silently
    turn this into a test of the PENDING path while still passing its own setup.
    """
    config = caption_config(_write_clips(tmp_path))
    dataset = ensure_caption_fields(config.input.clips_lance_uri, storage_options=None)
    model = resolve_registered_model(config.model.variant)
    digest = caption_contract_digest(
        model,
        resolve_context_fields(dataset),
        generation=config.generation,
        video=config.video,
        media=config.media,
    )
    # Every member that must be non-null for a successful terminal row is
    # populated; caption_error remains NULL, as the status requires.
    dataset.update(
        {
            CAPTION_COLUMN: "'a caption'",
            CAPTION_STATUS_COLUMN: f"'{CaptionStatus.SUCCESS}'",
            CAPTION_DIGEST_COLUMN: f"'{digest}'",
            CAPTION_MODEL_COLUMN: f"'{model.model_id}'",
        }
    )
    return config


@pytest.fixture
def malformed_current_clips(
    tmp_path: pathlib.Path,
    caption_config: CaptionConfig,
) -> ResolvedRobotActionCaptionConfig:
    """Return a current-digest table whose caption and model remain NULL."""
    config = caption_config(_write_clips(tmp_path))
    dataset = ensure_caption_fields(config.input.clips_lance_uri, storage_options=None)
    model = resolve_registered_model(config.model.variant)
    digest = caption_contract_digest(
        model,
        resolve_context_fields(dataset),
        generation=config.generation,
        video=config.video,
        media=config.media,
    )
    dataset.update(
        {
            CAPTION_STATUS_COLUMN: f"'{CaptionStatus.SUCCESS}'",
            CAPTION_DIGEST_COLUMN: f"'{digest}'",
        }
    )
    return config


@pytest.fixture
def clips_without_a_uri_column(tmp_path: pathlib.Path) -> str:
    """Return a table carrying clip ids but no media location, so no row is captionable."""
    return _write_clips(tmp_path, with_uri=False)


@pytest.fixture
def clips_with_a_non_text_context_column(tmp_path: pathlib.Path) -> str:
    """Return a valid table whose one supported context column holds integers."""
    return _write_clips(tmp_path, context_type=pa.int64())


def _write_clips(
    root: pathlib.Path,
    *,
    with_uri: bool = True,
    context_type: pa.DataType | None = None,
    selection: str | None = CURATE_SELECTED_VALUE,
    omit_selection: bool = False,
) -> str:
    """Write a minimal clip table and return its URI.

    No media objects are created: nothing in this module reaches the media stage,
    and a fixture that wrote clip bytes would imply otherwise.

    Args:
        root: Directory to hold the table.
        with_uri: Whether to write the structural ``clip_uri`` column.
        context_type: Arrow type for one supported context column, or ``None`` to
            omit context entirely. An absent context column is a supported source
            shape, so omitting it keeps every other fixture on the valid path.
        selection: Curation verdict written to every row, including ``None``.
        omit_selection: Omit the curation column for pre-curation table tests.

    Returns:
        The clip table URI.

    """
    fields = [pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False)]
    data: dict[str, pa.Array] = {CLIP_ID_COLUMN: pa.array([f"c{index}" for index in range(CLIPS)], pa.string())}
    if with_uri:
        fields.append(pa.field(CLIP_URI_COLUMN, pa.large_string(), nullable=False))
        data[CLIP_URI_COLUMN] = pa.array([f"{root}/media/c{index}.mp4" for index in range(CLIPS)], pa.large_string())
    if context_type is not None:
        fields.append(pa.field(_CONTEXT_COLUMN, context_type, nullable=True))
        data[_CONTEXT_COLUMN] = pa.array(range(CLIPS), context_type)
    if not omit_selection:
        fields.append(pa.field(CURATE_SELECTION_REASON, pa.string(), nullable=True))
        data[CURATE_SELECTION_REASON] = pa.array([selection] * CLIPS, pa.string())
    table = pa.table(data, schema=pa.schema(fields))
    uri = str(root / "clips.lance")
    lance.write_dataset(table, uri, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    return uri
