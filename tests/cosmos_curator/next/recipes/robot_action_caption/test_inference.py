# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase A's silent-failure mechanisms: checkpoint wiring, frame budget, stage requests, pool bounds.

The engine itself is not covered here and cannot be: ``build_vllm_transform``
calls ``ray.init()`` and downloads tokenizer files, so it needs a cluster and
staged weights. What IS covered is the decisions whose failure produces no error
message at all - a checkpoint keyed on the wrong column or written through a
second filesystem records nothing while the run appears to work, a frame budget
sent per request leaves the decode on a library default while the contract
digest still claims the configured frame count, a media stage that asks for the wrong
resources either serializes every read or waits for capacity that will never
come, and a replica ceiling derived from a startup GPU snapshot pins a growing
cluster to the fraction of itself it had at build time.

The stage-request tests read the plan Phase A actually builds: ``write_parquet``
is replaced by a recorder, so every operator is assembled with the arguments
production passes and then discarded unexecuted. The recorder also snapshots the
active checkpoint configuration, which exists only INSIDE the phase's context
manager.

``installed_checkpoint``'s own contract - refusing to replace an installed
config, verifying the one it installed became active, clearing on the way out,
and passing a filesystem through - belongs to the generic checkpoint helper and is
covered in ``tests/cosmos_curator/next/core/test_ray_checkpoints.py``. What is
covered here is the RECIPE's use of it: which column, which path, whose
filesystem.
"""

import math
import pathlib
from collections import Counter
from collections.abc import Iterator
from typing import Any

import attrs
import pyarrow as pa
import pyarrow.fs as pafs
import pytest
import ray
from ray.data.checkpoint import CheckpointConfig

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.models.registry import RegisteredModel
from cosmos_curator.next.recipes.robot_action_caption import inference
from cosmos_curator.next.recipes.robot_action_caption.config import (
    ResolvedRobotActionCaptionConfig,
    RobotActionCaptionExecutionConfig,
    RobotActionCaptionGenerationConfig,
    RobotActionCaptionVideoConfig,
)
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_DIGEST_COLUMN,
    CAPTION_MODEL_COLUMN,
    CAPTION_STATUS_COLUMN,
    DO_SAMPLE_FRAMES,
    REQUEST_CONTENT_TYPES,
    VIDEO_CONTENT_TYPE,
    CaptionStatus,
    caption_contract_digest,
)
from cosmos_curator.next.recipes.robot_action_caption.datasource import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    PROMPT_COLUMN,
    SCAN_SCHEMA,
    CaptionScanPlan,
    FragmentRead,
    plan_caption_scan,
)
from cosmos_curator.next.recipes.robot_action_caption.engine import ACTOR_RESTART_BOUNDS
from cosmos_curator.next.recipes.robot_action_caption.inference import (
    _FetchVersusEngineCapacity,
    _MediaFetchCapacity,
    _verify_media_resources,
    processor_config,
    run_inference_phase,
)
from cosmos_curator.next.recipes.robot_action_caption.media import MEDIA_ERROR_COLUMN, MEDIA_URL_COLUMN
from cosmos_curator.next.recipes.robot_action_caption.request import (
    GENERATED_TEXT_COLUMN,
    GENERATED_TOKENS_COLUMN,
    INFERENCE_ERROR_COLUMN,
    INPUT_TOKENS_COLUMN,
    build_request,
)
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CLIP_ID_COLUMN,
    CLIP_URI_COLUMN,
    caption_work_filter,
    ensure_caption_fields,
    resolve_context_fields,
)
from cosmos_curator.next.recipes.robot_action_caption.workspace import CaptionWorkspace
from tests.cosmos_curator.next.conftest import row_address
from tests.cosmos_curator.next.recipes.robot_action_caption.caption_harness import (
    CaptionConfig,
    ClipsTable,
)

# A test-owned model identity. The registry's contents are not the subject here,
# and resolving a real entry would couple every assertion below to whichever
# model the shipped registry pins today.
MODEL = RegisteredModel(key="model-under-test", model_id="org/model", revision="rev", precision="bfloat16")

# Distinct values ensure these tests observe configuration propagation rather
# than passing by coincidence against the shipped defaults.
GENERATION = RobotActionCaptionGenerationConfig(
    temperature=0.17,
    top_p=0.63,
    max_output_tokens=73,
    max_model_len=509,
)
# Every field carries a value no default in this recipe produces, for the reason
# stated above: a bound left at its default coincides with any literal a call site
# might hardcode instead of reading the config, so the propagation assertions
# would pass against precisely the bug they exist to catch. The cap sits above the
# declared per-frame floor and the frame bounds are whole temporal patches, as the
# section validates.
VIDEO = RobotActionCaptionVideoConfig(
    min_frames=8,
    max_frames=12,
    fps=4.25,
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28,
)

# Enough pending rows that the derived ceiling exceeds one replica whatever the
# module's inference batch size is, so no test here has to name that constant.
_MANY_PENDING_ROWS = 100_000

# A frame size no default in this recipe produces, so a reservation derived from
# some other number cannot coincide with the expected one.
_SOURCE_FRAME_PIXELS = 800 * 600

# A GPU count high enough that a snapshot-derived ceiling would be visibly
# different from a work-derived one.
_PLENTIFUL_GPUS = 64.0

# A cap no default in this recipe produces, so a stage built with some other
# number cannot coincide with it. Within the config's own 1 MiB..256 MiB bounds.
_CONFIGURED_CAP_BYTES = 3 * 1024 * 1024 + 7

# Likewise not the shipped attempt budget, so a stage built from the field's own
# default cannot pass the assertion by coincidence.
_CONFIGURED_MEDIA_ATTEMPTS = 7

# Distinct from both vLLM's inherited engine-context default and this recipe's
# configured context length, so propagation cannot pass by coincidence.
_CONFIGURED_SCHEDULER_BUDGET = 27_011

# Neither the field's own default nor the degenerate width of one, so a stage
# that ignored the setting cannot satisfy the coupling asserted below. Kept under
# the slot count the test cluster advertises, which is what the phase checks it
# against.
_CONFIGURED_FETCH_WORKERS = 3

# Distinct from both the scan default and the fixture fragment sizes, so the
# diagnostics must read the configured row target rather than infer either.
_FETCH_ROWS_PER_BLOCK = 13

# Every execution value differs from its field and downstream library defaults,
# so a propagation assertion cannot pass when the phase silently falls back.
EXECUTION = RobotActionCaptionExecutionConfig(
    storage_profile="default",
    inference_concurrency="auto",
    inference_batch_size=7,
    max_concurrent_batches=3,
    tensor_parallel_size=2,
    pipeline_parallel_size=3,
    max_num_batched_tokens=_CONFIGURED_SCHEDULER_BUDGET,
    media_attempts=_CONFIGURED_MEDIA_ATTEMPTS,
    media_fetch_workers=_CONFIGURED_FETCH_WORKERS,
    media_cpus_per_fetch=0.5,
    lance_read_batch_size=_FETCH_ROWS_PER_BLOCK,
    parquet_rows_per_file=17,
    source_frame_pixels=_SOURCE_FRAME_PIXELS,
)


def _fetched_row() -> dict[str, Any]:
    """Return one successful row under the media stage's declared schema."""
    return {
        CLIP_ID_COLUMN: "c0",
        FRAGMENT_ID_COLUMN: 0,
        CLIP_URI_COLUMN: "s3://bucket/c0.mp4",
        PROMPT_COLUMN: "describe",
        CHECKPOINT_ID_COLUMN: row_address(0, 0),
        MEDIA_URL_COLUMN: "data:video/mp4;base64,AA==",
        MEDIA_ERROR_COLUMN: None,
    }


@attrs.frozen
class PhasePlan:
    """What one Phase A call configured, captured without executing any of it.

    Attributes:
        media_kwargs: ``map_batches`` keyword arguments of the media stage.
        write_kwargs: ``write_parquet`` keyword arguments of the terminal sink.
        checkpoint: The Ray Data checkpoint configuration active while the sink
            was configured, or ``None`` if the phase installed none.
        workspace: The workspace the phase was given.
        digest: The semantic contract digest bound into terminal rows.
        config: The resolved config passed through the phase boundary.
        processor: The engine configuration handed to the shared vLLM layer.
        preprocess: The request hook bound by the phase.
        canonicalizer: The terminal batch hook bound by the phase.
        scan_batch_size: Rows requested from each Lance scan batch.
        repartition_kwargs: Row-block repartition arguments applied before media
            fetch, or ``None`` when the phase skipped the operator.
        preprocess_generations: Generation arguments observed at the preprocess boundary.
        canonicalize_generations: Generation arguments observed at the canonicalization boundary.

    """

    media_kwargs: dict[str, Any]
    write_kwargs: dict[str, Any]
    checkpoint: CheckpointConfig | None
    workspace: CaptionWorkspace
    digest: str
    config: ResolvedRobotActionCaptionConfig
    processor: Any
    preprocess: Any
    canonicalizer: Any
    scan_batch_size: int
    repartition_kwargs: dict[str, Any] | None
    preprocess_generations: list[RobotActionCaptionGenerationConfig]
    canonicalize_generations: list[RobotActionCaptionGenerationConfig]


@attrs.define
class _PhasePlanRecorders:
    """Captured operator arguments from one unexecuted Phase A plan."""

    map_functions: list[Any] = attrs.Factory(list)
    media_kwargs: list[dict[str, Any]] = attrs.Factory(list)
    write_kwargs: list[dict[str, Any]] = attrs.Factory(list)
    active_checkpoints: list[CheckpointConfig | None] = attrs.Factory(list)
    processors: list[Any] = attrs.Factory(list)
    preprocess_hooks: list[Any] = attrs.Factory(list)
    scan_batch_sizes: list[int] = attrs.Factory(list)
    repartition_kwargs: list[dict[str, Any]] = attrs.Factory(list)
    preprocess_generations: list[RobotActionCaptionGenerationConfig] = attrs.Factory(list)
    canonicalize_generations: list[RobotActionCaptionGenerationConfig] = attrs.Factory(list)


def _install_phase_plan_recorders(monkeypatch: pytest.MonkeyPatch) -> _PhasePlanRecorders:
    """Replace Phase A's Dataset and hook callables with recorders that skip execution."""
    recorders = _PhasePlanRecorders()
    real_map_batches = ray.data.Dataset.map_batches
    real_repartition = ray.data.Dataset.repartition
    real_canonicalize = inference.canonicalize
    real_datasource = inference.CaptionScanDatasource

    def record_repartition(self: ray.data.Dataset, **kwargs: Any) -> ray.data.Dataset:  # noqa: ANN401 -- records Ray's public keyword surface
        recorders.repartition_kwargs.append(kwargs)
        return real_repartition(self, **kwargs)

    def record_map_batches(self: ray.data.Dataset, fn: Any, **kwargs: Any) -> ray.data.Dataset:  # noqa: ANN401 -- records whatever the phase passes
        recorders.map_functions.append(fn)
        recorders.media_kwargs.append(kwargs)
        return real_map_batches(self, fn, **kwargs)

    def record_write_parquet(_self: ray.data.Dataset, _path: str, **kwargs: Any) -> None:  # noqa: ANN401 -- records whatever the phase passes
        recorders.write_kwargs.append(kwargs)
        # Read here rather than after the call: the configuration is cleared when
        # the phase's context manager exits, so this is the only point at which
        # the plan's own checkpoint is observable from the driver.
        recorders.active_checkpoints.append(ray.data.DataContext.get_current().checkpoint_config)

    monkeypatch.setattr(ray.data.Dataset, "repartition", record_repartition)
    monkeypatch.setattr(ray.data.Dataset, "map_batches", record_map_batches)
    monkeypatch.setattr(ray.data.Dataset, "write_parquet", record_write_parquet)

    def record_build_request(
        row: dict[str, Any],
        *,
        generation: RobotActionCaptionGenerationConfig,
    ) -> dict[str, Any]:
        recorders.preprocess_generations.append(generation)
        return build_request(row, generation=generation)

    def record_canonicalize(
        batch: pa.Table,
        *,
        digest: str,
        model_id: str,
        generation: RobotActionCaptionGenerationConfig,
    ) -> pa.Table:
        recorders.canonicalize_generations.append(generation)
        return real_canonicalize(
            batch,
            digest=digest,
            model_id=model_id,
            generation=generation,
        )

    monkeypatch.setattr(inference, "build_request", record_build_request)
    monkeypatch.setattr(inference, "canonicalize", record_canonicalize)

    def record_datasource(scan: Any, fragments: Any) -> Any:  # noqa: ANN401 -- records a recipe-owned constructor
        recorders.scan_batch_sizes.append(scan.batch_size)
        return real_datasource(scan, fragments)

    monkeypatch.setattr(inference, "CaptionScanDatasource", record_datasource)

    def record_transform(processor: Any, *, preprocess: Any, postprocess: Any) -> Any:  # noqa: ANN401 -- records a framework callable surface
        del postprocess
        recorders.processors.append(processor)
        recorders.preprocess_hooks.append(preprocess)
        return lambda dataset: dataset

    monkeypatch.setattr(inference, "build_vllm_transform", record_transform)
    return recorders


@pytest.fixture(autouse=True)
def clean_data_context() -> Iterator[None]:
    """Leave the process-global Ray Data checkpoint config unset around each test."""
    ray.data.DataContext.get_current().checkpoint_config = None
    yield
    ray.data.DataContext.get_current().checkpoint_config = None


@pytest.fixture
def phase_plan(
    curator_io_cluster: None,
    tmp_path: pathlib.Path,
    caption_clips: ClipsTable,
    caption_config: CaptionConfig,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> PhasePlan:
    """Return the plan Phase A builds for a real clip table, with nothing executed.

    The plan is built and then thrown away: the terminal ``write_parquet`` is
    replaced by a recorder, so nothing runs, but every operator is configured
    exactly as production configures it. That is what makes the requests
    assertable - a constant read straight from the module would stay correct
    while the call site stopped passing it.
    """
    del curator_io_cluster
    uri = caption_clips(rows=2, rows_per_file=2)
    config = caption_config(
        uri,
        max_clip_bytes=_CONFIGURED_CAP_BYTES,
        media_attempts=_CONFIGURED_MEDIA_ATTEMPTS,
    )
    config = config.model_copy(
        update={
            "generation": GENERATION,
            "video": VIDEO,
            "execution": EXECUTION.model_copy(
                update={
                    "max_num_batched_tokens": getattr(
                        request,
                        "param",
                        _CONFIGURED_SCHEDULER_BUDGET,
                    ),
                    "media_fetch_workers": _CONFIGURED_FETCH_WORKERS,
                },
            ),
        }
    )
    # Returns the widened handle, so the plan below is built against the version
    # that carries the caption columns the pending predicate filters on.
    dataset = ensure_caption_fields(uri, storage_options=None)
    context_fields = resolve_context_fields(dataset)
    digest = caption_contract_digest(
        MODEL,
        context_fields,
        generation=config.generation,
        video=config.video,
        media=config.media,
    )
    workspace = CaptionWorkspace(root_uri=str(tmp_path / "ws"), filesystem=pafs.LocalFileSystem())
    recorders = _install_phase_plan_recorders(monkeypatch)
    row_filter = caption_work_filter(digest, MODEL.model_id, selected_only=True)

    run_inference_phase(
        dataset,
        plan_caption_scan(dataset, row_filter, shard=None),
        config,
        MODEL,
        workspace,
        context_fields=context_fields,
        digest=digest,
        row_filter=row_filter,
        storage_options=None,
    )

    return PhasePlan(
        media_kwargs=recorders.media_kwargs[0],
        write_kwargs=recorders.write_kwargs[0],
        checkpoint=recorders.active_checkpoints[0],
        workspace=workspace,
        digest=digest,
        config=config,
        processor=recorders.processors[0],
        preprocess=recorders.preprocess_hooks[0],
        canonicalizer=recorders.map_functions[-1],
        scan_batch_size=recorders.scan_batch_sizes[0],
        repartition_kwargs=recorders.repartition_kwargs[0] if recorders.repartition_kwargs else None,
        preprocess_generations=recorders.preprocess_generations,
        canonicalize_generations=recorders.canonicalize_generations,
    )


@pytest.mark.integration
def test_the_phase_checkpoints_the_scanned_ids_in_its_own_workspace(phase_plan: PhasePlan) -> None:
    """Progress is recorded under the scan's own id column, inside this run's namespace.

    Both halves fail silently. A checkpoint keyed on any other column filters
    nothing and skips nothing, and one written outside the workspace would be
    read by a run whose semantics differ - or by none at all, which costs a full
    re-run of a GPU phase to discover.
    """
    assert phase_plan.checkpoint is not None
    assert phase_plan.checkpoint.id_column == CHECKPOINT_ID_COLUMN
    assert phase_plan.checkpoint.checkpoint_path == phase_plan.workspace.checkpoints_uri


@pytest.mark.integration
def test_the_phase_splits_scan_rows_before_media_fetch(phase_plan: PhasePlan) -> None:
    """Use the scan batch size as a non-strict execution-block target."""
    assert phase_plan.repartition_kwargs == {
        "target_num_rows_per_block": phase_plan.config.execution.lance_read_batch_size,
        "strict": False,
    }


@pytest.mark.integration
def test_the_checkpoint_and_the_sink_share_one_filesystem(phase_plan: PhasePlan) -> None:
    """One credential set reaches both, so the checkpoint can read what the sink wrote.

    Two independently resolved filesystems let the sink write where the
    checkpoint cannot read, and that surfaces as a resume that silently redoes
    everything rather than as an access error. Identity is asserted rather than
    equality: two equal handles resolved separately are exactly the failure.
    """
    assert phase_plan.checkpoint is not None
    assert phase_plan.checkpoint.filesystem is phase_plan.write_kwargs["filesystem"]


@pytest.mark.integration
def test_the_terminal_sink_routes_every_execution_block_by_fragment(phase_plan: PhasePlan) -> None:
    """Partition staged rows by their retained fragment identity."""
    assert phase_plan.write_kwargs["partition_cols"] == [FRAGMENT_ID_COLUMN]


@pytest.mark.integration
def test_the_media_stage_reserves_one_io_token_per_fetch_thread(phase_plan: PhasePlan) -> None:
    """An actor reserves exactly the concurrency it runs.

    One ``curator_io`` slot means one concurrent read, so an actor fetching N
    clips at once must hold N. Reserving fewer makes the node's advertised budget
    a fiction and silently multiplies the load every other stage sized itself
    against; reserving more strands capacity nobody can use. Neither shows up as
    a failure anywhere.
    """
    assert phase_plan.media_kwargs["resources"] == {
        environment.CURATOR_IO_RESOURCE_NAME: float(_CONFIGURED_FETCH_WORKERS)
    }


def test_the_media_guard_reports_the_concurrent_read_ceiling_it_derives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fetch ceiling is the figure the engine pool's replica floor is judged against.

    Two nodes whose capacity is bound by DIFFERENT resources - IO slots on one,
    CPU on the other - and whose totals therefore cannot be recovered from either
    node alone or from the advertised slot sum. The width divides no advertised
    figure evenly, so a ceiling that dropped the per-actor multiplication or read
    slots directly cannot match.
    """
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [
            {"Alive": True, "Resources": {environment.CURATOR_IO_RESOURCE_NAME: 16.0, "CPU": 96.0}},
            {"Alive": True, "Resources": {environment.CURATOR_IO_RESOURCE_NAME: 16.0, "CPU": 1.5}},
        ],
    )

    # Five actors fit the first node (16 // 3 slots, CPU no constraint) and two
    # the second (1.5 // 0.75 CPUs), each running three reads.
    assert _verify_media_resources(3, cpus_per_fetch=0.25) == _MediaFetchCapacity(
        placeable_actors=7,
        read_ceiling=21,
    )


# The shape a four-node Slurm allocation advertises at the default IO budget. The
# default width divides 16 exactly, so no wider width can hold more reads than
# the default already does on this cluster.
_UNIFORM_NODES = [{"Alive": True, "Resources": {environment.CURATOR_IO_RESOURCE_NAME: 16.0, "CPU": 192.0}}] * 4


def test_a_fetch_width_holding_no_more_reads_than_the_default_warns(
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A wider actor re-partitions the node budget rather than adding to it.

    Width 8 against 16 slots places two actors per node where the default places
    four, for the same 64 reads - the configuration a run spent two hours on
    before being cancelled with a third of its GPUs idle. The schedulability
    guard passes at exactly this width, so placement cannot be what reports it.
    """
    monkeypatch.setattr(ray, "nodes", lambda: _UNIFORM_NODES)

    _verify_media_resources(8, cpus_per_fetch=0.25)

    (warning,) = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    assert "64 concurrent read(s) against 64" in warning
    assert "8 actor(s) here against 16 at the default" in warning
    assert "--ray-io-slots-per-node" in warning


def test_a_fetch_width_that_buys_reads_warns_about_nothing(
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The counterfactual, without which the warning above is not attributable.

    A budget of 18 strands two slots at the default width 4 (four actors, 16
    reads); width 6 places three actors and spends all 18. A warning keyed on the
    width having been raised, rather than on the two ceilings, fires here too.
    """
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [{"Alive": True, "Resources": {environment.CURATOR_IO_RESOURCE_NAME: 18.0, "CPU": 192.0}}],
    )

    assert _verify_media_resources(6, cpus_per_fetch=0.25) == _MediaFetchCapacity(
        placeable_actors=3,
        read_ceiling=18,
    )

    assert [record for record in loguru_records if record["level"].name == "WARNING"] == []


def test_a_fetch_width_that_strands_io_slots_reports_the_budget_it_cannot_spend(
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """Integer division rounds the width down against each node, and says nothing.

    The two nodes strand different remainders - 4 of 16 and 1 of 13 at width 6 -
    so a total recoverable from one node, or from the advertised slot sum alone,
    cannot match.
    """
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [
            {"Alive": True, "Resources": {environment.CURATOR_IO_RESOURCE_NAME: 16.0, "CPU": 192.0}},
            {"Alive": True, "Resources": {environment.CURATOR_IO_RESOURCE_NAME: 13.0, "CPU": 192.0}},
        ],
    )

    _verify_media_resources(6, cpus_per_fetch=0.25)

    (reported,) = [record["message"] for record in loguru_records if record["level"].name == "INFO"]
    assert f"leaving 5 {environment.CURATOR_IO_RESOURCE_NAME} slot(s) unusable" in reported


# A pending geometry in which every reported figure differs from every other, so
# a line that printed the total where the pending count belongs, the sum where
# the largest belongs, or the smallest fragment where the largest belongs, cannot
# satisfy the assertions by coincidence. The largest is deliberately close to the
# sum without equalling it, which is the shape a resume's remainder really has.
_PENDING_FRAGMENT_ROWS = (5, 41, 2617)

# Fragments in the run's scope. Larger than the pending count so the two are
# distinguishable in the reported ratio, and not a multiple of it so a line that
# derived one from the other could not match.
_OWNED_FRAGMENTS = 17


def _plan(*fragment_rows: int, owned_fragments: int = _OWNED_FRAGMENTS) -> CaptionScanPlan:
    """Return a plan holding one pending fragment per row count given.

    The owned set is widened to ``owned_fragments`` ids beyond the pending ones,
    which is the real shape: a run owns fragments that owe nothing. Keeping the
    two equal would make every assertion about the pending-versus-owned ratio
    pass under a reporter that printed either number.
    """
    fragments = tuple(
        FragmentRead(fragment_id=index, num_rows=num_rows, input_files=(f"data/{index}.lance",))
        for index, num_rows in enumerate(fragment_rows)
    )
    return CaptionScanPlan(
        fragments=fragments,
        owned_fragment_ids=tuple(range(max(owned_fragments, len(fragments)))),
    )


def _scan_tables(*fragment_rows: int) -> list[pa.Table]:
    """Return one complete scan-schema block per synthetic fragment."""
    return [
        pa.Table.from_pylist(
            [
                {
                    CLIP_ID_COLUMN: f"f{fragment_id}-c{row_offset}",
                    FRAGMENT_ID_COLUMN: fragment_id,
                    CLIP_URI_COLUMN: f"s3://bucket/f{fragment_id}-c{row_offset}.mp4",
                    PROMPT_COLUMN: f"describe f{fragment_id}-c{row_offset}",
                    CHECKPOINT_ID_COLUMN: row_address(fragment_id, row_offset),
                }
                for row_offset in range(num_rows)
            ],
            schema=SCAN_SCHEMA,
        )
        for fragment_id, num_rows in enumerate(fragment_rows)
    ]


def _repartitioned_scan_blocks(
    *fragment_rows: int,
    repartition_kwargs: dict[str, Any],
) -> tuple[pa.Table, list[pa.Table]]:
    """Return the input rows and materialized non-strict output blocks."""
    source_tables = _scan_tables(*fragment_rows)
    source = pa.concat_tables(source_tables)
    repartitioned = ray.data.from_arrow(source_tables).repartition(**repartition_kwargs)
    output_blocks = list(
        repartitioned.materialize().iter_batches(
            batch_size=None,
            batch_format="pyarrow",
            prefetch_batches=0,
        )
    )
    return source, output_blocks


@pytest.mark.integration
def test_repartition_preserves_checkpoint_id_multiset_and_row_count(curator_io_cluster: None) -> None:
    """Every checkpoint identity crosses the execution-block boundary exactly once."""
    del curator_io_cluster
    source, output_blocks = _repartitioned_scan_blocks(
        5,
        7,
        repartition_kwargs={"target_num_rows_per_block": 4, "strict": False},
    )
    output = pa.concat_tables(output_blocks)

    assert output.num_rows == source.num_rows
    assert Counter(output[CHECKPOINT_ID_COLUMN].to_pylist()) == Counter(source[CHECKPOINT_ID_COLUMN].to_pylist())


@pytest.mark.integration
def test_repartition_preserves_clip_to_fragment_routing(curator_io_cluster: None) -> None:
    """Each clip keeps the fragment identifier publication later groups by."""
    del curator_io_cluster
    source, output_blocks = _repartitioned_scan_blocks(
        5,
        7,
        repartition_kwargs={"target_num_rows_per_block": 4, "strict": False},
    )
    output = pa.concat_tables(output_blocks)

    expected = dict(zip(source[CLIP_ID_COLUMN].to_pylist(), source[FRAGMENT_ID_COLUMN].to_pylist(), strict=True))
    actual = dict(zip(output[CLIP_ID_COLUMN].to_pylist(), output[FRAGMENT_ID_COLUMN].to_pylist(), strict=True))
    assert actual == expected


@pytest.mark.integration
def test_non_strict_repartition_keeps_each_output_block_fragment_local(phase_plan: PhasePlan) -> None:
    """Do not coalesce adjacent fragment blocks while splitting large ones."""
    assert phase_plan.repartition_kwargs is not None
    _, output_blocks = _repartitioned_scan_blocks(
        3,
        3,
        repartition_kwargs=phase_plan.repartition_kwargs,
    )

    assert [set(block[FRAGMENT_ID_COLUMN].to_pylist()) for block in output_blocks] == [{0}, {1}]


@pytest.mark.integration
@pytest.mark.parametrize(
    "fragment_rows",
    [
        pytest.param((31,), id="single-large-fragment"),
        pytest.param((15, 27), id="several-large-fragments"),
        pytest.param((1, 1, 1), id="sparse-resume"),
    ],
)
def test_non_strict_repartition_splits_each_fragment_into_exact_row_blocks(
    phase_plan: PhasePlan,
    fragment_rows: tuple[int, ...],
) -> None:
    """Derive fetch work blocks from pending rows without coalescing fragments."""
    assert phase_plan.repartition_kwargs is not None
    _, output_blocks = _repartitioned_scan_blocks(
        *fragment_rows,
        repartition_kwargs=phase_plan.repartition_kwargs,
    )
    rows_per_block = phase_plan.config.execution.lance_read_batch_size

    assert len(output_blocks) == sum(math.ceil(rows / rows_per_block) for rows in fragment_rows)


def test_the_plan_report_states_the_scan_tail_and_derived_fetch_blocks(
    loguru_records: list[dict[str, Any]],
) -> None:
    """Pending rows determine fetch blocks while fragments still describe scan tail.

    The row target splits each pending fragment independently, so the exact
    fetch-block count is available before execution. The largest pending
    fragment still identifies the source task whose scan can finish last, but
    it no longer traps all of that fragment's media work on one actor.
    """
    fetch_blocks = inference._report_pending_fragments(
        _plan(*_PENDING_FRAGMENT_ROWS),
        owned_fragments=_OWNED_FRAGMENTS,
        rows_per_block=_FETCH_ROWS_PER_BLOCK,
    )

    (reported,) = [record["message"] for record in loguru_records if record["level"].name == "INFO"]
    assert fetch_blocks == sum(math.ceil(rows / _FETCH_ROWS_PER_BLOCK) for rows in _PENDING_FRAGMENT_ROWS)
    assert f"3 of {_OWNED_FRAGMENTS} owned fragment(s) pending" in reported
    assert f"holding {sum(_PENDING_FRAGMENT_ROWS):,} row(s)" in reported
    assert f"with {max(_PENDING_FRAGMENT_ROWS):,} in the largest" in reported
    assert f"produce {fetch_blocks} fetch work block(s) at {_FETCH_ROWS_PER_BLOCK} row(s) per block" in reported
    assert "media pool tops out" not in reported


def test_a_pending_count_far_below_the_allocated_width_does_not_escalate(
    loguru_records: list[dict[str, Any]],
) -> None:
    """One pending fragment against a wide allocation stays at INFO.

    This is the geometry that got a run cancelled, and it is also the geometry of
    every healthy nearly-complete resume - the two differ only in how much work
    is left, and how long that takes is a per-clip service time this phase
    declines to invent. A level keyed on the fragment count alone therefore
    cannot separate them, so it would warn on the ordinary case and teach
    operators to skip the line.
    """
    inference._report_pending_fragments(
        _plan(131_602),
        owned_fragments=_OWNED_FRAGMENTS,
        rows_per_block=_FETCH_ROWS_PER_BLOCK,
    )

    assert [record for record in loguru_records if record["level"].name != "INFO"] == []


def _report_io_bound_shortfall() -> None:
    """Report a supply the IO budget caps well below the engine floor's appetite.

    Every figure is distinct, so a line printing one where another belongs cannot
    pass. The block ceiling (500 x 4 = 2000) sits far above the 64-read IO
    ceiling, so the IO budget is unambiguously what binds. The floor sits far
    below its own ceiling, so it is the GPU count that pinned it. And the two
    capacities divide to exactly 16, a value absent from every other figure here
    - which is what lets the multiplier guard below mean anything.
    """
    inference._report_fetch_against_engine_capacity(
        _FetchVersusEngineCapacity(
            read_ceiling=64,
            input_blocks=500,
            fetch_workers=4,
            rows_per_block=_FETCH_ROWS_PER_BLOCK,
            batch_size=32,
            concurrency=(32, 4113),
        )
    )


def test_the_capacity_report_reads_the_two_stocks_per_replica_as_well_as_per_cluster(
    loguru_records: list[dict[str, Any]],
) -> None:
    """The per-replica form is what makes the same two numbers legible.

    64 reads against 1,024 rows does not look wrong; two reads behind a 32-row
    batch does. Both readings describe one configuration, and only the second
    puts the shortfall at a scale an operator can judge at a glance.
    """
    _report_io_bound_shortfall()

    (reported,) = [record["message"] for record in loguru_records if record["level"].name == "INFO"]
    assert "2.00 read(s) in flight behind a 32-row batch" in reported


def test_the_capacity_report_states_no_bare_shortfall_multiplier(
    loguru_records: list[dict[str, Any]],
) -> None:
    """The two capacities are stocks, so their quotient is not a shortfall factor.

    Dividing them yields 16 here while the throughput shortfall is set by the two
    stages' service times, which startup cannot know. A run that published the
    quotient would overstate the gap several-fold and point an operator at the
    wrong remedy, so the message must state neither the number nor a rate.
    """
    _report_io_bound_shortfall()

    reported = " ".join(record["message"] for record in loguru_records)
    # Both spellings the quotient would take, and the unit a fabricated rate
    # would carry. 16 is absent from these figures for every other reason.
    assert "16x" not in reported
    assert " 16 " not in reported
    assert "rows/s" not in reported
    assert "row(s) per second" not in reported


def test_a_supply_far_under_the_engine_floor_neither_warns_nor_prescribes_a_bigger_budget(
    loguru_records: list[dict[str, Any]],
) -> None:
    """The most lopsided input the report accepts still draws no remedy.

    64 reads against a floor wanting 1,024 rows is a sixteen-fold gap, and it is
    exactly the shape a guard on ``supply < engine_demand`` would fire on. It is
    not evidence of starvation: completed fetches queue ahead of the engine, so
    the read count bounds an arrival rate rather than the rows a replica can draw
    on, and the two service times that would settle it are unmeasured here.
    Naming ``--ray-io-slots-per-node`` on this input sends an operator to enlarge
    a budget whose growth funds more CPU-stage actors, not more captions.
    """
    _report_io_bound_shortfall()

    assert [record for record in loguru_records if record["level"].name != "INFO"] == []
    (reported,) = [record["message"] for record in loguru_records]
    assert "--ray-io-slots-per-node" not in reported


def test_the_capacity_report_states_that_completed_fetches_queue_ahead_of_the_engine(
    loguru_records: list[dict[str, Any]],
) -> None:
    """Without the queue, the two printed stocks invite the comparison they deny.

    A reader given "64 reads" beside "1,024 rows the floor needs" will subtract
    them unless told why the smaller number is not a shortage. The queue is that
    reason, and it belongs in the operator-facing line rather than only in the
    source, because the line is what reaches a run's logs.
    """
    _report_io_bound_shortfall()

    # Scoped to INFO so this fails for its own reason. Reading every record would
    # also break on an added WARNING, which is the neighbouring test's concern.
    (reported,) = [record["message"] for record in loguru_records if record["level"].name == "INFO"]
    assert "completed fetches queue ahead of the engine" in reported


def test_a_block_bound_supply_names_the_row_target_as_its_lever(
    loguru_records: list[dict[str, Any]],
) -> None:
    """The one remedy the report still offers, and the input that earns it.

    One input block at width 4 supplies 4 reads while the node budgets would have
    allowed 2,000, so the block count binds and a larger IO budget would change
    nothing. Unlike the fetch/floor comparison this bound is structural: fewer
    work blocks than actors leaves actors with no task to be given, whatever the
    service times turn out to be.
    """
    inference._report_fetch_against_engine_capacity(
        _FetchVersusEngineCapacity(
            read_ceiling=2000,
            input_blocks=1,
            fetch_workers=4,
            rows_per_block=_FETCH_ROWS_PER_BLOCK,
            batch_size=32,
            concurrency=(32, 4113),
        )
    )

    (reported,) = [record["message"] for record in loguru_records if record["level"].name == "INFO"]
    assert f"lower execution.lance_read_batch_size below {_FETCH_ROWS_PER_BLOCK}" in reported


def test_the_video_policy_echo_carries_the_geometry_the_engine_is_built_with(
    loguru_records: list[dict[str, Any]],
) -> None:
    """A digest proves two runs agree; only this line says what geometry ran.

    Every value below differs from the field's own default, so a line that
    hand-assembled the mapping from the wrong attributes, or echoed a default
    instead of the resolved config, cannot satisfy these literals. Recovering
    them otherwise means inverting the engine's encoder-profiling arithmetic.
    """
    inference._report_video_policy(VIDEO)

    (reported,) = [record["message"] for record in loguru_records if record["level"].name == "INFO"]
    # VIDEO's values written out, not read back through the accessors under test.
    assert "'fps': 4.25" in reported
    assert "'min_frames': 8" in reported
    assert "'max_frames': 12" in reported
    assert "'min_pixels': 100352" in reported
    assert "'max_pixels': 200704" in reported


def test_an_undeclared_video_group_echoes_as_delegation_rather_than_as_empty(
    loguru_records: list[dict[str, Any]],
) -> None:
    """An inherited pixel cap is what an unexplained context overflow arrives through.

    Both groups are left at their defaults deliberately: the assertion is about
    where the values come from, not what they are, so only the branch that
    recognises an undeclared group can satisfy it. Printing a bare mapping would
    read as "no cap applies" when a cap does - the artifact's.
    """
    inference._report_video_policy(RobotActionCaptionVideoConfig())

    (reported,) = [record["message"] for record in loguru_records if record["level"].name == "INFO"]
    assert "the loader default decides frame sampling" in reported
    assert "the model artifact decides the pixel bounds" in reported


def test_the_media_guard_names_the_actor_count_as_what_the_budgets_can_place(
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A count read as the pool's size cannot be reconciled with the pool's own report.

    Width 16 against 16 slots places one actor per node, four in all. The line
    has to say that is what the budgets hold simultaneously, because the same
    number read as a prediction of the pool's size disagrees with every other
    actor figure a run produces.
    """
    monkeypatch.setattr(ray, "nodes", lambda: _UNIFORM_NODES)

    _verify_media_resources(16, cpus_per_fetch=0.25)

    reported = " ".join(record["message"] for record in loguru_records if record["level"].name == "INFO")
    assert "place 4 media actor(s) at once" in reported


@pytest.mark.integration
def test_the_media_pool_is_capped_at_the_actors_the_budgets_can_place(phase_plan: PhasePlan) -> None:
    """An actor the pool cannot place still counts in the pool's own occupancy.

    Left unbounded the pool autoscales toward the block count, and each actor
    past what the slots allow stays pending for the run while Ray Data keeps
    dividing work in flight across it. A saturated fetch stage then reports
    slack, which on a fetch-bound run is the number that sends an operator
    looking at the wrong stage. The cap is the placeable actor count (16 slots
    and 4 CPUs at width 3 and 0.5 CPUs per fetch place 2), never the read ceiling
    those actors hold.
    """
    compute = phase_plan.media_kwargs["compute"]

    assert (compute.min_size, compute.max_size) == (1, 2)


@pytest.mark.integration
def test_the_media_batch_matches_the_fetch_width(phase_plan: PhasePlan) -> None:
    """A batch holds exactly one clip per thread.

    The two numbers are one decision, and only their equality is checked, because
    either inequality wastes the reservation the test above pins: a narrower batch
    leaves threads idle while their slots stay claimed, and a wider one queues the
    surplus behind them, holding rows the actor cannot yet start on.
    """
    assert phase_plan.media_kwargs["batch_size"] == _CONFIGURED_FETCH_WORKERS
    assert phase_plan.media_kwargs["fn_constructor_kwargs"]["fetch_workers"] == _CONFIGURED_FETCH_WORKERS


@pytest.mark.integration
def test_the_configured_media_cpu_cost_is_reserved_per_fetch(phase_plan: PhasePlan) -> None:
    """Multiply the configured per-read CPU cost by the actor's fetch width."""
    assert phase_plan.media_kwargs["num_cpus"] == 1.5


@pytest.mark.integration
def test_the_configured_lance_batch_size_reaches_the_pinned_scan(phase_plan: PhasePlan) -> None:
    """Pass the configured scan batch size to each fragment reader."""
    assert phase_plan.scan_batch_size == 13


@pytest.mark.integration
def test_the_configured_parquet_row_target_reaches_both_sink_hints(phase_plan: PhasePlan) -> None:
    """Use one configured staged-file target for both Ray sink hints."""
    assert (
        phase_plan.write_kwargs["min_rows_per_file"],
        phase_plan.write_kwargs["max_rows_per_file"],
    ) == (17, 17)


@pytest.mark.integration
def test_the_configured_inference_batch_size_reaches_the_processor(phase_plan: PhasePlan) -> None:
    """Pass the configured non-default row count to the vLLM processor."""
    assert phase_plan.processor.batch_size == 7


@pytest.mark.integration
def test_the_configured_max_concurrent_batches_reaches_the_processor(phase_plan: PhasePlan) -> None:
    """Pass the configured non-default in-flight depth to each replica."""
    assert phase_plan.processor.max_concurrent_batches == 3


def test_the_configured_tensor_parallel_size_reaches_the_processor() -> None:
    """Pass the configured non-default tensor-parallel factor to the engine."""
    resolved = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=8.0,
    )

    assert resolved.tensor_parallel_size == 2


def test_the_configured_pipeline_parallel_size_reaches_the_processor() -> None:
    """Pass the configured non-default pipeline-parallel factor to the engine."""
    resolved = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=8.0,
    )

    assert resolved.pipeline_parallel_size == 3


def test_the_replica_floor_divides_the_free_gpus_by_the_replica_width() -> None:
    """Claim whole replicas, not GPUs: eight free GPUs at width six floor one.

    The floor is what reserves GPUs for the run's duration, so a derivation that
    read either parallel factor alone would reserve replicas the engine cannot
    place.
    """
    resolved = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=8.0,
    )

    assert resolved.concurrency[0] == 1


@pytest.mark.integration
def test_the_media_stage_carries_both_restart_bounds(phase_plan: PhasePlan) -> None:
    """Spending the restarts while task retries stay unbounded stalls the stage silently.

    Read with ``get`` rather than by subscript so a stage that passes one bound
    and drops the other fails as a diff naming the absent key, not as a
    ``KeyError`` from the test's own bookkeeping.
    """
    passed = {key: phase_plan.media_kwargs.get(key) for key in ACTOR_RESTART_BOUNDS}

    assert passed == dict(ACTOR_RESTART_BOUNDS)


@pytest.mark.integration
def test_the_media_stage_is_built_with_the_configured_byte_cap(phase_plan: PhasePlan) -> None:
    """The cap the digest was computed from is the cap the actor enforces.

    ``max_clip_bytes`` decides whether a clip becomes a caption or a terminal
    error row, and it reaches the contract digest. A stage constructed with any
    other value would judge the corpus under semantics no published row records,
    and every row would still look internally consistent.
    """
    assert phase_plan.media_kwargs["fn_constructor_kwargs"]["max_clip_bytes"] == _CONFIGURED_CAP_BYTES


@pytest.mark.integration
def test_the_media_stage_is_built_with_the_configured_attempt_budget(phase_plan: PhasePlan) -> None:
    """The operator's retry budget reaches the actor that spends it.

    It is the only lever over an intermittently failing object store, and
    spending it ends the whole attempt. A stage constructed from the code's own
    default would leave a raised budget with no effect, and a run would still
    die at the same clip with nothing reporting that the setting was ignored.
    """
    assert phase_plan.media_kwargs["fn_constructor_kwargs"]["attempts"] == _CONFIGURED_MEDIA_ATTEMPTS


def test_variable_frame_rate_policy_reaches_the_loader_unchanged() -> None:
    """Pass rate and bounds through without recipe-local duration arithmetic.

    The exact mapping is asserted, and ``num_frames`` is named as absent, because
    the loader owns sampling timestamps for both constant- and variable-frame-rate
    media. Curator must not derive a source-FPS-dependent frame count.
    """
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=1,
    ).engine_kwargs

    assert engine_kwargs["media_io_kwargs"] == {"video": {"min_frames": 8, "max_frames": 12, "fps": 4.25}}


def test_the_per_frame_pixel_cap_reaches_the_engine_as_a_min_max_pair() -> None:
    """Both bounds together, because the video processor drops either one alone.

    ``Qwen2VLVideoProcessor._standardize_kwargs`` rebuilds its size dict only
    when ``min_pixels`` and ``max_pixels`` are BOTH present, and both names are
    bound parameters there rather than forwarded through ``**kwargs``. A
    declaration carrying one key therefore reaches no resize at all, so the exact
    mapping is asserted rather than each key's presence: a test that checked only
    ``max_pixels`` would pass against precisely the payload that silently applies
    nothing.
    """
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=1,
    ).engine_kwargs

    assert engine_kwargs["mm_processor_kwargs"] == {
        "min_pixels": VIDEO.min_pixels,
        "max_pixels": VIDEO.max_pixels,
    }


def test_omitted_video_controls_reach_the_engine_as_empty_mappings() -> None:
    """Leave both optional override groups to the selected backend."""
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=RobotActionCaptionVideoConfig(),
        execution=EXECUTION,
        pending_rows=1,
    ).engine_kwargs

    assert engine_kwargs["media_io_kwargs"] == {}
    assert engine_kwargs["mm_processor_kwargs"] == {}


def test_temporal_controls_do_not_imply_pixel_overrides() -> None:
    """Keep the two optional groups independent at the engine boundary."""
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=RobotActionCaptionVideoConfig(fps=2.75, min_frames=5, max_frames=33),
        execution=EXECUTION,
        pending_rows=1,
    ).engine_kwargs

    assert engine_kwargs["media_io_kwargs"] == {"video": {"fps": 2.75, "min_frames": 5, "max_frames": 33}}
    assert engine_kwargs["mm_processor_kwargs"] == {}


def test_complete_runtime_video_controls_reach_the_engine_unchanged() -> None:
    """Forward both explicit groups without inspecting model identity."""
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=1,
    ).engine_kwargs

    assert engine_kwargs["media_io_kwargs"] == VIDEO.media_io_kwargs()
    assert engine_kwargs["mm_processor_kwargs"] == VIDEO.mm_processor_kwargs()


@pytest.mark.integration
def test_the_phase_binds_the_configured_pixel_cap_into_the_engine_pair(phase_plan: PhasePlan) -> None:
    """The cap the digest stamps is the cap the engine is built with.

    The gap this closes had both halves looking correct in isolation: a config
    field and a digest entry declaring a bound, and an engine payload that never
    received it. Reading the cap off the resolved config rather than a literal is
    what makes the two halves the same value.
    """
    assert phase_plan.processor.engine_kwargs["mm_processor_kwargs"] == {
        "min_pixels": phase_plan.config.video.min_pixels,
        "max_pixels": phase_plan.config.video.max_pixels,
    }


def test_the_context_limit_reaches_the_engine_config() -> None:
    """Verify the processor executes the digest-recorded context limit."""
    resolved = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=1,
    )

    assert resolved.max_model_len == 509


def test_an_omitted_scheduler_budget_preserves_the_engine_default() -> None:
    """Leave vLLM's scheduler policy untouched when the YAML omits the knob."""
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION.model_copy(update={"max_num_batched_tokens": None}),
        pending_rows=1,
    ).engine_kwargs

    assert "max_num_batched_tokens" not in engine_kwargs


def _scheduler_budget_report(records: list[dict[str, Any]]) -> str:
    """Return the one scheduler-budget line among the build's diagnostics."""
    (reported,) = [
        record["message"]
        for record in records
        if record["level"].name == "INFO" and "scheduler token budget" in record["message"]
    ]
    return reported


def test_an_omitted_scheduler_budget_is_reported_as_the_dependencys_own(
    loguru_records: list[dict[str, Any]],
) -> None:
    """An omitted key is a non-trivial dependency default, not the absence of a setting.

    This one is the scheduler's whole per-step token budget, and nothing
    downstream can tell an inherited value from a chosen one once it is a plain
    int inside the engine -- a grep of this repo for the knob returns nothing,
    which reads as "not relevant" rather than "inherited". So the value AND its
    provenance are both reported, and the value alone would not have closed it.

    The 2048 is a literal rather than the module constant: the claim under test
    is what the pinned vLLM actually defaults to, and re-reading the constant
    would only prove it equals itself.
    """
    processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION.model_copy(update={"max_num_batched_tokens": None}),
        pending_rows=1,
    )

    assert _scheduler_budget_report(loguru_records).startswith(
        "Engine scheduler token budget: 2048 token(s) per step, from vLLM's own default"
    )


def test_a_configured_scheduler_budget_is_reported_as_the_operators_choice(
    loguru_records: list[dict[str, Any]],
) -> None:
    """The other side of the provenance branch, which the report exists to separate.

    Pinned as a whole clause rather than by the number alone: a report naming
    the right value but crediting the wrong source is the defect that sends an
    operator looking for a YAML key they never set, and only the source phrase
    can tell the two branches apart -- the field's NAME appears in both.
    """
    processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=1,
    )

    assert _scheduler_budget_report(loguru_records).startswith(
        f"Engine scheduler token budget: {_CONFIGURED_SCHEDULER_BUDGET} token(s) per step, "
        f"from execution.max_num_batched_tokens."
    )


def test_a_configured_gpu_memory_fraction_reaches_the_engine() -> None:
    """The residency knob must arrive, because nothing else in the run can set it.

    What is left of the device after the weights becomes KV cache, which fixes
    how many requests a replica can hold at once. The value differs from vLLM's
    0.9 and from the 0.92 a later release moved it to, so an engine built on the
    dependency default cannot satisfy this.
    """
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION.model_copy(update={"gpu_memory_utilization": 0.83}),
        pending_rows=1,
    ).engine_kwargs

    assert engine_kwargs["gpu_memory_utilization"] == 0.83


def test_an_omitted_gpu_memory_fraction_leaves_the_key_off_the_engine() -> None:
    """The paired control: absence must preserve vLLM's policy, not substitute one.

    The test above passes identically whether the field is forwarded or a
    recipe-side constant is inserted, so only the omission case can tell
    forwarding from an invented fallback. A key present with any value here
    would override the dependency on every run that never mentioned the knob.
    """
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION.model_copy(update={"gpu_memory_utilization": None}),
        pending_rows=1,
    ).engine_kwargs

    assert "gpu_memory_utilization" not in engine_kwargs


def test_the_default_run_builds_an_engine_with_prefix_caching_off() -> None:
    """Reuse is unreachable for this request shape, so the feature can only cost.

    Every prompt opens with its own clip's vision tokens, because
    ``REQUEST_CONTENT_TYPES`` renders video before text, so two requests diverge
    at the first token and no block is ever a shared prefix. Asserted against
    ``False`` rather than merely "present": the default at the pinned version is
    on, so a dropped key and a wrong key are the same outcome, and neither
    raises -- the cost shows only as block hashing over every prompt and
    retained blocks counted against reported KV usage.
    """
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=1,
    ).engine_kwargs

    assert engine_kwargs["enable_prefix_caching"] is False


def test_an_enabled_prefix_cache_reaches_the_engine() -> None:
    """The configured value must arrive, not the recipe's opinion of it.

    Set to ``True`` deliberately: the field's own default is ``False``, so an
    engine built from a hardcoded literal -- which is what this field replaced --
    satisfies every assertion made at the default and only this one can tell
    forwarding from a constant. Nothing raises either way, so the assertion is
    the only thing standing between a knob and a decoration.
    """
    engine_kwargs = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION.model_copy(update={"enable_prefix_caching": True}),
        pending_rows=1,
    ).engine_kwargs

    assert engine_kwargs["enable_prefix_caching"] is True


def test_prefix_caching_stays_off_however_the_request_order_is_read() -> None:
    """Pins the premise the decision rests on, where that premise actually lives.

    The pin above is sound only while the clip is rendered FIRST. Text first
    would give every request a genuinely shared prefix and make the setting
    worth revisiting -- and that is a change to ``REQUEST_CONTENT_TYPES``, which
    the caption contract digests, so it cannot happen quietly. This is the
    assertion that would redden if it did.
    """
    assert REQUEST_CONTENT_TYPES[0] == VIDEO_CONTENT_TYPE


def test_the_engine_actors_run_with_expandable_allocator_segments() -> None:
    """The engine actors carry the allocator setting the recipe's fragmentation needs."""
    resolved = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=1,
    )

    assert resolved.runtime_env is not None
    assert resolved.runtime_env["env_vars"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_the_allocator_setting_is_not_taken_from_the_shared_gpu_environment() -> None:
    """Sits on the other side of the scoping comparison the test above asserts.

    The assertion above passes identically whether the setting is scoped to this
    recipe or applied to every Ray Data GPU stage in every recipe, because the
    shared mapping is merged into whatever the caller supplies. Only the shared
    environment's own contents can tell the two apart.
    """
    shared = ray_data_gpu_runtime_env("default")

    assert "PYTORCH_CUDA_ALLOC_CONF" not in shared["env_vars"]


@pytest.mark.integration
@pytest.mark.parametrize("phase_plan", [None], indirect=True)
def test_the_phase_omits_an_unconfigured_scheduler_budget(phase_plan: PhasePlan) -> None:
    """Preserve vLLM's scheduler default through the top-level phase caller."""
    assert phase_plan.processor.max_model_len == GENERATION.max_model_len
    assert "max_num_batched_tokens" not in phase_plan.processor.engine_kwargs


@pytest.mark.integration
def test_the_phase_forwards_the_configured_scheduler_budget(phase_plan: PhasePlan) -> None:
    """Pass an execution-only scheduler budget from resolved config to vLLM."""
    assert phase_plan.processor.engine_kwargs["max_num_batched_tokens"] == _CONFIGURED_SCHEDULER_BUDGET


def test_the_frame_budget_is_not_sent_as_a_per_request_processor_argument() -> None:
    """A per-request copy would leave the decode on the library default, silently.

    The other half of the pin above: the budget belongs to the processor build,
    and the only per-request media argument is the flag that stops the model
    processor re-sampling the frames Ray's multimodal stage already selected.
    """
    request = build_request(_fetched_row(), generation=GENERATION)

    assert request["mm_processor_kwargs"] == {"do_sample_frames": DO_SAMPLE_FRAMES}


def test_the_processor_config_asks_for_the_media_preparation_stage() -> None:
    """A media-backed recipe hands its private executor a True flag.

    Pins only what this function decides. Whether the flag then reaches the
    pinned processor is the engine wiring's contract, covered against a stub
    ``ray.data.llm`` in ``test_engine_wiring.py``.
    """
    assert (
        processor_config(
            MODEL,
            generation=GENERATION,
            video=VIDEO,
            execution=EXECUTION,
            pending_rows=1,
        ).prepare_multimodal
        is True
    )


@pytest.mark.integration
def test_the_bound_request_hooks_execute_the_configured_generation(phase_plan: PhasePlan) -> None:
    """Execute one resolved generation object through both bound hooks."""
    request = phase_plan.preprocess(_fetched_row())
    assert request["sampling_params"] == {
        "temperature": phase_plan.config.generation.temperature,
        "top_p": phase_plan.config.generation.top_p,
        "max_tokens": phase_plan.config.generation.max_output_tokens,
    }

    generated_token_count = 2
    terminal = phase_plan.canonicalizer(
        pa.Table.from_pylist(
            [
                {
                    CLIP_ID_COLUMN: "c0",
                    FRAGMENT_ID_COLUMN: 0,
                    CHECKPOINT_ID_COLUMN: row_address(0, 0),
                    GENERATED_TEXT_COLUMN: "caption",
                    GENERATED_TOKENS_COLUMN: generated_token_count,
                    INPUT_TOKENS_COLUMN: phase_plan.config.generation.max_model_len - generated_token_count,
                    INFERENCE_ERROR_COLUMN: "",
                }
            ]
        )
    )

    assert terminal.column(CAPTION_STATUS_COLUMN).to_pylist() == [CaptionStatus.TRUNCATED]
    assert terminal.column(CAPTION_DIGEST_COLUMN).to_pylist() == [phase_plan.digest]
    assert terminal.column(CAPTION_MODEL_COLUMN).to_pylist() == [MODEL.model_id]
    assert phase_plan.preprocess_generations == [phase_plan.config.generation]
    assert phase_plan.preprocess_generations[0] is phase_plan.config.generation
    assert phase_plan.canonicalize_generations == [phase_plan.config.generation]
    assert phase_plan.canonicalize_generations[0] is phase_plan.config.generation


@pytest.mark.integration
def test_the_phase_binds_generation_and_video_into_the_processor(phase_plan: PhasePlan) -> None:
    """Execute context and frame values from the resolved top-level config."""
    video = phase_plan.config.video

    assert phase_plan.processor.max_model_len == phase_plan.config.generation.max_model_len
    assert phase_plan.processor.engine_kwargs["media_io_kwargs"] == {
        "video": {"min_frames": video.min_frames, "max_frames": video.max_frames, "fps": video.fps}
    }


def test_the_actor_ceiling_derives_from_pending_work_not_startup_gpus() -> None:
    """The pool may grow past the GPUs Ray reported at build time.

    A snapshot taken before the pool starts is the wrong bound: on a cluster
    still scaling up it reads near zero, and a ceiling derived from it would pin
    the whole run to a fraction of the hardware it is about to receive - with
    nothing reporting the loss but a slower run. Ray scales between the two
    bounds as blocks queue, so the snapshot may set the FLOOR only.
    """
    starved = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=0.0,
    )
    plentiful = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=_PLENTIFUL_GPUS,
    )
    floor, ceiling = starved.concurrency

    assert floor == 1
    assert ceiling > floor
    assert ceiling == plentiful.concurrency[1]


def test_a_small_backlog_never_licenses_more_replicas_than_work() -> None:
    """A large cluster with one pending row starts and stays at one replica.

    The other direction of the same derivation: without the pending-work bound a
    plentiful snapshot would license replicas there is no work to fill, each
    paying a full model load to process nothing.
    """
    assert processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=1,
        available_gpus=_PLENTIFUL_GPUS,
    ).concurrency == (1, 1)


def test_an_explicit_concurrency_caps_the_ceiling() -> None:
    """An operator's pinned replica count bounds the pool however much work is pending."""
    resolved = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION.model_copy(update={"inference_concurrency": 3}),
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=_PLENTIFUL_GPUS,
    )

    assert resolved.concurrency == (3, 3)


def test_the_replica_pool_never_resolves_to_zero() -> None:
    """A pool of zero replicas would stall the plan rather than wait for capacity."""
    assert processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=0,
        available_gpus=0.0,
    ).concurrency == (1, 1)


def test_a_non_positive_configured_replica_cap_is_rejected() -> None:
    """min(ceiling, 0) would return a pool of zero, which stalls the plan."""
    with pytest.raises(ValueError, match="configured replica cap must be a positive integer"):
        inference._inference_concurrency(
            pending_rows=64,
            available_gpus=8.0,
            configured=0,
            batch_size=32,
            gpus_per_replica=1,
        )


def test_the_media_budget_is_the_frame_ceiling_times_the_frame_size() -> None:
    """The reservation is derived from the configured ceiling, not a stored constant.

    Pinned against a literal so the assertion cannot be satisfied by re-deriving
    the formula under test. Every input differs from its shipped default -- VIDEO
    caps at 12 frames where the example config caps at 64, and _SOURCE_FRAME_PIXELS
    is 800x600 where the default is 1280x720 -- so a build that ignored either
    argument, or dropped the bytes-per-pixel factor, cannot land on this number.
    """
    resolved = processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=_PLENTIFUL_GPUS,
    )

    # 12 frames x (800 x 600) px x 3 bytes of uint8 RGB.
    assert resolved.media_bytes_per_row == 17_280_000


def test_an_unbounded_temporal_group_falls_back_to_the_loader_frame_default() -> None:
    """Without a configured ceiling the budget rests on vLLM's own frame default.

    Sits on the other side of the branch from the test above. The bound is then
    read off a dependency rather than this run's config, which is why the caller
    is warned; here it is enough that the fallback is reachable and produces a
    different, larger budget than the 12-frame configuration would.
    """
    resolved = processor_config(
        MODEL,
        generation=GENERATION,
        video=RobotActionCaptionVideoConfig(),
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=_PLENTIFUL_GPUS,
    )

    # 32 frames x (800 x 600) px x 3 bytes, VideoMediaIO's num_frames default.
    assert resolved.media_bytes_per_row == 46_080_000


def test_an_unbounded_temporal_group_warns_that_the_budget_is_not_this_runs_own(
    loguru_records: list[dict[str, Any]],
) -> None:
    """Delegating temporal behavior must not silently borrow a dependency's bound.

    The warning is the whole remedy for the fallback: it stays a supported mode,
    so nothing else tells an operator their reservation was sized by vLLM rather
    than by their own config.
    """
    processor_config(
        MODEL,
        generation=GENERATION,
        video=RobotActionCaptionVideoConfig(),
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=_PLENTIFUL_GPUS,
    )

    (warning,) = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    assert "video.max_frames is unset" in warning


def test_a_configured_frame_ceiling_draws_no_warning(loguru_records: list[dict[str, Any]]) -> None:
    """A run that bounds its own frames has nothing to be warned about.

    The negative control for the test above, on the other side of the same
    branch: a warning emitted unconditionally would train operators to ignore the
    one case where the bound really is borrowed.
    """
    processor_config(
        MODEL,
        generation=GENERATION,
        video=VIDEO,
        execution=EXECUTION,
        pending_rows=_MANY_PENDING_ROWS,
        available_gpus=_PLENTIFUL_GPUS,
    )

    assert [record for record in loguru_records if record["level"].name == "WARNING"] == []
