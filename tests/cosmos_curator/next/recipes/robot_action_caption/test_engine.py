# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the robot-action-caption vLLM config and actor restart bounds.

The restart-bound tests run against a local stub processor rather than a real
one: ``build_processor`` calls ``ray.init`` and downloads tokenizer files, so
building a real processor needs a live cluster and a network. The stub
reproduces the only surface the injection touches -- stage names and each
stage's ``map_batches`` kwargs.
"""

import collections
import pickle
import sys
import types
from collections.abc import Callable
from typing import Any

import attrs
import pytest
from ray.data import ActorPoolStrategy

from cosmos_curator.next.recipes.robot_action_caption import engine
from cosmos_curator.next.recipes.robot_action_caption.config import (
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_SOURCE_FRAME_PIXELS,
)
from cosmos_curator.next.recipes.robot_action_caption.engine import (
    StageHeapReservations,
    VllmProcessorConfig,
    apply_restart_bounds,
    build_vllm_transform,
    is_actor_backed,
)


@attrs.define
class _StubStage:
    """Stage that accepts a kwargs assignment, like the pinned pydantic stage."""

    map_batches_kwargs: dict[str, Any]


class _WriteResistantStage:
    """Stage that silently drops a kwargs assignment.

    Stands in for the failure the build-time assertion exists to catch: a stage
    whose scheduling knobs cannot be re-configured, which would make the
    injection loop a no-op discovered only as an unbounded restart loop.
    """

    def __init__(self, map_batches_kwargs: dict[str, Any]) -> None:
        self._kwargs = map_batches_kwargs

    @property
    def map_batches_kwargs(self) -> dict[str, Any]:
        """Return the stage's scheduling knobs."""
        return self._kwargs

    @map_batches_kwargs.setter
    def map_batches_kwargs(self, value: dict[str, Any]) -> None:
        """Drop the assignment."""


@attrs.define
class _StubProcessor:
    """Minimal stand-in exposing the two accessors the injection relies on."""

    stages: dict[str, Any]

    def list_stage_names(self) -> list[str]:
        """Return the stage names in insertion order."""
        return list(self.stages)

    def get_stage_by_name(self, name: str) -> Any:  # noqa: ANN401 -- stubs vary by test
        """Return one stage by name."""
        return self.stages[name]


@pytest.fixture
def make_config() -> Callable[..., VllmProcessorConfig]:
    """Return a factory building a valid config with per-test overrides."""

    def _factory(**overrides: Any) -> VllmProcessorConfig:  # noqa: ANN401 -- passthrough overrides
        defaults: dict[str, Any] = {
            "model": "stub/model",
            "max_model_len": 4096,
            "prepare_multimodal": False,
        }
        return VllmProcessorConfig(**(defaults | overrides))

    return _factory


def test_config_rejects_non_positive_tensor_parallel_size(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """A replica must shard across at least one GPU."""
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        make_config(tensor_parallel_size=0)


def test_config_rejects_non_positive_pipeline_parallel_size(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """A replica must contain at least one pipeline stage."""
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        make_config(pipeline_parallel_size=0)


def test_config_rejects_a_non_string_model(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """A model source must use the documented string category."""
    with pytest.raises(TypeError, match="model"):
        make_config(model=["stub/model"])


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("max_model_len", 1.5),
        ("max_model_len", True),
        ("tensor_parallel_size", 1.5),
        ("tensor_parallel_size", True),
        ("pipeline_parallel_size", 1.5),
        ("pipeline_parallel_size", True),
        ("batch_size", 1.5),
        ("batch_size", True),
        ("max_concurrent_batches", 1.5),
        ("max_concurrent_batches", True),
    ],
)
def test_config_rejects_non_integer_values_for_positive_integer_fields(
    make_config: Callable[..., VllmProcessorConfig],
    field_name: str,
    invalid_value: object,
) -> None:
    """Positive-integer fields reject comparison-compatible wrong types."""
    with pytest.raises(TypeError, match=field_name):
        make_config(**{field_name: invalid_value})


@pytest.mark.parametrize(
    "field_name",
    ["max_model_len", "tensor_parallel_size", "pipeline_parallel_size", "batch_size", "max_concurrent_batches"],
)
def test_config_accepts_one_for_positive_integer_fields(
    make_config: Callable[..., VllmProcessorConfig],
    field_name: str,
) -> None:
    """One remains the valid lower boundary for every positive-integer field."""
    assert getattr(make_config(**{field_name: 1}), field_name) == 1


@pytest.mark.parametrize("invalid_value", [1, "false"])
def test_config_rejects_non_boolean_multimodal_choices(
    make_config: Callable[..., VllmProcessorConfig],
    invalid_value: object,
) -> None:
    """The required media-preparation choice must be an actual boolean."""
    with pytest.raises(TypeError, match="prepare_multimodal"):
        make_config(prepare_multimodal=invalid_value)


@pytest.mark.parametrize(
    ("prepare_multimodal", "media_bytes_per_row"),
    [(False, None), (True, 11 * 1024 * 1024)],
)
def test_config_accepts_boolean_multimodal_choices(
    make_config: Callable[..., VllmProcessorConfig],
    prepare_multimodal: object,
    media_bytes_per_row: int | None,
) -> None:
    """Both explicit media-preparation choices remain valid, each with its own budget.

    The budget is parametrized alongside the flag because the two are accepted
    only in agreement; the pairing itself is asserted elsewhere.
    """
    config = make_config(prepare_multimodal=prepare_multimodal, media_bytes_per_row=media_bytes_per_row)
    assert config.prepare_multimodal is prepare_multimodal


def test_media_reservations_cover_every_batch_in_flight_and_the_resident_cost(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Each reservation must cover both in-flight batches plus the resident cost.

    Pinned against literals rather than the formula: an assertion re-deriving
    its expectation from the same constants would hold just as well if the
    in-flight factor or the resident term were dropped, which is the regression
    that packs twice the actors a node can hold and OOM-kills it.
    """
    config = make_config(prepare_multimodal=True, batch_size=7, media_bytes_per_row=11 * 1024 * 1024)

    # 3 GiB resident, plus 7 rows x 11 MiB x 2 batches in flight x the copies held.
    assert config.stage_heap_reservations() == StageHeapReservations(
        preparation=3_705_667_584.0,
        forwarding=3_544_186_880.0,
    )


def test_the_published_reservation_figures_still_match_the_shipped_defaults(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """The design doc's worked figures must stay true of the shipped defaults.

    Those figures are what an operator sizes a node from, and they are a product
    of four private constants and two config defaults -- none of which any other
    test ties to them, so all six can move while every gate stays green. The
    inputs are read from the shipped constants and only the published results
    are literals, so this reddens on a change to any of the six.
    """
    config = make_config(
        prepare_multimodal=True,
        batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
        # 64 frames of decoded RGB. Not a default -- video.max_frames is unset
        # unless configured -- so the doc names it beside the two that are.
        media_bytes_per_row=64 * DEFAULT_SOURCE_FRAME_PIXELS * 3,
    )

    # Published as 18.82 GiB and 13.55 GiB in curator-next-robot-action-caption.md.
    assert config.stage_heap_reservations() == StageHeapReservations(
        preparation=20_208_156_672.0,
        forwarding=14_545_846_272.0,
    )


@pytest.fixture
def short_hook_report_interval(monkeypatch: pytest.MonkeyPatch) -> int:
    """Shorten the timed hook's reporting interval and return the value set.

    The shipped interval is a log-volume choice rather than a contract, so
    driving thousands of rows to observe one line would pin the constant instead
    of the accounting it gates.
    """
    interval = 3
    monkeypatch.setattr(engine, "_HOOK_REPORT_EVERY_ROWS", interval, raising=True)
    return interval


def test_a_timed_hook_returns_what_the_wrapped_hook_returned() -> None:
    """Timing must be transparent: the row the engine sees is the hook's own.

    The wrapper sits on the request-construction and result-canonicalization
    path, so a row it rebuilt, copied, or replaced with its own bookkeeping
    would change what the engine is asked to generate.
    """
    produced = {"prompt": "describe", "sampling_params": {"temperature": 0.0}}

    returned = engine._timed_hook(lambda _row: produced, "preprocess")({"clip_id": "c0"})

    assert returned is produced


def test_a_timed_hook_propagates_the_wrapped_hooks_failure() -> None:
    """A failing row must still fail. Timing may not convert a fault into a row."""

    def explode(row: dict[str, Any]) -> dict[str, Any]:
        del row
        msg = "request could not be built"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="request could not be built"):
        engine._timed_hook(explode, "preprocess")({"clip_id": "c0"})


def test_a_timed_hook_accounts_a_row_that_raised(
    short_hook_report_interval: int,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A raising row must contribute its cost, which is what the ``finally`` buys.

    A hook that fails slowly is exactly the case worth seeing, and accounting
    only successful rows hides it: the tally then advances on the surviving rows
    alone. Driven so the LAST row of the interval is the one that raises, so the
    report can only appear if that row was counted -- accounting placed after
    the wrapped call leaves the tally one short and emits nothing.
    """
    calls = iter(range(short_hook_report_interval))

    def failing_last(row: dict[str, Any]) -> dict[str, Any]:
        if next(calls) == short_hook_report_interval - 1:
            msg = "the slow one"
            raise ValueError(msg)
        return row

    hook = engine._timed_hook(failing_last, "preprocess")
    for _ in range(short_hook_report_interval - 1):
        hook({"clip_id": "c0"})
    with pytest.raises(ValueError, match="the slow one"):
        hook({"clip_id": "c0"})

    (reported,) = [record["message"] for record in loguru_records if record["level"].name == "INFO"]
    assert f"preprocess hook after {short_hook_report_interval} row(s)" in reported


def test_a_timed_hook_stays_silent_below_its_reporting_interval(
    short_hook_report_interval: int,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The negative control for the interval, on the other side of its comparison.

    Every assertion above is satisfied by a hook that logged on each row, which
    at this recipe's batch sizes is hundreds of thousands of lines. Only a row
    count BELOW the interval can tell periodic reporting from unconditional.
    """
    hook = engine._timed_hook(lambda row: row, "postprocess")
    for _ in range(short_hook_report_interval - 1):
        hook({"clip_id": "c0"})

    assert [record for record in loguru_records if record["level"].name == "INFO"] == []


def test_reservations_refuse_a_preparation_figure_that_does_not_exceed_forwarding() -> None:
    """The carrier of the two figures must reject them equal or inverted.

    Both roles derived from one copy count is the defect the split exists to
    prevent, and it is silent: it either strands budget on the two cheap stages
    or overcommits the node on the expensive one. Enforced at construction so
    no caller has to remember which figure belongs to which stage.
    """
    with pytest.raises(ValueError, match="preparation must reserve more"):
        StageHeapReservations(preparation=1.0, forwarding=1.0)


def test_shrinking_the_batch_leaves_the_resident_cost_standing(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Only the payload term scales with the batch; the resident cost does not.

    A purely per-batch formula trends to zero as the batch shrinks, which is
    where the interpreter, tokenizer and vLLM ``ModelConfig`` are most of an
    actor's footprint. Two batch sizes are compared because a single one cannot
    distinguish a fixed term from a proportional one.
    """
    kwargs = {"prepare_multimodal": True, "media_bytes_per_row": 11 * 1024 * 1024}
    wide = make_config(batch_size=7, **kwargs).stage_heap_reservations()
    narrow = make_config(batch_size=1, **kwargs).stage_heap_reservations()
    assert wide is not None
    assert narrow is not None

    # Six rows fewer, at 11 MiB each, over 2 batches in flight x 3 copies decoded.
    assert wide.preparation - narrow.preparation == 6 * 11 * 1024 * 1024 * 2 * 3
    # What remains under the smallest possible batch is the resident cost alone.
    assert narrow.preparation - 1 * 11 * 1024 * 1024 * 2 * 3 == 3 * 1024**3


def test_config_rejects_a_scalar_concurrency(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """The pool is always a range, so a bare replica count is not a valid shape.

    A fixed-size pool is expressed as equal bounds. Accepting a scalar too would
    give one concept two spellings, and every reader of the field would have to
    handle both. The zero-replica case is covered by the range tests.
    """
    with pytest.raises(TypeError, match="concurrency"):
        make_config(concurrency=1)


def test_config_rejects_non_positive_batch_size(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """A zero-row batch would hand the engine nothing."""
    with pytest.raises(ValueError, match="batch_size"):
        make_config(batch_size=0)


def test_config_accepts_an_autoscaling_replica_range(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """A run that starts on partial capacity must still be able to use more when it arrives."""
    assert make_config(concurrency=(1, 4)).concurrency == (1, 4)


def test_config_rejects_an_inverted_replica_range(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """A ceiling below the initial size would ask Ray for a pool it may not hold.

    Both bounds are individually valid here, so only the ordering rule can
    produce this refusal.
    """
    with pytest.raises(ValueError, match="initial <= ceiling"):
        make_config(concurrency=(4, 1))


def test_config_rejects_a_replica_range_of_the_wrong_arity(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """A 3-tuple is not a range; accepting it would silently ignore an element."""
    with pytest.raises(TypeError, match="2-tuple"):
        make_config(concurrency=(1, 2, 3))


@pytest.mark.parametrize("invalid_member", [1.0, True])
def test_config_rejects_non_integer_replica_range_members(
    make_config: Callable[..., VllmProcessorConfig],
    invalid_member: object,
) -> None:
    """Replica range members must be actual integers, not numeric substitutes."""
    with pytest.raises(TypeError, match="concurrency"):
        make_config(concurrency=(invalid_member, 4))


def test_config_rejects_a_zero_initial_replica_count_in_a_range(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """A pool starting at zero replicas never makes the first request that would grow it."""
    with pytest.raises(ValueError, match="initial replica count"):
        make_config(concurrency=(0, 4))


@pytest.mark.parametrize("invalid_model", ["", " \t\n"])
def test_config_rejects_blank_model(
    make_config: Callable[..., VllmProcessorConfig],
    invalid_model: str,
) -> None:
    """A model source needs at least one non-whitespace character."""
    with pytest.raises(ValueError, match="model"):
        make_config(model=invalid_model)


def test_config_refuses_to_infer_whether_media_preparation_is_needed() -> None:
    """Omitting the media-preparation choice fails at construction, not at inference.

    A caller who leaves it out would otherwise inherit "off" and get an engine
    request carrying vision placeholders with no media, which nothing rejects.
    """
    with pytest.raises(TypeError, match="prepare_multimodal"):
        VllmProcessorConfig(model="stub/model", max_model_len=4096)  # type: ignore[call-arg]


def test_config_rejects_engine_kwargs_shadowing_a_typed_field(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Two sources of truth for one engine argument must fail at construction."""
    with pytest.raises(ValueError, match="max_model_len"):
        make_config(engine_kwargs={"max_model_len": 8192})


def test_config_rejects_engine_side_data_parallelism(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """Ray Data owns replication, so the engine must not fan out as well."""
    with pytest.raises(ValueError, match="data_parallel_size"):
        make_config(engine_kwargs={"data_parallel_size": 2})


def test_config_survives_a_pickle_round_trip(make_config: Callable[..., VllmProcessorConfig]) -> None:
    """The config crosses a Ray boundary, so it must pickle unchanged."""
    config = make_config(concurrency=(3, 3), engine_kwargs={"gpu_memory_utilization": 0.85})
    assert pickle.loads(pickle.dumps(config)) == config  # noqa: S301 -- round-tripping our own value


def test_config_deep_copies_caller_supplied_engine_kwargs(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Keep later nested caller mutation out of the stored engine config."""
    caller_kwargs: dict[str, Any] = {"media_io_kwargs": {"video": {"num_frames": 32}}}
    config = make_config(engine_kwargs=caller_kwargs)
    caller_kwargs["media_io_kwargs"]["video"]["num_frames"] = 8
    assert config.engine_kwargs["media_io_kwargs"]["video"]["num_frames"] == 32


def test_config_deep_copies_caller_supplied_runtime_env(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Keep later nested caller mutation out of the stored runtime environment."""
    caller_runtime_env: dict[str, Any] = {"env_vars": {"MODEL_REVISION": "pinned"}}
    config = make_config(runtime_env=caller_runtime_env)
    caller_runtime_env["env_vars"]["MODEL_REVISION"] = "changed"
    assert config.runtime_env == {"env_vars": {"MODEL_REVISION": "pinned"}}


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("engine_kwargs", [("gpu_memory_utilization", 0.8)]),
        ("runtime_env", [("env_vars", {"MODEL_REVISION": "pinned"})]),
    ],
)
def test_config_rejects_non_mapping_inputs_before_copying(
    make_config: Callable[..., VllmProcessorConfig],
    field_name: str,
    invalid_value: object,
) -> None:
    """Iterable pairs are not mappings even though ``dict`` can consume them."""
    with pytest.raises(TypeError, match=field_name):
        make_config(**{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "mapping"),
    [
        ("engine_kwargs", collections.UserDict({"gpu_memory_utilization": 0.8})),
        ("runtime_env", collections.UserDict({"env_vars": {"MODEL_REVISION": "pinned"}})),
    ],
)
def test_config_accepts_mapping_inputs_and_stores_plain_dicts(
    make_config: Callable[..., VllmProcessorConfig],
    field_name: str,
    mapping: collections.UserDict[str, Any],
) -> None:
    """Retain the mapping input contract while taking ownership as a dict."""
    stored = getattr(make_config(**{field_name: mapping}), field_name)
    assert stored == dict(mapping)
    assert type(stored) is dict


def test_the_transform_hands_the_processor_a_copy_of_the_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """A processor that mutates the runtime env cannot reach back into the config."""
    captured: list[dict[str, Any]] = []

    def capture_engine_config(**kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - mirrors pydantic kwargs
        captured.append(kwargs)
        return kwargs

    fake_llm = types.ModuleType("ray.data.llm")
    fake_llm.PrepareMultimodalStageConfig = _unasserted_stage_config
    fake_llm.ChatTemplateStageConfig = _unasserted_stage_config
    fake_llm.TokenizerStageConfig = _unasserted_stage_config
    fake_llm.vLLMEngineProcessorConfig = capture_engine_config
    fake_llm.build_processor = lambda _config, **_kwargs: _StubProcessor(
        {"vLLMEngineStage": _StubStage({"concurrency": 1})}
    )
    monkeypatch.setitem(sys.modules, "ray.data.llm", fake_llm)

    config = make_config(runtime_env={"env_vars": {"MODEL_REVISION": "pinned"}})
    build_vllm_transform(config, preprocess=lambda row: row, postprocess=lambda row: row)

    captured[0]["runtime_env"]["env_vars"]["MODEL_REVISION"] = "mutated"
    assert config.runtime_env == {"env_vars": {"MODEL_REVISION": "pinned"}}


def test_config_engine_kwargs_payload_is_deeply_fresh_on_every_call(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Keep nested consumer mutation out of the config and sibling payloads."""
    config = make_config(engine_kwargs={"media_io_kwargs": {"video": {"num_frames": 32}}})
    first = config.engine_kwargs_payload()
    second = config.engine_kwargs_payload()

    first["media_io_kwargs"]["video"]["num_frames"] = 8
    assert config.engine_kwargs["media_io_kwargs"]["video"]["num_frames"] == 32
    assert second["media_io_kwargs"]["video"]["num_frames"] == 32

    second["media_io_kwargs"]["video"]["num_frames"] = 16
    assert first["media_io_kwargs"]["video"]["num_frames"] == 8
    assert config.engine_kwargs_payload()["media_io_kwargs"]["video"]["num_frames"] == 32


def _unasserted_stage_config(**kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - mirrors pydantic kwargs
    """Stand in for a stage config these tests do not assert on.

    The chat-template and tokenizer stage configs are constructed on every
    build, so a stub module must supply something callable for them even when
    the test under way only cares about the engine config.
    """
    return kwargs


def test_config_engine_kwargs_payload_carries_the_typed_engine_fields(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Typed fields reach the engine through the payload, not a second channel."""
    payload = make_config(
        max_model_len=2048,
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
    ).engine_kwargs_payload()
    assert payload["max_model_len"] == 2048
    assert payload["tensor_parallel_size"] == 2
    assert payload["pipeline_parallel_size"] == 3


def test_repeated_builds_give_preparation_and_engine_independent_media_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Give every build and consumer independent nested media kwargs."""

    @attrs.define
    class CapturedPreparationConfig:
        """Capture the preparation-stage model configuration."""

        enabled: bool
        model_config_kwargs: dict[str, Any]
        memory: float | None = None

    captured_engine_configs: list[dict[str, Any]] = []

    def capture_engine_config(**kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - mirrors pydantic kwargs
        captured_engine_configs.append(kwargs)
        return kwargs

    def build_processor(
        _config: dict[str, Any],
        **_kwargs: Any,  # noqa: ANN401 - mirrors the external builder
    ) -> _StubProcessor:
        return _StubProcessor({"vLLMEngineStage": _StubStage({"concurrency": 1})})

    fake_llm = types.ModuleType("ray.data.llm")
    fake_llm.PrepareMultimodalStageConfig = CapturedPreparationConfig
    fake_llm.ChatTemplateStageConfig = _unasserted_stage_config
    fake_llm.TokenizerStageConfig = _unasserted_stage_config
    fake_llm.vLLMEngineProcessorConfig = capture_engine_config
    fake_llm.build_processor = build_processor
    monkeypatch.setitem(sys.modules, "ray.data.llm", fake_llm)

    config = make_config(
        prepare_multimodal=True,
        media_bytes_per_row=11 * 1024 * 1024,
        engine_kwargs={"media_io_kwargs": {"video": {"num_frames": 32}}},
    )
    build_vllm_transform(
        config,
        preprocess=lambda row: row,
        postprocess=lambda row: row,
    )

    first_engine_config = captured_engine_configs[0]
    first_preparation_config = first_engine_config["prepare_multimodal_stage"].model_config_kwargs
    first_engine_media_kwargs = first_engine_config["engine_kwargs"]["media_io_kwargs"]
    first_preparation_media_kwargs = first_preparation_config["media_io_kwargs"]

    first_engine_media_kwargs["video"]["num_frames"] = 8
    assert first_preparation_media_kwargs["video"]["num_frames"] == 32

    first_preparation_media_kwargs["video"]["num_frames"] = 16
    assert first_engine_media_kwargs["video"]["num_frames"] == 8

    build_vllm_transform(
        config,
        preprocess=lambda row: row,
        postprocess=lambda row: row,
    )

    second_engine_config = captured_engine_configs[1]
    second_preparation_config = second_engine_config["prepare_multimodal_stage"].model_config_kwargs
    assert config.engine_kwargs["media_io_kwargs"]["video"]["num_frames"] == 32
    assert second_engine_config["engine_kwargs"]["media_io_kwargs"]["video"]["num_frames"] == 32
    assert second_preparation_config["media_io_kwargs"]["video"]["num_frames"] == 32


def test_build_propagates_vllm_model_compatibility_errors(
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Preserve the engine's native incompatibility diagnostic."""

    def incompatible_config(**_kwargs: Any) -> None:  # noqa: ANN401 - mirrors pydantic kwargs
        msg = "registered artifact is incompatible with this vLLM engine"
        raise RuntimeError(msg)

    fake_llm = types.ModuleType("ray.data.llm")
    fake_llm.PrepareMultimodalStageConfig = object
    fake_llm.ChatTemplateStageConfig = _unasserted_stage_config
    fake_llm.TokenizerStageConfig = _unasserted_stage_config
    fake_llm.vLLMEngineProcessorConfig = incompatible_config
    fake_llm.build_processor = pytest.fail
    monkeypatch.setitem(sys.modules, "ray.data.llm", fake_llm)

    with pytest.raises(RuntimeError, match="registered artifact is incompatible"):
        build_vllm_transform(
            make_config(),
            preprocess=lambda row: row,
            postprocess=lambda row: row,
        )


def test_build_hands_the_replica_range_to_the_engine_pool_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Both bounds reach Ray Data as given, so element 0 is the pool's floor.

    Two layers describe this one pair - the caller that derives it from the GPUs
    it can see, and this facade that reports the floor as a reservation - and
    both accounts are only true if the range arrives unrewritten. A build that
    lowered the lower bound to let the pool ramp, or raised it to fill a
    cluster, would leave each layer stating a GPU count the other does not hold.
    """
    captured: list[dict[str, Any]] = []

    def capture_engine_config(**kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - mirrors pydantic kwargs
        captured.append(kwargs)
        return kwargs

    fake_llm = types.ModuleType("ray.data.llm")
    fake_llm.PrepareMultimodalStageConfig = object
    fake_llm.ChatTemplateStageConfig = _unasserted_stage_config
    fake_llm.TokenizerStageConfig = _unasserted_stage_config
    fake_llm.vLLMEngineProcessorConfig = capture_engine_config
    fake_llm.build_processor = lambda _config, **_kwargs: _StubProcessor(
        {"vLLMEngineStage": _StubStage({"concurrency": 1})}
    )
    monkeypatch.setitem(sys.modules, "ray.data.llm", fake_llm)

    build_vllm_transform(
        # Distinct from each other and from the (1, 1) config default, so neither
        # bound can be satisfied by the default nor by the other's value.
        make_config(concurrency=(3, 7)),
        preprocess=lambda row: row,
        postprocess=lambda row: row,
    )

    assert captured[0]["concurrency"] == (3, 7)


def test_actor_pool_strategy_compute_is_recognized_as_actor_backed() -> None:
    """The pinned processor expresses its pool as an ActorPoolStrategy."""
    assert is_actor_backed({"compute": ActorPoolStrategy(size=2)})


def test_bare_concurrency_is_recognized_as_actor_backed() -> None:
    """A class-shaped UDF with only a concurrency knob still runs on actors."""
    assert is_actor_backed({"concurrency": 1})


def test_kwargs_without_a_pool_signal_are_not_actor_backed() -> None:
    """A task-backed stage has no actor restarts to bound."""
    assert not is_actor_backed({"batch_size": 8})


def test_restart_bound_applies_to_every_actor_backed_stage() -> None:
    """A bound covering only the engine leaves the stage that dies unbounded."""
    processor = _StubProcessor(
        {
            "PrepareMultimodalStage": _StubStage({"compute": ActorPoolStrategy(size=1)}),
            "vLLMEngineStage": _StubStage({"compute": ActorPoolStrategy(size=1)}),
        }
    )

    bounded = apply_restart_bounds(processor)

    assert bounded == ("PrepareMultimodalStage", "vLLMEngineStage")
    for name in bounded:
        kwargs = processor.get_stage_by_name(name).map_batches_kwargs
        assert kwargs["max_restarts"] == 5
        assert kwargs["max_task_retries"] == 5


def test_restart_bound_preserves_the_stage_scheduling_kwargs() -> None:
    """Injecting a bound must not drop the resource knobs the stage already had."""
    processor = _StubProcessor({"vLLMEngineStage": _StubStage({"concurrency": 1, "num_gpus": 1, "batch_size": 32})})

    apply_restart_bounds(processor)

    kwargs = processor.get_stage_by_name("vLLMEngineStage").map_batches_kwargs
    assert kwargs["num_gpus"] == 1
    assert kwargs["batch_size"] == 32


def test_restart_bound_leaves_a_task_backed_stage_alone() -> None:
    """A stage with no actor pool is reported as unbounded by neither list."""
    processor = _StubProcessor(
        {
            "TaskStage": _StubStage({"batch_size": 8}),
            "vLLMEngineStage": _StubStage({"concurrency": 1}),
        }
    )

    assert apply_restart_bounds(processor) == ("vLLMEngineStage",)
    assert "max_restarts" not in processor.get_stage_by_name("TaskStage").map_batches_kwargs


def test_restart_bound_rejects_a_stage_that_ignores_the_assignment() -> None:
    """Break the injection mechanism and the build must refuse, not proceed."""
    processor = _StubProcessor({"vLLMEngineStage": _WriteResistantStage({"concurrency": 1})})

    with pytest.raises(ValueError, match="did not accept a restart bound"):
        apply_restart_bounds(processor)


def test_restart_bound_rejects_a_processor_with_no_actor_backed_stage() -> None:
    """An injection that could not match any stage is a stale layout, not a pass."""
    processor = _StubProcessor({"TaskStage": _StubStage({"batch_size": 8})})

    with pytest.raises(ValueError, match="no actor-backed stage found"):
        apply_restart_bounds(processor)


def test_restart_bound_rejects_an_empty_stage_list() -> None:
    """A renamed accessor surfaces here as a processor that exposes no stages."""
    with pytest.raises(ValueError, match="no actor-backed stage found"):
        apply_restart_bounds(_StubProcessor({}))
