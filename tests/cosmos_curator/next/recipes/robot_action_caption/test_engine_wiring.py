# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for what ``build_vllm_transform`` hands the pinned processor builder.

Every mechanism covered here is silent when it breaks: a media-preparation
stage that is off produces text describing nothing, an engine failure that is
swallowed publishes a dead engine's output as ordinary rows, and media
arguments that reach only the engine let the stage decode something else. None
of them raises, so only an assertion on the objects handed to
``ray.data.llm`` can tell a correct wiring from a plausible one.

``build_processor`` calls ``ray.init`` and downloads tokenizer files, so a
real processor cannot be built without a cluster and a network. The seam used
here is the deferred ``from ray.data.llm import ...`` inside the function: a
stub module standing in for it records the config objects it is handed, and
the assertions read what was recorded.
"""

import sys
import types
from collections.abc import Callable
from typing import Any

import attrs
import pytest
from ray.data import ActorPoolStrategy

from cosmos_curator.next.recipes.robot_action_caption.engine import (
    ACTOR_RESTART_BOUNDS,
    VllmProcessorConfig,
    build_vllm_transform,
)


@attrs.define
class _StubStage:
    """Stage that accepts a kwargs assignment, like the pinned pydantic stage."""

    map_batches_kwargs: dict[str, Any]


@attrs.define
class _StubProcessor:
    """Minimal stand-in exposing the two accessors the injection relies on."""

    stages: dict[str, _StubStage]

    def list_stage_names(self) -> list[str]:
        """Return the stage names in insertion order."""
        return list(self.stages)

    def get_stage_by_name(self, name: str) -> _StubStage:
        """Return one stage by name."""
        return self.stages[name]


@attrs.define
class _RecordedPrepareMultimodalConfig:
    """Stand-in for the pinned media-preparation stage config.

    Mirrors only the fields this layer sets; the real class carries more.
    """

    enabled: bool
    model_config_kwargs: dict[str, Any]
    memory: float | None = None


@attrs.define
class _RecordedCarryingStageConfig:
    """Stand in for a media-carrying processor stage config.

    The chat-template stage also receives template kwargs; the tokenizer leaves
    them unset.
    """

    enabled: bool
    memory: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None


@attrs.define
class _RecordedEngineConfig:
    """Stand-in for the pinned engine config, keeping its keyword arguments."""

    kwargs: dict[str, Any]


@attrs.define
class _RecordedBuild:
    """Everything one ``build_vllm_transform`` call handed to the stub module.

    Attributes:
        engine_config: Keyword arguments the engine config was constructed with.
        processor: The stub processor the builder returned, after the function
            finished with it, so its stage kwargs show any injected bounds.

    """

    engine_config: _RecordedEngineConfig
    processor: _StubProcessor


TransformBuilder = Callable[[VllmProcessorConfig], _RecordedBuild]


def _passthrough(row: dict[str, Any]) -> dict[str, Any]:
    """Return the row unchanged; the hooks' behavior is not under test here."""
    return row


def _actor_backed_stages() -> dict[str, _StubStage]:
    """Return two actor-backed stages named after the pinned stage layout."""
    return {
        "PrepareMultimodalStage": _StubStage({"compute": ActorPoolStrategy(size=1)}),
        "vLLMEngineStage": _StubStage({"compute": ActorPoolStrategy(size=1)}),
    }


@pytest.fixture
def build_transform(monkeypatch: pytest.MonkeyPatch) -> TransformBuilder:
    """Return a factory that builds one transform against a stub ``ray.data.llm``.

    The stub is installed in ``sys.modules`` rather than patched onto the real
    module so nothing imports ``transformers`` or attempts ``vllm``, keeping
    the test on ordinary CPU CI.
    """

    def _build(config: VllmProcessorConfig) -> _RecordedBuild:
        recorded_engine_config: list[_RecordedEngineConfig] = []
        recorded_processor: list[_StubProcessor] = []

        def _engine_config(**kwargs: Any) -> _RecordedEngineConfig:  # noqa: ANN401 -- records whatever it is handed
            engine_config = _RecordedEngineConfig(kwargs=kwargs)
            recorded_engine_config.append(engine_config)
            return engine_config

        def _build_processor(
            engine_config: _RecordedEngineConfig,
            *,
            preprocess: Callable[[dict[str, Any]], dict[str, Any]],
            postprocess: Callable[[dict[str, Any]], dict[str, Any]],
        ) -> _StubProcessor:
            del engine_config, preprocess, postprocess
            processor = _StubProcessor(_actor_backed_stages())
            recorded_processor.append(processor)
            return processor

        module = types.ModuleType("ray.data.llm")
        module.PrepareMultimodalStageConfig = _RecordedPrepareMultimodalConfig  # type: ignore[attr-defined]
        module.ChatTemplateStageConfig = _RecordedCarryingStageConfig  # type: ignore[attr-defined]
        module.TokenizerStageConfig = _RecordedCarryingStageConfig  # type: ignore[attr-defined]
        module.vLLMEngineProcessorConfig = _engine_config  # type: ignore[attr-defined]
        module.build_processor = _build_processor  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ray.data.llm", module)

        build_vllm_transform(config, _passthrough, _passthrough)
        return _RecordedBuild(
            engine_config=recorded_engine_config[0],
            processor=recorded_processor[0],
        )

    return _build


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


def test_transform_enables_the_media_preparation_stage_when_media_is_requested(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """A config asking for media preparation must reach the processor as enabled.

    The pinned processor treats the stage as off by default and nothing else
    populates the media payload, so a request would arrive as text carrying
    vision placeholder tokens and generate text describing nothing.
    """
    recorded = build_transform(make_config(prepare_multimodal=True, media_bytes_per_row=11 * 1024 * 1024))

    stage = recorded.engine_config.kwargs["prepare_multimodal_stage"]
    assert isinstance(stage, _RecordedPrepareMultimodalConfig)
    assert stage.enabled is True


def test_transform_leaves_the_media_preparation_stage_off_for_text_only_rows(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """A text-only config must not pay for a media-preparation stage."""
    recorded = build_transform(make_config(prepare_multimodal=False))

    assert recorded.engine_config.kwargs["prepare_multimodal_stage"] is False


def test_transform_copies_media_io_kwargs_into_the_preparation_stage_config(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """The preparation stage must be built from the engine's own media arguments.

    It constructs its own model config, so an uncopied argument lets it decode
    media the engine was not configured for, with neither side wrong alone.

    Asserted as the WHOLE mapping rather than one key, so a stage argument added
    here without a test cannot arrive unnoticed. The context length is the
    other member today, and the test below is the one that pins its value.
    """
    media_io_kwargs = {"video": {"num_frames": 8}}
    recorded = build_transform(
        make_config(
            prepare_multimodal=True,
            media_bytes_per_row=11 * 1024 * 1024,
            engine_kwargs={"media_io_kwargs": media_io_kwargs},
        ),
    )

    stage = recorded.engine_config.kwargs["prepare_multimodal_stage"]
    assert stage.model_config_kwargs == {"media_io_kwargs": media_io_kwargs, "max_model_len": 4096}


def test_the_preparation_stage_is_built_with_the_engines_own_context_length(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """The preparation stage must budget against the window the engine admits.

    Left unpassed, the stage builds its ModelConfig on the checkpoint's native
    maximum instead, and the two halves of one request size for different
    windows while only the engine's limit is ever enforced. Nothing raises: the
    disagreement is visible solely as the stage logging a context length the
    engine never uses.

    The configured value is not a power of two and matches no default on the
    path -- neither this fixture's 4096, nor the recipe's shipped 32768, nor the
    262144 a Qwen3-VL checkpoint reports -- so a build that hardcoded any of
    them, or read the checkpoint instead of the config, cannot satisfy it.
    """
    configured_context = 20_481

    recorded = build_transform(
        make_config(
            prepare_multimodal=True,
            max_model_len=configured_context,
            media_bytes_per_row=11 * 1024 * 1024,
        ),
    )

    stage = recorded.engine_config.kwargs["prepare_multimodal_stage"]
    assert stage.model_config_kwargs["max_model_len"] == configured_context
    # The engine half of the same number, so the test fails if the two are ever
    # sourced separately -- agreeing is the entire point of passing it.
    assert recorded.engine_config.kwargs["engine_kwargs"]["max_model_len"] == configured_context


def test_transform_pins_an_engine_side_failure_as_fatal_to_the_run(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """Continuing past an engine failure would publish a dead engine's output."""
    recorded = build_transform(make_config())

    assert recorded.engine_config.kwargs["should_continue_on_error"] is False


def test_transform_disables_thinking_in_the_chat_template(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """Keep reasoning traces out of the generated caption text."""
    recorded = build_transform(make_config())

    stage = recorded.engine_config.kwargs["chat_template_stage"]
    assert stage.chat_template_kwargs == {"enable_thinking": False}


def test_media_reservation_reaches_every_stage_that_carries_the_payload(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """All three stages between decode and the engine must carry the reservation.

    Ray's resolver copies batch_size, concurrency, runtime_env and model_source
    from the processor level but not ``memory``, so a stage this layer forgets
    silently keeps Ray's generic per-actor footprint and is packed as though it
    held a prompt string rather than a batch of decoded video.
    """
    config = make_config(prepare_multimodal=True, batch_size=7, media_bytes_per_row=11 * 1024 * 1024)
    reservations = config.stage_heap_reservations()
    assert reservations is not None, "fixture must request media, or this test asserts nothing"

    kwargs = build_transform(config).engine_config.kwargs

    # Asserted per stage rather than against one shared figure: the preparation
    # stage decodes and the other two only carry the result, so a build handing
    # any of them the wrong one of the two is the defect this pins. The carrier
    # guarantees the two figures differ, so a swap cannot pass silently.
    assert kwargs["prepare_multimodal_stage"].memory == reservations.preparation
    assert kwargs["chat_template_stage"].memory == reservations.forwarding
    assert kwargs["tokenize_stage"].memory == reservations.forwarding


def test_text_only_rows_leave_the_carrying_stages_on_rays_own_footprint(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """Without media preparation no stage carries a payload worth reserving for.

    Sits on the other side of the comparison the reservation is keyed on, so an
    implementation that reserved unconditionally would inflate every text-only
    stage and land fewer actors per node for nothing.
    """
    config = make_config(prepare_multimodal=False, batch_size=7)
    assert config.stage_heap_reservations() is None

    kwargs = build_transform(config).engine_config.kwargs

    assert kwargs["chat_template_stage"].memory is None
    assert kwargs["tokenize_stage"].memory is None


def test_requesting_media_without_a_budget_is_rejected(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """Media preparation without a budget would leave the carrying stages unsized.

    The omission has no symptom at build time and no error at runtime; it simply
    packs actors on a text-sized footprint, so construction is the only place it
    can be caught.
    """
    with pytest.raises(ValueError, match="requires media_bytes_per_row"):
        make_config(prepare_multimodal=True)


def test_a_budget_without_media_preparation_is_rejected(
    make_config: Callable[..., VllmProcessorConfig],
) -> None:
    """A budget no stage can use reads as tuned while reserving nothing."""
    with pytest.raises(ValueError, match="no effect without prepare_multimodal"):
        make_config(prepare_multimodal=False, media_bytes_per_row=11 * 1024 * 1024)


def test_the_restart_bounds_are_finite_and_non_zero() -> None:
    """Both bounds are pinned to literals, because every other assertion is self-comparing.

    The tests that check the bounds reach a stage read the expected pair from
    ``ACTOR_RESTART_BOUNDS`` itself, so widening the constant to Ray Data's own
    unbounded defaults leaves them green while restoring the infinite restart
    loop the constant exists to prevent. Literals are what turn that widening
    red.

    The lower bound matters as much as the upper one. Zero is what a reader
    reaching for Ray's actor-restart warning about object-store constructor
    arguments would try, and it silences that warning by disabling the in-place
    restarts Ray Data's actor pool has no alternative to.
    """
    assert ACTOR_RESTART_BOUNDS["max_restarts"] == 5
    assert ACTOR_RESTART_BOUNDS["max_task_retries"] == 5


def test_transform_bounds_actor_restarts_on_every_stage_the_processor_created(
    make_config: Callable[..., VllmProcessorConfig],
    build_transform: TransformBuilder,
) -> None:
    """An unbounded actor-backed stage re-pays the model load forever on a fault.

    Reads the expected pair from the module rather than restating it, so the
    values stay owned by the layer that injects them.
    """
    recorded = build_transform(make_config(prepare_multimodal=True, media_bytes_per_row=11 * 1024 * 1024))

    for name in recorded.processor.list_stage_names():
        kwargs = recorded.processor.get_stage_by_name(name).map_batches_kwargs
        for key, value in ACTOR_RESTART_BOUNDS.items():
            assert kwargs[key] == value
