# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure the robot-action-caption vLLM inference transform.

This module owns the recipe's engine-facing knobs, processor construction,
horizontal replication, batching, and actor restart bounds. ``inference.py``
supplies the row-shaped request and canonicalization hooks. Checkpoint identity
and installation remain with the recipe datasource and Ray runtime.

::

  VllmProcessorConfig
        |
        v
  build_vllm_transform(config, preprocess, postprocess)
        |
        +--> vLLMEngineProcessorConfig --> build_processor
        |                                       |
        |                                       v
        +--> apply_restart_bounds(processor) --> every actor stage bounded
                                                |
                                                v
                                       Callable[[Dataset], Dataset]
"""

import copy
import time
import types
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol

import attrs
from loguru import logger
from ray.data import ActorPoolStrategy, Dataset

from cosmos_curator.next.recipes.robot_action_caption.contracts import CHAT_TEMPLATE_KWARGS

# Five restarts AFTER the initial actor - six lifetimes in total - and one task
# retry per permitted lifetime. Ray Data's ActorPoolMapOperator defaults
# ``max_restarts`` to -1 when a caller leaves it unset, which is an unbounded
# loop: a deterministic failure (weights that do not fit the GPU, a corrupt
# checkpoint) re-pays the whole model load forever while holding the allocation.
# A bound turns that into a job failure an operator sees.
#
# Exported as ONE mapping because the two keys are one decision. Ray Data forces
# ``max_task_retries`` to -1 whenever ``max_restarts`` is non-zero and that key
# is absent, so bounding the restarts alone leaves a task retrying forever
# against a pool with no replacements left to land on - a stage that neither
# progresses nor raises while holding its whole reservation. Anything adding an
# actor-backed stage spreads this mapping rather than picking a key from it.
#
# ``max_restarts`` must stay NON-ZERO, and zero is not available as a way to
# silence Ray's actor-restart warning about constructor arguments in the object
# store (https://github.com/ray-project/ray/issues/53727). That warning fires on
# every actor-backed Ray Data stage: the operator hands each worker its
# serialized transformer and DataContext as object references, so the condition
# is met whatever a caller's own constructor arguments weigh. Zero would remove
# the warning by removing in-place restarts, which Ray Data's actor pool assumes
# it has - the pool asserts a live actor is either alive or restarting, and
# recovers a dead worker no other way.
ACTOR_RESTART_BOUNDS: Final[Mapping[str, Any]] = types.MappingProxyType({"max_restarts": 5, "max_task_retries": 5})

# Ray Data owns horizontal replication through ``concurrency``. Letting engine
# data parallelism fan out beside it puts two schedulers on the same GPUs.
# Tensor and pipeline parallelism instead determine one replica's width.
_REJECTED_ENGINE_PARALLELISM: Final = frozenset({"data_parallel_size"})

_DEFAULT_BATCH_SIZE: Final = 32

# Matches the pinned vLLMEngineProcessorConfig default. Restated rather than left
# unpassed so a Ray upgrade cannot change a run's in-flight batch shape silently.
_DEFAULT_MAX_CONCURRENT_BATCHES: Final = 8

# The engine argument the media preparation stage and the engine itself must both
# be built from. Named here because this module copies it from one to the other;
# what goes IN it (frame budgets, decode backends) is owned by the recipe config.
_MEDIA_IO_KWARGS: Final = "media_io_kwargs"

# Copies of ONE batch's media a stage holds at its peak. Two for a stage that
# only forwards the payload: it holds the unpickled input while pickle builds
# the output buffer beside it. Every hand-off between these stages pays that
# round trip, because Arrow has no representation for an ndarray nested inside
# a struct, so the column falls back to pickle at each boundary. Three where the
# stage also decodes, which adds the resized output beside the decoded frames.
_FORWARDED_MEDIA_COPIES: Final = 2
_DECODED_MEDIA_COPIES: Final = 3

# Batches Ray Data keeps resident per actor, which multiplies the copies above.
# NOT a knob this layer owns, and the reason the reservation must restate it:
# ActorPoolMapOperator resolves the depth as the compute strategy's own value,
# else the DataContext's, else max_actor_concurrency * 2. ray.data.llm's
# build_cpu_stage_map_kwargs supplies none of the three and sets no
# max_concurrency, so a CPU stage lands on the default 1 * 2. Sizing a
# reservation for one batch therefore under-declares every such stage by half,
# and `memory` is a placement hint Ray never enforces, so the pool is admitted
# at twice the density its working set fits and the node OOMs rather than
# raising. The 2 is env_integer-backed
# (RAY_DATA_ACTOR_DEFAULT_MAX_TASKS_IN_FLIGHT_TO_MAX_CONCURRENCY_FACTOR), so it is
# a default rather than an invariant and a raised override under-declares in the
# direction that OOMs. Re-measure on a Ray upgrade or an override: only the
# factor can move, not the shape.
_BATCHES_IN_FLIGHT_PER_ACTOR: Final = 2

# Resident bytes an actor holds whatever the batch size: interpreter, tokenizer,
# and for the preparation stage the vLLM ModelConfig it builds for itself. A
# purely per-batch formula omits this and so scales to zero on a small batch,
# where the fixed cost is most of the footprint. Measured 2.45 GB on a settled
# chat-template actor between batches, rounded up for the heavier decode stage.
_ACTOR_RESIDENT_BYTES: Final = 3 * 1024 * 1024 * 1024

# Rows one worker's hook completes between timing summaries. Chosen so a run
# produces tens of lines per worker rather than one per row, which at the batch
# sizes this recipe uses would be hundreds of thousands of lines.
_HOOK_REPORT_EVERY_ROWS: Final = 2000


def _copied_mapping(
    value: object,
    field_name: str,
) -> dict[str, Any]:
    """Validate and deep-copy a mapping into caller-independent storage."""
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be a mapping, got {type(value).__name__}: {value!r}"
        raise TypeError(msg)
    return copy.deepcopy(dict(value))


def _copied_engine_kwargs(value: object) -> dict[str, Any]:
    """Validate and copy engine keyword arguments."""
    return _copied_mapping(value, "engine_kwargs")


def _optional_copied_runtime_env(value: object) -> dict[str, Any] | None:
    """Validate and deep-copy an optional mapping, preserving ``None``."""
    return None if value is None else _copied_mapping(value, "runtime_env")


def _validate_nonblank_string(
    _instance: object,
    attribute: "attrs.Attribute[Any]",
    value: object,
) -> None:
    """Reject non-string and blank field values."""
    if not isinstance(value, str):
        msg = f"{attribute.name} must be a str, got {type(value).__name__}: {value!r}"
        raise TypeError(msg)
    if not value.strip():
        msg = f"{attribute.name} must be a nonblank string, got {value!r}"
        raise ValueError(msg)


def _validate_strict_positive_int(
    _instance: object,
    attribute: "attrs.Attribute[Any]",
    value: object,
) -> None:
    """Reject non-integer and non-positive field values."""
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{attribute.name} must be an int, got {type(value).__name__}: {value!r}"
        raise TypeError(msg)
    if value < 1:
        msg = f"{attribute.name} must be at least 1, got {value}"
        raise ValueError(msg)


def _validate_concurrency(_instance: object, attribute: "attrs.Attribute[Any]", value: object) -> None:
    """Accept an ``(initial, ceiling)`` autoscaling range.

    Raises:
        TypeError: If the value is not a 2-tuple of ints.
        ValueError: If the initial count is below one, or the range is inverted.

    """
    # A bare bool needs no special case here even though it is an int subclass:
    # the shape check below rejects every non-tuple, bools included.
    if not (isinstance(value, tuple) and len(value) == 2):  # noqa: PLR2004 - (initial, ceiling)
        msg = f"{attribute.name} must be a 2-tuple of ints, got {value!r}"
        raise TypeError(msg)
    initial, ceiling = value
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        msg = f"{attribute.name} range must hold two ints, got {value!r}"
        raise TypeError(msg)
    if initial < 1:
        msg = f"{attribute.name} initial replica count must be at least 1, got {initial}"
        raise ValueError(msg)
    if initial > ceiling:
        msg = f"{attribute.name} range must be (initial <= ceiling), got {value!r}"
        raise ValueError(msg)


class ProcessorStage(Protocol):
    """One stage of a processor, seen only through its Ray scheduling knobs.

    Structural stand-in for the ``StatefulStage`` model that
    ``ray.data.llm.Processor.get_stage_by_name`` returns, as pinned at Ray
    2.58.0. Ray keeps that class internal to
    ``ray.llm._internal.batch.stages.base``, so re-verify this shape on an
    upgrade rather than trusting it to have survived.
    """

    map_batches_kwargs: dict[str, Any]


class StagedProcessor(Protocol):
    """A processor whose stages can be enumerated and re-configured by name.

    Structural stand-in for ``ray.data.llm.Processor`` as pinned at Ray 2.58.0.
    Declared rather than imported so this module costs nothing to import; both
    accessors are public there, so re-verify them on an upgrade.
    """

    def list_stage_names(self) -> list[str]:
        """Return the stage names in execution order."""
        ...

    def get_stage_by_name(self, name: str) -> ProcessorStage:
        """Return one stage by name."""
        ...


@attrs.frozen
class StageHeapReservations:
    """Per-worker heap reservations for the media-carrying CPU stages.

    Named fields rather than an ordered pair because the two are the same type
    and are told apart only by role. An ordered range validates its own order,
    so a transposition is loud; a role pair does not, and handing the
    preparation stage the smaller figure re-creates the under-declaration these
    reservations exist to prevent -- silently, and visible only as an OOM kill.

    Attributes:
        preparation: Bytes declared by the stage that decodes the media.
        forwarding: Bytes declared by a stage that only carries the decoded
            column onward, which holds fewer copies and so peaks below the
            preparation stage.

    """

    preparation: float
    forwarding: float

    def __attrs_post_init__(self) -> None:
        """Reject a pair whose preparation figure does not exceed forwarding."""
        # Equal or inverted figures mean both roles were derived from one copy
        # count, which is the defect rather than a tuning choice: it either
        # strands budget on the two cheap stages or overcommits the node on the
        # expensive one. Raised here so no caller has to remember the ordering.
        if self.preparation <= self.forwarding:
            msg = (
                f"preparation must reserve more than forwarding, got "
                f"preparation={self.preparation} forwarding={self.forwarding}"
            )
            raise ValueError(msg)


@attrs.frozen
class VllmProcessorConfig:
    """Engine-facing configuration for one vLLM inference transform.

    Crosses a Ray boundary, so every field is picklable. Common knobs are typed;
    everything else goes through ``engine_kwargs`` rather than growing one field
    per vLLM argument.

    Attributes:
        model: Model source the engine loads (repository id or local directory).
        max_model_len: Context length the engine is built with.
        tensor_parallel_size: Tensor-parallel factor within one engine replica.
        pipeline_parallel_size: Pipeline-parallel factor within one engine
            replica.
        concurrency: Engine replicas Ray Data may run, as ``(initial, ceiling)``:
            Ray starts at ``initial`` and autoscales up to ``ceiling``, so a
            cluster that gains capacity mid-run can still be used. Pass equal
            values to pin the pool to a fixed size.
        batch_size: Rows handed to the engine stage per call.
        max_concurrent_batches: In-flight Ray batch calls per replica, so a
            replica is never idle between calls. Costs host memory, not GPUs.
        prepare_multimodal: Whether to run the processor's media preparation
            stage. Any request whose message content references media rather than
            carrying a decoded payload needs it; a text-only request does not.
        media_bytes_per_row: Largest decoded media payload one row can carry,
            which sizes the heap reservations of the stages that carry it -- a
            larger one where the stage also decodes than where it only forwards
            the result. Required with ``prepare_multimodal``, and rejected
            without it. Pass a ceiling rather than an average: a reservation
            admitting the mean row is exceeded by every row above it.
        engine_kwargs: Additional engine arguments, validated at construction.
            Consumers take a copy through ``engine_kwargs_payload`` rather than
            reading this mapping directly.
        runtime_env: Ray runtime environment for the engine actors, copied
            before it is handed to the processor.

    """

    model: str = attrs.field(validator=_validate_nonblank_string)
    max_model_len: int = attrs.field(validator=_validate_strict_positive_int)
    tensor_parallel_size: int = attrs.field(default=1, validator=_validate_strict_positive_int)
    pipeline_parallel_size: int = attrs.field(default=1, validator=_validate_strict_positive_int)
    concurrency: tuple[int, int] = attrs.field(default=(1, 1), validator=_validate_concurrency)
    batch_size: int = attrs.field(default=_DEFAULT_BATCH_SIZE, validator=_validate_strict_positive_int)
    max_concurrent_batches: int = attrs.field(
        default=_DEFAULT_MAX_CONCURRENT_BATCHES,
        validator=_validate_strict_positive_int,
    )
    # Deliberately has no default, unlike every other optional field here. This
    # layer cannot derive the value: ``preprocess`` is an opaque row callable, so
    # nothing at construction time can see whether the rows it emits will
    # reference media. Defaulting to False (the pinned processor's own default)
    # would make forgetting it produce a request whose prompt carries the model's
    # vision placeholder tokens and no media - which no layer of the stack
    # rejects, so the engine generates fluent text describing nothing instead of
    # failing. Requiring the answer is what keeps that outcome unreachable by
    # omission.
    prepare_multimodal: bool = attrs.field(kw_only=True, validator=attrs.validators.instance_of(bool))
    # Paired with ``prepare_multimodal`` rather than defaulted, for that field's
    # own reason: only the caller knows the frame budget and source resolution
    # its rows decode to. A number invented here would be one corpus's
    # measurement applied to every later one, and since understating it is what
    # gets a node OOM-killed, there is no safe generic guess to fall back on.
    media_bytes_per_row: int | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(_validate_strict_positive_int),
    )
    engine_kwargs: dict[str, Any] = attrs.field(
        factory=dict,
        converter=_copied_engine_kwargs,
    )
    runtime_env: dict[str, Any] | None = attrs.field(
        default=None,
        converter=_optional_copied_runtime_env,
    )

    def __attrs_post_init__(self) -> None:
        """Reject unusable media budgets and conflicting engine arguments.

        Raises:
            ValueError: If the media budget and ``prepare_multimodal`` disagree,
                if ``engine_kwargs`` names a typed field of this class, or if it
                requests engine-side data parallelism.

        """
        # Enforced in both directions. Missing, the carrying stages silently keep
        # Ray's text-sized footprint, which is the packing error this field
        # exists to correct. Present without media preparation, it is a number
        # that reads as tuned and reserves nothing, since no stage carries a
        # payload for it to describe.
        if self.prepare_multimodal and self.media_bytes_per_row is None:
            msg = (
                "prepare_multimodal=True requires media_bytes_per_row: the stages between "
                "preparation and the engine carry the decoded payload, and Ray would otherwise "
                "schedule them on its text-sized default footprint"
            )
            raise ValueError(msg)
        if not self.prepare_multimodal and self.media_bytes_per_row is not None:
            msg = (
                "media_bytes_per_row has no effect without prepare_multimodal=True, because no "
                "stage then carries decoded media; drop it or enable media preparation"
            )
            raise ValueError(msg)
        # Derived from the class rather than a hand-kept list, so a new typed
        # field is reserved automatically. Two sources of truth for a value like
        # max_model_len surfaces as an engine crash an hour into a run.
        typed_fields = {field.name for field in attrs.fields(type(self))}
        shadowed = sorted(typed_fields & self.engine_kwargs.keys())
        if shadowed:
            msg = (
                f"engine_kwargs must not duplicate typed VllmProcessorConfig fields {shadowed}; "
                "set them as config fields instead"
            )
            raise ValueError(msg)
        fanout = sorted(_REJECTED_ENGINE_PARALLELISM & self.engine_kwargs.keys())
        if fanout:
            msg = (
                f"engine_kwargs must not set {fanout}; Ray Data owns replication through "
                "concurrency, so the engine must not replicate beside it"
            )
            raise ValueError(msg)

    def engine_kwargs_payload(self) -> dict[str, Any]:
        """Return the full engine argument mapping, typed fields merged in.

        Returns:
            A fresh deep copy on every call.

        """
        # Fresh and deep per call because the pinned vLLM stack mutates both the
        # top-level mapping and nested media processor kwargs. Sharing either
        # layer would carry engine state into the next build from this config.
        return {
            **copy.deepcopy(self.engine_kwargs),
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }

    def stage_heap_reservations(self) -> StageHeapReservations | None:
        """Return the per-worker heap reservations for the media-carrying stages.

        Returns:
            The figures each stage role declares to the scheduler, or ``None``
            when no stage carries media and Ray's own default footprint applies
            to all of them. Ray treats these as logical resources for placement,
            not enforced caps, so exceeding one overcommits the node rather than
            raising.

        """
        # Reads the budget rather than the flag: construction already rejects the
        # two disagreeing, so this stays a single source of truth.
        if self.media_bytes_per_row is None:
            return None
        payload = self.batch_size * self.media_bytes_per_row

        def reservation(copies_held: int) -> float:
            return float(_ACTOR_RESIDENT_BYTES + _BATCHES_IN_FLIGHT_PER_ACTOR * copies_held * payload)

        return StageHeapReservations(
            preparation=reservation(_DECODED_MEDIA_COPIES),
            forwarding=reservation(_FORWARDED_MEDIA_COPIES),
        )


def is_actor_backed(map_batches_kwargs: Mapping[str, Any]) -> bool:
    """Return whether a stage's ``map_batches`` kwargs schedule an actor pool.

    Args:
        map_batches_kwargs: One stage's Ray Data ``map_batches`` keyword
            arguments.

    Returns:
        True when the stage runs on actors and therefore has restarts to bound.

    """
    # Two independent signals because the pin expresses the pool one way and a
    # future version may express it the other: an explicit ActorPoolStrategy in
    # ``compute``, or a bare ``concurrency`` on a class-shaped UDF.
    if isinstance(map_batches_kwargs.get("compute"), ActorPoolStrategy):
        return True
    return map_batches_kwargs.get("concurrency") is not None


def _carries_restart_bound(map_batches_kwargs: Mapping[str, Any]) -> bool:
    """Return whether every bound has the value this module sets."""
    return all(map_batches_kwargs.get(key) == value for key, value in ACTOR_RESTART_BOUNDS.items())


def apply_restart_bounds(processor: StagedProcessor) -> tuple[str, ...]:
    """Bound actor restarts and task retries on every actor-backed stage.

    Bounds every stage rather than only the engine: a decoded-frame OOM lands in
    whichever stage prepares the payload, and a bound covering the engine alone
    leaves the stage that actually dies unbounded.

    Args:
        processor: Processor whose stages are re-configured in place.

    Returns:
        The names of the stages that were bounded, in execution order.

    Raises:
        ValueError: If the processor exposes no actor-backed stage, or if a
            stage still lacks a bound after assignment.

    """
    stage_names = processor.list_stage_names()
    for name in stage_names:
        stage = processor.get_stage_by_name(name)
        if not is_actor_backed(stage.map_batches_kwargs):
            continue
        stage.map_batches_kwargs = {**stage.map_batches_kwargs, **ACTOR_RESTART_BOUNDS}

    # Re-read the stages instead of trusting the loop above. Without this check a
    # renamed accessor, a restructured stage list, or a stage that ignores the
    # assignment turns the loop into a silent no-op, and the missing bound is
    # discovered as an unbounded restart loop in production instead of at build.
    bounded = []
    unbounded = []
    for name in stage_names:
        kwargs = processor.get_stage_by_name(name).map_batches_kwargs
        if not is_actor_backed(kwargs):
            continue
        if _carries_restart_bound(kwargs):
            bounded.append(name)
        else:
            unbounded.append(name)
    if unbounded:
        msg = f"actor-backed stage(s) {unbounded} did not accept a restart bound; refusing to run unbounded"
        raise ValueError(msg)
    if not bounded:
        msg = (
            f"no actor-backed stage found among {stage_names}; the restart-bound injection would be a no-op, "
            "which means the processor stage layout is no longer the one this module understands"
        )
        raise ValueError(msg)
    return tuple(bounded)


def _timed_hook(
    hook: Callable[[dict[str, Any]], dict[str, Any]], label: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a row hook so its cumulative cost is reported periodically.

    These two hooks are the ONLY seam this recipe owns inside the processor's
    stage graph: every other stage is Ray's, runs in its own actor, and emits no
    timing. Without this, a phase that spends its time in request construction
    or result canonicalization is indistinguishable from one that spends it in
    the engine.

    Args:
        hook: The row callable to wrap.
        label: Name used in the diagnostic.

    Returns:
        A callable with the same contract as ``hook``.

    """
    state = {"rows": 0, "seconds": 0.0}

    def timed(row: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            return hook(row)
        finally:
            # In a finally so a raising row still contributes its cost: a hook
            # that fails slowly is exactly the case worth seeing.
            state["seconds"] += time.monotonic() - started
            state["rows"] += 1
            if not state["rows"] % _HOOK_REPORT_EVERY_ROWS:
                logger.info(
                    f"Engine {label} hook after {state['rows']} row(s): {state['seconds']:.1f}s "
                    f"cumulative, {state['seconds'] / state['rows'] * 1000:.2f} ms/row. Wall time in "
                    f"this worker only; Ray runs several of them per stage."
                )

    return timed


def build_vllm_transform(
    config: VllmProcessorConfig,
    preprocess: Callable[[dict[str, Any]], dict[str, Any]],
    postprocess: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[Dataset], Dataset]:
    """Build a bounded vLLM inference transform from row-shaped hooks.

    This module treats both hooks as opaque so request and result semantics
    remain owned by the rest of the recipe.

    Args:
        config: Engine-facing configuration.
        preprocess: Maps one input row onto the engine's expected keys.
        postprocess: Maps one generated row onto the caller's output keys.

    Returns:
        A callable applicable to a ``Dataset``.

    """
    # Kept inside the function: ``ray.data.llm`` re-exports
    # ``vLLMEngineProcessorConfig`` from the PEP 562-lazy
    # ``ray.llm._internal.batch.processor``, and that re-export is the attribute
    # access that loads ``processor/vllm_engine_proc.py``, whose top-level
    # ``import transformers`` would otherwise land on every CLI and config layer
    # that merely builds a ``VllmProcessorConfig``.
    from ray.data.llm import (  # noqa: PLC0415 - heavyweight engine stack
        ChatTemplateStageConfig,
        PrepareMultimodalStageConfig,
        TokenizerStageConfig,
        build_processor,
        vLLMEngineProcessorConfig,
    )

    engine_kwargs = config.engine_kwargs_payload()
    # Two figures, not one. The preparation stage decodes and so peaks above the
    # two that only carry the decoded column onward; giving all three the larger
    # figure would strand budget on the cheap stages, and all three the smaller
    # would overcommit the node on the expensive one. None throughout under a
    # text-only config, which leaves every stage on Ray's own footprint.
    reservations = config.stage_heap_reservations()
    processor_config = vLLMEngineProcessorConfig(
        model_source=config.model,
        batch_size=config.batch_size,
        # Pass the validated (initial, ceiling) tuple through unchanged so this
        # facade and Ray share one interpretation of the autoscaling bounds.
        concurrency=config.concurrency,
        max_concurrent_batches=config.max_concurrent_batches,
        # Copied for the same reason the engine kwargs are: the processor may
        # mutate what it is handed, and the config outlives one build.
        runtime_env=copy.deepcopy(config.runtime_env),
        # The preparation stage builds its own vLLM ModelConfig, so its media
        # arguments come from the engine mapping rather than a second caller
        # setting. It still receives a separate deep copy because both vLLM
        # consumers may mutate nested processor kwargs while they initialize.
        prepare_multimodal_stage=(
            PrepareMultimodalStageConfig(
                enabled=True,
                memory=reservations.preparation if reservations else None,
                model_config_kwargs={
                    _MEDIA_IO_KWARGS: copy.deepcopy(engine_kwargs.get(_MEDIA_IO_KWARGS, {})),
                    # Passed so the preparation stage budgets against the SAME
                    # context length the engine admits. Left out, it builds its
                    # ModelConfig on the checkpoint's native maximum instead -
                    # measured at 262,144 against an engine configured for
                    # 32,768 - so the two halves of one request disagree about
                    # the window they are sizing for, and only the engine's
                    # limit is ever enforced.
                    "max_model_len": config.max_model_len,
                },
            )
            if config.prepare_multimodal
            else False
        ),
        # Declared here only to carry the reservation: both stages are enabled by
        # default under this pin, and the resolver still fills batch_size,
        # concurrency, runtime_env and model_source from the processor-level
        # settings above. It does NOT propagate ``memory``, which is why each
        # stage that holds the media has to name it.
        #
        # Detokenize is absent on purpose. It runs downstream of the engine,
        # which has already replaced the media with generated text, so its rows
        # are text-sized and Ray's default footprint is the right one.
        chat_template_stage=ChatTemplateStageConfig(
            enabled=True,
            memory=reservations.forwarding if reservations else None,
            chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS),
        ),
        tokenize_stage=TokenizerStageConfig(
            enabled=True,
            memory=reservations.forwarding if reservations else None,
        ),
        engine_kwargs=engine_kwargs,
        # Passed explicitly even though it matches the pinned default: True makes
        # the engine stage swallow request failures and continue, which is right
        # for one unusable input row and wrong for a dead engine. Callers that
        # need per-row error rows pre-seed them in ``preprocess`` instead,
        # because Ray bypasses ``postprocess`` for a failed row entirely.
        #
        # The consequence is that under this pin the stage RE-RAISES a per-row
        # failure and never emits an `__inference_error__` row of its own, so an
        # engine fault ends the run. That is the intended trade, and what makes
        # it affordable is a caller-installed row checkpoint bounding the loss to
        # the rows not yet committed. A caller that has no checkpoint pays a full
        # re-run per fault; flipping this flag to buy that back instead would
        # publish a corrupt engine's output as ordinary per-row data.
        should_continue_on_error=False,
    )
    processor = build_processor(
        processor_config,
        preprocess=_timed_hook(preprocess, "preprocess"),
        postprocess=_timed_hook(postprocess, "postprocess"),
    )
    bounded = apply_restart_bounds(processor)
    # ``min_size`` is a FLOOR, not a starting point: Ray Data never scales an
    # actor pool below it, so those GPUs are reserved for the whole run whether
    # or not work ever reaches them. Spelled out in GPU terms because that is the
    # figure an idle-GPU reaper reports back, and reconstructing it from a
    # replica count and a parallel size is exactly the arithmetic this line
    # exists to spare an operator.
    floor, ceiling = config.concurrency
    gpus_per_replica = config.tensor_parallel_size * config.pipeline_parallel_size
    logger.info(
        f"Built vLLM transform for {config.model} with an engine pool floored at {floor} replica(s) and "
        f"capped at {ceiling}, each on {gpus_per_replica} GPU(s); the floor pins "
        f"{floor * gpus_per_replica} GPU(s) for the run's duration. Bounded restarts on "
        f"stage(s) {list(bounded)}",
    )
    return processor
