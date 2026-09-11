# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Phase A as one checkpointed Read-to-Parquet plan.

The terminal write is both Phase A's only durable act and Ray's checkpoint
commit point. Completed stochastic captions are recovered from staged rows,
never regenerated, and this phase never mutates Lance. Engine and IO geometry
come from the execution config so deployments can tune throughput without
changing caption identity.

See ``docs/curator/design/curator-next-robot-action-caption.md`` for the
operator flow, two-phase checkpoint contract, and artifact rationale.
"""

import math
from collections.abc import Sequence
from functools import partial
from typing import Any, Final

import attrs
import lance
import ray
from loguru import logger
from ray.data import ActorPoolStrategy, SaveMode

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.models.registry import RegisteredModel
from cosmos_curator.next.core.ray_runtime import installed_checkpoint
from cosmos_curator.next.recipes.robot_action_caption.config import (
    DEFAULT_MEDIA_FETCH_WORKERS,
    ResolvedRobotActionCaptionConfig,
    RobotActionCaptionExecutionConfig,
    RobotActionCaptionGenerationConfig,
    RobotActionCaptionVideoConfig,
)
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    PREPARE_MULTIMODAL,
    VIDEO_ITEMS_PER_PROMPT,
)
from cosmos_curator.next.recipes.robot_action_caption.datasource import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    CaptionScanDatasource,
    CaptionScanPlan,
    PinnedScan,
)
from cosmos_curator.next.recipes.robot_action_caption.engine import (
    ACTOR_RESTART_BOUNDS,
    VllmProcessorConfig,
    build_vllm_transform,
)
from cosmos_curator.next.recipes.robot_action_caption.media import ClipMediaFetch
from cosmos_curator.next.recipes.robot_action_caption.prompt import PromptContextField
from cosmos_curator.next.recipes.robot_action_caption.request import (
    build_request,
    canonicalize,
    read_generation,
)
from cosmos_curator.next.recipes.robot_action_caption.workspace import CaptionWorkspace

# The loader hands over uint8 RGB, so one pixel costs three bytes in the payload
# the media-carrying stages queue. This is the payload format, not a tuning knob.
_BYTES_PER_DECODED_PIXEL: Final = 3

# Reporting units only: a per-row payload reads naturally in MiB and a per-worker
# reservation in GiB, and mixing them in one line is what makes it scannable.
# These constants format diagnostics; they do not alter execution geometry.
_BYTES_PER_MIB: Final = 1024 * 1024
_BYTES_PER_GIB: Final = 1024 * _BYTES_PER_MIB

# VideoMediaIO's own ``num_frames`` default, which it hands to whichever backend
# resolves. Restated here because the payload ceiling depends on it and nothing
# in this repo would look different if a vLLM release moved it. It bounds the
# payload only for a backend that HONOURS num_frames; one deriving its count from
# fps and duration ignores it, and the derived reservation is then too small
# rather than merely imprecise. Configuring the temporal group avoids the
# question entirely, which is why taking this path warns. It is not configurable:
# an override here would size a reservation without changing the backend default.
_LOADER_DEFAULT_NUM_FRAMES: Final = 32


def _media_fetch_resources(fetch_workers: int) -> dict[str, float]:
    """Return the logical IO reservation for one media actor.

    ONE slot per concurrent READ, which is what the node-advertised
    ``curator_io`` budget means: a node publishing N slots sustains N concurrent
    source reads, and this reservation is how an actor claims its share. The
    budget is therefore PARTITIONED by ``execution.media_fetch_workers``, not
    enlarged by it - four workers per actor buy a quarter as many actors for the
    same read capacity, trading processes and their storage clients for threads.
    Reserving less than the actor actually runs would make the node's published
    figure a fiction and silently multiply the load on the object store, and the
    same convention is relied on by ``video_split`` and ``video_caption``.

    Raising total capacity is a different action, taken at node registration
    (``--ray-io-slots-per-node``, see ``ray_runtime.curator_io_resources``).

    Args:
        fetch_workers: Clips one actor reads at once.

    Returns:
        The custom-resource mapping for the media stage.

    """
    return {environment.CURATOR_IO_RESOURCE_NAME: float(fetch_workers)}


def _media_fetch_cpus(fetch_workers: int, *, cpus_per_fetch: float) -> float:
    """Return the CPU reservation for one media actor."""
    return cpus_per_fetch * fetch_workers


def _media_actor_capacities(
    advertised: Sequence[tuple[float, float]],
    fetch_workers: int,
    *,
    cpus_per_fetch: float,
) -> list[int]:
    """Return how many media actors each live node could host at one fetch width.

    BOTH terms scale with the width - one IO slot and ``cpus_per_fetch`` CPUs
    per concurrent read - which is why a wider actor re-partitions a node's
    budget rather than drawing more from it.

    Args:
        advertised: One ``(curator_io slots, CPUs)`` pair per live node.
        fetch_workers: Clips one actor reads at once.
        cpus_per_fetch: CPU reservation for one concurrent clip read.

    Returns:
        Per-node actor capacity, in the order the nodes were read.

    """
    required_io = float(fetch_workers)
    required_cpus = _media_fetch_cpus(fetch_workers, cpus_per_fetch=cpus_per_fetch)
    return [min(int(io_slots // required_io), int(cpus // required_cpus)) for io_slots, cpus in advertised]


def _warn_if_width_adds_no_concurrency(
    advertised: Sequence[tuple[float, float]],
    *,
    fetch_workers: int,
    cpus_per_fetch: float,
    actors: int,
    read_ceiling: int,
) -> None:
    """Warn when a raised fetch width did not raise total read concurrency.

    Widening an actor partitions the same per-node budget more coarsely, so the
    total is unchanged at best and truncated at worst. An operator who raises the
    width expects more reads, and the schedulability guard says only that the
    wider actor can be placed - which stays true at exactly the width that gained
    nothing. Both figures come from the same node list, so a cluster still
    assembling scales them together and this verdict survives a partial view even
    though the absolute ceilings do not.

    Args:
        advertised: One ``(curator_io slots, CPUs)`` pair per live node.
        fetch_workers: Clips one actor reads at once.
        cpus_per_fetch: CPU reservation for one concurrent clip read.
        actors: Media actors the live nodes could host at that width.
        read_ceiling: Concurrent reads those actors would hold.

    """
    # Only a RAISED width is reported. At the default the comparison is trivially
    # equal, and below it the operator traded reads for more actors deliberately -
    # a choice this warning's remedy does not address, so firing there is the
    # noise that gets a warning filtered out before the run that needs it.
    if fetch_workers <= DEFAULT_MEDIA_FETCH_WORKERS:
        return
    default_actors = sum(
        _media_actor_capacities(
            advertised,
            DEFAULT_MEDIA_FETCH_WORKERS,
            cpus_per_fetch=cpus_per_fetch,
        )
    )
    default_ceiling = default_actors * DEFAULT_MEDIA_FETCH_WORKERS
    if read_ceiling > default_ceiling:
        return
    logger.warning(
        f"Media fetch width {fetch_workers} does not raise total read concurrency above the default "
        f"width {DEFAULT_MEDIA_FETCH_WORKERS}: {read_ceiling} concurrent read(s) against "
        f"{default_ceiling}. Each read reserves one {environment.CURATOR_IO_RESOURCE_NAME} slot, so a "
        f"wider actor claims a larger share of the same per-node budget - {actors} actor(s) here "
        f"against {default_actors} at the default - instead of adding to it. Raise the budget with "
        f"--ray-io-slots-per-node to raise the total; raise the width to shorten the tail one actor "
        f"works through on its last block. This run raised the width alone, so it holds no more read "
        f"capacity than the default does, in fewer and faster actors."
    )


@attrs.define(frozen=True)
class _MediaFetchCapacity:
    """What one cluster snapshot's resource budgets permit the media pool.

    Both figures come from the same node reading, so they cannot describe
    different clusters, and ``read_ceiling`` is ``placeable_actors`` times the
    fetch width by construction rather than by a second derivation.

    Attributes:
        placeable_actors: Media actors the live nodes could hold at once. Not the
            pool's eventual size - an actor pool left to autoscale requests
            actors it cannot place, and counts them in its own occupancy.
        read_ceiling: Concurrent source reads those actors would hold.

    """

    placeable_actors: int
    read_ceiling: int


def _verify_media_resources(fetch_workers: int, *, cpus_per_fetch: float) -> _MediaFetchCapacity:
    """Refuse a fetch width no node can host with all required resources.

    Ray does not fail a task whose combined CPU and custom-resource request fits
    no node - it leaves it PENDING forever. Unchecked, an over-wide reservation
    therefore presents as a run that starts, builds its engines, and then
    produces nothing, with no error anywhere to explain it.

    Schedulability needs one sufficient node, not every node, so the comparison
    is against each node's resource pair: independently sufficient maxima can
    live on different nodes and still leave the actor impossible to place.

    Args:
        fetch_workers: Clips one actor reads at once.
        cpus_per_fetch: CPU reservation for one concurrent clip read.

    Returns:
        The actors this cluster snapshot could hold at once and the reads they
        would run. Ceilings, not predictions - the input block count caps the
        pool below them - and a snapshot, so nodes joining later are not counted.

    Raises:
        ValueError: If no live node can satisfy the actor's CPU and IO request.

    """
    advertised = [
        (
            float(node.get("Resources", {}).get(environment.CURATOR_IO_RESOURCE_NAME, 0.0)),
            float(node.get("Resources", {}).get("CPU", 0.0)),
        )
        for node in ray.nodes()  # type: ignore[no-untyped-call]  # Missing in Ray stubs.
        if node.get("Alive", False)
    ]
    required_io = float(fetch_workers)
    required_cpus = _media_fetch_cpus(fetch_workers, cpus_per_fetch=cpus_per_fetch)
    host_capacities = _media_actor_capacities(
        advertised,
        fetch_workers,
        cpus_per_fetch=cpus_per_fetch,
    )
    if not any(host_capacities):
        largest_io = max((io_slots for io_slots, _ in advertised), default=0.0)
        largest_cpus = max((cpus for _, cpus in advertised), default=0.0)
        msg = (
            f"execution.media_fetch_workers={fetch_workers} makes each media actor request "
            f"{required_io:g} {environment.CURATOR_IO_RESOURCE_NAME} slots and {required_cpus:g} CPUs, "
            f"but no live node advertises both; the independent maxima are {largest_io:g} slots and "
            f"{largest_cpus:g} CPUs. No media actor could ever be scheduled. Lower the fetch width, "
            f"raise the IO budget with --ray-io-slots-per-node, or allocate more CPUs per node."
        )
        raise ValueError(msg)
    # Passing the check above only proves ONE node can host an actor. On a
    # heterogeneous cluster a width that fits on the largest node alone confines
    # the whole stage there, which reads downstream as idle GPUs on every other
    # node - a placement artifact wearing the costume of a throughput problem.
    # Reported rather than refused, because a single-node media stage is a valid
    # choice; it just must not be an invisible one.
    hosts = sum(capacity > 0 for capacity in host_capacities)
    actors = sum(host_capacities)
    read_ceiling = actors * fetch_workers
    # Reported wherever it occurs, not only when the width is also inert: a width
    # can strand slots while still gaining on the default, and a stranded slot is
    # budget the job holds and cannot spend. Only the IO remainder is counted, so
    # capacity a node loses to its CPU term is never mislabelled as truncation.
    stranded = int(sum(io_slots % required_io for io_slots, _ in advertised))
    remainder = (
        f" The width does not divide every node's budget evenly, leaving {stranded} "
        f"{environment.CURATOR_IO_RESOURCE_NAME} slot(s) unusable cluster-wide."
        if stranded
        else ""
    )
    # The denominator says "registered so far" because it counts whichever nodes
    # had joined by the time this ran, not the nodes the job was allocated. Stated
    # on this line rather than left to the cluster snapshot logged earlier, which
    # is far enough away that a reader quoting this one line loses the caveat.
    # Says what the figure COUNTS. It is the actors the per-node IO and CPU budgets
    # can hold simultaneously - not a prediction of the pool's size, and not the
    # actors Ray would otherwise request. The caller caps the pool at this number
    # for that reason; see the ``max_size`` comment at the media stage.
    logger.info(
        f"Media fetch width {fetch_workers} fits on {hosts} of the {len(advertised)} node(s) registered "
        f"so far. The per-node budgets place {actors} media actor(s) at once, holding {read_ceiling} "
        f"concurrent read(s), and the pool is capped there so it never requests an actor no node can "
        f"host; the input block count caps it further.{remainder}"
    )
    _warn_if_width_adds_no_concurrency(
        advertised,
        fetch_workers=fetch_workers,
        cpus_per_fetch=cpus_per_fetch,
        actors=actors,
        read_ceiling=read_ceiling,
    )
    return _MediaFetchCapacity(placeable_actors=actors, read_ceiling=read_ceiling)


def _report_pending_fragments(
    plan: CaptionScanPlan,
    *,
    total_fragments: int,
    rows_per_block: int,
) -> int:
    """Log pending scan geometry and return the derived fetch-block count.

    The scan starts from one source block per pending fragment, then non-strict
    repartition splits each source block independently. Fragment geometry still
    describes source-read startup and tail, while the derived row-block count
    describes downstream media work.

    Args:
        plan: The pending fragments and their post-filter row counts.
        total_fragments: Fragments the table holds at the pinned version.
        rows_per_block: Target rows per fetch work block.

    Returns:
        Exact fetch-block count produced by non-strict repartition.

    """
    pending = len(plan.fragments)
    # The fragment's PENDING rows, not its physical size: on a resume the two
    # differ by everything already captioned, and only the remainder is still
    # read. This is why the shared table shape cannot supply this figure.
    largest = max((fragment.num_rows for fragment in plan.fragments), default=0)
    fetch_blocks = sum(math.ceil(fragment.num_rows / rows_per_block) for fragment in plan.fragments)
    # INFO however narrow the resulting work set is. Whether it is harmful also
    # depends on service time and the cluster's idle policy, neither of which
    # this phase owns or can infer at startup.
    logger.info(
        f"Caption plan: {pending} of {total_fragments} fragment(s) pending, holding {plan.num_rows:,} "
        f"row(s) with {largest:,} in the largest. Non-strict row repartition will produce "
        f"{fetch_blocks} fetch work block(s) at {rows_per_block} row(s) per block without coalescing "
        f"fragments. The largest pending fragment still bounds one source scan task's tail, but no "
        f"longer confines that fragment's media fetches to one actor."
    )
    return fetch_blocks


def _report_video_policy(video: RobotActionCaptionVideoConfig) -> None:
    """Echo the resolved video geometry the engine will be built with.

    The contract digest proves two runs agree; it cannot say what geometry ran,
    and that is the question asked after a prompt overruns the context window.
    Recovering the pixel cap from the run alone otherwise means inverting the
    engine's encoder-profiling arithmetic.

    Read from the SAME accessors the digest consumes, so the echo cannot describe
    a policy other than the one that ships. An empty group means this config
    declared none and the layer below decides - the case an inherited oversized
    cap arrives through, so it is spelled out rather than printed as ``{}``.

    Args:
        video: The run's resolved video settings.

    """
    media_io = video.media_io_kwargs()
    processor = video.mm_processor_kwargs()
    logger.info(
        f"Caption video policy reaching the engine: "
        f"media_io_kwargs={media_io or 'unset, so the loader default decides frame sampling'}, "
        f"mm_processor_kwargs={processor or 'unset, so the model artifact decides the pixel bounds'}."
    )


@attrs.define(frozen=True)
class _FetchVersusEngineCapacity:
    """The two stocks the capacity log compares, each still in its own units.

    Attributes:
        read_ceiling: Concurrent reads the live nodes' resource budgets permit.
        input_blocks: Blocks the row repartition emits, which is the media
            pool's task ceiling - Ray dispatches one task per block, so actors
            beyond this count have no work to be given.
        fetch_workers: Clips one actor reads at once.
        rows_per_block: Configured row target that produced ``input_blocks``.
        batch_size: Rows handed to one engine replica per Ray batch.
        concurrency: The engine pool's ``(floor, ceiling)`` replica bounds.

    """

    read_ceiling: int
    input_blocks: int
    fetch_workers: int
    rows_per_block: int
    batch_size: int
    concurrency: tuple[int, int]


def _report_fetch_against_engine_capacity(capacity: _FetchVersusEngineCapacity) -> None:
    """Log the fetch ceiling and the engine floor together, in their own units.

    The two sides of this phase are sized by unrelated inputs - the fetch side by
    node budgets and repartitioned work blocks, the engine side by pending rows -
    so neither number alone tells an operator what the run was configured to
    hold. Both figures are stated in ROWS IN FLIGHT,
    which is the one unit they share, and per replica as well as per cluster. No
    throughput rate is derived: per-clip fetch and generation latencies are
    unknown at startup, and a fabricated rate would be read as measured.

    Args:
        capacity: Fetch-side ceilings and engine-side floor in one snapshot, so
            the comparison cannot mix numbers from different configurations.

    """
    read_ceiling = capacity.read_ceiling
    input_blocks = capacity.input_blocks
    fetch_workers = capacity.fetch_workers
    rows_per_block = capacity.rows_per_block
    batch_size = capacity.batch_size
    floor, ceiling = capacity.concurrency
    # BOTH fetch bounds are reported, not just the smaller one: which of the two
    # binds decides what an operator should change, and they call for opposite
    # actions - a bigger node IO budget versus a smaller row-block target.
    block_ceiling = input_blocks * fetch_workers
    supply = min(read_ceiling, block_ceiling)
    # A LOWER bound on the pool's appetite, deliberately: a replica may hold
    # several batches at once (``execution.max_concurrent_batches``), so one
    # full batch each is the least that keeps every floored replica fed. It is
    # printed rather than compared against, so understating it costs no accuracy
    # and keeps the figure one an operator can re-derive by hand.
    engine_demand = floor * batch_size
    # Two decimals rather than integer division. A floor holding more replicas
    # than the supply has reads is the severe case, and it is the one integer
    # division renders as "0" - an absence of supply rather than the shortfall it
    # actually is. The cluster-scale pair reads as merely large or merely small;
    # this is the form in which the same two numbers are legible.
    reads_per_replica = f"{supply / floor:.2f}"
    logger.info(
        f"Caption supply versus capacity, as two stocks measured at this instant and not as rates: "
        f"media fetch holds at most {supply} clip read(s) in flight - a ceiling, the lower of "
        f"{read_ceiling} from the live nodes' IO and CPU budgets and {block_ceiling} from "
        f"{input_blocks} input block(s) at width {fetch_workers}. The engine pool's floor of {floor} "
        f"replica(s) holds at least {engine_demand} row(s) in flight to give each replica one full "
        f"batch ({floor} x batch size {batch_size}); the pool may scale to {ceiling}. Per "
        f"replica that is {reads_per_replica} read(s) in flight behind a {batch_size}-row "
        f"batch. How far the fetch side actually falls behind is set by the two stages' service times, "
        f"which this run has not measured yet, so neither stock divided by the other is that factor. "
        f"Concurrent reads cap the rate rows arrive at rather than the rows a replica can draw on - "
        f"completed fetches queue ahead of the engine - so a supply under the floor's appetite is not "
        f"itself starvation. If the block ceiling binds, lower execution.lance_read_batch_size below "
        f"{rows_per_block} to create more fetch work blocks without rewriting the Lance table."
    )


def run_inference_phase(  # noqa: PLR0913 - a phase boundary; each input is independently meaningful
    dataset: lance.LanceDataset,
    plan: CaptionScanPlan,
    config: ResolvedRobotActionCaptionConfig,
    model: RegisteredModel,
    workspace: CaptionWorkspace,
    *,
    context_fields: tuple[PromptContextField, ...],
    digest: str,
    row_filter: str,
    storage_options: dict[str, str] | None,
) -> None:
    """Stage one attempt's terminal caption rows under the workspace.

    Args:
        dataset: The clip table, open at the run's pinned version.
        plan: The pending fragments and their row counts.
        config: The run's resolved semantic and execution settings.
        model: The resolved model identity.
        workspace: The namespace this attempt stages into and checkpoints against.
        context_fields: The context subset resolved for this run, rendered into
            each prompt. Passed alongside ``digest`` rather than re-resolved here
            so the prompts and the digest cannot describe different field sets.
        digest: This run's contract digest.
        row_filter: Predicate selecting the rows this attempt owes.
        storage_options: Lance credentials for the read tasks.

    Raises:
        RuntimeError: If a checkpoint configuration is already installed on the
            ``DataContext``, or if the one this phase installed is not the one
            that ends up active.

    """
    execution = config.execution
    fetch_workers = execution.media_fetch_workers
    # Before anything is built, because the alternative failure is a silent one.
    media_capacity = _verify_media_resources(
        fetch_workers,
        cpus_per_fetch=execution.media_cpus_per_fetch,
    )
    source = CaptionScanDatasource(
        PinnedScan.for_dataset(
            dataset,
            storage_options=storage_options,
            batch_size=execution.lance_read_batch_size,
            context_fields=context_fields,
            row_filter=row_filter,
        ),
        plan.fragments,
    )
    # Held in a local so the resolved replica bounds can be reported next to the
    # fetch ceiling above; the two numbers are only meaningful together.
    engine_config = processor_config(
        model,
        generation=config.generation,
        video=config.video,
        execution=execution,
        pending_rows=plan.num_rows,
        available_gpus=float(ray.available_resources().get("GPU", 0.0)),  # type: ignore[no-untyped-call]
    )
    # get_fragments() resolves from the open manifest and reads no column data,
    # so the table total costs nothing against remote storage.
    fetch_input_blocks = _report_pending_fragments(
        plan,
        total_fragments=len(dataset.get_fragments()),
        rows_per_block=execution.lance_read_batch_size,
    )
    _report_video_policy(config.video)
    _report_fetch_against_engine_capacity(
        _FetchVersusEngineCapacity(
            read_ceiling=media_capacity.read_ceiling,
            # Non-strict repartition splits every source block independently, so
            # this driver-side count is the exact number the media pool receives.
            input_blocks=fetch_input_blocks,
            fetch_workers=fetch_workers,
            rows_per_block=execution.lance_read_batch_size,
            batch_size=execution.inference_batch_size,
            concurrency=engine_config.concurrency,
        )
    )
    transform = build_vllm_transform(
        engine_config,
        preprocess=partial(build_request, generation=config.generation),
        postprocess=read_generation,
    )
    # The checkpoint is a CORRECTNESS requirement here, not a cost optimisation.
    # The engine stage is pinned to treat a per-row generation failure as
    # systemic: it re-raises rather than emitting an error row, so one unusable
    # clip ends the whole attempt. That pin is deliberate - a swallowed failure
    # cannot be told apart from a dead engine, and a table of silently-errored
    # captions is worse than a stopped run - and it is affordable ONLY because
    # the rows already finished are recorded here and skipped on the next
    # attempt. Removing or bypassing this wiring turns every engine fault into a
    # full re-run of the phase; loosening the pin instead is the other half of
    # the same mistake.
    #
    # The filesystem comes off the workspace so the checkpoint and the sink below
    # are reached through ONE credential set. Two independently resolved
    # filesystems let the sink write where the checkpoint cannot read, and that
    # surfaces as a resume that silently redoes everything rather than as an
    # access error.
    with installed_checkpoint(
        id_column=CHECKPOINT_ID_COLUMN,
        checkpoint_path=workspace.checkpoints_uri,
        filesystem=workspace.filesystem,
    ):
        fetch_inputs = ray.data.read_datasource(
            source,
            override_num_blocks=len(plan.fragments),
        ).repartition(
            target_num_rows_per_block=execution.lance_read_batch_size,
            # Non-strict processes each source block independently: it splits a
            # large fragment without coalescing or mixing adjacent fragments.
            # Therefore the fetch-block count remains exactly the sum of each
            # fragment's row count rounded up to this target. Strict mode
            # rebundles across source blocks and would erase that locality.
            strict=False,
        )
        fetched = fetch_inputs.map_batches(
            ClipMediaFetch,
            fn_constructor_kwargs={
                "storage_profile": execution.storage_profile,
                "attempts": execution.media_attempts,
                "max_clip_bytes": config.media.max_clip_bytes,
                "fetch_workers": fetch_workers,
            },
            batch_format="pyarrow",
            # One row per thread, so a batch fills the actor's fetch width
            # exactly. A wider batch would queue rows behind the threads without
            # adding overlap, and hold their encoded bytes for the wait; a
            # narrower one would leave threads idle and pay a Ray hand-off per
            # clip anyway. The reservation below is derived from the SAME number,
            # so the width an actor runs is always the width it paid for.
            batch_size=fetch_workers,
            num_cpus=_media_fetch_cpus(
                fetch_workers,
                cpus_per_fetch=execution.media_cpus_per_fetch,
            ),
            resources=_media_fetch_resources(fetch_workers),
            # Capped at the actors the node budgets can actually place. Left
            # unbounded, the pool autoscales toward the block count, and every
            # actor past this number stays PENDING for the whole run because one
            # concurrent read reserves one ``curator_io`` slot and the slots are
            # already spoken for. Those phantom actors are not merely idle: Ray
            # Data's occupancy divides work in flight across every actor it has
            # REQUESTED, so a fetch stage running flat out reports slack it does
            # not have - on a fetch-bound run, that is the one number an operator
            # would use to rule this stage out.
            #
            # The cost is that the cap comes from the same driver-start snapshot
            # as the engine floor, so a node registering later cannot be used by
            # this pool. That is the existing exposure, not a new one, and it
            # buys an occupancy figure whose denominator is real.
            compute=ActorPoolStrategy(min_size=1, max_size=media_capacity.placeable_actors),
            # The recipe executor bounds only the stages it creates. This module
            # must bound the media stage it adds, because an unbounded media actor loops
            # forever on a permanently rejected credential. Spread whole because
            # the two keys are one decision - see ``ACTOR_RESTART_BOUNDS``.
            **ACTOR_RESTART_BOUNDS,
        )
        terminal = transform(fetched).map_batches(
            partial(
                canonicalize,
                digest=digest,
                model_id=model.model_id,
                generation=config.generation,
            ),
            batch_format="pyarrow",
            batch_size=execution.inference_batch_size,
        )
        terminal.write_parquet(
            workspace.results_uri,
            filesystem=workspace.filesystem,
            partition_cols=[FRAGMENT_ID_COLUMN],
            # Append, so a resume adds to what an earlier attempt staged instead
            # of replacing it. The checkpoint guarantees those two sets are
            # disjoint, and Ray's two-phase commit removes any shard that became
            # durable without its checkpoint entry. Passed explicitly even though
            # it is Ray's current default: were that default ever to become
            # OVERWRITE, a resume would silently delete the captions it exists to
            # preserve, and unseeded generation makes them unrecoverable.
            mode=SaveMode.APPEND,
            # Both bounds are HINTS in Ray's sink, not guarantees, and are set to
            # one value only to keep a staged file small enough that a crash
            # mid-phase leaves usable partial work. No repartition is added to
            # enforce them: paying a shuffle to hit an exact file size would cost
            # more than the uneven files it avoids.
            min_rows_per_file=execution.parquet_rows_per_file,
            max_rows_per_file=execution.parquet_rows_per_file,
        )
    logger.info(
        f"Phase A staged terminal caption rows for {len(plan.fragments)} fragment(s) under {workspace.results_uri}."
    )


def _media_bytes_per_row(video: RobotActionCaptionVideoConfig, source_frame_pixels: int) -> int:
    """Return the largest decoded media payload one row can carry.

    Args:
        video: The run's video settings, whose frame ceiling bounds the payload.
        source_frame_pixels: Pixels in one decoded source frame.

    Returns:
        Bytes of decoded RGB one row carries at its frame ceiling.

    """
    # The CEILING, not a typical row: this sizes a memory reservation, and one
    # admitting the average row is exceeded by every row above it.
    if video.max_frames is not None:
        return video.max_frames * source_frame_pixels * _BYTES_PER_DECODED_PIXEL

    # Warn rather than refuse: delegating temporal behavior to the backend stays
    # a supported mode, but the bound below is then read off a dependency default
    # instead of this run's own config, so an operator should see which of the
    # two produced the reservation they are being scheduled against.
    logger.warning(
        f"video.max_frames is unset, so the media reservation assumes vLLM's own "
        f"{_LOADER_DEFAULT_NUM_FRAMES}-frame default. That bounds the payload only for a backend "
        f"honoring num_frames; one sampling by fps and duration can exceed it and overrun the "
        f"reservation. Set video.fps, video.min_frames and video.max_frames to bound it here."
    )
    return _LOADER_DEFAULT_NUM_FRAMES * source_frame_pixels * _BYTES_PER_DECODED_PIXEL


def processor_config(  # noqa: PLR0913 - phase-facing semantic and execution inputs stay explicit
    model: RegisteredModel,
    *,
    generation: RobotActionCaptionGenerationConfig,
    video: RobotActionCaptionVideoConfig,
    execution: RobotActionCaptionExecutionConfig,
    pending_rows: int,
    available_gpus: float = 0.0,
) -> VllmProcessorConfig:
    """Build the engine-facing configuration for this run.

    Args:
        model: The resolved model identity, whose staged directory the engine loads.
        generation: Generation settings whose context limit builds the engine.
        video: The run's video settings, which supply both multimodal mappings.
        execution: Non-semantic engine and scheduling policy for this run.
        pending_rows: Rows this attempt owes, which bounds how many replicas the
            work could ever occupy.
        available_gpus: GPUs Ray reports free, which sets the pool's replica
            floor and so the GPUs it reserves for the run. Passed in rather than
            read here so building the configuration never depends on a live
            cluster.

    Returns:
        The configuration the generic transform is built from.

    """
    engine_kwargs: dict[str, Any] = {
        # Use the same translation methods as the digest rather than
        # reconstructing these mappings. Each optional group is either
        # complete or empty; empty preserves backend/artifact ownership.
        "media_io_kwargs": video.media_io_kwargs(),
        "limit_mm_per_prompt": {"video": VIDEO_ITEMS_PER_PROMPT},
        # A configured pair passes through in processor-native units.
        # Curator never derives it from the selected model identity.
        "mm_processor_kwargs": video.mm_processor_kwargs(),
    }
    if execution.max_num_batched_tokens is not None:
        # This is a capacity override, not a semantic default. vLLM's Qwen2-VL
        # profile can underestimate a caller's media frame policy, while other
        # processors may have different memory/concurrency trade-offs. Preserve
        # dependency defaults unless the runtime config explicitly opts in.
        engine_kwargs["max_num_batched_tokens"] = execution.max_num_batched_tokens

    media_bytes_per_row = _media_bytes_per_row(video, execution.source_frame_pixels)
    resolved = VllmProcessorConfig(
        model=str(model.runtime_model_dir),
        max_model_len=generation.max_model_len,
        tensor_parallel_size=execution.tensor_parallel_size,
        pipeline_parallel_size=execution.pipeline_parallel_size,
        concurrency=_inference_concurrency(
            pending_rows=pending_rows,
            available_gpus=available_gpus,
            configured=None if execution.inference_concurrency == "auto" else execution.inference_concurrency,
            batch_size=execution.inference_batch_size,
            gpus_per_replica=execution.tensor_parallel_size * execution.pipeline_parallel_size,
        ),
        batch_size=execution.inference_batch_size,
        max_concurrent_batches=execution.max_concurrent_batches,
        prepare_multimodal=PREPARE_MULTIMODAL,
        media_bytes_per_row=media_bytes_per_row,
        engine_kwargs=engine_kwargs,
        runtime_env=ray_data_gpu_runtime_env(
            "default",
            # The caching allocator's large pool fragments under this recipe's
            # concurrent multimodal batches: each batch in flight drives a
            # differently sized encoder activation, so a new multi-GiB segment
            # finds no cached block to reuse and no contiguous unreserved memory
            # to claim. Without expandable segments the allocator recovers by
            # flushing its whole cache and retrying, which synchronizes the
            # device. Growing one virtual-address-backed arena per stream removes
            # the CONTIGUITY requirement instead - it adds no device memory, and
            # adding memory is not what a fragmentation mechanism needs.
            #
            # Scoped to this recipe's engine actors deliberately. The shared
            # ``_RAY_DATA_GPU_ENV_VARS`` reaches every Ray Data GPU stage in
            # every recipe, and none of the others has been measured here.
            env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        ),
    )
    # Reported because it decides how densely three CPU stages pack onto a node,
    # and understating execution.source_frame_pixels has no symptom until one is
    # OOM-killed. Every part is given: an operator checks the per-row ceiling
    # against their own footage and each per-worker figure against node memory.
    # The guard is not dead under a text-only variant of this recipe, where there
    # is no reservation to report because no stage carries a payload.
    reservations = resolved.stage_heap_reservations()
    if reservations is not None:
        # The in-flight factor is deliberately described rather than printed:
        # it is a private constant of the engine module, and a number restated
        # in prose beside the figures it produced is a second definition that
        # no test can redden when the two drift apart.
        logger.info(
            f"Media-carrying stages reserve {reservations.preparation / _BYTES_PER_GIB:.2f} GiB per "
            f"preparation worker and {reservations.forwarding / _BYTES_PER_GIB:.2f} GiB per forwarding "
            f"worker, from a {media_bytes_per_row / _BYTES_PER_MIB:.1f} MiB per-row ceiling. Each figure "
            "covers every batch Ray Data keeps in flight per actor, not just one, plus a fixed resident "
            "cost that does not shrink with the batch - so halving execution.inference_batch_size does "
            "not halve it."
        )
    return resolved


def _inference_concurrency(
    *,
    pending_rows: int,
    available_gpus: float,
    configured: int | None,
    batch_size: int,
    gpus_per_replica: int,
) -> tuple[int, int]:
    """Return ``(floor, ceiling)`` replicas for the engine pool.

    Element 0 becomes Ray Data's ``min_size``: an absolute floor the pool is
    never scaled below, so it reserves that many GPUs for the run's duration
    whether or not work reaches them. Only the inputs-complete drain releases
    them, at the end.

    The ceiling is derived from PENDING WORK rather than from the GPUs Ray
    reports, which on a large cluster would license more replicas than there is
    work to fill. Ray scales the pool between the two bounds as blocks queue.

    Args:
        pending_rows: Rows this attempt owes.
        available_gpus: GPUs Ray reports free at this instant.
        configured: The operator's cap, or ``None`` for auto.
        batch_size: Rows one replica receives in a Ray batch.
        gpus_per_replica: GPUs reserved by each replica.

    Returns:
        ``(floor, ceiling)``, both at least one. Never zero: a pool of zero
        stalls the plan rather than waiting for capacity.

    """
    ceiling = max(1, math.ceil(pending_rows / batch_size))
    if configured is not None:
        # The typed config already rejects a non-positive cap; this guard is for
        # direct callers. min(ceiling, 0) would return a pool of zero, which
        # stalls the plan rather than waiting for capacity.
        if configured < 1:
            msg = f"configured replica cap must be a positive integer, got {configured}"
            raise ValueError(msg)
        ceiling = min(ceiling, configured)
    # The floor claims the visible GPUs up front rather than starting at one and
    # letting the pool grow into the allocation, because neither half of that
    # alternative works here. Ray Data adds about one replica per actor-
    # construction round-trip, since actors pending construction count against
    # the utilization it scales on and so throttle their own ramp; at minutes per
    # engine load a low floor serialises those loads across the run's early
    # hours. And a pool that later shrank
    # would not hand anything back, since Slurm holds the allocation for the
    # job's lifetime and a utilization-keyed idle-GPU reaper reads a released GPU
    # exactly like a resident idle one.
    floor = min(ceiling, max(1, int(max(0.0, available_gpus) // gpus_per_replica)))
    logger.info(
        f"Caption inference will hold a floor of {floor} replica(s) and scale to at most {ceiling}, "
        f"each on {gpus_per_replica} GPU(s), for {pending_rows} pending row(s) against the "
        f"{available_gpus} GPU(s) Ray reports free right now. That GPU reading is a snapshot of "
        f"whichever nodes have registered so far, and this floor is fixed for the run: nodes joining "
        f"later let the pool scale toward its ceiling but never raise the floor, so a floor taken from "
        f"a partial cluster under-uses the allocation until the run ends."
    )
    return floor, ceiling
