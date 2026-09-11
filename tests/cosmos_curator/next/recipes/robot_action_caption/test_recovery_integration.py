# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole attempts, measured by how much inference the next one pays for.

Every test here runs the real pipeline against a real Lance table with only the
engine faked, because the properties are not visible at any function boundary.
The one observable that separates recovery working from recovery merely appearing
to work is HOW MANY CLIPS REACH THE ENGINE: a run that re-captions everything
still publishes a complete, plausible table - with different captions, since
generation is unseeded. Captions therefore carry a per-attempt tag, so a
substituted one cannot pass unnoticed.

The shared fixture holds eight pending clips in two fragments. Each fragment is
split into bounded fetch blocks, but keeping two source fragments makes the
failure boundary deterministic while still exercising publication's fragment
fan-in. Four captions survive the attempt's failure for a retry to recover::

    attempt 1   fragment 0 (c0..c3)  split -> captioned -> checkpointed
                fragment 1 (c4..c7)  split -> engine refuses -> attempt fails
                                     clips.lance UNCHANGED, staged rows intact
    attempt 2   fragment 0           dropped by Ray's checkpoint filter
                fragment 1 (c4..c7)  captioned -> staged
                publication          4 recovered + 4 new -> ONE commit

What each starting state costs the next attempt:

    starting state        clips reaching the engine on the next attempt
    --------------------  --------------------------------------------
    empty                 all pending
    partial (4 of 8)      the 4 that did not finish
    complete              none, and the attempt still publishes
    moved source version  all pending, from a different namespace
    widened input scope   the 4 that did not finish; scope is not a namespace input
    changed semantics     all pending, and the captions are overwritten
    changed execution     the 4 that did not finish, exactly as before
    published             none, and nothing is committed

The last two rows are the digest's whole purpose, and they are the pair that must
be tested together: a prompt or model change makes completed rows unreusable, and
a throughput change must not. Getting either wrong is invisible in the published
table, which is complete and plausible in both cases.

The moved-version row is the exact-version identity: staged rows are keyed by
Lance row address, which only means anything inside the snapshot that produced
it, so a moved version must share nothing. That address is the whole of a staged
row's identity - there is no per-row request fingerprint to fall back on.

Further concerns are only observable end to end, which is why they live here
rather than beside the code they constrain:

- A REPLAYED shard, which the in-memory fold tests in ``test_publication.py``
  cannot reach: a second attempt appends its Parquet beside the first attempt's,
  so a duplicate row arrives as a second FILE in one partition rather than as a
  second row in one table.
- Columns the recipe does not own. Every stage returns a narrow key set by
  design - the recipe never carries source columns through the engine, it
  re-reads them at publication - so only the committed table can show that an
  unrelated column kept its per-row values.
- Workspace cleanup, which is ordered AFTER the commit and is nonfatal. Both
  halves need a real attempt: one to show a successful run leaves nothing
  staged, the other to show a run whose cleanup failed is still a success.
"""

import pathlib
import shutil

import attrs
import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ray.exceptions import RayTaskError

from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.next.recipes.robot_action_caption import pipeline, prompt
from cosmos_curator.next.recipes.robot_action_caption.config import (
    DEFAULT_MEDIA_FETCH_WORKERS,
    DEFAULT_MODEL_VARIANT,
    ResolvedRobotActionCaptionConfig,
)
from cosmos_curator.next.recipes.robot_action_caption.contracts import CAPTION_COLUMN, CaptionStatus
from cosmos_curator.next.recipes.robot_action_caption.pipeline import run_config
from cosmos_curator.next.recipes.robot_action_caption.publication import CaptionPublicationError
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CLIP_ID_COLUMN,
    CURATE_SELECTED_VALUE,
    CURATE_SELECTION_REASON,
    caption_work_filter,
)
from tests.cosmos_curator.next.recipes.robot_action_caption import caption_harness
from tests.cosmos_curator.next.recipes.robot_action_caption.caption_harness import (
    CaptionConfig,
    ClipsTable,
    FakeEngine,
    InferenceLedger,
)

# Rows one fetch block carries. Small enough that a fixture-scale corpus still
# spans several staged files, which is what makes partial durable progress
# reachable at all.
FETCH_BLOCK_ROWS = 2

# Rows one engine task consumes, and therefore the rows one checkpoint commits.
# It is the WIDEST row minimum downstream of the read, and that is the media
# fetch stage's, because geometry_rows shrinks the inference batch and the Lance
# read batch but NOT media_fetch_workers. Derived from the production default so
# that raising the default re-sizes the fixture instead of silently moving the
# failure boundary off the fragment boundary these tests assert against.
DURABLE_ROWS = FETCH_BLOCK_ROWS * -(-DEFAULT_MEDIA_FETCH_WORKERS // FETCH_BLOCK_ROWS)

# A fragment must be a whole number of DURABLE_ROWS, and that is a REQUIREMENT
# of the scenario, not a convenience. Being a whole number of fetch BLOCKS is not
# sufficient: Ray's map bundler will not dispatch fewer rows than its minimum, so
# a fragment that ends mid-unit strands its trailing rows in a bundle that waits
# and merges with the NEXT fragment's first block. The refused clip and the
# previous fragment's finished clips then share one engine task, and those
# captions die with the refusal. Measured: 5- and 6-row fragments both commit
# only the first 4 rows, and 6 is three whole 2-row blocks. A fragment SMALLER
# than one unit is worse - no bundle ever fills, nothing commits, and the
# harness barrier times out rather than reaching the refusal.
ROWS_PER_FRAGMENT = DURABLE_ROWS

# One finished fragment and one refused fragment.
CLIPS = ROWS_PER_FRAGMENT * 2
ALL_CLIPS = tuple(f"c{index}" for index in range(CLIPS))

FINISHED = ALL_CLIPS[:ROWS_PER_FRAGMENT]
UNFINISHED = ALL_CLIPS[ROWS_PER_FRAGMENT:]

FIRST_ATTEMPT_TAG = "first"
RETRY_TAG = "retry"
MODEL_ID = resolve_registered_model(DEFAULT_MODEL_VARIANT).model_id

# Names the injected publication failure so a test asserting it cannot pass on an
# unrelated error from the real publication path.
_INJECTED_PUBLICATION_FAILURE = "injected failure before the caption commit"

# Names the injected cleanup failure, on the same terms: a real storage error
# reaching this point would otherwise be indistinguishable from the injected one.
_INJECTED_CLEANUP_FAILURE = "injected failure after the caption commit"

# A column the recipe has never heard of, carrying a DISTINCT value per row. A
# constant would survive a bug that rebound the column to the caption file and
# read back one repeated value; per-row values do not.
UNRELATED_COLUMN = "operator_note"


@attrs.frozen
class Attempt:
    """One finished attempt, and what a following attempt needs to know about it.

    Attributes:
        uri: The clip table it read and would have published to.
        config: The configuration a retry reuses unchanged.
        read_version: The version it pinned, and the version a retry must pin.
        workspace_root: The namespace it staged into.
        ledger: The clips its engine was asked to caption.
        staged: Its staged terminal rows.

    """

    uri: str
    config: ResolvedRobotActionCaptionConfig
    read_version: int
    workspace_root: str
    ledger: InferenceLedger
    staged: tuple[caption_harness.StagedRow, ...]


@pytest.fixture
def local_run(
    curator_io_cluster: None,
    staged_weights: pathlib.Path,
    caption_clips: ClipsTable,
    caption_config: CaptionConfig,
) -> tuple[str, ResolvedRobotActionCaptionConfig]:
    """Return a table of eight pending clips in two fragments, and its configuration."""
    del curator_io_cluster, staged_weights
    uri = caption_clips(rows=CLIPS, rows_per_file=ROWS_PER_FRAGMENT)
    return uri, caption_config(uri, geometry_rows=FETCH_BLOCK_ROWS)


@pytest.fixture
def crashed_attempt(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    fake_engine: FakeEngine,
) -> Attempt:
    """Run an attempt that captions the whole first fragment and then refuses.

    Waiting for one committed checkpoint is waiting for the whole finished set,
    because the fixture makes the finished fragment exactly one write bundle. A
    wider fragment would leave a bundle in flight when the engine raises, and the
    refusal aborts the executor, so those rows would be cancelled unstaged.
    """
    uri, config = local_run
    ledger = fake_engine(
        fails_on=UNFINISHED,
        caption_tag=FIRST_ATTEMPT_TAG,
        barrier_root=config.staging_root,
    )

    with pytest.raises(RayTaskError, match="fake engine refused clip"):
        run_config(config)

    return _remains_of(uri, config, ledger)


@pytest.fixture
def fully_staged_unpublished(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    fake_engine: FakeEngine,
) -> Attempt:
    """Run an attempt that captions every clip and then fails before the commit.

    The failure is injected at ``publish_captions`` rather than inside inference,
    because this state - every clip checkpointed, nothing published - is what a
    crash in the publication phase leaves, and it is the state that must cost no
    further GPU time.

    The patch is scoped to the failing attempt alone, so the following attempt in
    the same test publishes through the real path.
    """
    uri, config = local_run

    def refuse_to_publish(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(_INJECTED_PUBLICATION_FAILURE)

    ledger = fake_engine(caption_tag=FIRST_ATTEMPT_TAG)
    with pytest.MonkeyPatch.context() as publication:
        publication.setattr(pipeline, "publish_captions", refuse_to_publish)
        with pytest.raises(RuntimeError, match=_INJECTED_PUBLICATION_FAILURE):
            run_config(config)

    return _remains_of(uri, config, ledger)


@pytest.fixture
def published_once(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    fake_engine: FakeEngine,
) -> Attempt:
    """Run one attempt through to its commit, so the table owes nothing afterwards."""
    uri, config = local_run
    ledger = fake_engine(caption_tag=FIRST_ATTEMPT_TAG)

    result = run_config(config)
    assert result.published_rows == CLIPS

    return _remains_of(uri, config, ledger)


def test_an_empty_checkpoint_sends_every_pending_clip_to_inference(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    fake_engine: FakeEngine,
) -> None:
    """A first attempt has nothing to recover, so the whole pending set is captioned."""
    uri, config = local_run
    ledger = fake_engine(caption_tag=FIRST_ATTEMPT_TAG)

    result = run_config(config)

    assert ledger.clip_ids == list(ALL_CLIPS)
    assert result.published_rows == CLIPS
    assert _pending_rows(uri, result.contract_digest, version=result.ending_version) == 0


def test_a_crash_during_inference_publishes_nothing(crashed_attempt: Attempt) -> None:
    """The commit is the sole visibility boundary, so a failed attempt leaves no caption."""
    assert lance.dataset(crashed_attempt.uri).count_rows(filter=f"{CAPTION_COLUMN} IS NOT NULL") == 0


def test_a_crash_during_inference_leaves_the_finished_clips_staged(crashed_attempt: Attempt) -> None:
    """The captions the first fragment generated are durable, which is what a retry recovers."""
    assert sorted(row.clip_id for row in crashed_attempt.staged) == list(FINISHED)


def test_a_retry_infers_only_the_clips_the_crash_did_not_finish(
    crashed_attempt: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """Ray's checkpoint filter drops the finished rows at the read, so no GPU work repeats."""
    retry = fake_engine(caption_tag=RETRY_TAG)

    run_config(crashed_attempt.config)

    assert retry.clip_ids == list(UNFINISHED)
    assert retry.calls == len(UNFINISHED)


def test_a_widened_retry_reuses_the_narrower_attempt_it_widens(
    crashed_attempt: Attempt,
    caption_config: CaptionConfig,
    fake_engine: FakeEngine,
) -> None:
    """Broadening scope after a crash recovers the staged captions instead of stranding them.

    Selection scope is not a namespace input, so the widened attempt resolves to
    the directory the selected-only crash staged into. Digesting the flag would
    send this retry to an empty namespace and recaption all eight clips - with
    different text, since generation is unseeded.
    """
    retry = fake_engine(caption_tag=RETRY_TAG)
    widened = caption_config(crashed_attempt.uri, geometry_rows=FETCH_BLOCK_ROWS, selected_only=False)

    result = run_config(widened)

    assert retry.clip_ids == list(UNFINISHED)
    captions = caption_harness.published_captions(crashed_attempt.uri, version=result.ending_version)
    assert [captions[clip_id] for clip_id in FINISHED] == [
        f"{FIRST_ATTEMPT_TAG} caption for {clip_id}" for clip_id in FINISHED
    ]


def test_recovery_uses_row_identity_when_one_fragment_spans_fetch_blocks(
    curator_io_cluster: None,
    staged_weights: pathlib.Path,
    caption_clips: ClipsTable,
    caption_config: CaptionConfig,
    fake_engine: FakeEngine,
) -> None:
    """Reprocess exactly the unstaged rows after splitting one source fragment.

    One Lance fragment is divided into two-row fetch blocks, which the bundler
    regroups into engine tasks of DURABLE_ROWS. The failure waits until an
    earlier task is checkpointed, then the retry must infer the exact complement
    of whatever became durable; the fragment boundary cannot reconstruct that
    set, because the durable split falls inside the fragment.
    """
    del curator_io_cluster, staged_weights
    uri = caption_clips(rows=CLIPS, rows_per_file=CLIPS)
    config = caption_config(uri, geometry_rows=FETCH_BLOCK_ROWS)
    fake_engine(
        fails_on=("c7",),
        caption_tag=FIRST_ATTEMPT_TAG,
        barrier_root=config.staging_root,
    )

    with pytest.raises(RayTaskError, match="fake engine refused clip"):
        run_config(config)

    assert len(lance.dataset(uri).get_fragments()) == 1
    staged_clip_ids = {row.clip_id for row in caption_harness.staged_rows(config.staging_root)}
    assert 0 < len(staged_clip_ids) < CLIPS
    retry = fake_engine(caption_tag=RETRY_TAG)
    read_version = int(lance.dataset(uri).version)

    result = run_config(config)

    expected_retry = sorted(set(ALL_CLIPS) - staged_clip_ids)
    assert retry.clip_ids == expected_retry
    assert retry.calls == len(expected_retry)
    assert result.ending_version == read_version + 1
    assert result.published_rows == CLIPS


def test_a_retry_publishes_every_clip_in_exactly_one_commit(
    crashed_attempt: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """Recovered and newly generated rows become visible together, in one table state."""
    fake_engine(caption_tag=RETRY_TAG)

    result = run_config(crashed_attempt.config)

    assert result.ending_version == crashed_attempt.read_version + 1
    assert (result.source_version, result.published_rows) == (crashed_attempt.read_version, CLIPS)
    published = lance.dataset(crashed_attempt.uri, version=result.ending_version)
    assert (
        published.count_rows(
            filter=caption_work_filter(
                result.contract_digest,
                MODEL_ID,
                selected_only=crashed_attempt.config.input.selected_only,
            )
        )
        == 0
    )
    assert published.count_rows(filter=f"caption_status = '{CaptionStatus.SUCCESS}'") == CLIPS


def test_a_retry_publishes_the_captions_the_crashed_attempt_generated(
    crashed_attempt: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """A recovered clip keeps its original caption; regenerating it would answer differently."""
    fake_engine(caption_tag=RETRY_TAG)

    result = run_config(crashed_attempt.config)

    captions = caption_harness.published_captions(crashed_attempt.uri, version=result.ending_version)
    assert [captions[clip_id] for clip_id in FINISHED] == [
        f"{FIRST_ATTEMPT_TAG} caption for {clip_id}" for clip_id in FINISHED
    ]
    assert [captions[clip_id] for clip_id in UNFINISHED] == [
        f"{RETRY_TAG} caption for {clip_id}" for clip_id in UNFINISHED
    ]


def test_the_contract_digest_is_identical_across_the_crash_and_the_retry(
    crashed_attempt: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """One digest over both attempts is what let the retry reuse the first's rows at all.

    Derived from the rows and from the run rather than compared against a literal:
    the digest covers the prompt and the model, so pinning its value here would
    fail the test on every deliberate semantic change.
    """
    fake_engine(caption_tag=RETRY_TAG)

    result = run_config(crashed_attempt.config)

    assert {row.digest for row in crashed_attempt.staged} == {result.contract_digest}
    assert caption_harness.published_digests(crashed_attempt.uri, version=result.ending_version) == {
        result.contract_digest
    }


def test_a_complete_checkpoint_publishes_without_returning_to_inference(
    fully_staged_unpublished: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """Every clip is already recorded, so the retry captions none of them and still commits."""
    retry = fake_engine(caption_tag=RETRY_TAG)

    result = run_config(fully_staged_unpublished.config)

    assert retry.calls == 0
    assert result.published_rows == CLIPS
    assert _captions_in_clip_order(fully_staged_unpublished.uri, version=result.ending_version) == [
        f"{FIRST_ATTEMPT_TAG} caption for {clip_id}" for clip_id in ALL_CLIPS
    ]


def test_a_moved_source_version_reuses_nothing(
    crashed_attempt: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """Staged rows are keyed by row address, which means nothing outside their snapshot."""
    lance.dataset(crashed_attempt.uri).add_columns(pa.schema([pa.field("unrelated", pa.string())]))
    retry = fake_engine(caption_tag=RETRY_TAG)

    result = run_config(crashed_attempt.config)

    assert retry.clip_ids == list(ALL_CLIPS)
    assert result.published_rows == CLIPS
    assert _captions_in_clip_order(crashed_attempt.uri, version=result.ending_version) == [
        f"{RETRY_TAG} caption for {clip_id}" for clip_id in ALL_CLIPS
    ]


def test_a_changed_prompt_contract_reuses_nothing_from_the_crashed_attempt(
    crashed_attempt: Attempt,
    fake_engine: FakeEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different semantics land in a different namespace, so every clip is captioned again.

    The prompt version identifies prompt semantics, so changing it changes the
    digest, and the digest names the directory holding staged progress. Reuse
    across that boundary would publish captions the new prompt never produced.

    Patched on ``prompt``, which owns the constant, rather than on the digest
    module that reads it: the digest reads it through the module attribute, so
    the owning module is the only place a patch reaches every reader.
    """
    monkeypatch.setattr(prompt, "PROMPT_VERSION", "a_different_prompt_contract")
    retry = fake_engine(caption_tag=RETRY_TAG)

    result = run_config(crashed_attempt.config)

    assert retry.clip_ids == list(ALL_CLIPS)
    assert result.contract_digest != crashed_attempt.staged[0].digest
    assert _captions_in_clip_order(crashed_attempt.uri, version=result.ending_version) == [
        f"{RETRY_TAG} caption for {clip_id}" for clip_id in ALL_CLIPS
    ]


def test_a_changed_model_revision_reuses_nothing_from_the_crashed_attempt(
    crashed_attempt: Attempt,
    fake_engine: FakeEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repinned model is different semantics even under an unchanged model id.

    The revision is what actually decides which weights generate the caption, so
    reusing rows across a repin would publish text the new weights never
    produced - and ``caption_model``, which records only the id, could not tell
    the two populations apart afterwards.
    """
    repinned = attrs.evolve(resolve_registered_model(DEFAULT_MODEL_VARIANT), revision="a-different-commit")
    monkeypatch.setattr(pipeline, "resolve_registered_model", lambda _key: repinned)
    retry = fake_engine(caption_tag=RETRY_TAG)

    result = run_config(crashed_attempt.config)

    assert retry.clip_ids == list(ALL_CLIPS)
    assert result.contract_digest != crashed_attempt.staged[0].digest
    assert _captions_in_clip_order(crashed_attempt.uri, version=result.ending_version) == [
        f"{RETRY_TAG} caption for {clip_id}" for clip_id in ALL_CLIPS
    ]


def test_an_execution_only_change_still_recovers_the_crashed_attempts_rows(
    crashed_attempt: Attempt,
    caption_config: CaptionConfig,
    fake_engine: FakeEngine,
) -> None:
    """Retuning throughput must not discard completed GPU work.

    The counterpart to the semantic tests above, and the reason the digest draws
    the line where it does: ``media_attempts`` is a retry budget that changes no
    caption's text, so a retry that raised it has to land in the same namespace
    and recover the same rows. Folding execution policy into the digest
    would make every retune a recaption of the corpus.
    """
    retry = fake_engine(caption_tag=RETRY_TAG)

    result = run_config(caption_config(crashed_attempt.uri, media_attempts=5))

    assert retry.clip_ids == list(UNFINISHED)
    assert result.contract_digest == crashed_attempt.staged[0].digest
    assert _captions_in_clip_order(crashed_attempt.uri, version=result.ending_version) == [
        f"{FIRST_ATTEMPT_TAG} caption for {clip_id}" if clip_id in FINISHED else f"{RETRY_TAG} caption for {clip_id}"
        for clip_id in ALL_CLIPS
    ]


def test_a_stale_digest_row_is_recaptioned_and_overwritten(
    published_once: Attempt,
    fake_engine: FakeEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A published row under a superseded contract is owed again, and the new text wins.

    This is the latest-contract-wins half of the pending predicate, and it is the
    half a status-only predicate cannot express: every row here already carries
    ``caption_status = 'success'``, so only the digest comparison can select it.
    """
    monkeypatch.setattr(prompt, "PROMPT_VERSION", "a_superseded_prompt_contract")
    retry = fake_engine(caption_tag=RETRY_TAG)

    result = run_config(published_once.config)

    assert retry.clip_ids == list(ALL_CLIPS)
    assert result.published_rows == CLIPS
    assert _captions_in_clip_order(published_once.uri, version=result.ending_version) == [
        f"{RETRY_TAG} caption for {clip_id}" for clip_id in ALL_CLIPS
    ]
    assert caption_harness.published_digests(published_once.uri, version=result.ending_version) == {
        result.contract_digest
    }


def test_a_rerun_under_an_unchanged_contract_owes_nothing(
    published_once: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """The idempotent case: nothing is captioned, nothing is committed, nothing fails.

    A monitoring loop re-runs this constantly, so it must be both free and quiet.
    ``ending_version`` still names where the table is canonical, which is the
    version the previous attempt committed.
    """
    retry = fake_engine(caption_tag=RETRY_TAG)

    result = run_config(published_once.config)

    assert (retry.calls, result.published_rows) == (0, 0)
    assert result.ending_version == published_once.read_version
    assert _captions_in_clip_order(published_once.uri, version=None) == [
        f"{FIRST_ATTEMPT_TAG} caption for {clip_id}" for clip_id in ALL_CLIPS
    ]


def test_broadening_scope_captions_only_the_previously_unselected_rows(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    caption_config: CaptionConfig,
    fake_engine: FakeEngine,
) -> None:
    """A broader attempt preserves valid captions and fills only the untouched rows."""
    uri, selected_config = local_run
    initially_unselected = ("c1", "c3", "c5", "c7")
    lance.dataset(uri).update(
        {CURATE_SELECTION_REASON: "'duplicate'"},
        where=f"{CLIP_ID_COLUMN} IN ({', '.join(repr(clip_id) for clip_id in initially_unselected)})",
    )
    selected_ledger = fake_engine(caption_tag=FIRST_ATTEMPT_TAG)

    selected_result = run_config(selected_config)
    broad_ledger = fake_engine(caption_tag=RETRY_TAG)
    broad_result = run_config(caption_config(uri, selected_only=False))

    assert selected_ledger.clip_ids == ["c0", "c2", "c4", "c6"]
    assert broad_ledger.clip_ids == list(initially_unselected)
    assert broad_result.contract_digest == selected_result.contract_digest
    assert _captions_in_clip_order(uri, version=broad_result.ending_version) == [
        f"{RETRY_TAG if clip_id in initially_unselected else FIRST_ATTEMPT_TAG} caption for {clip_id}"
        for clip_id in ALL_CLIPS
    ]


def test_a_newly_selected_row_becomes_work_without_a_new_caption_contract(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    fake_engine: FakeEngine,
) -> None:
    """A newer Lance version changes scope while preserving per-row caption semantics."""
    uri, config = local_run
    lance.dataset(uri).update(
        {CURATE_SELECTION_REASON: "'duplicate'"},
        where=f"{CLIP_ID_COLUMN} = 'c0'",
    )
    fake_engine(caption_tag=FIRST_ATTEMPT_TAG)
    first = run_config(config)
    lance.dataset(uri).update(
        {CURATE_SELECTION_REASON: repr(CURATE_SELECTED_VALUE)},
        where=f"{CLIP_ID_COLUMN} = 'c0'",
    )
    newly_selected = fake_engine(caption_tag=RETRY_TAG)

    second = run_config(config)

    assert newly_selected.clip_ids == ["c0"]
    assert second.contract_digest == first.contract_digest
    assert caption_harness.published_captions(uri, version=second.ending_version)["c0"] == (
        f"{RETRY_TAG} caption for c0"
    )


def test_deselecting_a_captioned_row_retains_its_caption(
    published_once: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """Selection controls future work and never destructively clears metadata."""
    before = caption_harness.published_captions(published_once.uri, version=None)["c0"]
    lance.dataset(published_once.uri).update(
        {CURATE_SELECTION_REASON: "'duplicate'"},
        where=f"{CLIP_ID_COLUMN} = 'c0'",
    )
    retry = fake_engine(caption_tag=RETRY_TAG)

    result = run_config(published_once.config)

    assert (retry.calls, result.published_rows) == (0, 0)
    assert caption_harness.published_captions(published_once.uri, version=result.ending_version)["c0"] == before


def test_a_replayed_shard_of_identical_rows_publishes_one_caption_per_clip(
    fully_staged_unpublished: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """A shard written twice is one answer written twice, so publication accepts it."""
    fake_engine(caption_tag=RETRY_TAG)
    _replay_a_shard(fully_staged_unpublished.config.staging_root)

    result = run_config(fully_staged_unpublished.config)

    assert result.published_rows == CLIPS
    assert _captions_in_clip_order(fully_staged_unpublished.uri, version=result.ending_version) == [
        f"{FIRST_ATTEMPT_TAG} caption for {clip_id}" for clip_id in ALL_CLIPS
    ]


def test_a_replayed_shard_that_disagrees_refuses_publication(
    fully_staged_unpublished: Attempt,
    fake_engine: FakeEngine,
) -> None:
    """Two differing captions for one clip are two answers, and neither is the right one."""
    fake_engine(caption_tag=RETRY_TAG)
    _replay_a_shard(fully_staged_unpublished.config.staging_root, caption="a disagreeing caption")

    with pytest.raises(CaptionPublicationError, match="conflicting staged caption rows"):
        run_config(fully_staged_unpublished.config)

    assert lance.dataset(fully_staged_unpublished.uri).count_rows(filter=f"{CAPTION_COLUMN} IS NOT NULL") == 0


def test_a_published_attempt_removes_the_namespace_it_staged_into(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    fake_engine: FakeEngine,
) -> None:
    """Cleanup follows the commit, so a successful attempt leaves no staged rows.

    Ordered after the commit rather than before it because until then the staged
    rows are the sole copy of finished GPU work, and generation is unseeded:
    deleting them early would not cost a re-run, it would cost different captions.
    """
    _, config = local_run
    fake_engine()

    result = run_config(config)

    assert result.published_rows == CLIPS
    assert caption_harness.namespaces(config.staging_root) == []


def test_a_cleanup_failure_still_reports_the_canonical_commit(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    fake_engine: FakeEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage error after the commit costs storage, not the run.

    The captions are already canonical by the time cleanup runs, so raising here
    would send an operator to investigate an attempt that in fact succeeded, and
    a retry would find nothing left to do. The leftover namespace is asserted
    because a swallowed failure that had actually deleted the tree would make
    this test pass while proving nothing.
    """
    uri, config = local_run

    def refuse_to_clean(*_args: object, **_kwargs: object) -> None:
        raise OSError(_INJECTED_CLEANUP_FAILURE)

    fake_engine()
    monkeypatch.setattr(pipeline, "cleanup_workspace", refuse_to_clean)

    result = run_config(config)

    assert result.published_rows == CLIPS
    assert _pending_rows(uri, result.contract_digest, version=result.ending_version) == 0
    assert caption_harness.namespaces(config.staging_root) != []


def test_the_configured_progress_setting_reaches_ray_data(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    caption_config: CaptionConfig,
    fake_engine: FakeEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator asking for progress output gets it configured, once, before the read.

    Recorded rather than called through: the real call mutates the process-wide
    Ray Data context, which would then leak into every later test in the session.
    What is under test is the wiring - that the configured value is the one
    handed over - not Ray's own honouring of the flag.
    """
    uri, _ = local_run
    observed: list[bool] = []
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", lambda *, progress: observed.append(progress))
    fake_engine()

    result = run_config(caption_config(uri, progress=True))

    assert result.published_rows == CLIPS
    assert observed == [True]


def test_publishing_captions_leaves_an_unrelated_source_column_unchanged(
    local_run: tuple[str, ResolvedRobotActionCaptionConfig],
    caption_config: CaptionConfig,
    fake_engine: FakeEngine,
) -> None:
    """A column the recipe does not own keeps its per-row values across the commit.

    What holds the property is the completeness of the fragment descriptor the
    commit carries, not the narrowness of ``fields_modified``: widening the field
    id list to a superset was measured NOT to detach an unrelated column, because
    the descriptor still names its file. So this guards against a redesign toward
    a whole-row rewrite, not against a one-token slip in the transaction.

    The column is added through Lance's own SQL-expression form so its values are
    computed per row and land in their own column file, which is the layout the
    commit's field scoping has to respect. Adding it also moves the version, so
    the config is rebuilt against the widened table.
    """
    uri, _ = local_run
    lance.dataset(uri).add_columns({UNRELATED_COLUMN: f"{CLIP_ID_COLUMN} || '-note'"})
    fake_engine()
    expected = _unrelated_values(uri, version=None)

    result = run_config(caption_config(uri))

    # Without this the assertion below is vacuous: an attempt that published
    # nothing also leaves the column alone.
    assert result.published_rows == CLIPS
    assert _unrelated_values(uri, version=result.ending_version) == expected


def _remains_of(
    uri: str,
    config: ResolvedRobotActionCaptionConfig,
    ledger: InferenceLedger,
) -> Attempt:
    """Return what an attempt left behind, read from the table and the staging root.

    The namespace list is read rather than recomputed, so a test comparing two
    attempts' namespaces is comparing what the runs actually produced. A
    successful attempt cleans its own namespace up, which is why the list can be
    empty here.
    """
    claimed = caption_harness.namespaces(config.staging_root)
    return Attempt(
        uri=uri,
        config=config,
        read_version=int(lance.dataset(uri).version),
        workspace_root=claimed[0] if claimed else "",
        ledger=ledger,
        staged=tuple(caption_harness.staged_rows(config.staging_root)),
    )


def _replay_a_shard(staging_root: str, *, caption: str | None = None) -> pathlib.Path:
    """Copy one staged shard beside itself, optionally rewriting its captions.

    Reproduces what a replayed write task leaves behind: the staged tree is
    appended to, so the second copy is a second file in the same fragment
    partition. With ``caption`` set, the copy answers differently.
    """
    original = next(iter(sorted(pathlib.Path(staging_root).rglob("results/**/*.parquet"))))
    replay = original.with_name(f"replay-{original.name}")
    if caption is None:
        shutil.copyfile(original, replay)
        return replay
    staged = pq.read_table(original)
    rewritten = staged.set_column(
        staged.schema.get_field_index(CAPTION_COLUMN),
        staged.schema.field(CAPTION_COLUMN),
        pa.array([caption] * staged.num_rows, staged.schema.field(CAPTION_COLUMN).type),
    )
    pq.write_table(rewritten, replay)
    return replay


def _pending_rows(uri: str, digest: str, *, version: int | None, selected_only: bool = True) -> int:
    """Return how many rows the table still owes a caption under one contract."""
    return int(
        lance.dataset(uri, version=version).count_rows(
            filter=caption_work_filter(digest, MODEL_ID, selected_only=selected_only)
        )
    )


def _captions_in_clip_order(uri: str, *, version: int | None) -> list[str | None]:
    """Return every clip's caption at one version, in clip order."""
    captions = caption_harness.published_captions(uri, version=version)
    return [captions[clip_id] for clip_id in ALL_CLIPS]


def _unrelated_values(uri: str, *, version: int | None) -> dict[str, str]:
    """Return the unrelated column keyed by clip id at one version."""
    table = lance.dataset(uri, version=version).to_table(columns=[CLIP_ID_COLUMN, UNRELATED_COLUMN])
    return {str(row[CLIP_ID_COLUMN]): str(row[UNRELATED_COLUMN]) for row in table.to_pylist()}
