# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify published caption verdicts, null-invariants, and semantic identity.

``caption_status`` is an on-disk format. Every assertion about it below therefore
compares against a LITERAL rather than against ``CaptionStatus`` itself: an enum
re-based onto different strings, or onto a shape whose members stringify as
``CaptionStatus.SUCCESS``, would keep a self-comparison green while every stored
row and every operator query changed underneath it.

``terminal_row`` accepts ``CaptionStatus`` by contract; it does not perform a
runtime type check. Its runtime refusals enforce the pair of null-invariants.

Most digest tests assert MOVEMENT: each digest is compared with another digest
built from one deliberately altered semantic input, so the direction of the
dependency is pinned. One literal golden digest additionally pins canonical
serialization; changing it requires an explicit contract-format decision. Both
the model identity and context fields are test-owned synthetic values, so a
registry or supported-column change elsewhere cannot turn these red.
"""

import enum
import json

import pyarrow as pa
import pytest

from cosmos_curator.models.registry import RegisteredModel
from cosmos_curator.next.recipes.robot_action_caption import contracts
from cosmos_curator.next.recipes.robot_action_caption.config import (
    KIND_NAME,
    ResolvedRobotActionCaptionConfig,
    RobotActionCaptionExecutionConfig,
    RobotActionCaptionGenerationConfig,
    RobotActionCaptionMediaConfig,
    RobotActionCaptionVideoConfig,
)
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_FIELDS,
    CAPTION_STATUS_COLUMN,
    TERMINAL_POLICY_VERSION,
    CaptionStatus,
    caption_contract_digest,
    caption_status,
    normalized_caption_contract,
    terminal_row,
)
from cosmos_curator.next.recipes.robot_action_caption.prompt import PromptContextField
from cosmos_curator.next.recipes.robot_action_caption.table import caption_work_filter

# Opaque to ``terminal_row``, which records the digest without inspecting it, so
# this deliberately is NOT a real digest: a well-formed one here would suggest the
# row builder validates a shape it never looks at.
DIGEST = "contract-digest"
MODEL_ID = "org/model"

# A pinned identity that belongs to no registry entry, so nothing below depends on
# which model the recipe currently ships.
MODEL = RegisteredModel(
    key="model_under_test",
    model_id="org/model-under-test",
    revision="a" * 40,
    precision="bfloat16",
)

# Two synthetic context fields standing in for whatever a snapshot resolves.
CONTEXT_FIELDS = (
    PromptContextField(column="task", label="Task"),
    PromptContextField(column="motion", label="Camera motion"),
)

# A cap inside the configurable range, chosen well away from the shipped default
# so a test that accidentally read the default would not agree with one that read
# this.
MAX_CLIP_BYTES = 8 * 1024 * 1024

# Distinctive explicit values keep contract tests independent of production
# defaults, whose own contract belongs to test_config.py.
GENERATION = RobotActionCaptionGenerationConfig(
    temperature=0.25,
    top_p=0.6,
    max_output_tokens=96,
    max_model_len=4096,
)
MEDIA = RobotActionCaptionMediaConfig(max_clip_bytes=MAX_CLIP_BYTES)

# Every video field is distinct, so a digest test that read the wrong one would
# not coincide with the right one.
VIDEO = RobotActionCaptionVideoConfig(
    min_frames=8,
    max_frames=12,
    fps=3.75,
    min_pixels=64 * 28 * 28,
    max_pixels=384 * 28 * 28,
)


def _normalized_contract(
    *,
    context_fields: tuple[PromptContextField, ...] = CONTEXT_FIELDS,
    generation: RobotActionCaptionGenerationConfig = GENERATION,
    video: RobotActionCaptionVideoConfig = VIDEO,
    media: RobotActionCaptionMediaConfig = MEDIA,
) -> dict[str, object]:
    """Build the normalized contract from test-owned semantic inputs."""
    return normalized_caption_contract(
        MODEL,
        context_fields,
        generation=generation,
        video=video,
        media=media,
    )


def _contract_digest(
    *,
    model: RegisteredModel = MODEL,
    context_fields: tuple[PromptContextField, ...] = CONTEXT_FIELDS,
    generation: RobotActionCaptionGenerationConfig = GENERATION,
    video: RobotActionCaptionVideoConfig = VIDEO,
    media: RobotActionCaptionMediaConfig = MEDIA,
) -> str:
    """Build the digest from test-owned semantic inputs."""
    return caption_contract_digest(
        model,
        context_fields,
        generation=generation,
        video=video,
        media=media,
    )


def test_every_caption_status_value_is_its_published_literal() -> None:
    """The three strings, and only those three, are what a caption row can carry.

    The whole member set is pinned rather than one value at a time so a fourth
    verdict cannot appear, and an existing one cannot vanish, without this test
    naming the change.
    """
    assert {member.name: str(member.value) for member in CaptionStatus} == {
        "SUCCESS": "success",
        "TRUNCATED": "truncated",
        "ERROR": "error",
    }


def test_a_caption_status_renders_as_its_value_when_interpolated() -> None:
    """An interpolated status yields the stored string, not the member's qualified name.

    This is the property that separates ``enum.StrEnum`` from the pre-3.11
    ``class X(str, Enum)`` shape, and it is not interchangeable: the latter's
    ``__str__`` returns ``CaptionStatus.SUCCESS``, so a filter built by
    interpolation would compare the column against that text and match no row at
    all while still executing successfully.
    """
    assert f"{CAPTION_STATUS_COLUMN} = '{CaptionStatus.SUCCESS}'" == "caption_status = 'success'"


def test_a_published_status_reaches_the_table_as_a_plain_string() -> None:
    """The value stored in the Arrow column is the literal, at the frozen column type."""
    row = terminal_row(status=CaptionStatus.SUCCESS, digest=DIGEST, model_id=MODEL_ID, caption="a caption")

    column = pa.Table.from_pylist([row], schema=CAPTION_FIELDS).column(CAPTION_STATUS_COLUMN)

    assert column.to_pylist() == ["success"]
    assert column.type == pa.string()
    assert type(column.to_pylist()[0]) is str


def test_a_successful_row_without_caption_text_is_refused() -> None:
    """A caption-less success has no correct publication, so it cannot be built.

    Exactly one invariant is violated - the error field stays null - so the
    message that fires attributes the refusal to the missing text.
    """
    with pytest.raises(ValueError, match="nonblank string caption"):
        terminal_row(status=CaptionStatus.SUCCESS, digest=DIGEST, model_id=MODEL_ID, caption="")


def test_an_error_row_carrying_a_caption_is_refused() -> None:
    """A row that both failed and captioned records two answers, and neither is safe.

    The error message is present and non-empty, so the only violated invariant is
    the non-null caption.
    """
    with pytest.raises(ValueError, match="null caption"):
        terminal_row(
            status=CaptionStatus.ERROR,
            digest=DIGEST,
            model_id=MODEL_ID,
            caption="a caption",
            error="it failed",
        )


def test_caption_status_is_a_str_enum() -> None:
    """Members are ``str`` instances, which is what lets one column carry both types.

    Production compares a status read back out of Arrow (a plain ``str``) against
    a member, and stamps a member into a row Arrow then stores. Both directions
    rest on this, and neither would raise if it stopped holding - the comparison
    would simply always be false.
    """
    assert issubclass(CaptionStatus, str)
    assert isinstance(CaptionStatus.SUCCESS, str)
    assert isinstance(CaptionStatus.SUCCESS, enum.StrEnum)


def test_an_empty_generation_is_an_error() -> None:
    """No text is a failure to caption, not a caption that happens to be empty."""
    assert (
        caption_status(
            generated_token_count=10,
            prompt_token_count=100,
            text="",
            generation=GENERATION,
        )
        is CaptionStatus.ERROR
    )


def test_a_whitespace_only_generation_is_an_error() -> None:
    """Text that renders as nothing is treated the same as no text at all."""
    assert (
        caption_status(
            generated_token_count=10,
            prompt_token_count=100,
            text="  \n\t ",
            generation=GENERATION,
        )
        is CaptionStatus.ERROR
    )


def test_an_empty_generation_that_reached_a_bound_is_still_an_error() -> None:
    """Text absence outranks either truncation bound, and has to.

    A bound-reaching empty generation classified as ``truncated`` would carry a
    null caption into ``terminal_row``, which refuses that pair - so the order of
    these two checks decides between one honest ``error`` row and an exception
    that ends the run.
    """
    status = caption_status(
        generated_token_count=GENERATION.max_output_tokens,
        prompt_token_count=GENERATION.max_model_len,
        text="",
        generation=GENERATION,
    )

    assert status is CaptionStatus.ERROR


def test_a_generation_at_the_token_cap_is_truncated() -> None:
    """The cap itself counts as truncated, because the two cases are indistinguishable.

    The pinned engine stage does not surface ``finish_reason``, so a generation
    that ended naturally on the cap's last token cannot be told apart from one
    the cap cut off. Calling it truncated is conservative; the other way round
    publishes a cut caption as complete, which is wrong data.
    """
    status = caption_status(
        generated_token_count=GENERATION.max_output_tokens,
        prompt_token_count=100,
        text="a caption",
        generation=GENERATION,
    )

    assert status is CaptionStatus.TRUNCATED


def test_a_generation_below_the_token_cap_is_a_success() -> None:
    """One token short of the cap is a complete caption, which pins the boundary's side."""
    status = caption_status(
        generated_token_count=GENERATION.max_output_tokens - 1,
        prompt_token_count=100,
        text="a caption",
        generation=GENERATION,
    )

    assert status is CaptionStatus.SUCCESS


def test_a_generation_that_filled_the_context_length_is_truncated() -> None:
    """A prompt that leaves too little room stops the generation below the cap.

    The generated count here is a small fraction of the sampling cap, so the cap
    alone reads this row as complete. What actually happened is that the prompt
    and its video left the generation nowhere to go, and a caption cut off by the
    context length is no more complete than one cut off by the cap.
    """
    generated = GENERATION.max_output_tokens // 4
    status = caption_status(
        generated_token_count=generated,
        prompt_token_count=GENERATION.max_model_len - generated,
        text="a caption",
        generation=GENERATION,
    )

    assert status is CaptionStatus.TRUNCATED


def test_a_generation_one_token_short_of_the_context_length_is_a_success() -> None:
    """Room left over means the generation stopped on its own, which pins the boundary."""
    generated = GENERATION.max_output_tokens // 4
    status = caption_status(
        generated_token_count=generated,
        prompt_token_count=GENERATION.max_model_len - generated - 1,
        text="a caption",
        generation=GENERATION,
    )

    assert status is CaptionStatus.SUCCESS


def test_the_same_inputs_produce_the_same_digest() -> None:
    """One set of semantics maps to one digest, or nothing downstream of it works.

    The predicate, the resume, and recovery reuse all compare digests for
    equality, so an unstable digest would recaption the corpus on every run while
    reporting success.
    """
    first = _contract_digest()
    second = _contract_digest()

    assert first == second


def test_the_contract_digest_matches_the_published_golden() -> None:
    """Keep canonical serialization of the test-owned contract stable.

    The video block stores the same produced mappings passed to vLLM. Every
    intentional format change requires updating this literal and recaptioning
    rows produced under the earlier semantic contract.
    """
    assert _contract_digest() == "ab6464be917033704c5fb68e94f1f466197d042a81c7746a39af39cb7195f358"


def test_the_digested_pixel_floor_is_the_floor_the_engine_is_given() -> None:
    """The floor is inside the identity key, because it can change what a caption says.

    ``smart_resize`` upscales a frame whose area falls below the floor, so for a
    low-resolution source the floor and not the cap is the bound that decides
    what the model sees. Asserting via the digest rather than the payload key is
    what gives this power: a floor recorded nowhere would leave two populations
    in one table with nothing to tell them apart.
    """
    raised = VIDEO.model_copy(update={"min_pixels": VIDEO.max_pixels})

    assert _contract_digest(video=raised) != _contract_digest()


def test_the_digest_is_accepted_by_the_pending_predicate() -> None:
    """The digest's shape must satisfy the validator its only consumer applies.

    A cross-module invariant, and the one place the two halves meet: the
    predicate refuses anything but 64 lowercase hex characters, so a producer
    that emitted a different shape would fail every run at the first query rather
    than at review.
    """
    digest = _contract_digest()

    assert caption_work_filter(digest, MODEL.model_id, selected_only=False).count(digest) == 1


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        pytest.param("generation", "temperature", 0.1, id="temperature"),
        pytest.param("generation", "top_p", 0.9, id="top-p"),
        pytest.param("generation", "max_output_tokens", 256, id="output-token-cap"),
        pytest.param("generation", "max_model_len", 16_384, id="model-context"),
        pytest.param("video", "min_frames", 4, id="frame-floor"),
        pytest.param("video", "max_frames", 16, id="frame-ceiling"),
        pytest.param("video", "fps", 1.0, id="sampling-rate"),
        pytest.param("video", "min_pixels", 96 * 28 * 28, id="pixel-floor"),
        pytest.param("video", "max_pixels", 512 * 28 * 28, id="pixel-cap"),
        pytest.param("media", "max_clip_bytes", MAX_CLIP_BYTES * 2, id="media-cap"),
    ],
)
def test_each_semantic_setting_changes_the_digest(section: str, field: str, value: float) -> None:
    """Invalidate prior work when any result-defining value changes."""
    generation = GENERATION
    video = VIDEO
    media = MEDIA
    if section == "generation":
        generation = GENERATION.model_copy(update={field: value})
    elif section == "video":
        video = VIDEO.model_copy(update={field: value})
    else:
        media = MEDIA.model_copy(update={field: value})

    digest = _contract_digest(generation=generation, video=video, media=media)

    assert digest != _contract_digest()


def test_the_resolved_context_subset_changes_the_digest() -> None:
    """Captioning with fewer context fields is captioning under different semantics.

    The subset is resolved against the source snapshot, so this is how a column
    landing upstream recaptions the corpus: the resolved set grows, the digest
    moves, and the existing rows are honestly marked as produced without it.
    """
    both = _contract_digest()
    narrowed = _contract_digest(context_fields=CONTEXT_FIELDS[:1])

    assert both != narrowed


def test_reordering_the_context_fields_changes_the_digest() -> None:
    """Order is rendered, so it is semantic and not merely presentational.

    The fields become labelled prompt lines in the order given, so a reorder
    changes the text the model reads. The digest has to follow, which means the
    context block cannot be normalized to a set.
    """
    forward = _contract_digest()
    reversed_order = _contract_digest(context_fields=tuple(reversed(CONTEXT_FIELDS)))

    assert forward != reversed_order


def test_relabelling_a_context_field_changes_the_digest() -> None:
    """The label reaches the model, so changing it changes the caption.

    Distinct from the reorder above: this holds the field set and its order fixed
    and moves only the text in front of one value, which pins that the label is
    digested rather than just the column name.
    """
    first, *rest = CONTEXT_FIELDS
    original = _contract_digest()
    relabelled = _contract_digest(
        context_fields=(PromptContextField(column=first.column, label="Objective"), *rest),
    )

    assert original != relabelled


def test_a_different_model_revision_changes_the_digest() -> None:
    """The pinned revision is the weights, so the same repository at two commits differs.

    Asserted on the revision rather than the model id because the id is the field
    an operator notices: a registry re-pin that keeps the id and moves the commit
    is the change most likely to go unnoticed, and it produces different captions.
    """
    original = _contract_digest()
    re_pinned = _contract_digest(
        model=RegisteredModel(key=MODEL.key, model_id=MODEL.model_id, revision="b" * 40, precision=MODEL.precision),
    )

    assert original != re_pinned


def test_alias_spelling_does_not_change_resolved_model_identity() -> None:
    """Share recovery when two aliases resolve to the same artifact identity."""
    alternate_alias = RegisteredModel(
        key="alternate_alias",
        model_id=MODEL.model_id,
        revision=MODEL.revision,
        precision=MODEL.precision,
    )

    assert _contract_digest(model=alternate_alias) == _contract_digest()


@pytest.mark.parametrize(
    "name",
    ["TEMPERATURE", "TOP_P", "MAX_OUTPUT_TOKENS", "MAX_MODEL_LEN", "NUM_FRAMES"],
)
def test_configurable_behavior_has_no_contracts_module_global(name: str) -> None:
    """Use config models as the single source for configurable ML values."""
    assert not hasattr(contracts, name)


def test_the_request_skeleton_is_read_from_the_constants_it_stamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing what the model is shown must move the digest by construction.

    A skeleton restated as literals here would hash identically however the
    request was edited, and the damage is silent and permanent: admitting a
    second video item changes every caption while the pending predicate keeps
    skipping the rows the new setting would caption differently, leaving one
    table holding two populations with nothing to tell them apart.
    """
    original = _contract_digest()

    monkeypatch.setattr(contracts, "VIDEO_ITEMS_PER_PROMPT", contracts.VIDEO_ITEMS_PER_PROMPT + 1)

    assert _contract_digest() != original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("storage_profile", "profile_under_test", id="storage-profile"),
        pytest.param("inference_concurrency", 4241, id="inference-concurrency"),
        pytest.param("pipeline_parallel_size", 3, id="pipeline-parallel-size"),
        pytest.param("max_num_batched_tokens", 27_011, id="scheduler-token-budget"),
        pytest.param("media_attempts", 7, id="media-attempts"),
        pytest.param("progress", True, id="progress"),
    ],
)
def test_each_execution_setting_leaves_the_digest_unchanged(field: str, value: object) -> None:
    """Keep completed recovery work valid while retuning execution policy."""
    baseline = ResolvedRobotActionCaptionConfig.model_validate(
        {
            "schema_version": 1,
            "kind": KIND_NAME,
            "input": {"clips_lance_uri": "s3://bucket-under-test/clips.lance"},
        }
    )
    changed = baseline.model_copy(
        update={"execution": baseline.execution.model_copy(update={field: value})},
    )

    baseline_digest = _contract_digest(
        generation=baseline.generation,
        video=baseline.video,
        media=baseline.media,
    )
    changed_digest = _contract_digest(
        generation=changed.generation,
        video=changed.video,
        media=changed.media,
    )

    assert changed.execution != baseline.execution
    assert changed_digest == baseline_digest


def test_execution_fields_and_values_never_enter_the_contract() -> None:
    """Exclude execution names and distinctive values from the payload."""
    execution = RobotActionCaptionExecutionConfig(
        storage_profile="profile_under_test",
        inference_concurrency=4241,
        pipeline_parallel_size=3,
        max_num_batched_tokens=27_011,
        media_attempts=987_654_321,
        progress=True,
    )
    payload = json.dumps(_normalized_contract(), sort_keys=True)

    assert [name for name in type(execution).model_fields if name in payload] == []
    assert execution.storage_profile not in payload
    assert str(execution.inference_concurrency) not in payload
    assert str(execution.max_num_batched_tokens) not in payload
    assert str(execution.media_attempts) not in payload


def test_the_digested_video_block_is_the_mapping_pair_the_engine_receives() -> None:
    """Digest the produced kwargs, not the fields they were derived from.

    Every factor of the visual token term reaches the model through exactly these
    two mappings, so hashing them covers the frame bounds, the sampling rate and
    the pixel pair together - and it covers them under the key names vLLM reads,
    which is what makes a rename of one a digest change rather than a silent
    re-interpretation. Pinned by CONTENT here, while
    ``test_each_semantic_setting_changes_the_digest`` pins that a changed value
    moves the digest: a block built from constants would satisfy only one.
    """
    video = _normalized_contract()["video"]

    assert isinstance(video, dict)
    assert set(video) == {"media_io_kwargs", "mm_processor_kwargs", "do_sample_frames"}
    assert video["media_io_kwargs"] == VIDEO.media_io_kwargs()
    assert video["mm_processor_kwargs"] == VIDEO.mm_processor_kwargs()


def test_omitted_video_controls_are_digested_as_empty_mappings() -> None:
    """Record backend-owned temporal and pixel behavior without invented values."""
    inherited = RobotActionCaptionVideoConfig()

    video = _normalized_contract(video=inherited)["video"]

    assert isinstance(video, dict)
    assert video["media_io_kwargs"] == {}
    assert video["mm_processor_kwargs"] == {}


def test_temporal_overrides_do_not_invent_digested_pixel_controls() -> None:
    """Keep independently omitted pixel behavior backend-owned in the digest."""
    temporal_only = RobotActionCaptionVideoConfig(fps=2.25, min_frames=5, max_frames=33)

    video = _normalized_contract(video=temporal_only)["video"]

    assert isinstance(video, dict)
    assert video["media_io_kwargs"] == temporal_only.media_io_kwargs()
    assert video["mm_processor_kwargs"] == {}


def test_explicit_runtime_video_controls_move_the_digest_from_backend_ownership() -> None:
    """Do not reuse captions when runtime YAML takes video-policy ownership."""
    inherited_digest = _contract_digest(video=RobotActionCaptionVideoConfig())
    explicit_digest = _contract_digest(
        video=RobotActionCaptionVideoConfig(
            fps=3.0,
            min_frames=4,
            max_frames=64,
            min_pixels=3_136,
            max_pixels=602_112,
        ),
    )

    assert explicit_digest != inherited_digest


def test_the_contract_payload_carries_exactly_the_published_blocks() -> None:
    """A new top-level block is a corpus-wide recaption, so it cannot arrive silently.

    The key set is pinned rather than derived: every block here is a deliberate
    digest input, and adding one is exactly the change whose cost a reviewer needs
    to be forced to acknowledge.
    """
    assert set(_normalized_contract()) == {
        "model",
        "prompt",
        "request",
        "sampling",
        "video",
        "media",
        "terminal",
    }


def test_the_terminal_block_declares_both_truncation_bounds() -> None:
    """A verdict rule the digest omits would let two rules stamp rows identically.

    The classifier measures a generation against the sampling cap and against the
    context length, so the block that describes the verdict has to answer for
    both. The context length being a sampling input is not enough on its own -
    that records only which engine was built, not how its output was judged. This
    test pins bound presence by value; the literal golden digest separately pins
    terminal key names.
    """
    terminal = _normalized_contract()["terminal"]

    assert isinstance(terminal, dict)
    assert GENERATION.max_output_tokens in terminal.values()
    assert GENERATION.max_model_len in terminal.values()


def test_the_terminal_policy_version_is_digested() -> None:
    """Digest classifier semantics independently of numeric bounds."""
    terminal = _normalized_contract()["terminal"]

    assert isinstance(terminal, dict)
    assert terminal["policy_version"] == TERMINAL_POLICY_VERSION == 1


def test_prompt_normalization_identity_is_exact() -> None:
    """Name whitespace collapse and null-or-blank omission exactly."""
    prompt = _normalized_contract()["prompt"]

    assert isinstance(prompt, dict)
    assert prompt["normalization"] == "collapse_whitespace_drop_null_or_blank"


def test_the_contract_payload_survives_a_canonical_json_round_trip() -> None:
    """The payload must be JSON-serializable, because the digest is taken over its text.

    An enum member or other non-primitive would hash as whatever ``json`` made of
    it, or fail at the first run rather than here.
    """
    payload = _normalized_contract()

    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
