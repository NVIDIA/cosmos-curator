# Multimodal AV Captioning

## Summary

`multimodal-caption` enriches the canonical `episodes.lance` table produced by
[`multimodal-split`](multimodal-av-episode-publication.md). The recipes are independently runnable: splitting publishes
complete episode rows, and captioning later selects one explicitly configured camera from each row. Captioning does not
invoke splitting, rediscover source recordings, or assemble camera sets.

The initial `video_only` variant runs the existing complete-MP4 free-form Qwen caption treatment once per episode. It
uses the same prompt, 2 FPS sampling, model pins, terminal states, durable checkpointing, recovery, staging, and
fragment-atomic Lance publication engine documented for [`video-caption`](curator-next-video-captioning.md).
`video-caption` retains its existing `CLIP_SCHEMA`, `clip_id`, and single-MP4 public contract; the recipes share only
private execution machinery.

## Input and camera binding

The input must be a canonical CVC multimodal episode table with globally unique `episode_id` values. The recipe pins a
Lance input version before camera preflight and requires one non-empty `context.camera_key`. It resolves exactly one
matching descriptor from every row's complete, lexicographically serialized `camera_media` list. Missing or duplicate
camera keys fail before inference; list position is never a camera-selection policy.

Before inference, each selected complete MP4 is checked against its descriptor's exact byte size and SHA-256. The
visual binding carried through checkpointing and publication is:

```text
(episode_id, materialization_digest, camera_key,
 media_location, media_size_bytes, media_sha256)
```

`episode_id` provides the episode join. `materialization_digest` binds the result to the complete published camera set,
while the selected-camera values bind the exact bytes used by the model.

## Public configuration

Configuration is strict, frozen, and versioned. Unknown fields are rejected. For example:

```yaml
schema_version: 1
kind: multimodal-caption

input:
  episodes_lance_uri: s3://example-bucket/curated/multimodal-split/episodes.lance

context:
  variant: video_only
  camera_key: front_wide  # Example only; there is no default camera.

model:
  variant: qwen3_8_27b_fp8

output: {}

execution:
  storage_profile: default
```

`context.variant` is required and currently accepts only `video_only`. `output.staging_root_uri` defaults to
`<episodes.lance parent>/staging/multimodal-caption`. Model weights must already exist under the standard model mount on
every inference worker; the recipe never downloads them.

## Result contract

The recipe atomically registers one nullable caption field and its nullable metadata field, owned by
`multimodal-caption`:

- `caption_video_only_qwen3_8_27b_fp8_v1` and its `_metadata` field; or
- `caption_video_only_qwen3_8_27b_v1` and its `_metadata` field.

Each terminal metadata value records the schema version, `video_only` variant, caption contract digest, episode and
materialization identities, selected camera key and MP4 hash, terminal status, pinned model identity and revision,
prompt, token counts, and error details. The result identity is:

```text
(episode_id, materialization_digest, video_only, caption_contract_digest)
```

Terminal values follow the shared `success`, `truncated`, and `error` semantics. Registration, checkpoint reuse, and
publication fail closed if the episode binding, schema, field ownership, contract, or physical source-row identity has
changed. A successful fragment commit remains canonical even if later work fails.

Each model/variant field set holds terminal results for only one camera contract. Changing `camera_key` changes the
caption contract digest and derived workspace, but it still targets the same two Lance columns because camera names are
not part of the public field names. If those columns already contain terminal rows from another camera contract,
fragment classification rejects the rerun as incompatible. Publishing another camera requires a separately named and
versioned field set.

## Extension boundary

Eligible later free-form context variants can publish sibling versioned contracts alongside `video_only`; they do not
replace it in place. Multiview layouts, audio, split-and-caption orchestration, train-of-cognition reasoning, and
Auto-Labeler integration are outside this recipe. Those capabilities remain independent consumers joined through the
canonical episode spine.
