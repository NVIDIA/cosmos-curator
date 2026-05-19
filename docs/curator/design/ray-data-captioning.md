# Ray Data Captioning Design

## Summary

The Ray Data splitting pipeline generates Qwen captions by combining two existing strengths:

- Ray Data LLM owns the vLLM engine actors, GPU scheduling, and continuous batching.
- The existing Xenna/Qwen preparation path still owns video windowing, prompt construction, and model-specific
  multimodal input construction.

The main compromise is a private bridge between those layers. Clip rows carry a nested `caption_windows` list. Each
window stores decoded frames in Arrow-friendly fields (`video_frame_bytes`, `video_frame_shape`, `video_frame_dtype`,
and optional metadata). A private Ray vLLM engine-stage shim expands those nested windows into vLLM requests inside the
actor, reconstructs vLLM's nested `multimodal_data` immediately before inference, then collapses the outputs back into
one captioned clip row.

This keeps the Ray Data pipeline aligned with upstream Ray primitives while avoiding Ray Data's Python object-extension
fallback for nested tensors.

## Scope

The first version supports the default Qwen2.5-VL captioning path for the Ray Data fixed-stride splitting pipeline.

In scope:

- Default model: `qwen` (`Qwen/Qwen2.5-VL-7B-Instruct`).
- Default captioning enabled, with `--no-generate-captions` preserving splitting-only behavior.
- Ray Data LLM processor for vLLM execution.
- Xenna/Qwen frame preparation via the shared `vllm_interface` helpers.
- Per-window caption metadata, per-clip metadata JSON, and `summary.json` token totals.
- Xenna-style caption throughput logging.

Out of scope for the first version:

- Model selection, prompt selection, or sampling CLI flags.
- Stage-2 captioning.
- Gemini, OpenAI, Nemotron, Qwen3, Cosmos Reason, filters, T5, previews, or enhancement support.
- A custom vLLM GPU actor.
- Changes to the existing `vllm_async` path.

## Pipeline Shape

```text
video paths
  -> read video
  -> fixed-stride split spans
  -> transcode clips
  -> write MP4 bytes
  -> attach caption_windows to each clip row
  -> ray.data.llm vLLM processor (expand/collapse windows inside the actor)
  -> distributed per-clip metadata JSON write
  -> write summary and token stats
```

The clip writer has two modes:

- Splitting-only mode writes the MP4 and stub metadata JSON, then drops `clip_bytes`.
- Captioning mode writes the MP4, keeps `clip_bytes` and clip metadata on the row, and defers final metadata JSON until
  caption windows have been generated.

Captioning stays in a single Ray Data branch. A previous `filter(...needs caption...)` plus `filter(...skip...)` plus
`union(...)` shape caused Ray Data to re-execute the shared parent and duplicate clip decode/window preparation. No-window
clips now move through the same processor branch with a skipped window entry and a non-empty `__inference_error__`, which
Ray's LLM stages already treat as a pass-through row. The final normalizer converts those windows back into the
skipped-caption metadata shape.

The captioning branch keeps one row per clip. This avoids the earlier `groupby("clip_uuid").map_groups(...)` shuffle
that was needed after fanning out to one row per caption window. Metadata JSON writing is now a distributed per-clip
`map`, and the driver only collects small per-clip rows for `summary.json`.

## Key Decisions

### Use `ray.data.llm`, not a custom vLLM actor

Ray Data LLM already provides the hard parts of an LLM inference stage: actor lifecycle, GPU scheduling, continuous
batching, and vLLM request overlap. A custom GPU actor that owns vLLM would need to reimplement enough request queueing
and cross-call continuous batching to be competitive.

The cost is relying on a private shim around Ray's vLLM engine stage. For the first version, that is smaller and less
risky than owning a custom LLM serving layer.

### Reuse Xenna/Qwen input preparation

The Ray Data path intentionally uses the existing Qwen defaults:

```python
VllmConfig(model_variant="qwen", preprocess=False, num_gpus=1)
WindowConfig(model_does_preprocess=False, preprocess_dtype="float16")
VllmSamplingConfig()
```

It also reuses `split_video_into_windows`, `make_metadata`, `make_model_inputs`, and the default prompt helper. Ray's
higher-level chat-template, tokenization, and multimodal-preparation stages are disabled because the shared Xenna path
already constructs the vLLM-ready inputs.

### Keep multimodal frames Arrow-friendly

Passing raw nested `multimodal_data` through Ray Data blocks forces Ray to fall back to Python object-extension Arrow
serialization for tensors. The Ray Data captioning path avoids that by storing each nested caption window's video frames
as:

- raw frame bytes in a large-binary scalar;
- shape;
- dtype;
- optional video metadata.

The vLLM engine-stage shim reconstructs the numpy array and vLLM `multimodal_data` dict inside the actor, immediately
before Ray creates the vLLM request.

### Keep one row per clip through captioning

Ray's vLLM stage normally expects one Ray row per vLLM request, but the durable artifact boundary is the clip metadata
JSON. Fanning out to one row per caption window makes vLLM integration simple, but it requires a later regroup by
`clip_uuid` before writing metadata. At benchmark scale that shuffle can fail, and driver-side metadata writing is too
expensive.

The current path therefore keeps one Ray row per clip and stores the per-window requests in `caption_windows`. The
private vLLM stage shim expands those windows only inside the actor, submits them to vLLM concurrently, and returns one
updated clip row. This preserves Ray Data LLM's actor lifecycle and vLLM ownership while avoiding the shuffle.

### Keep the CLI narrow

The first version exposes only behavior that is already needed:

- `--no-generate-captions` to preserve splitting-only runs.
- `--model-weights-path` for the shared model-download path.
- `--vllm-max-num-seqs`, defaulting to `64`. Ray Data batch size is derived from this value and Ray's default per-actor
  in-flight task count. The default works well on A100/H100/B200 benchmark runs; GB200 benchmark runs benefited from
  higher values.
- `--progress/--no-progress`, defaulting to `--no-progress`, so redirected or tee'd logs stay readable.

There is no verbose-caption logging flag. Caption debug logging through Ray workers was too noisy and inconsistent to
include in the first version. There is also no `max_num_batched_tokens` CLI flag; benchmark sweeps did not show enough
throughput sensitivity to justify exposing it, so it stays fixed at `32768`.

### Let GPUs, not Ray memory resources, limit vLLM actors

Ray may warn that the vLLM engine-stage task uses several GiB of host memory without requesting a Ray `memory=`
resource. The first version does not set a custom memory resource. For the default single-GPU vLLM actor path, Ray's
scheduler is already GPU-bound, and an empirical memory reservation would add a tuning knob without changing expected
concurrency.

## Private Compatibility Shims

The Ray Data captioning helper keeps all Ray/vLLM compatibility logic private to
`cosmos_curator/pipelines/ray_data/_vllm_caption.py`.

Current shims:

- Replace Ray's vLLM engine-stage UDF with a subclass that recognizes one-clip rows with nested `caption_windows`,
  expands those windows into internal vLLM requests, reconstructs `multimodal_data` from Arrow-friendly frame fields,
  and collapses outputs back into one clip row.
- Patch `vllm.inputs.data.TextPrompt` and `TokensPrompt` inside the actor for the pinned Ray/vLLM namespace mismatch.
- Set `CUDA_VISIBLE_DEVICES` from `ray.get_gpu_ids()` before vLLM spawns engine processes.
- Use Ray LLM's `__inference_error__` pass-through behavior for no-window rows.

These shims are deliberately narrow, local, and covered by focused tests. They are the main technical debt in this
design.

## Metadata and Stats

Caption windows use the existing normalized outcome vocabulary:

- `success`
- `truncated`
- `error`

`truncated` is inferred when non-empty text is produced and `num_generated_tokens >= VllmSamplingConfig().max_tokens`.
Successful and truncated windows include prompt/output token counts. Error windows include the failure reason and error
text where available.

`summary.json` includes raw caption totals:

- `total_num_clips_with_caption`
- `total_num_caption_windows`
- `total_prompt_tokens`
- `total_output_tokens`
- `output_tokens_per_s`

When captioning runs, the pipeline also logs a Xenna-style throughput block for easy comparison.

## Follow-Ups

- Track Ray/vLLM internal API stability around `vLLMEngineStage`, `stage.fn`, and `__inference_error__` pass-through.
- Consider filing a Ray Data feature request for prebuilt multimodal vLLM inputs that can carry Arrow-native frame
  payloads without a private engine-stage shim.
- Add a GPU budget allocator before introducing more GPU stages, so captioning does not automatically claim every
  visible GPU.
- Revisit driver-side summary aggregation (`take_all()`) if per-clip summary row volume becomes large enough to matter.
- Keep MP4 path/data URL or JPEG-frame payloads as fallback options if the private shim becomes too brittle, but do not
  prefer them while the Arrow-native frame bridge is stable.
