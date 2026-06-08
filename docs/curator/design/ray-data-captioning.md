# Ray Data Captioning Design

## Summary

The Ray Data splitting pipeline captions clips with Ray Data LLM, vLLM, and the existing Xenna/Qwen preparation path.
Ray Data LLM owns the vLLM actors and GPU scheduling. The Xenna/Qwen helpers own video windowing, prompt construction,
and model-specific multimodal input construction.

Captioning keeps one Ray Data row per clip. Each clip row carries a nested `caption_windows` list. A private Ray vLLM
engine-stage shim expands those windows into vLLM requests inside the actor, reconstructs vLLM `multimodal_data` just
before inference, and writes the outputs back into the same clip row.

## Pipeline Shape

```text
video paths
  -> read video
  -> selected split spans (TransNetV2 by default, fixed-stride optionally)
  -> transcode clips
  -> write MP4 bytes
  -> attach caption_windows to each clip row
  -> ray.data.llm vLLM processor
  -> distributed per-clip metadata JSON write
  -> write summary and token stats
```

In captioning mode, the clip writer writes the MP4 and keeps `clip_bytes` plus clip metadata on the row. Final metadata
JSON is written after caption windows are generated.

## Caption Row Shape

The caption preparation stage uses `ds.map(...)` to turn each clip row into one clip row with `caption_windows`.

Each caption window stores vLLM-ready request fields and Arrow-friendly video payload fields:

- `video_frame_bytes`
- `video_frame_shape`
- `video_frame_dtype`
- optional `video_metadata`
- prompt/token fields and sampling params

The Ray LLM vLLM engine stage uses `map_batches(...)`. Its input batch contains clip rows. The shim expands those rows
inside the actor:

```text
1 Ray batch = N clip rows
N clip rows = M caption windows
M caption windows = M vLLM requests/sequences
```

Rows with no captionable windows carry a skipped window entry and a non-empty `__inference_error__`, which Ray LLM
treats as a pass-through row. The final normalizer converts the row into the standard skipped-caption metadata shape.

## Model Defaults

The Ray Data path uses the existing Qwen defaults:

```python
VllmConfig(model_variant="qwen", preprocess_mode="curator", num_gpus=1, batch_size=32)
WindowConfig()
VllmSamplingConfig()
```

The processor disables Ray LLM's chat-template, tokenization, detokenization, image-preparation, and
multimodal-preparation stages because the Xenna/Qwen helpers already produce vLLM-ready inputs.

The vLLM engine kwargs mirror the existing Qwen path where needed:

- `limit_mm_per_prompt={"images": 0, "video": 1}`
- `max_model_len=32768`
- `gpu_memory_utilization=0.85`
- `mm_processor_cache_gb=4.0` unless disabled
- `max_num_batched_tokens=32768`
- `tensor_parallel_size=VllmConfig.num_gpus`
- `performance_mode=VllmConfig.performance_mode`

## Scheduling Parameters

### `concurrency`

`concurrency` is the Ray Data LLM actor-pool size. The captioning path uses:

```python
concurrency=(1, caption_workers)
```

Each actor owns one vLLM engine replica. `caption_workers` is derived from the cluster GPU count returned by
`download_models()` and `VllmConfig.num_gpus`. On a two-GPU machine with one GPU per actor, `caption_workers=2`, so
captioning can scale from one actor to two actors.

### `batch_size`

`batch_size` is the Ray LLM `map_batches` input size. In this pipeline the unit is clip rows, not caption windows and
not vLLM sequences. The default value comes from `VllmConfig.batch_size`, currently `32`. The CLI
`--caption-batch-size` can override it for benchmark and throughput tuning runs.

With `batch_size=32`, one vLLM actor call receives up to 32 clip rows. Those clips may expand into more than 32
caption-window requests.

### `max_concurrent_batches`

`max_concurrent_batches` is per-actor Ray method concurrency for the vLLM stage. It limits how many Ray batch calls may
run at the same time on one vLLM actor.

The caption path uses Ray LLM's default value. In the currently pinned Ray version, that default is `8`.

### `max_tasks_in_flight_per_actor`

`max_tasks_in_flight_per_actor` is Ray Data actor-pool queue depth per actor. It is the number of Ray batch tasks that
may be assigned to an actor, including running plus queued tasks. It does not increase execution concurrency beyond
`max_concurrent_batches`.

The caption path uses Ray LLM's default value. In the currently pinned Ray version, that default is `16`.

This queue depth is also what lets Ray Data's autoscaling actor pool observe enough pressure to add vLLM actors.

## Practical Sizing Model

With Ray LLM defaults, `batch_size=32`, and one GPU per vLLM actor:

```text
per actor:
  max_concurrent_batches = 8
  max_tasks_in_flight_per_actor = 16
  actively running clip rows <= 8 * 32 = 256
  assigned clip rows <= 16 * 32 = 512
```

Those clip rows expand into caption-window requests inside the actor. vLLM then admits the requests according to its
sequence, token, and memory limits.

## Public CLI

Captioning keeps a narrow CLI surface:

- `--no-generate-captions`
- `--model-weights-path`
- `--caption-batch-size`
- `--progress/--no-progress`

Ray LLM scheduling knobs and vLLM scheduler capacity use their library defaults.

## Compatibility Shims

Most private compatibility logic lives in `cosmos_curator/pipelines/ray_data/_vllm_caption.py`.

Current shims:

- Replace Ray's vLLM engine-stage UDF with a subclass that understands clip rows with nested `caption_windows`.
- Reconstruct `multimodal_data` from Arrow-friendly frame fields inside the actor.
- Patch `vllm.inputs.data.TextPrompt` and `TokensPrompt` inside the actor for the pinned Ray/vLLM namespace mismatch.
- Use Ray LLM's `__inference_error__` pass-through behavior for no-window rows.

GPU Ray Data stages use `ray_data_gpu_runtime_env(...)`, which sets
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0` and restores Ray's `CUDA_VISIBLE_DEVICES` masking while the shared Xenna
image still defaults the flag to `1`. The helper is shared with the TransNetV2 splitter.

## Metadata and Stats

Caption windows use the normalized outcome vocabulary:

- `success`
- `truncated`
- `error`

`truncated` is inferred when non-empty text is produced and `num_generated_tokens >= VllmSamplingConfig().max_tokens`.
Successful and truncated windows include prompt/output token counts. Error windows include the failure reason and error
text where available.

`summary.json` includes caption totals:

- `total_num_clips_with_caption`
- `total_num_caption_windows`
- `total_prompt_tokens`
- `total_output_tokens`
- `output_tokens_per_s`

When captioning runs, the pipeline also logs a Xenna-style throughput block.
