# vLLM Async Captioning Guide

## Overview

The `vllm_async` captioning algorithm runs an in-process `AsyncLLM`
engine within each Ray worker actor.  Per-model concerns
(engine construction, request formatting, output decoding) are owned by
**`VllmPlugin`** subclasses (`cosmos_curator/models/vllm_*.py`) -- the
**same** plugins the synchronous pipeline uses.  Per-variant numeric
tuning (`MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`,
`MAX_NUM_BATCHED_TOKENS`, `TRUST_REMOTE_CODE`, `LIMIT_MM_PER_PROMPT_*`)
lives as module-scope constants on each plugin and is read by both
sync `model()` and async `model_async()` - single source of truth, no
drift between sync and async pipelines.

`VllmAsyncConfig` (in `cosmos_curator/pipelines/video/utils/data_model.py`)
holds **only user-tunable knobs**: `num_gpus`, `data_parallel_size`,
`fp8`, `disable_mmcache`, async-only engine knobs (`async_scheduling`,
`distributed_executor_backend`, etc.), and runtime tuning
(`max_num_seqs`, `kv_cache_dtype`, etc.).

## Architecture

Two-stage pipeline (three with previews enabled):

```
VllmPrepStage (CPU; reused from sync)  -->  VllmAsyncCaptionStage (GPU, continuous mode)
  decode + smart_resize frames                 vllm_caption_async (in vllm_interface)
  processor.apply_chat_template -> tokens      -> _AsyncCaptioner: per-window tenacity
  plugin.make_llm_input (final shape)             retry, on_window_done/on_window_error
  write to window.model_input[variant]            callbacks, plugin.make_refined_llm_request
                                                  for stage-2
```

Prep ships the **fully-formed vLLM input dict** (prompt token IDs plus
zero-copy frame tensors in `multi_modal_data`) under
`window.model_input[serve_config.model_variant]`. The async caption
stage reuses sync's `VllmPrepStage` verbatim, so smart-resize and
tokenization are deterministic and identical across both pipelines --
**no renderer call happens on the GPU side**.

Captions and token counts are always written under the constant
user-facing key `window.caption["vllm_async"]` (and
`window.token_counts["vllm_async"]`), independent of the underlying
model variant. The split between the read key (`model_variant`) and
the write key (`"vllm_async"`) keeps the output schema stable across
variant swaps without touching sync prep.

The caption stage is a thin `cosmos_xenna.ray_utils.continuous_stage.ContinuousInterface`
shell driven by Xenna's `run_continuous` loop rather than per-batch
`process_data` calls. Per-window orchestration -- request lifecycle,
retry, stage-2 refinement, output decode -- lives inside
`cosmos_curator.models.vllm_interface.vllm_caption_async`, which
internally constructs an `_AsyncCaptioner` to drive concurrent
`engine.generate()` calls.

### Per-window dispatch flow (inside the GPU actor)

The `run_continuous` loop alternates between awaiting at least one
in-flight task with `_await_and_reap`, emitting completed pipe-tasks
via `_emit_completed_tasks`, and pulling more input from the queue.
When nothing is in flight it blocks on `input_queue.get()` with an
`_INPUT_GET_TIMEOUT_S` ceiling so `stop_event` is observed promptly.
Termination is driven exclusively by `stop_event` (set by Xenna's
`_watch_stop_flag`); there is no in-band sentinel.

`_register_task` is the synchronous bookkeeping entry point. It
type-checks each `SplitPipeTask`, takes ownership timestamps for
`--perf-profile`, and either (a) emits synchronously when no clip in
the task has any windows (so the pipeline never stalls on empty
inputs), or (b) inserts a `_ContinuousTaskTracker` and spawns
`_caption_pipe_tasks` as one asyncio task.

`_caption_pipe_tasks` calls `_gather_inputs` to collect the per-window
`llm_input` dicts from `window.model_input[self._input_key]`, then
delegates to `vllm_caption_async`. It returns when every window has
either succeeded, exhausted retries (sentinel emission), or -- for
`EngineDeadError` -- propagated the failure up so the caller can
escalate to actor restart.

```
                +-----------------------------------+
                | input_queue.get()                 |
                | (block, _INPUT_GET_TIMEOUT_S)     |
                +-----------------+-----------------+
                                  |
                                  v
                +-----------------------------------+
                | _register_task                    |
                |   classify, perf bookkeeping      |
                +--------+----------------+---------+
                         |                |
                  windows: yes      windows: zero
                         |                |
                         v                v
        +-------------------------+   +-----------------------------+
        | _caption_pipe_tasks     |   | output_queue.put(emit)      |
        |   _gather_inputs ->     |   | (synchronous; no tracker)   |
        |   vllm_caption_async    |   +-----------------------------+
        |     -> _AsyncCaptioner  |
        +-----------+-------------+
                    |
                    v
        +-------------------------+
        | _await_and_reap (raise  |
        |  on EngineDeadError)    |
        +-----------+-------------+
                    |
                    v
        +-------------------------+
        | _emit_completed_tasks   |   (runs unconditionally
        +-------------------------+    on every loop tick)
```

### Per-window concurrency model

`_AsyncCaptioner` (defined in `cosmos_curator/models/vllm_interface.py`)
implements the per-window contract:

| Concern | How it works |
|---------|-------------|
| Concurrency cap | Caller-owned `asyncio.Semaphore` (sized by `_effective_max_concurrent_requests`). Held only while `engine.generate` iterates -- never during retry backoff or stage-2 build, so slow / retrying windows never starve healthy siblings. |
| Retry policy | `tenacity.retry(stop=stop_after_attempt(max_retries), reraise=True, retry=retry_if_not_exception_type(EngineDeadError))`. Each attempt regenerates the engine `request_id` because vLLM rejects reusing an in-flight id. |
| Per-window error isolation | After retry exhaustion, non-`EngineDeadError` exceptions emit a `VLLM_UNKNOWN_CAPTION` sentinel via `_emit_unknown`, fire `on_window_error(idx, phase, exc)`, and continue. Siblings on the same actor are unaffected. |
| Engine death | `EngineDeadError` is **never** retried. It propagates unchanged out of `_AsyncCaptioner.run` -> `_caption_pipe_tasks` -> `_await_and_reap`, which crashes the actor; Xenna restarts it. |
| Stage-2 refinement | If `stage2_prompt` is non-`None`, `plugin.make_refined_llm_request(req, processor, stage2_prompt)` builds the refined request and `_spawn` schedules it with `phase="stage2"`. Token counts accumulate (stage-1 + stage-2) per sync's contract. |
| Payload mutation safety | Every attempt rebuilds the outer dict + `multi_modal_data` mapping via `_fresh_prompt_payload`. vLLM/HF/Transformers may mutate the outer shell; tensor references are zero-copy preserved. |

### Failure semantics

**Per-window failures are contained.** When a window's `engine.generate`
call raises a non-`EngineDeadError` exception, `tenacity` retries up to
`max_retries` attempts (default 3), regenerating the engine `request_id`
each time. If every attempt fails, `_AsyncCaptioner._emit_unknown(idx)`
records a sentinel `VLLM_UNKNOWN_CAPTION` result and fires
`on_window_error(idx, phase, exc)` so the stage can log the failure.
Sibling windows on the same actor continue running. Stage-2 build
failures (`plugin.make_refined_llm_request` raising) are also contained
through the same sentinel path with `phase="stage2_build"`.

**`EngineDeadError` triggers an actor restart.**
`_AsyncCaptioner._handle_completed` re-raises `EngineDeadError`
unchanged. It propagates out of `vllm_caption_async`, out of
`_caption_pipe_tasks`, and into `_await_and_reap`, which re-raises
from `run_continuous`. Xenna sees the actor crash and starts a fresh
one. Other in-flight windows on the failing actor are forfeited;
sibling actors are unaffected.

For persisted metadata, async vLLM implements a subset of the
[normalized caption-outcome contract](../design/vllm-interface.md#caption-outcomes-and-metadata)
via sync's shared `_normalize_vllm_result` helper, which `_scatter_one`
invokes for every window:

- `caption_status = "success"` -- non-empty caption, normal finish
- `caption_status = "truncated"` -- non-empty caption but vLLM stopped
  on `finish_reason == "length"` (request hit `max_tokens`)
- `caption_status = "error"` with `caption_failure_reason = "exception"`
  on retry exhaustion (sentinel caption text) or on empty / `"length"`
  output with no usable text
- It does **not** emit `blocked` or `skipped`, and does **not**
  evaluate caption-quality flags (those are produced only by sync
  `VllmCaptionStage`)

Engine-level failures restart the actor rather than writing a distinct
`caption_status` value for the lost in-actor copies; Xenna re-dispatches
the original input on the restarted actor.

#### Memory hygiene + actor-restart safety

Per-window cleanup happens inside `_scatter_one`, called via the
`on_window_done` callback as each result finalizes:

```python
window.model_input.pop(self._input_key, None)
window.mp4_bytes.drop()
```

This releases the cached vLLM input dict and the underlying frame
buffer **per window**, not at the end of the pipe-task, so the
slowest sibling cannot pin the fastest sibling's memory. The
`_caption_pipe_tasks` `finally` block also calls
`_free_vllm_inputs(windows, self._input_key, keep_mp4=False)` as
defence-in-depth, so any window that never reached `on_window_done`
(e.g. `vllm_caption_async` returning early on cancellation) still has
its inputs released. This mirrors sync's final `_free_vllm_inputs` call.

Two invariants make per-window in-actor mutation safe across a Xenna
actor restart:

1. **Ray pickles task arguments at the actor boundary.** In-actor
   mutation of a deserialized `SplitPipeTask` never propagates back
   to the upstream-queue copy.
2. **Continuous mode acks on emit.** A task is removed from the
   upstream queue only after `_emit_completed_tasks` puts a matching
   `ContinuousTaskOutput`. If the actor dies first, Xenna
   re-dispatches a fresh deserialization of the original input.

```
upstream queue --pickle-----> in-actor task --(_scatter_one)--> dies
upstream queue --re-pickle--> fresh actor (cleanup never observed)
```

### N-Actors vs DP Mode

```
data_parallel_size <= 1 (default)  -->  N-ACTORS MODE
data_parallel_size > 1             -->  DP MODE
```

**N-Actors** (default): Multiple independent workers, each with its
own `AsyncLLM` engine and `num_gpus` GPUs. No drain-refill barrier.

**DP Mode**: Single actor owns all GPUs, vLLM's built-in DP routes
requests internally.

| Config | Mode | GPUs/actor | Backend |
|--------|------|------------|---------|
| `--num-gpus 1` | N-actors | 1 | mp |
| `--num-gpus 2` | N-actors | 2 | ray |
| `--num-gpus 1 --dp 7` | DP | 7 (total) | ray |

Worker count: `--vllm-async-num-workers-per-node` (`0` = Xenna
autoscale, `> 0` = fixed count).

## Usage

### Basic

```bash
cosmos-curator local launch --curator-path . -- pixi run --as-is python -m \
    cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path /config/input \
    --output-clip-path /config/output \
    --captioning-algorithm vllm_async \
    --vllm-async-model-name qwen
```

### Multi-GPU (tensor parallel)

```bash
--vllm-async-model-name qwen3_vl_30b \
--vllm-async-num-gpus 4
```

### Data parallelism

DP mode runs a **single actor** that owns all GPUs and lets vLLM's
built-in data-parallel router fan requests across `data_parallel_size`
internal engine replicas. Total GPU footprint per actor is
`num_gpus * data_parallel_size` (TP per replica `*` replica count).

#### How TP and DP combine inside one actor

The two axes are orthogonal and serve different purposes:

| Axis | What it does | What it enables |
|------|--------------|-----------------|
| TP (`--vllm-async-num-gpus`) | Shards **weights + activations** of one engine across N GPUs (NCCL all-reduce per layer) | **Fitting** models that exceed a single GPU's VRAM |
| DP (`--vllm-async-data-parallel-size`) | **Replicates** the (already-built) engine M times; vLLM router dispatches per-request | **Throughput** - M concurrent decode streams sharing one scheduler |

Per-actor VRAM pool = `TP * DP * per_GPU_VRAM`. TP is what unlocks
large weights; DP is what keeps those expensive replicas busy:

```
                  +-----------------------------+
                  | one Xenna actor             |
                  | (vLLM AsyncLLM, 1 scheduler)|
                  +--------------+--------------+
                                 |
        +-----------+------------+------------+-----------+
        v           v                         v           v
   +----+----+ +----+----+               +----+----+ +----+----+
   | replica | | replica |   ... DP-1    | replica | | replica |
   |  (TP=N) | |  (TP=N) |               |  (TP=N) | |  (TP=N) |
   | g0..gN-1| | gN..2N-1|               |         | |  ...8N-1|
   +----+----+ +----+----+               +----+----+ +----+----+
     ^^^^^^^^^^                                       ^^^^^^^^^^
     full weight copy per replica                     same weights
     KV cache:  per-replica, fed by shared scheduler  (replicated)
```

Worked example - approaching ~1TB per-actor VRAM on 8xH200 (141GB):

- TP=8, DP=1: 8 GPUs, ~1128 GB pooled, **single replica only**.
  Use when weights + KV cache need every GPU. Throughput limited
  to one decode stream.
- TP=4, DP=2: 8 GPUs, ~1128 GB pooled, **2 replicas of 4 GPUs**.
  Each replica must fit in 4xH200 = ~564 GB. Doubles concurrent
  decode streams; halves per-replica KV budget.
- TP=2, DP=4: 8 GPUs, ~1128 GB pooled, **4 replicas of 2 GPUs**.
  Each replica must fit in 2xH200 = ~282 GB (e.g. 72B FP8 + KV).

Sizing rule: pick the **smallest TP** at which weights + headroom
for KV cache fit in `TP * per_GPU_VRAM`, then spend remaining GPUs
on DP. Going TP-heavier than necessary trades throughput for
unused VRAM; going DP-heavier than weights allow simply OOMs at
engine init.

```bash
# 72B FP8 on 8xH100: TP=2 per replica, 4 replicas in one actor (8 GPUs)
--vllm-async-model-name qwen2_5_vl_72b \
--vllm-async-num-gpus 2 \
--vllm-async-data-parallel-size 4 \
--vllm-async-fp8

# Smaller DP example: 1 GPU per replica, 2 replicas
--vllm-async-num-gpus 1 \
--vllm-async-data-parallel-size 2
```

### Quantized models

```bash
--vllm-async-fp8
```

The plugin's `model_async()` translates `config.fp8=True` into
`quantization="fp8"` (matches sync's derivation).  Other quantization
schemes (awq, gptq, etc.) require editing the plugin file directly.

### Stage-2 caption refinement

Enable a second refinement pass over each stage-1 caption. Stage-2
runs through the same engine + plugin; token counts accumulate
(stage-1 + stage-2) and the final caption text replaces the stage-1
text.

```bash
--vllm-async-stage2-caption \
--vllm-async-stage2-prompt-text "Improve and refine the following caption..."
```

Omitting `--vllm-async-stage2-prompt-text` falls back to the plugin's
built-in refinement prompt. The flag is only honoured when
`--vllm-async-stage2-caption` is set. Programmatic equivalent:

```python
VllmAsyncCaptionConfig(
    stage2_caption=True,
    stage2_prompt_text="Improve and refine the following...",
)
```

## GPU Scaling Recommendations

| Model size | Recommended | Config |
|------------|-------------|--------|
| 7B (Qwen2.5-VL) | N-actors TP=1 | `--num-gpus 1` |
| 27B FP8 (Qwen3.5-27B-FP8, Qwen3.6-27B-FP8) | N-actors TP=1 | `--num-gpus 1` (H100 80GB) |
| 27B BF16 (Qwen3.6-27B) | N-actors TP=2 | `--num-gpus 2` (H100 80GB) or `--num-gpus 1` (H200 141GB) |
| 30B (Qwen3-VL) | N-actors TP=1 or TP=2 | `--num-gpus 1` (H100 80GB) or `--num-gpus 2` |
| 72B (Qwen2.5-VL-72B) | N-actors TP=2 | `--num-gpus 2` (FP8) |
| 235B+ | TP=4 or TP=8 | `--num-gpus 4` or `--num-gpus 8` |

Memory estimate: `weight_bytes = params * bytes_per_param`,
`total_vram ~= weight_bytes * 1.2`. BF16 = 2 B/param, FP8 = 1 B/param,
INT4 = 0.5 B/param.

## Troubleshooting

### Out of GPU memory

Lower the engine's GPU memory budget on the command line:

```bash
--vllm-async-gpu-memory-utilization 0.7
```

To change the *default* itself (so `unset` picks up a new value),
edit `GPU_MEMORY_UTILIZATION` in the plugin file
(`cosmos_curator/models/vllm_<variant>.py`); both sync and async
pick up the new constant.

### Encoder cache ValueError

`ValueError: exceeds the pre-allocated encoder cache size` means
`max_num_batched_tokens` is too small.  Edit `MAX_NUM_BATCHED_TOKENS`
in the plugin file (`cosmos_curator/models/vllm_<variant>.py`).

### Engine failures

Preprocessing (smart resize + tokenization) is done deterministically
by sync's `VllmPrepStage` on the CPU side -- there is **no inline
renderer** in the async path, so renderer-class failures cannot
surface during captioning. Failures that surface during
`engine.generate` are split into two paths:

- **Retried then sentinelled.** `tenacity` retries up to `max_retries`
  attempts (default 3, controlled by `VllmAsyncConfig.max_retries` /
  `--vllm-async-max-retries`). On exhaustion, the window's caption is
  set to `VLLM_UNKNOWN_CAPTION`, `caption_status = "error"`,
  `caption_failure_reason = "exception"`, and the rest of the pipe-task
  continues unaffected. The original exception is logged via
  `on_window_error` so the failure is visible in actor logs.
- **Escalated to actor restart.** `EngineDeadError` (raised by vLLM V1
  when the engine-core process exits, e.g. after OOM) skips retry and
  crashes the actor. Xenna restarts it with a fresh `AsyncLLM`; other
  in-flight windows on the failing actor are lost; sibling actors are
  unaffected.

### CUBLAS_STATUS_INVALID_VALUE

CUDA library mismatch -- system cuBLAS loaded instead of PyTorch's
bundled version. The `unified` pixi environment resolves this.

### Extra environment variables

```bash
--vllm-async-extra-env-vars '{"VLLM_LOGGING_LEVEL": "DEBUG"}'
--vllm-async-extra-env-vars '{"CUDA_LAUNCH_BLOCKING": "1"}'
--vllm-async-extra-env-vars '{"NCCL_DEBUG": "TRACE"}'
```
