# vLLM Async Captioning Guide

## Overview

The `vllm_async` captioning algorithm runs an in-process `AsyncLLM`
engine within each Ray worker actor. It generates captions via async
`engine.generate()` calls with prompt formatting handled by
`transformers.AutoProcessor.apply_chat_template`.

**When to use `vllm_async`:**

- Simple model integration without custom vLLM plugin code
- Native data parallelism (`data_parallel_size`)
- Testing a model variant by adding entries to `_MODEL_DEFAULTS`
  and `get_vllm_model_id()`

**When to use the in-process path (`qwen`, `nemotron`, etc.):**

- Fine-grained control over input construction
- Running in production with existing in-process model code

## Architecture

Two-stage pipeline (three with previews enabled):

```
VllmAsyncPrepStage (CPU) --> VllmAsyncCaptionStage (GPU, continuous mode)
  decode frames, build         inline render under asyncio.Lock + asyncio.to_thread,
  TextPrompt + frames          AsyncLLM.generate(), assign captions
```

Render is performed inline inside `VllmAsyncCaptionStage`, which
implements `cosmos_xenna.ray_utils.continuous_stage.ContinuousInterface`
and is driven by Xenna's `run_continuous` loop instead of per-batch
`process_data` calls.

### Per-window dispatch flow (inside the GPU actor)

The `run_continuous` loop alternates between reaping completed in-flight
work and pulling more input from the queue. When nothing is in flight it
blocks on `input_queue.get()` with a `_INPUT_GET_TIMEOUT_S` ceiling so
`stop_event` is observed promptly. Termination is driven exclusively by
`stop_event` (set by Xenna's `_watch_stop_flag`); there is no in-band
sentinel.

`_register_task` first calls `_extract_prepared_windows`. If the task
yields zero prepared windows (upstream prep produced nothing), the
output is emitted synchronously to `output_queue` and no tracker is
inserted -- the pipeline never stalls on inputs whose render budget is
empty. Otherwise the tracker is inserted *after* extract succeeds and
one stage-1 task per window is spawned. `_emit_completed_tasks` runs
unconditionally on every loop tick as defense-in-depth.

```
                +-----------------------------------+
                | input_queue.get()                 |
                | (block, _INPUT_GET_TIMEOUT_S)     |
                +-----------------+-----------------+
                                  |
                                  v
                +-----------------------------------+
                | _register_task                    |
                |   _extract_prepared_windows       |
                +--------+----------------+---------+
                         |                |
                  windows: yes      windows: zero
                         |                |
                         v                v
        +-------------------------+   +-----------------------------+
        | _generate_and_assign    |   | output_queue.put(emit)      |
        | (render via Lock + to_  |   | (synchronous; no tracker)   |
        |  thread; AsyncLLM.gen)  |   +-----------------------------+
        +-----------+-------------+
                    |
                    v
        +-------------------------+
        | _await_and_reap (raise) |
        +-----------+-------------+
                    |
                    v
        +-------------------------+
        | _emit_completed_tasks   |   (runs unconditionally
        +-------------------------+    on every loop tick)
```

The renderer is invoked inline via:

```python
async with self._render_lock:
    rendered = await asyncio.to_thread(self._engine.renderer.render_cmpl, payload)
```

`asyncio.Lock` serialises HF tokenizer calls (which raise
`RuntimeError("Already borrowed")` under concurrent use), and
`asyncio.to_thread` keeps the event loop free while the blocking
renderer runs on an OS thread. A fresh payload dict and a fresh
`multi_modal_data` mapping are built per call so the renderer's
in-place mutations never leak across windows.

### Failure semantics

There is **no per-window error isolation** in continuous mode. Any
exception inside a stage-1 or stage-2 task propagates out of
`_await_and_reap`, exits `run_continuous`, and triggers a Xenna
actor restart. Multi-actor parallelism cushions the blast radius,
but other in-flight windows on the failing actor are forfeited.

#### Actor-restart cleanup contract

The eager in-actor cleanup in `_generate_and_assign`
(`finally: del rendered_prompt`) and `_extract_prepared_windows`
(`window.model_input.pop(variant, None)`) is **horizontal memory
hygiene**, not recovery-path mutation. Two invariants make it safe
across a Xenna actor restart:

1. **Ray pickles task arguments at the actor boundary.** In-actor
   mutation of a deserialized `SplitPipeTask` never propagates back
   to the upstream-queue copy.
2. **Continuous mode acks on emit.** A task is removed from the
   upstream queue only after `_emit_completed_tasks` puts a matching
   `ContinuousTaskOutput`. If the actor dies first, Xenna
   re-dispatches a fresh deserialization of the original input.

```
upstream queue --pickle-----> in-actor task --(del / pop)--> dies
upstream queue --re-pickle--> fresh actor (cleanup never observed)
```

`del rendered_prompt` releases only the per-window vLLM-rendered
tensors; the raw `prompt_text` and `decoded_rgb_frames` remain reachable
through `_PreparedWindow` for the lifetime of the in-flight task.
`window.model_input.pop(variant, None)` evicts the upstream cache
entry *after* its values have been copied into a `_PreparedWindow`;
the upstream-queue copy is untouched. Neither cleanup is reachable
from the recovery path, so no work is lost on restart.

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
    cosmos_curator.pipelines.video.splitting_pipeline \
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

```bash
--vllm-async-num-gpus 1 \
--vllm-async-data-parallel-size 2
```

### Quantized models

```bash
--vllm-async-dtype float16 \
--vllm-async-quantization fp8
```

### Stage-2 caption refinement

Available via programmatic config (CLI flags not yet wired):

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
| 30B (Qwen3-VL) | N-actors TP=1 or TP=2 | `--num-gpus 1` (H100 80GB) or `--num-gpus 2` |
| 72B (Qwen2.5-VL-72B) | N-actors TP=2 | `--num-gpus 2` (FP8) |
| 235B+ | TP=4 or TP=8 | `--num-gpus 4` or `--num-gpus 8` |

Memory estimate: `weight_bytes = params * bytes_per_param`,
`total_vram ~= weight_bytes * 1.2`. BF16 = 2 B/param, FP8 = 1 B/param,
INT4 = 0.5 B/param.

## Troubleshooting

### Out of GPU memory

```bash
--vllm-async-gpu-memory-utilization 0.80
--vllm-async-num-gpus 2
```

### Encoder cache ValueError

`ValueError: exceeds the pre-allocated encoder cache size` means
`max_num_batched_tokens` is too small. For qwen, `_MODEL_DEFAULTS`
sets it to `32768`. For other models:

```bash
--vllm-async-max-num-batched-tokens 32768
```

### Engine / renderer failures

In continuous mode the caption stage no longer special-cases
`EngineDeadError` (or any other per-window error). Any uncaught
exception during render or generate -- engine OOM, GPU CUDA fault,
renderer error -- propagates out of `run_continuous` and crashes
the Ray actor. Xenna restarts it with a fresh `AsyncLLM` engine.
Other in-flight windows on the failing actor are lost; sibling
actors are unaffected.

### CUBLAS_STATUS_INVALID_VALUE

CUDA library mismatch -- system cuBLAS loaded instead of PyTorch's
bundled version. The `unified` pixi environment resolves this.

### Extra environment variables

```bash
--vllm-async-extra-env-vars '{"VLLM_LOGGING_LEVEL": "DEBUG"}'
--vllm-async-extra-env-vars '{"CUDA_LAUNCH_BLOCKING": "1"}'
--vllm-async-extra-env-vars '{"NCCL_DEBUG": "TRACE"}'
```
