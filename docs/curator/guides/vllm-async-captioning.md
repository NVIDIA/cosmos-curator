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

Three-stage pipeline:

```
VllmAsyncPrepStage (CPU) --> VllmAsyncPromptRenderStage (CPU) --> VllmAsyncCaptionStage (GPU)
  decode frames, build         render TextPrompt to                engine.generate()
  TextPrompt                   ProcessorInputs                    assign captions
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

### EngineDeadError

GPU OOM-kill of the `EngineCore` subprocess. The stage re-raises
`EngineDeadError` to crash the Ray actor; Xenna restarts it with a
fresh engine.

### CUBLAS_STATUS_INVALID_VALUE

CUDA library mismatch -- system cuBLAS loaded instead of PyTorch's
bundled version. The `unified` pixi environment resolves this.

### Extra environment variables

```bash
--vllm-async-extra-env-vars '{"VLLM_LOGGING_LEVEL": "DEBUG"}'
--vllm-async-extra-env-vars '{"CUDA_LAUNCH_BLOCKING": "1"}'
--vllm-async-extra-env-vars '{"NCCL_DEBUG": "TRACE"}'
```
