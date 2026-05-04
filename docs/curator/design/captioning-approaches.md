# Captioning Approaches: In-Process vLLM vs vLLM Async

Two captioning approaches exist in Cosmos Curator. Both use the same
underlying vLLM engine; the difference is how that engine is hosted
and how models are integrated.

## Architecture Overview

```
            IN-PROCESS (VllmCaptionStage)
+---------------------------------------------------+
|  VllmPrepStage         VllmCaptionStage            |
|  - extract frames      - LLM() lives in-process   |
|  - build llm_inputs    - llm.generate(inputs)      |
|  GPU: CuratorStageResource(gpus=N)                 |
|  Engine: single LLM() instance, tensor-parallel    |
+---------------------------------------------------+

            VLLM ASYNC (3-stage pipeline)
+---------------------------------------------------------------+
|  VllmAsyncPrepStage     VllmAsyncPrompt     VllmAsyncCaption  |
|  (CPU, unified)         RenderStage          Stage             |
|  - decode frames        (CPU, unified)      (GPU, unified)     |
|  - build TextPrompt     - Renderer (CPU)    - engine.generate  |
|                         - TextPrompt ->       (ProcessorInputs)|
|                           ProcessorInputs   - stage-2 refine   |
|  GPU: CuratorStageResource(gpus=num_gpus) per caption actor    |
|  Modes: N-actors (1 GPU each) or DP (all GPUs, 1 actor)       |
+---------------------------------------------------------------+
```

## New-Model Integration

### In-Process

Adding a new model requires: model class (`ModelInterface`), prep
utilities, model registration, vLLM interface glue, stage wiring.
Each model tightly couples to vLLM Python internals.

**Difficulty**: Medium-High.

### vLLM Async

Adding a new model requires: model registration in `all_models.py`,
HuggingFace ID in `get_vllm_model_id()`, and optionally a
`_MODEL_DEFAULTS` entry for non-default engine parameters.

**Difficulty**: Low. `AutoProcessor.apply_chat_template` provides a
uniform interface. No model class or prep utilities needed.

**Trade-off**: In-process allows fine-grained control over input
construction. Async delegates to vLLM's built-in pipeline -- simpler
but less customizable.

## Performance Comparison

Both use the same vLLM engine for GPU inference. Core model forward
pass, KV cache management, and continuous batching are identical.

| Aspect | In-Process | vLLM Async |
|--------|-----------|------------|
| Engine | `LLM()` in-process | `AsyncLLM` in-process |
| Batching | Inflight batching or explicit chunks | Continuous (semaphore-bounded async) |
| GPU allocation | `tensor_parallel_size=N` | TP + optional `data_parallel_size=M` |
| Input format | Python dicts (zero-copy) | TextPrompt -> ProcessorInputs (pre-rendered on CPU) |
| Data parallelism | TP only | Native TP + DP |

### GPU Allocation on Multi-GPU Servers

Both default to 1 GPU for captioning. On a 2-GPU server, GPU 0 runs
captioning while GPU 1 handles other stages (TransNetV2, embedding, etc.).

To use both GPUs for captioning:

- **In-process**: `--qwen-num-gpus-per-worker 2` (TP=2)
- **vLLM async**: `--vllm-async-num-gpus 2` (TP=2) or
  `--vllm-async-num-gpus 1 --vllm-async-data-parallel-size 2` (DP=2,
  generally higher throughput for 7B-class models)

## When to Use Which

| Scenario | Recommended |
|----------|-------------|
| New model integration (quick) | vLLM async |
| Custom input preprocessing | In-process |
| Data parallelism needed | vLLM async |
| Existing pipeline integration | In-process (already wired) |
