# vLLM Interface Design Document

## Overview

The `vllm_interface` module provides a unified, plugin-based abstraction layer for integrating vLLM-powered vision-language models (VLMs) into the Cosmos Curator video curation pipeline. It enables efficient, GPU-accelerated video captioning with support for multiple model backends, two-stage caption refinement, and flexible batching strategies.

The `vllm_interface` provides separation of concerns, enabling CuratorStage classes to focus on data routing while `vllm_interface` handles model setup, caption generation, and in-flight batching. This eliminates the need for stages to re-implement this functionality.

**Location**: `cosmos_curator/models/vllm_interface.py`

**Primary Purpose**: Abstract away model-specific implementation details and provide a consistent interface for video captioning using vLLM's high-performance inference engine.

**📚 Documentation Structure**:
- **This document** (vllm-interface.md): Architecture, API reference, configuration, usage examples
- **[vllm-interface-plugin.md](../guides/vllm-interface-plugin.md)**: Step-by-step guide for adding new model plugins
- **[vllm-interface-debug.md](../guides/vllm-interface-debug.md)**: Code flow tracing, debugging scenarios, troubleshooting

## Architecture

### High-Level Design

```text
┌─────────────────────────────────────────────────────────────┐
│                   Pipeline Stage Layer                      │
│              (VllmCaptionStage, VllmPrepStage)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  vllm_interface.py                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Public API Functions                                │   │
│  │  • vllm_model()        • auto_processor()            │   │
│  │  • sampling_params()   • make_model_inputs()         │   │
│  │  • vllm_caption()      • vllm_generate()             │   │
│  └──────────────────────────────────────────────────────┘   │
│                       │                                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Plugin Registry: _VLLM_PLUGINS                      │   │
│  │  • VllmNemotronNano12Bv2VL • VllmQwen7B              │   │
│  │  • VllmCosmosReason1VL                               │   │
│  │  • VllmCosmosReason2VL                               │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   VllmPlugin Interface                      │
│           (vllm_plugin.py - Abstract Base Class)            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Abstract Methods:                                   │   │
│  │  • model_variant()    • processor()                  │   │
│  │  • model()            • make_llm_input()             │   │
│  │  • decode()           • make_refined_llm_request()   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                        Concrete Plugin Implementations                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ VllmNemotron │  │  VllmQwen7B  │  │ VllmCosmosR1VL   │  │ VllmCosmosR2VL   │   │
│  │(vllm_nemotron)│ │ (vllm_qwen)  │  │(vllm_cosmos_r1)  │  │(vllm_cosmos_r2)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘  └──────────────────┘   │
└──────────────────────┬────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Library                             │
│  (LLM engine, SamplingParams, RequestOutput)                │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Plugin Architecture**: New models can be added by implementing the `VllmPlugin` interface without modifying core logic
2. **Separation of Concerns**: Model instantiation, input preparation, and captioning logic are cleanly separated
3. **Flexibility**: Supports multiple batching strategies, two-stage captioning, and model-specific preprocessing
4. **Performance**: Leverages vLLM's optimized inference engine with optional inflight batching
5. **Type Safety**: Comprehensive type hints for better IDE support and error detection

## Core Components

### 1. Plugin Registry

**Location**: `vllm_interface.py`

```python
_VLLM_PLUGINS = {
    VllmNemotronNano12Bv2VL.model_variant(): VllmNemotronNano12Bv2VL,
    VllmQwen7B.model_variant(): VllmQwen7B,
    VllmCosmosReason1VL.model_variant(): VllmCosmosReason1VL,
    VllmCosmosReason2VL.model_variant(): VllmCosmosReason2VL,
}
```

**Purpose**: Maps model variant names (strings) to their corresponding plugin implementations.

**Extension Point**: To add a new model:
1. Implement a new class inheriting from `VllmPlugin`
2. Register it in `_VLLM_PLUGINS`
3. Add the model ID to `vllm_model_ids.py`

### 2. VllmPlugin Interface

**Location**: `cosmos_curator/models/vllm_plugin.py`

**Required Methods**:
- `model_variant() -> str`: Returns unique identifier for the model
- `model_id() -> str`: Returns HuggingFace model ID (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
- `model_path(config: VllmConfig) -> Path`: Returns local path to model weights
- `processor(config: VllmConfig) -> AutoProcessor`: Returns the HuggingFace processor for the model
- `model(config: VllmConfig) -> LLM`: Instantiates the vLLM model with configuration
- `make_llm_input(prompt, frames, metadata, processor, config) -> dict`: Creates model-specific input format
- `make_refined_llm_request(request, processor, refine_prompt) -> VllmCaptionRequest`: Creates refined captioning request for stage 2
- `decode(vllm_output) -> str`: Extracts caption text from vLLM output

### 3. Configuration and Data Objects

#### VllmConfig

**Location**: `cosmos_curator/pipelines/video/utils/data_model.py`

```python
@attrs.define
class VllmConfig:
    model_variant: str                    # Which model to use (e.g., "qwen", "nemotron")
    use_image_input: bool = False         # Use image modality instead of video
    prompt_variant: str = "default"       # Prompt template variant
    prompt_text: str | None = None        # Custom prompt text
    fp8: bool = False                     # Enable FP8 quantization
    preprocess_mode: PreprocessMode = PreprocessMode.CURATOR
                                           # Owner of visual preprocessing
    disable_mmcache: bool = False         # Disable multimodal cache
    num_cpus_for_prepare: float = 2.0     # CPUs for preparation stage
    num_gpus: int = 1                     # GPUs for inference
    batch_size: int = 4                   # Inference batch size
    stage2_caption: bool = False          # Enable two-stage captioning
    stage2_prompt_text: str | None = None # Custom stage 2 prompt
    max_retries: int = 3                  # Caption retry attempts
    copy_weights_to: Path | None = None   # Optional model weight copy directory
    sampling_config: VllmSamplingConfig = attrs.Factory(VllmSamplingConfig)
    performance_mode: str | None = "throughput"
    debug_save_frames: bool = False       # Save debug frame PNGs
    debug_frames_output_dir: Path | None = None
```

`VllmSamplingConfig` owns generation sampling settings such as `temperature`,
`top_p`, `top_k`, `repetition_penalty`, `min_tokens`, and `max_tokens`.
`max_tokens` defaults to `8192`.

#### VllmCaptionRequest

**Location**: `cosmos_curator/pipelines/video/utils/data_model.py`

```python
@attrs.define
class VllmCaptionRequest:
    request_id: str                       # Unique request identifier
    inputs: dict[str, Any]                # Model-specific inputs
    caption: str | None = None            # Generated caption (None = needs generation)
    stage2_prompt: str | None = None      # Refinement prompt (if applicable)
    prompt_tokens: int = 0                # Prompt tokens consumed by vLLM
    output_tokens: int = 0                # Output tokens generated by vLLM
    finish_reason: str | None = None      # vLLM finish reason, such as "length"
```

**State Transitions**:
- **Initial**: `caption=None, stage2_prompt=None` → Needs initial captioning
- **Stage 1 Complete**: `caption="...", stage2_prompt=None` → Final caption ready
- **Stage 2 Pending**: `caption="...", stage2_prompt="..."` → Needs refinement
- **Stage 2 Complete**: `caption="...", stage2_prompt=None` → Refined caption ready

#### VllmWindowResult

**Location**: `cosmos_curator/models/vllm_interface.py`

```python
@attrs.define
class VllmWindowResult:
    text: str
    finish_reason: str | None
    token_counts: TokenCounts
```

`VllmWindowResult` is the terminal per-window sync vLLM result returned by
`vllm_caption()` before stage-level status normalization. `text` is the generated
caption text, `finish_reason` carries the terminal vLLM finish reason, and
`token_counts` stores accumulated prompt/output tokens.

## Public API Functions

### Model Setup Functions

#### `vllm_model(config: VllmConfig) -> LLM`

Creates and initializes a vLLM model instance.

**Parameters**:
- `config`: Configuration for the vLLM model

**Returns**: Initialized vLLM `LLM` object

**Usage**:
```python
config = VllmConfig(model_variant="qwen", num_gpus=1)
llm = vllm_model(config)
```

#### `auto_processor(config: VllmConfig) -> AutoProcessor`

Gets the HuggingFace processor for the specified model.

**Parameters**:
- `config`: Configuration specifying the model variant

**Returns**: AutoProcessor for tokenization and preprocessing

#### `sampling_params(config: VllmSamplingConfig) -> SamplingParams`

Creates vLLM sampling parameters from configuration.

**Parameters**:
- `config`: Sampling configuration

**Returns**: vLLM `SamplingParams` object configured with:
- Temperature, top_p, repetition_penalty
- Max output tokens
- Output kind (FINAL_ONLY for efficiency)

### Input Preparation Functions

#### `make_model_inputs(videos, metadata, config, processor, prompt, *, debug_window_ids=None) -> list[dict[str, Any]]`

Converts decoded video frames into model-ready inputs.

**Parameters**:
- `videos`: List of decoded video tensors (torch.Tensor)
- `metadata`: Per-window metadata dictionaries, such as frame count, size, and fps
- `config`: VllmConfig specifying the model variant
- `processor`: AutoProcessor for the model
- `prompt`: Text prompt for captioning
- `debug_window_ids`: Optional IDs used to organize saved debug frames

**Returns**: List of model-specific input dictionaries

**Model-Specific Formats**:
- **Qwen**: `{"prompt_token_ids": [...], "multi_modal_data": {"video": [(frames, metadata)]}}`
- **Nemotron**: `{"prompt_token_ids": [...], "multi_modal_data": {"video": (video_np, metadata)}}`
- **CosmosReason1VL/CosmosReason2VL**: `{"prompt": "...", "multi_modal_data": {"video": [(frames, metadata)]}}`

### Inference Functions

#### `vllm_generate(llm, sampling_params, requests, batch_size) -> list[RequestOutput]`

Performs batched inference on a list of caption requests.

**Parameters**:
- `llm`: The vLLM model instance
- `sampling_params`: Sampling configuration
- `requests`: List of VllmCaptionRequest objects
- `batch_size`: Number of requests per batch

**Returns**: List of RequestOutput objects (same order as inputs)

**Implementation Details**:
- Splits requests into chunks of `batch_size`
- Preserves request ordering
- Updates request IDs in outputs to match inputs

#### `vllm_caption(model_inputs, llm, processor, sampling_params, vllm_config, max_inflight_requests, *, inflight_batching, stage2_prompts) -> list[VllmWindowResult]`

**High-level function for video captioning with support for two-stage refinement.**

**Parameters**:
- `model_inputs`: List of model-ready inputs
- `llm`: vLLM model instance
- `processor`: AutoProcessor
- `sampling_params`: Sampling configuration
- `vllm_config`: Model configuration
- `max_inflight_requests`: Max concurrent requests (0 = unlimited)
- `inflight_batching`: Enable inflight batching (default: False)
- `stage2_prompts`: Optional list of refinement prompts (None = no refinement)

**Returns**: One `VllmWindowResult` per input window, in input order. Each result
contains:

- `text`: generated caption text
- `finish_reason`: vLLM finish reason, such as `"length"` when output hit the
  requested token limit
- `token_counts`: `TokenCounts(prompt_tokens, output_tokens)` accumulated across
  stage-1 and stage-2 generation

**Validation**:
- Raises `ValueError` if `max_inflight_requests < 0`
- Raises `ValueError` if `stage2_prompts` length doesn't match `model_inputs`

**Behavior**:
- Dispatches to `_caption_inflight_batching()` or `_caption_no_inflight_batching()` based on flag
- Handles two-stage captioning workflow automatically
- Returns raw per-window vLLM results; `VllmCaptionStage` normalizes those results
  into persisted caption status and failure metadata

## Caption Outcomes and Metadata

`vllm_caption()` is the sync vLLM result API. Public pipeline outputs should use the
normalized caption-outcome contract emitted by caption stages and writers, not raw
caption text as an outcome signal.

### Normalized Status Fields

Caption-capable pipeline paths write:

- `caption_status`: one of `success`, `truncated`, `blocked`, `error`, `skipped`,
  or `null`
- `caption_failure_reason`: `exception`, `timeout`, or `null`

`null` means the caption stage did not run for that row. `skipped` is part of the
shared vocabulary but is currently reserved; no backend emits it today.

`caption_failure_reason` is set only when `caption_status == "error"`. `timeout`
is produced by OpenAI-compatible and Gemini API paths when an SDK-specific timeout
exception is observed. Sync vLLM, image vLLM, and async vLLM currently use only
`exception` for errors.

### Current Backend Emission Set

| Backend | success | truncated | blocked | error | skipped | Failure reasons |
|---|---|---|---|---|---|---|
| sync vLLM video (`VllmCaptionStage`) | yes | yes | no | yes | reserved | `exception` |
| sync vLLM image | yes | yes | no | yes | reserved | `exception` |
| async vLLM video | yes | no | no | yes | reserved | `exception` |
| OpenAI-compatible / Gemini, video and image | yes | yes | yes | yes | reserved | `exception`, `timeout` |

### Video Quality Flags

The video split pipeline can also emit caption-quality annotations for the sync
vLLM path:

- `flag_length_outlier`
- `flag_repetition`
- `flag_near_duplicate`
- top-level `caption_quality_flags_enabled`

The flags are annotations only. They do not change `caption_status`, trigger
retries, block export, or enforce policy. They are evaluated only when
`caption_status in {"success", "truncated"}`.

When `caption_quality_flags_enabled` is `true`, each video metadata window row
includes the three `flag_*` keys. A `null` flag value means the schema is enabled
but no flag value was evaluated for that window. When
`caption_quality_flags_enabled` is `false`, the per-window `flag_*` keys are
omitted. Image pipeline metadata does not emit these video-only flag fields or the
marker.

### Writer Behavior

Video clip metadata stores `caption_status` and `caption_failure_reason` on every
`windows[]` row. Caption text contributes to `has_caption`, and to the
Cosmos-Predict per-window dataset export, only when
`caption_status in {"success", "truncated"}`. Video token counts are stored as flat
per-window fields such as `<model>_prompt_tokens` and `<model>_output_tokens`, plus
clip-level `total_prompt_tokens` and `total_output_tokens` aggregates.

Image metadata stores `caption_status`, `caption_failure_reason`, and nested
per-model `token_counts`; it does not include the video quality flag fields.

## Batching Strategies

### 1. Inflight Batching (`_caption_inflight_batching`) - **Typical Production Path**

**Use Case**: Maximum throughput, continuous processing (recommended for production)

**Flow**:
```
1. Maintain request queue and inflight requests dict
2. While len(results) < total_requests:
   a. If queue has requests AND under inflight limit:
      - Pop request, submit to engine, track in-flight
   b. engine.step() → Process one inference step
   c. Check for finished requests
   d. For finished requests:
      - If no stage2_prompt: Add per-window result
      - If stage2_prompt: Create refined request, add to queue
3. Return all per-window results
```

**Characteristics**:
- Continuous request submission as capacity allows
- Interleaved stage 1 and stage 2 processing
- Lower latency for stage 2 requests
- More complex state management
- Better GPU utilization
- **This is the typical code path used in production**

**Configuration**:
- `max_inflight_requests=0`: Unlimited inflight (trust vLLM's scheduler)
- `max_inflight_requests>0`: Cap concurrent requests (memory control)

### 2. Standard Batching (`_caption_no_inflight_batching`) - **Fallback/Testing Path**

**Use Case**: Simple batching for debugging, testing, or edge cases where continuous processing isn't suitable

**Flow**:
```
1. Create VllmCaptionRequest objects for all inputs
2. Batch process stage 1 requests → Initial captions
3. Filter: requests with stage2_prompt → needs refinement
4. Batch process stage 2 requests → Refined captions
5. Combine and return all per-window results
```

**Characteristics**:
- All stage 1 requests complete before stage 2 starts
- Fixed batch sizes via `vllm_generate()`
- Simpler logic, easier to debug
- Higher latency for stage 2 requests
- **Rarely used - mainly for testing and debugging**

## Two-Stage Captioning Workflow

### Motivation

Generate initial captions, then refine them with additional context for higher quality.

### Stage 1: Initial Captioning

**Input**: Video frames + initial prompt (e.g., "Describe this video in detail")

**Output**: First-pass caption

### Stage 2: Caption Refinement

**Input**:
- Original video frames
- Stage 1 caption
- Refinement prompt (e.g., "Improve and refine following video description...")

**Output**: Enhanced caption with better detail and coherence

### Implementation

**Request Creation**:
```python
# Stage 1
request = VllmCaptionRequest(
    request_id="abc123",
    inputs=model_inputs,
    stage2_prompt="Refine this caption: ..."  # If stage2_caption=True
)

# After Stage 1 completes with caption="A person walking"
# Plugin creates Stage 2 request:
refined_request = plugin.make_refined_llm_request(
    request=request,
    processor=processor,
    refine_prompt=stage2_prompt
)
# New request: caption=None, stage2_prompt=None (reset for stage 2)
```

**Model-Specific Refinement**:
- **Qwen**: Concatenates refine prompt + stage1 caption, reuses video tensor
- **Nemotron**: Concatenates refine prompt + stage1 caption, reuses video payload + metadata

### Configuration

Enable two-stage captioning:
```python
config = VllmConfig(
    model_variant="qwen",
    stage2_caption=True,
    stage2_prompt_text="Improve and refine following video description..."  # Optional
)
```

## Integration with Pipeline Stages

### VllmPrepStage

**Purpose**: Prepares video windows by decoding frames and creating model inputs

**Responsibilities**:
1. Decode video windows into tensor frames
2. Get prompt from prompt variant
3. Call `make_model_inputs()` to create model-ready inputs
4. Store inputs in `Window.model_input[model_variant]`

**Key Code** (simplified):
```python
videos = [decode_frames(window.mp4_bytes) for window in windows]
metadata = make_metadata(videos, window_config)
processor = auto_processor(config)
inputs = make_model_inputs(videos, metadata, config, processor, prompt)
for window, input_dict in zip(windows, inputs):
    window.model_input[config.model_variant] = input_dict
```

### VllmCaptionStage

**Purpose**: Generates captions for prepared windows using vLLM

**Responsibilities**:
1. Load model, processor, sampling params (in `stage_setup()`)
2. Gather model inputs from windows
3. Set up stage 2 prompts if enabled
4. Call `vllm_caption()` to generate per-window results
5. Scatter result text, token counts, status, and failure reason back to windows
6. Apply sync video caption-quality flags when enabled

**Key Code** (simplified):
```python
def stage_setup(self):
    self._llm = vllm_model(self._vllm_config)
    self._processor = auto_processor(self._vllm_config)
    self._sampling_params = sampling_params(self._vllm_config.sampling_config)

def process_data(self, tasks):
    windows = gather_windows_from_tasks(tasks)
    model_inputs = [w.model_input[self._vllm_config.model_variant] for w in windows]
    stage2_prompts = get_stage2_prompts(self._vllm_config, len(windows))
    
    results = vllm_caption(
        model_inputs,
        self._llm,
        self._processor,
        self._sampling_params,
        self._vllm_config,
        max_inflight_requests=self._max_inflight_requests,
        inflight_batching=self._inflight_batching,
        stage2_prompts=stage2_prompts,
    )
    
    scatter_captions_to_windows(windows, results, self._vllm_config.model_variant)
    apply_caption_quality_flags_if_enabled(tasks)
    return tasks
```

### Error Handling and Retries

**VllmCaptionStage** uses `tenacity` for automatic retries:
```python
@tenacity.retry(stop=tenacity.stop_after_attempt(max_retries), reraise=True)
def _vllm_caption(model_inputs, stage2_prompts):
    try:
        return vllm_caption(...)
    except Exception:
        # On retry: teardown and reinitialize vLLM
        del self._llm, self._processor, self._sampling_params
        self.destroy()
        self.stage_setup()
        raise
```

## Adding New Models

**Want to add support for a new vLLM vision-language model?**

See **[`vllm-interface-plugin.md`](../guides/vllm-interface-plugin.md)** for the complete step-by-step guide covering:
- ✅ Plugin interface overview and requirements
- ✅ Step-by-step implementation for each of the 6 required methods
- ✅ Testing checklist (unit tests, integration tests, e2e tests)
- ✅ Common plugin bugs and how to avoid them
- ✅ Best practices and code patterns
- ✅ Complete working example (VideoLLaMA plugin)

**Quick Summary:**
1. Create plugin file: `cosmos_curator/models/vllm_mymodel.py` (~150 lines)
2. Implement 6 methods: `model_variant()`, `processor()`, `model()`, `make_llm_input()`, `decode()`, `make_refined_llm_request()`
3. Register in `vllm_interface.py` (1 line) and `vllm_model_ids.py` (1 line)
4. Write tests and verify end-to-end captioning works

**Time estimate:** 2-4 hours for a well-understood model

## Configuration Examples

### Basic Video Captioning

```python
config = VllmConfig(
    model_variant="qwen",
    num_gpus=1,
    batch_size=8,
    sampling_config=VllmSamplingConfig(max_tokens=256),
)
```

### High-Quality Two-Stage Captioning

```python
config = VllmConfig(
    model_variant="nemotron",
    num_gpus=1,
    batch_size=4,
    stage2_caption=True,
    stage2_prompt_text="Refine this caption with more visual details: ",
    sampling_config=VllmSamplingConfig(temperature=0.7, top_p=0.9, max_tokens=512),
)
```

### Maximum Throughput with Inflight Batching

```python
# In VllmCaptionStage initialization
stage = VllmCaptionStage(
    vllm_config=VllmConfig(model_variant="qwen", batch_size=16),
    inflight_batching=True,
    max_inflight_requests=0,  # Unlimited (let vLLM manage)
)
```

### Memory-Constrained Environment

```python
config = VllmConfig(
    model_variant="qwen",
    num_gpus=1,
    batch_size=2,
    fp8=True,  # Quantization to reduce memory
    disable_mmcache=True,  # Disable multimodal cache
)

stage = VllmCaptionStage(
    vllm_config=config,
    inflight_batching=True,
    max_inflight_requests=4,  # Limit concurrent requests
)
```

## Error Handling

### Request-Level Errors

- **Empty or failed vLLM result**: normalized by `VllmCaptionStage` to
  `caption_status="error"` and `caption_failure_reason="exception"`
- **Length-limited vLLM result with text**: normalized to `caption_status="truncated"`
- **Invalid inputs**: raises `ValueError` for malformed inputs
- **Stage 2 errors**: raises `ValueError` if caption is `None` when creating a
  refined request

### Model-Level Errors

Handled by `VllmCaptionStage` with automatic retry and model restart:

```python
try:
    results = vllm_caption(...)
except Exception:
    logger.exception("Error generating captions, retrying...")
    # Teardown and reinitialize model
    del self._llm, self._processor, self._sampling_params
    self.destroy()
    self.stage_setup()
    raise  # tenacity will retry
```

### Configuration Errors

- **Unsupported Model Variant**: Raises `ValueError` via `_get_vllm_plugin()`
- **Negative max_inflight_requests**: Raises `ValueError` in `vllm_caption()`
- **Mismatched stage2_prompts Length**: Raises `ValueError` in `vllm_caption()`

## Performance Considerations

### GPU Memory Management

1. **FP8 Quantization**: Reduces memory by ~50% with possible quality loss
   ```python
   config = VllmConfig(fp8=True)
   ```

2. **Multimodal Cache**: Disabling saves memory but slows preprocessing
   ```python
   config = VllmConfig(disable_mmcache=True)
   ```

3. **Tensor Parallelism**: Distribute model across multiple GPUs
   ```python
   config = VllmConfig(num_gpus=2)
   ```

### Throughput Optimization

1. **Batch Size**: Larger batches improve GPU utilization
   - Start with 8-16, tune based on GPU memory
   - Formula: `batch_size * sequence_length * num_frames` should fill GPU

2. **Inflight Batching**: Reduces idle time between batches
   - Enable for continuous workloads
   - Monitor memory usage if using `max_inflight_requests=0`

3. **Two-Stage Captioning**: Doubles inference time
   - Only enable if quality improvement is worth the cost
   - Consider selective refinement (not all clips)

### Latency Optimization

1. **Disable Two-Stage**: Halves end-to-end latency
2. **Smaller Batch Sizes**: Reduces wait time for batch formation
3. **Inflight Batching**: Starts processing immediately without waiting for full batch

## Dependencies

### Core Dependencies

- **vllm**: High-performance LLM inference engine
  - `LLM`: Model loading and inference
  - `SamplingParams`: Generation configuration
  - `RequestOutput`: Raw inference results
  - `PoolingOutput`, `PoolingRequestOutput`: Not supported (will raise TypeError)

- **transformers**: HuggingFace library
  - `AutoProcessor`: Tokenization and preprocessing

- **torch**: PyTorch framework
  - Used for video tensor handling

### Internal Dependencies

- `cosmos_curator.core.utils.misc.grouping`: Batch splitting utilities
- `cosmos_curator.pipelines.video.utils.data_model`: Configuration, request, status, and token-count objects
- Plugin implementations: `vllm_nemotron.py`, `vllm_qwen.py`, `vllm_cosmos_reason1_vl.py`, `vllm_cosmos_reason2_vl.py`

### Pixi Environment

vLLM models require the **`default`** Pixi environment:
```bash
pixi run --as-is -e default pytest -m env tests/
```

## Testing Strategies

### Unit Testing

1. **Plugin Interface Compliance**
   ```python
   def test_plugin_implements_interface():
       plugin = VllmQwen7B()
       assert hasattr(plugin, 'model_variant')
       assert hasattr(plugin, 'make_llm_input')
       # ... test all required methods
   ```

2. **Input/Output Format Validation**
   ```python
   def test_make_llm_input_format():
       frames = torch.rand(8, 3, 224, 224)
       metadata = make_metadata([frames], WindowConfig(sampling_fps=1.0))[0]
       config = VllmConfig(model_variant="qwen")
       processor = auto_processor(config)
       input_dict = VllmQwen7B.make_llm_input("prompt", frames, metadata, processor, config)
       assert "prompt_token_ids" in input_dict
       assert "multi_modal_data" in input_dict
   ```

3. **Two-Stage Request Creation**
   ```python
   def test_refined_request():
       request = VllmCaptionRequest(..., caption="Initial")
       refined = plugin.make_refined_llm_request(request, processor)
       assert refined.caption is None  # Reset for stage 2
   ```

### Integration Testing

1. **End-to-End Captioning**
   ```python
   @pytest.mark.env("default")  # GPU required
   def test_vllm_caption_e2e():
       config = VllmConfig(model_variant="qwen")
       llm = vllm_model(config)
       processor = auto_processor(config)
       
       # Mock video inputs
       videos = [torch.rand(8, 3, 224, 224) for _ in range(4)]
       metadata = make_metadata(videos, WindowConfig(sampling_fps=1.0))
       inputs = make_model_inputs(videos, metadata, config, processor, "Describe this")
       
       results = vllm_caption(
           inputs, llm, processor,
           sampling_params(config.sampling_config), config,
           max_inflight_requests=0,
           inflight_batching=False
       )
       
       assert len(results) == 4
       assert all(isinstance(result.text, str) for result in results)
       assert all(result.token_counts.prompt_tokens >= 0 for result in results)
   ```

2. **Batching Strategy Comparison**
   ```python
   def test_batching_strategies_equivalent():
       # Run with and without inflight batching
       results_standard = vllm_caption(..., inflight_batching=False)
       results_inflight = vllm_caption(..., inflight_batching=True)
       assert len(results_standard) == len(results_inflight)
   ```

### Performance Testing

1. **Throughput Benchmarking**
   ```python
   def benchmark_batching():
       configs = [
           (False, 4),   # Standard, batch_size=4
           (True, 4),    # Inflight, batch_size=4
           (True, 0),    # Inflight, unlimited
       ]
       for inflight, max_inflight in configs:
           start = time.time()
           results = vllm_caption(..., inflight_batching=inflight)
           throughput = len(results) / (time.time() - start)
           print(f"Inflight={inflight}, max={max_inflight}: {throughput:.2f} captions/sec")
   ```

## Debugging and Code Flow Tracing

**Need to understand how the code works internally or debug an issue?**

See **[`vllm-interface-debug.md`](../guides/vllm-interface-debug.md)** for:
- ✅ Complete code flow example tracing a video through the system
- ✅ Request state lifecycle and transitions
- ✅ Common debugging scenarios (captions failing, stage 2 not triggering, requests dropped)
- ✅ Where to set breakpoints and what to log
- ✅ Testing checklist for new plugins
- ✅ Advanced profiling and memory debugging

**Quick debugging tips:**
- Set breakpoint in `vllm_caption()` and step through to see full flow
- Check `VllmCaptionRequest.stage2_prompt`: `None` = done, `"..."` = needs refinement
- Log request IDs to track requests through batching and refinement
- Use `inflight_batching=False` for simpler debugging (easier to step through)
- Inspect `VllmWindowResult.finish_reason` and `token_counts` before stage-level
  status normalization

## References

- vLLM Documentation: https://docs.vllm.ai/
- Cosmos-Xenna (Ray pipeline framework): https://github.com/nvidia-cosmos/cosmos-xenna
- [Pipeline Design Guide](../guides/pipeline-design.md)
- [Model Interface](../../../cosmos_curator/core/interfaces/model_interface.py)
- [VllmCaptionStage](../../../cosmos_curator/pipelines/video/captioning/vllm_caption_stage.py)
