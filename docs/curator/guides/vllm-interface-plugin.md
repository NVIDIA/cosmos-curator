# vLLM Interface Plugin Guide

## Overview

This guide walks you through adding a new vLLM model to cosmos-curator
by implementing a plugin.  Plugins are the **adapter layer** between
cosmos-curator pipeline data and per-model request format /
engine construction.  A single `VllmPlugin` subclass serves both the
**synchronous** captioning pipeline (`model()` returns `LLM`) and the
**asynchronous** captioning pipeline (`model_async()` returns
`AsyncEngineArgs`) - per-variant numeric tuning lives as
**module-scope constants** on the plugin and is the **single source of
truth** read by both methods.  No drift between sync and async tunings.

**Prerequisites:**
- Familiarity with vLLM library and the model you want to add
- Understanding of the model's input format and tokenization requirements
- Model weights downloaded locally (see model download guide)

**Related Documentation:**
- **[vllm-interface.md](../design/vllm-interface.md)**: Architecture and API reference
- **[vllm-interface-debug.md](vllm-interface-debug.md)**: Debugging and troubleshooting
- **[vllm-async-captioning.md](vllm-async-captioning.md)**: Async pipeline using these plugins

## Quick Start

Adding a new model touches 4 implementation files plus tests and ~260 lines of code:

1. **Create plugin**: `cosmos_curator/models/vllm_mymodel.py` (~170 lines)
   - Module-scope constants for per-variant tuning (`MAX_MODEL_LEN`, etc.)
   - `model()` for sync, `model_async()` for async - both read those constants
2. **Register plugin**: Add to `cosmos_curator/models/vllm_interface.py` (2 lines)
3. **Add model ID**: Add to `cosmos_curator/models/vllm_model_ids.py` (1 line)
4. **Add model info**: Add to `cosmos_curator/configs/all_models.json` (7 lines)
5. **Test plugin**: `tests/models/test_vllm_mymodel.py` (~100 lines)

**Time estimate:** 2-4 hours for a well-understood model

---

## Plugin Interface Overview

Every plugin inherits from `VllmPlugin` and implements these methods:

```python
class VllmPlugin(ABC):
    @staticmethod
    @abstractmethod
    def model_variant() -> str:
        """Return unique identifier (e.g., "qwen", "nemotron")"""

    @classmethod
    def model_id(cls) -> str:
        """Return HuggingFace model ID (inherited, no need to override)"""

    @classmethod
    def model_path(cls, config: VllmConfig) -> Path:
        """Return local path to model weights (inherited, no need to override)"""

    @classmethod
    @abstractmethod
    def processor(cls, config: VllmConfig) -> AutoProcessor:
        """Return HuggingFace processor for tokenization"""

    @classmethod
    @abstractmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate vLLM ``LLM`` (sync pipeline).
        Reads per-variant module-scope constants on this plugin."""

    @classmethod
    @abstractmethod
    def model_async(cls, config: VllmAsyncConfig) -> AsyncEngineArgs:
        """Build ``AsyncEngineArgs`` for in-process ``AsyncLLM`` (async pipeline).
        Reads the SAME per-variant module-scope constants as ``model()``.
        Bakes async-only invariants such as
        ``mm_processor_kwargs={"do_sample_frames": False, ...}``."""

    @staticmethod
    @abstractmethod
    def make_llm_input(
        prompt: str,
        frames: torch.Tensor,
        metadata: dict[str, Any],
        processor: AutoProcessor,
        config: VllmConfig,
    ) -> dict[str, Any]:
        """Convert prompt + frames + metadata to model-specific input format"""

    @staticmethod
    @abstractmethod
    def make_refined_llm_request(
        request: VllmCaptionRequest,
        processor: AutoProcessor,
        refine_prompt: str | None = None,
    ) -> VllmCaptionRequest:
        """Create stage 2 refinement request from stage 1 caption"""
    
    @staticmethod
    @abstractmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Extract caption string from vLLM output"""
```

**Only need to implement 7 methods** - `model_id()` and `model_path()` are inherited.
The plugin abstract surface is `model_variant`, `processor`, `model`, `model_async`,
`make_llm_input`, `make_refined_llm_request`, and `decode`.
`make_llm_input()` must accept both `metadata` and `config`. Plugins that support
both image and video should use `config.use_image_input` to choose the modality.

---

## Step-by-Step Plugin Creation

### Step 1: Create Plugin File

Create `cosmos_curator/models/vllm_mymodel.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MyModel vLLM plugin."""

import secrets
from typing import Any

from transformers import AutoProcessor
from vllm import LLM

from cosmos_curator.models.vllm_plugin import VllmPlugin
from cosmos_curator.pipelines.video.utils.data_model import (
    VllmCaptionRequest,
    VllmConfig,
)


class VllmMyModel(VllmPlugin):
    """MyModel vLLM plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "mymodel"

    # Implement other methods below...
```

### Step 2: Implement `processor()`

Return a HuggingFace `AutoProcessor` for your model:

```python
@classmethod
def processor(cls, config: VllmConfig) -> AutoProcessor:
    """Return the AutoProcessor for the model."""
    processor = AutoProcessor.from_pretrained(
        cls.model_path(config),
        trust_remote_code=True,  # Set based on model requirements
    )
    return processor
```

**Common patterns:**
- Most models: `AutoProcessor.from_pretrained(cls.model_path(config))`
- Custom processor: Override specific components after loading
- Tokenizer-only models: Return tokenizer wrapped as processor

**Testing:**
```python
config = VllmConfig(model_variant="mymodel")
processor = VllmMyModel.processor(config)
assert processor is not None
tokens = processor.tokenizer("test prompt")
assert len(tokens.input_ids) > 0
```

### Step 3: Implement `model()`

Instantiate the vLLM `LLM` object with your model's configuration:

```python
@classmethod
def model(cls, config: VllmConfig) -> LLM:
    """Instantiate the vLLM model."""
    quantization = "fp8" if config.fp8 else None
    
    mm_processor_kwargs = {
        "do_resize": config.model_preprocess_enabled,
        "do_rescale": config.model_preprocess_enabled,
        "do_normalize": config.model_preprocess_enabled,
    }
    
    return LLM(
        model=str(cls.model_path(config)),
        quantization=quantization,
        tensor_parallel_size=config.num_gpus,
        max_model_len=32768,  # Model-specific
        gpu_memory_utilization=0.85,
        mm_processor_kwargs=mm_processor_kwargs,
        mm_processor_cache_gb=0.0 if config.disable_mmcache else 4.0,
        max_num_batched_tokens=32768,  # Model-specific
        trust_remote_code=True,  # Set based on model requirements
        limit_mm_per_prompt={"video": 1},  # Model-specific: video vs image
    )
```

**Key configuration options:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `quantization` | FP8/INT8 quantization | `"fp8"` if `config.fp8` else `None` |
| `tensor_parallel_size` | GPUs for model parallelism | `config.num_gpus` |
| `max_model_len` | Max sequence length | Model-dependent (8K-32K) |
| `gpu_memory_utilization` | GPU memory fraction | 0.85 (85%) |
| `limit_mm_per_prompt` | Multimodal limits | `{"video": 1}` or `{"image": N}` |
| `trust_remote_code` | Execute model code | `True` if needed |

**Model-specific considerations:**
- **Qwen**: Supports video, uses token IDs, no image limit
- **Nemotron**: Supports video with model-specific metadata payload
- **Your model**: Check vLLM docs for supported multimodal types

**Testing:**
```python
config = VllmConfig(model_variant="mymodel", num_gpus=1)
llm = VllmMyModel.model(config)
assert llm is not None
assert llm.llm_engine is not None
```

### Step 3b: Implement `model_async()`

Build the `AsyncEngineArgs` that the async pipeline passes to
`vllm.v1.engine.async_llm.AsyncLLM`. The signature is

```python
@classmethod
def model_async(cls, config: VllmAsyncConfig) -> AsyncEngineArgs: ...
```

Unlike `model()`, this method does NOT instantiate the engine - it
only returns the **arguments** the async stage will use to construct
one. All per-variant numeric tuning is read from the **same
module-scope constants** as `model()`, which is the project's
single-source-of-truth invariant: sync and async tunings must never
drift.

```python
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import AsyncEngineArgs

from cosmos_curator.pipelines.video.utils.data_model import VllmAsyncConfig


@classmethod
def model_async(cls, config: VllmAsyncConfig) -> AsyncEngineArgs:
    """Build ``AsyncEngineArgs`` for in-process ``AsyncLLM`` (async pipeline).

    Mirrors :meth:`model` - reads the SAME module-scope constants
    (single source of truth, no sync/async drift).
    """
    return AsyncEngineArgs(
        # cls.model_path() takes a sync VllmConfig; bridge with to_vllm_config().
        model=str(cls.model_path(config.to_vllm_config())),
        served_model_name=[config.model_variant],

        # --- Same module-scope constants as model() ---
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        trust_remote_code=TRUST_REMOTE_CODE,
        limit_mm_per_prompt=LIMIT_MM_PER_PROMPT_VIDEO,

        # --- Pass-through user knobs from VllmAsyncConfig ---
        tensor_parallel_size=int(config.num_gpus),
        data_parallel_size=max(1, config.data_parallel_size),
        max_num_seqs=config.max_num_seqs if config.max_num_seqs > 0 else None,
        enforce_eager=config.enforce_eager,
        kv_cache_dtype=config.kv_cache_dtype,
        async_scheduling=config.async_scheduling,
        enable_chunked_prefill=config.enable_chunked_prefill,
        long_prefill_token_threshold=config.long_prefill_token_threshold,
        stream_interval=config.stream_interval,
        distributed_executor_backend=config.distributed_executor_backend,
        skip_mm_profiling=config.skip_mm_profiling,
        disable_log_stats=config.disable_log_stats,
        enable_log_requests=config.enable_log_requests,
        mm_encoder_tp_mode=config.mm_encoder_tp_mode or None,
        mm_processor_cache_type=config.mm_processor_cache_type or None,
        disable_chunked_mm_input=config.disable_chunked_mm_input,

        # --- Mirror sync derivations of VllmConfig fields shared with
        # VllmAsyncConfig (fp8, disable_mmcache) ---
        quantization="fp8" if config.fp8 else None,
        mm_processor_cache_gb=0.0 if config.disable_mmcache else 4.0,

        # --- Async-only invariants ---
        # Frames are CPU-pre-sampled by VllmPrepStage (reused verbatim by
        # the async pipeline; the legacy VllmAsyncPrepStage class no
        # longer exists), so the processor MUST NOT re-sample them.
        # This flag is the contract between prep and engine; never set
        # it elsewhere.  ``do_resize``/``do_rescale``/``do_normalize``
        # are gated by ``config.model_preprocess_enabled`` so the single-owner
        # contract (CPU vs vLLM) stays explicit.
        mm_processor_kwargs={
            "do_sample_frames": False,
            "do_resize": config.model_preprocess_enabled,
            "do_rescale": config.model_preprocess_enabled,
            "do_normalize": config.model_preprocess_enabled,
        },

        compilation_config=CompilationConfig(cudagraph_mode="piecewise"),
        enable_prefix_caching=True,
        use_tqdm_on_load=False,
    )
```

### Step 4: Implement `make_llm_input()`

Convert prompt and video frames to your model's expected input format:

#### Pattern 1: Token IDs + Video Tensor (e.g., Qwen)

```python
@staticmethod
def make_llm_input(
    prompt: str,
    frames: torch.Tensor,
    metadata: dict[str, Any],
    processor: AutoProcessor,
    config: VllmConfig,
) -> dict[str, Any]:
    """Make LLM input for token-based models."""
    # Create message structure (model-specific)
    message = {
        "role": "user",
        "content": [
            {"type": "video"},  # Placeholder for video
            {"type": "text", "text": prompt},
        ],
    }
    
    # Apply chat template and tokenize
    prompt_ids = processor.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()
    
    return {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": {"video": [(frames, metadata)]},
    }
```

#### Pattern 2: Text Prompt + Images (image-only models)

```python
@staticmethod
def make_llm_input(
    prompt: str,
    frames: torch.Tensor,
    metadata: dict[str, Any],  # noqa: ARG004
    processor: AutoProcessor,
    config: VllmConfig,
) -> dict[str, Any]:
    """Make LLM input for text-based models with PIL images."""
    from PIL import Image
    
    # Convert tensor frames to PIL Images
    # frames shape: (num_frames, C, H, W) in range [0, 1]
    images = []
    for frame in frames:
        # Convert to (H, W, C) and scale to [0, 255]
        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(frame_np))
    
    # Build prompt with image placeholders
    prompt_with_placeholders = f"<|image_1|>" * len(images) + f"\n{prompt}"
    
    return {
        "prompt": prompt_with_placeholders,
        "multi_modal_data": {"image": images if config.use_image_input else [(frames, metadata)]},
    }
```

**Key points:**
1. **Input format varies by model**: Check vLLM's model-specific docs
2. **`multi_modal_data` key matters**: `"video"` vs `"image"` (singular vs plural)
3. **Frames format**: Expect `torch.Tensor` of shape `(num_frames, C, H, W)` in range `[0, 1]`
4. **Metadata is part of the plugin API**: include it even if your model ignores it
5. **`config.use_image_input` controls modality**: use it when one plugin supports both images and video
6. **Chat templates**: Use `processor.apply_chat_template()` if model supports it

**Testing:**
```python
import torch
frames = torch.rand(8, 3, 224, 224)  # 8 frames, RGB, 224x224
prompt = "Describe this video"
config = VllmConfig(model_variant="mymodel")
processor = VllmMyModel.processor(config)

inputs = VllmMyModel.make_llm_input(
    prompt,
    frames,
    {"fps": 1.0},
    processor,
    config,
)

# Verify structure
assert isinstance(inputs, dict)
assert "prompt_token_ids" in inputs or "prompt" in inputs
assert "multi_modal_data" in inputs
assert "video" in inputs["multi_modal_data"] or "image" in inputs["multi_modal_data"]
```

### Step 5: Implement `decode()`

Extract the caption string from vLLM's output:

```python
@staticmethod
def decode(vllm_output: RequestOutput) -> str:
    """Decode vllm output into a caption."""
    return vllm_output.outputs[0].text
```

**Common patterns:**

1. **Simple text extraction** (most models):
   ```python
   return vllm_output.outputs[0].text
   ```

2. **With post-processing** (strip special tokens):
   ```python
   text = vllm_output.outputs[0].text
   text = text.strip()
   text = text.replace("<|endoftext|>", "")
   return text
   ```

3. **Multi-output handling** (rare):
   ```python
   # Take best output based on score
   outputs = vllm_output.outputs
   best_output = max(outputs, key=lambda x: x.cumulative_logprob)
   return best_output.text
   ```

**Testing:**
```python
from vllm import RequestOutput
from vllm.outputs import CompletionOutput

mock_output = RequestOutput(
    request_id="test",
    prompt="",
    prompt_token_ids=[],
    outputs=[CompletionOutput(index=0, text="A person walking", token_ids=[])],
    finished=True,
)

caption = VllmMyModel.decode(mock_output)
assert isinstance(caption, str)
assert caption == "A person walking"
```

`decode()` is intentionally narrow: it returns caption text only. The shared
vLLM interface wraps the text with finish-reason and token-count metadata, the
caption stage assigns `caption_status` and `caption_failure_reason`, and the
metadata writer persists those fields. Plugin code should not write stage-level
or writer-level metadata.

### Step 6: Implement `make_refined_llm_request()`

Create a stage 2 refinement request that combines the stage 1 caption with a refinement prompt:

```python
# Default refinement prompt (customize for your model)
_DEFAULT_REFINE_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.
"""

@staticmethod
def make_refined_llm_request(
    request: VllmCaptionRequest,
    processor: AutoProcessor,
    refine_prompt: str | None = None,
) -> VllmCaptionRequest:
    """Make a refined LLM request for stage 2 captioning."""
    # Use provided prompt or default
    _refine_prompt = _DEFAULT_REFINE_PROMPT if refine_prompt is None else refine_prompt
    
    # Validate request has caption
    if request.caption is None:
        msg = "Request caption is None"
        raise ValueError(msg)
    
    # Combine refinement prompt with stage 1 caption
    final_prompt = _refine_prompt + request.caption
    
    # Extract original multimodal data (reuse frames!)
    if "multi_modal_data" not in request.inputs:
        msg = "Message does not contain multi_modal_data"
        raise ValueError(msg)
    
    # Get video frames or images from original request
    mm_data = request.inputs["multi_modal_data"]
    if "video" in mm_data:
        video_value = mm_data["video"]
        if isinstance(video_value, list) and video_value and isinstance(video_value[0], tuple):
            video_frames, metadata = video_value[0]
        else:
            video_frames = video_value
            metadata = {"fps": 1.0}
        # Reuse make_llm_input to create new inputs
        inputs = VllmMyModel.make_llm_input(
            final_prompt,
            video_frames,
            metadata,
            processor,
            VllmConfig(model_variant="mymodel"),
        )
    elif "image" in mm_data:
        # For image-based models, reconstruct from PIL images
        images = mm_data["image"]
        # Build new prompt (model-specific)
        prompt_with_placeholders = f"<|image_1|>" * len(images) + f"\n{final_prompt}"
        inputs = {
            "prompt": prompt_with_placeholders,
            "multi_modal_data": {"image": images},
        }
    else:
        msg = "Unknown multimodal data format"
        raise ValueError(msg)
    
    # Create new request with RESET caption and stage2_prompt
    return VllmCaptionRequest(
        request_id=secrets.token_hex(8),  # New unique ID
        inputs=inputs,
        # caption=None (default) - IMPORTANT!
        # stage2_prompt=None (default) - IMPORTANT!
    )
```

**Critical requirements:**
1. ✅ **Reuse multimodal data** from original request (don't re-decode video)
2. ✅ **Generate new `request_id`** (don't reuse stage 1 ID)
3. ✅ **Leave `caption=None`** (will be filled by stage 2 generation)
4. ✅ **Leave `stage2_prompt=None`** (no third stage)
5. ✅ **Combine prompts properly** (refine prompt + stage 1 caption)

**Testing:**
```python
# Create stage 1 request with caption
stage1_request = VllmCaptionRequest(
    request_id="stage1-abc",
    inputs={
        "prompt_token_ids": [123, 456],
        "multi_modal_data": {"video": [(torch.rand(8, 3, 224, 224), {"fps": 1.0})]}
    },
    caption="A person walking",
    stage2_prompt="Refine this caption",
)

# Create stage 2 request
config = VllmConfig(model_variant="mymodel")
processor = VllmMyModel.processor(config)
stage2_request = VllmMyModel.make_refined_llm_request(
    stage1_request, processor, "Refine this caption"
)

# Verify structure
assert stage2_request.request_id != stage1_request.request_id
assert stage2_request.caption is None  # CRITICAL
assert stage2_request.stage2_prompt is None  # CRITICAL
assert "multi_modal_data" in stage2_request.inputs

# Verify stage 1 caption is in new prompt
if "prompt_token_ids" in stage2_request.inputs:
    tokens = stage2_request.inputs["prompt_token_ids"]
    decoded = processor.tokenizer.decode(tokens)
    assert "A person walking" in decoded
```

### Step 7: Register Plugin

Add your plugin to the registry in `cosmos_curator/models/vllm_interface.py`:

```python
from cosmos_curator.models.vllm_mymodel import VllmMyModel

_VLLM_PLUGINS = {
    VllmNemotronNano12Bv2VL.model_variant(): VllmNemotronNano12Bv2VL,
    VllmQwen7B.model_variant(): VllmQwen7B,
    VllmCosmosReason1VL.model_variant(): VllmCosmosReason1VL,
    VllmCosmosReason2VL.model_variant(): VllmCosmosReason2VL,
    VllmMyModel.model_variant(): VllmMyModel,  # Add this line
}
```

### Step 8: Add Model ID Mapping

Add your model's HuggingFace ID to `cosmos_curator/models/vllm_model_ids.py`:

```python
_VLLM_MODELS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "nemotron": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "cosmos_r1": "nvidia/Cosmos-Reason1-7B",
    "cosmos_r2": "nvidia/Cosmos-Reason2-8B",
    "mymodel": "organization/my-model-name",  # Add this line
}
```

---

## Testing Your Plugin

### Unit Tests

Create `tests/models/test_vllm_mymodel.py`:

```python
"""Unit tests for MyModel vLLM plugin."""

import pytest
import torch
from vllm import RequestOutput
from vllm.outputs import CompletionOutput

from cosmos_curator.models.vllm_mymodel import VllmMyModel
from cosmos_curator.pipelines.video.utils.data_model import (
    VllmCaptionRequest,
    VllmConfig,
    VllmSamplingConfig,
    WindowConfig,
)


def test_model_variant():
    """Test model variant returns correct string."""
    assert VllmMyModel.model_variant() == "mymodel"


def test_model_id():
    """Test model ID is registered."""
    model_id = VllmMyModel.model_id()
    assert isinstance(model_id, str)
    assert len(model_id) > 0


@pytest.mark.skip(reason="Requires model weights downloaded")
def test_model_path():
    """Test model path exists."""
    config = VllmConfig(model_variant="mymodel")
    model_path = VllmMyModel.model_path(config)
    assert model_path.exists(), f"Model not found at {model_path}"


@pytest.mark.skip(reason="Requires model weights downloaded")
def test_processor():
    """Test processor can be loaded."""
    config = VllmConfig(model_variant="mymodel")
    processor = VllmMyModel.processor(config)
    assert processor is not None
    # Try tokenizing
    tokens = processor.tokenizer("test prompt")
    assert len(tokens.input_ids) > 0


@pytest.mark.env("default")
@pytest.mark.skip(reason="Requires model weights and GPU")
def test_model():
    """Test model can be instantiated."""
    config = VllmConfig(model_variant="mymodel", num_gpus=1)
    llm = VllmMyModel.model(config)
    assert llm is not None
    assert llm.llm_engine is not None


def test_make_llm_input():
    """Test LLM input creation."""
    pytest.skip("Requires processor - implement after model download")

    frames = torch.rand(8, 3, 224, 224)
    prompt = "Describe this video"
    config = VllmConfig(model_variant="mymodel")
    processor = VllmMyModel.processor(config)

    inputs = VllmMyModel.make_llm_input(
        prompt,
        frames,
        {"fps": 1.0},
        processor,
        config,
    )

    # Verify structure
    assert isinstance(inputs, dict)
    assert "prompt_token_ids" in inputs or "prompt" in inputs
    assert "multi_modal_data" in inputs

    mm_data = inputs["multi_modal_data"]
    assert "video" in mm_data or "image" in mm_data


def test_decode():
    """Test decoding vLLM output."""
    mock_output = RequestOutput(
        request_id="test",
        prompt="",
        prompt_token_ids=[],
        outputs=[CompletionOutput(index=0, text="A person walking", token_ids=[])],
        finished=True,
    )

    caption = VllmMyModel.decode(mock_output)
    assert isinstance(caption, str)
    assert caption == "A person walking"


def test_make_refined_llm_request():
    """Test stage 2 request creation."""
    pytest.skip("Requires processor - implement after model download")

    # Create stage 1 request
    stage1_request = VllmCaptionRequest(
        request_id="stage1-test",
        inputs={
            "prompt_token_ids": [123, 456],
            "multi_modal_data": {"video": [(torch.rand(8, 3, 224, 224), {"fps": 1.0})]}
        },
        caption="A person walking",
        stage2_prompt="Refine this",
    )

    config = VllmConfig(model_variant="mymodel")
    processor = VllmMyModel.processor(config)
    stage2_request = VllmMyModel.make_refined_llm_request(
        stage1_request, processor, "Refine this"
    )

    # Verify structure
    assert stage2_request.request_id != stage1_request.request_id
    assert stage2_request.caption is None
    assert stage2_request.stage2_prompt is None
    assert "multi_modal_data" in stage2_request.inputs
```

Run tests:
```bash
# Unit tests (no GPU required)
pytest tests/models/test_vllm_mymodel.py -v

# Integration tests (requires GPU and model weights)
cosmos-curator local launch --curator-path . -- \
    pixi run --as-is -e default pytest tests/models/test_vllm_mymodel.py -m env -v
```

### Integration Test

Test end-to-end captioning:

```python
@pytest.mark.env("default")
@pytest.mark.slow
def test_mymodel_e2e_captioning():
    """End-to-end test for MyModel plugin."""
    from cosmos_curator.models.vllm_interface import (
        auto_processor,
        make_metadata,
        make_model_inputs,
        sampling_params,
        vllm_caption,
        vllm_model,
    )

    config = VllmConfig(
        model_variant="mymodel",
        num_gpus=1,
        batch_size=2,
        sampling_config=VllmSamplingConfig(max_tokens=128),
    )

    # Setup
    llm = vllm_model(config)
    processor = auto_processor(config)
    samp_params = sampling_params(config.sampling_config)

    # Create test inputs
    frames = torch.rand(8, 3, 224, 224)
    metadata = make_metadata([frames], WindowConfig(sampling_fps=1.0))
    model_inputs = make_model_inputs([frames], metadata, config, processor, "Describe this video")

    # Test stage 1
    results = vllm_caption(
        model_inputs,
        llm,
        processor,
        samp_params,
        config,
        max_inflight_requests=0,
        inflight_batching=True,
    )

    assert len(results) == 1
    assert isinstance(results[0].text, str)
    assert results[0].text.strip()
    assert results[0].token_counts.prompt_tokens >= 0
    assert results[0].token_counts.output_tokens >= 0
    print(f"Stage 1 caption: {results[0].text}")

    # Test stage 2
    results_s2 = vllm_caption(
        model_inputs,
        llm,
        processor,
        samp_params,
        config,
        max_inflight_requests=0,
        inflight_batching=True,
        stage2_prompts=["Refine this caption"],
    )

    assert len(results_s2) == 1
    assert isinstance(results_s2[0].text, str)
    assert results_s2[0].text.strip()
    assert results_s2[0].token_counts.prompt_tokens >= results[0].token_counts.prompt_tokens
    print(f"Stage 2 caption: {results_s2[0].text}")
```

---

## Common Plugin Bugs and How to Avoid Them

### Bug 1: Forgot to Reset `caption` and `stage2_prompt` in Stage 2 Request

❌ **WRONG:**
```python
return VllmCaptionRequest(
    request_id=new_id,
    inputs=new_inputs,
    caption=request.caption,  # Should be None!
    stage2_prompt=request.stage2_prompt,  # Should be None!
)
```

✅ **CORRECT:**
```python
return VllmCaptionRequest(
    request_id=new_id,
    inputs=new_inputs,
    # caption and stage2_prompt default to None - don't set them!
)
```

**Why:** Stage 2 request needs `caption=None` so vLLM generates a new caption. If you copy the old caption, stage 2 won't run.

### Bug 2: Wrong Multimodal Data Key

❌ **WRONG:**
```python
return {
    "prompt_token_ids": [...],
    "multi_modal_data": {"video_frames": tensor},  # Wrong key!
}
```

✅ **CORRECT:**
```python
# For video models
return {
    "prompt_token_ids": [...],
    "multi_modal_data": {"video": [(tensor, metadata)]},  # vLLM expects "video"
}

# For image models
return {
    "prompt": "...",
    "multi_modal_data": {"image": [PIL.Image, ...]},  # vLLM expects "image"
}
```

**Why:** vLLM has specific key names it expects. Check your model's vLLM implementation.

### Bug 3: Not Reusing Video Frames in Stage 2

❌ **WRONG:**
```python
# This re-decodes the video - expensive and unnecessary!
new_frames = decode_video_from_bytes(video_bytes)
```

✅ **CORRECT:**
```python
# Reuse frames from stage 1 request
video_frames = request.inputs["multi_modal_data"]["video"]
```

**Why:** Decoding video is expensive. Stage 1 already has decoded frames - reuse them!

### Bug 4: `decode()` Returns Non-String

❌ **WRONG:**
```python
return vllm_output.outputs  # Returns list!
```

✅ **CORRECT:**
```python
return vllm_output.outputs[0].text  # Returns string
```

**Why:** The interface expects a string. Returning a list will cause type errors.

### Bug 5: Incorrect Tensor Shape or Data Type

❌ **WRONG:**
```python
# Expects (num_frames, C, H, W) but receives (C, num_frames, H, W)
frames = frames.transpose(0, 1)  # Wrong!
```

✅ **CORRECT:**
```python
# Assume input is correct: (num_frames, C, H, W)
# Only transform if your model requires a different format
```

**Why:** The interface contract is `(num_frames, C, H, W)` in range `[0, 1]`. Document if your model needs different format.

### Bug 6: Forgetting to Handle Edge Cases

❌ **WRONG:**
```python
def make_refined_llm_request(request, processor, refine_prompt):
    # No validation!
    final_prompt = refine_prompt + request.caption
    # Crashes if caption is None
```

✅ **CORRECT:**
```python
def make_refined_llm_request(request, processor, refine_prompt):
    if request.caption is None:
        msg = "Request caption is None - cannot create refined request"
        raise ValueError(msg)
    
    _refine_prompt = _DEFAULT_REFINE_PROMPT if refine_prompt is None else refine_prompt
    final_prompt = _refine_prompt + request.caption
```

**Why:** Defensive programming prevents cryptic errors. Validate inputs and provide clear error messages.

---

## Best Practices

### 1. Follow Existing Plugin Patterns

**Reference implementations:**
- **`vllm_qwen.py`**: Token-based model with video support (most complete example)
- **`vllm_nemotron.py`**: Video + metadata model format
- **`vllm_cosmos_reason1_vl.py`**: NVIDIA model example

Copy structure from the closest match to your model.

### 2. Add Comprehensive Docstrings

```python
@staticmethod
def make_llm_input(
    prompt: str,
    frames: torch.Tensor,
    metadata: dict[str, Any],
    processor: AutoProcessor,
    config: VllmConfig,
) -> dict[str, Any]:
    """Make LLM input for MyModel.
    
    Args:
        prompt: Text prompt for video description.
        frames: Video frames as torch.Tensor with shape (num_frames, C, H, W)
                in range [0, 1]. Typically 8 frames for MyModel.
        metadata: Per-window metadata supplied by ``make_metadata``.
        processor: AutoProcessor for tokenization.
        config: vLLM configuration. Use ``config.use_image_input`` for image vs video plugins.
    
    Returns:
        Dictionary with keys:
        - "prompt_token_ids": List of token IDs
        - "multi_modal_data": Dict with "video" key containing the model-specific video payload,
          usually frames plus metadata
    
    """
```

### 3. Add Model-Specific Constants

```python
# MyModel-specific configuration
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 32768
TRUST_REMOTE_CODE = True
LIMIT_MM_PER_PROMPT_VIDEO = {"video": 1}

_DEFAULT_REFINE_PROMPT = """
Model-specific refinement prompt that works well with MyModel's training...
"""
```

### 4. Handle Model Variants

If you have multiple sizes (e.g., 7B, 13B):

```python
class VllmMyModel7B(VllmMyModelBase):
    @staticmethod
    def model_variant() -> str:
        return "mymodel7b"

class VllmMyModel13B(VllmMyModelBase):
    @staticmethod
    def model_variant() -> str:
        return "mymodel13b"
    
    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        # Override with different settings for larger model
        return LLM(
            model=str(cls.model_path(config)),
            max_model_len=65536,  # Larger context
            ...
        )
```

### 5. Add Logging for Debugging

```python
from loguru import logger

@staticmethod
def make_llm_input(
    prompt: str,
    frames: torch.Tensor,
    metadata: dict[str, Any],  # noqa: ARG004
    processor: AutoProcessor,
    config: VllmConfig,  # noqa: ARG004
) -> dict[str, Any]:
    logger.debug(f"Creating input for {frames.shape[0]} frames with prompt: {prompt[:50]}...")
    
    inputs = {...}
    
    logger.debug(f"Created input with keys: {list(inputs.keys())}")
    return inputs
```

### 6. Test with Real Videos Early

Don't just test with random tensors:

```python
# Load a real video for testing
from cosmos_curator.pipelines.video.utils.video_utils import decode_video

video_path = "test_data/sample_video.mp4"
frames = decode_video(video_path, num_frames=8)
```

---

## Checklist: Plugin Implementation

Use this checklist to ensure your plugin is complete:

- [ ] **Step 1**: Created `vllm_mymodel.py` file with proper imports
- [ ] **Step 2**: Implemented `model_variant()` returning unique string
- [ ] **Step 3**: Implemented `processor()` loading AutoProcessor
- [ ] **Step 4**: Implemented `model()` with appropriate vLLM configuration
- [ ] **Step 5**: Implemented `make_llm_input()` converting frames to model format
- [ ] **Step 6**: Implemented `decode()` extracting text from RequestOutput
- [ ] **Step 7**: Implemented `make_refined_llm_request()` with proper validation
- [ ] **Step 8**: Registered plugin in `vllm_interface.py`
- [ ] **Step 9**: Added model ID to `vllm_model_ids.py`
- [ ] **Step 10**: Created unit tests in `tests/models/test_vllm_mymodel.py`
- [ ] **Step 11**: Tested with `pytest` (unit tests pass)
- [ ] **Step 12**: Downloaded model weights
- [ ] **Step 13**: Tested with GPU (integration tests pass)
- [ ] **Step 14**: Tested end-to-end captioning with real videos
- [ ] **Step 15**: Verified stage 2 refinement works correctly
- [ ] **Step 16**: Added docstrings to all methods
- [ ] **Step 17**: Added to documentation (if public model)

---

## Troubleshooting

### Model Won't Load

**Error:** `FileNotFoundError: Model not found at path/to/model`

**Solutions:**
1. Download model weights: `pixi run --as-is -e model-download python -m cosmos_curator.models.download_model mymodel`
2. Check `vllm_model_ids.py` has correct HuggingFace ID
3. Verify `model_path()` returns correct path

### vLLM Errors on Input Format

**Error:** `ValueError: Invalid input format for model`

**Solutions:**
1. Check vLLM's documentation for your model
2. Inspect vLLM's model implementation: `vllm/model_executor/models/yourmodel.py`
3. Compare with a working plugin (Qwen or Nemotron)
4. Verify `multi_modal_data` keys match what vLLM expects

### Caption Results Are Empty or Error-Normalized

**Error:** Generated text is empty, or downstream metadata records
`caption_status="error"`.

**Solutions:**
1. Check `decode()` implementation - ensure it returns string
2. Verify vLLM model generates output: add logging in `decode()`
3. Test `decode()` with mock RequestOutput
4. See [vllm-interface-debug.md](vllm-interface-debug.md) Scenario 1

### Stage 2 Not Working

**Error:** Stage 2 captions same as Stage 1

**Solutions:**
1. Verify `make_refined_llm_request()` sets `caption=None` and `stage2_prompt=None`
2. Check refined request has new `request_id`
3. Verify refinement prompt is combined with stage 1 caption
4. See [vllm-interface-debug.md](vllm-interface-debug.md) Scenario 2

---

## Examples

### Complete Plugin: Hypothetical VideoLLaMA

```python
"""VideoLLaMA vLLM plugin example."""

import secrets
from typing import Any, TypedDict

import torch
from transformers import AutoProcessor
from vllm import LLM, RequestOutput

from cosmos_curator.models.vllm_plugin import VllmPlugin
from cosmos_curator.pipelines.video.utils.data_model import (
    VllmCaptionRequest,
    VllmConfig,
)

# VideoLLaMA configuration
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
LIMIT_MM_PER_PROMPT = {"video": 1}

_DEFAULT_REFINE_PROMPT = "Improve the following video description: "


class VideoLLaMAMessage(TypedDict):
    """Message format for VideoLLaMA."""
    role: str
    content: list[dict[str, str]]


class VllmVideoLLaMA(VllmPlugin):
    """VideoLLaMA vLLM plugin."""

    @staticmethod
    def model_variant() -> str:
        return "videollama"

    @classmethod
    def processor(cls, config: VllmConfig) -> AutoProcessor:
        processor = AutoProcessor.from_pretrained(
            cls.model_path(config),
            trust_remote_code=True,
        )
        return processor

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        quantization = "fp8" if config.fp8 else None

        return LLM(
            model=str(cls.model_path(config)),
            quantization=quantization,
            tensor_parallel_size=config.num_gpus,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
            trust_remote_code=True,
        )

    @staticmethod
    def make_llm_input(
            prompt: str,
            frames: torch.Tensor,
            metadata: dict[str, Any],
            processor: AutoProcessor,
            config: VllmConfig,
    ) -> dict[str, Any]:
        message = VideoLLaMAMessage(
            role="user",
            content=[
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        )

        prompt_ids = processor.apply_chat_template(
            [message],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )[0].tolist()

        return {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {"video": [(frames, metadata)]},
        }

    @staticmethod
    def make_refined_llm_request(
            request: VllmCaptionRequest,
            processor: AutoProcessor,
            refine_prompt: str | None = None,
    ) -> VllmCaptionRequest:
        _refine_prompt = _DEFAULT_REFINE_PROMPT if refine_prompt is None else refine_prompt

        if request.caption is None:
            msg = "Request caption is None"
            raise ValueError(msg)

        final_prompt = _refine_prompt + request.caption
        video_frames = request.inputs["multi_modal_data"]["video"]

        message = VideoLLaMAMessage(
            role="user",
            content=[
                {"type": "video"},
                {"type": "text", "text": final_prompt},
            ],
        )

        prompt_ids = processor.apply_chat_template(
            [message],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )[0].tolist()

        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs={
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": {"video": video_frames},
            },
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        text = vllm_output.outputs[0].text
        # Post-processing: strip special tokens
        text = text.strip().replace("</s>", "").replace("<s>", "")
        return text
```

---

## References

- **Plugin Interface Definition**: `cosmos_curator/models/vllm_plugin.py`
- **Existing Plugin Examples**:
  - `cosmos_curator/models/vllm_qwen.py` - Token-based with video
  - `cosmos_curator/models/vllm_nemotron.py` - Video + metadata format
  - `cosmos_curator/models/vllm_cosmos_reason1_vl.py` - NVIDIA model
- **vLLM Documentation**: https://docs.vllm.ai/
- **Design Document**: [vllm-interface.md](../design/vllm-interface.md)
- **Debug Guide**: [vllm-interface-debug.md](vllm-interface-debug.md)
- **Profiling Guide**: [profiling.md](profiling.md)
