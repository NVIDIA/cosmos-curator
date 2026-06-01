# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Cosmos Curator vLLM interface.

═══════════════════════════════════════════════════════════════════════════════
READING THIS CODE? START HERE - Flow Trace Guide
═══════════════════════════════════════════════════════════════════════════════

Problem this solves:
- Without this interface, every CuratorStage that uses vLLM would need to reimplement:
  * Model loading with correct quantization/tensor parallelism settings
  * Two-stage captioning workflow (initial caption → refinement request → final caption)
  * Batching logic with proper request tracking
  * Model-specific input formatting (Qwen token IDs vs Nemotron metadata prompts vs ...)

This was leading to code duplication across the input preparation and captioning stages
for each model.

Design decision: Plugin architecture rather than if/elif chain because:
- Models have fundamentally different input formats
- Makes adding new models contained to a single file
- Allows model-specific refinement logic
- Supporting 5+ models, expanding to image/audio captioning
- Plugins can be removed from the code base, e.g. if the user of the code
  cannot have a specific model or related code in their code base

FLOW TRACE - How a video gets captioned (follow this to understand the code):

1. Entry: Pipeline calls vllm_caption() with model_inputs
   └─> Dispatches to _caption_inflight_batching() (typical) or _caption_no_inflight_batching() (fallback)

2. Request Creation: Wraps each input in VllmCaptionRequest
   └─> Includes unique request_id, inputs dict, and optional stage2_prompt

3. Continuous Processing: engine.step() processes requests as they arrive
   └─> Submits requests when capacity available
   └─> Allows interleaved stage 1 and stage 2 processing for better throughput

4. Output Decoding: process_vllm_output() extracts caption text
   └─> Calls plugin.decode() - e.g., cosmos_curator/models/vllm_qwen.py

5. Stage 2 (if needed): Creates refinement request, adds back to queue
   └─> Uses plugin.make_refined_llm_request()
   └─> Example: cosmos_curator/models/vllm_qwen.py

6. Return: List of caption strings

Note: No-inflight batching path (_caption_no_inflight_batching, line ~277) is a
fallback for simpler debugging and testing. Production code uses inflight batching.

Plugin Implementations (model-specific code):
- VllmCosmosReason1VL:     cosmos_curator/models/vllm_cosmos_reason1_vl.py
- VllmCosmosReason2VL:     cosmos_curator/models/vllm_cosmos_reason2_vl.py
- VllmCosmos3NanoReasonerVL:  cosmos_curator/models/vllm_cosmos_reason3_vl.py
- VllmCosmos3SuperReasonerVL: cosmos_curator/models/vllm_cosmos_reason3_vl.py
- VllmNemotronNano12Bv2VL: cosmos_curator/models/vllm_nemotron.py
- VllmQwen7B:              cosmos_curator/models/vllm_qwen.py
- VllmQwen3VL30B:          cosmos_curator/models/vllm_qwen.py
- VllmQwen3VL235B:         cosmos_curator/models/vllm_qwen.py
- Plugin Interface:        cosmos_curator/models/vllm_plugin.py (7 abstract methods)
- Registry:                _VLLM_PLUGINS dict

Public API (what pipeline stages call):
- vllm_model()       - Create model instance
- auto_processor()   - Get model processor
- sampling_params()  - Create sampling config
- make_model_inputs() - Convert frames to model-specific format
- vllm_caption()     - Main captioning function (entry point)
- vllm_generate()    - Lower-level batch generation (no inflight batching)

DEBUG TIP: Set breakpoint in vllm_caption() and step through for complete flow
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import contextlib
import secrets
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import (  # noqa: UP035, remove noqa when we drop support for python 3.10
    TYPE_CHECKING,
    Any,
    Deque,
    Iterable,
    TypeVar,
    cast,
)

import attrs
import numpy as np
import tenacity
import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, PoolingOutput, PoolingRequestOutput, RequestOutput, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.exceptions import EngineDeadError

from cosmos_curator.core.utils.infra.tracing import traced_span
from cosmos_curator.core.utils.misc import grouping
from cosmos_curator.models.vllm_cosmos_reason1_vl import VllmCosmosReason1VL
from cosmos_curator.models.vllm_cosmos_reason2_vl import VllmCosmosReason2VL
from cosmos_curator.models.vllm_cosmos_reason3_vl import VllmCosmos3NanoReasonerVL, VllmCosmos3SuperReasonerVL
from cosmos_curator.models.vllm_nemotron import VllmNemotronNano12Bv2VL
from cosmos_curator.models.vllm_plugin import VllmPlugin
from cosmos_curator.models.vllm_qwen import (
    VllmQwen3VL30B,
    VllmQwen3VL30BFP8,
    VllmQwen3VL235B,
    VllmQwen3VL235BFP8,
    VllmQwen7B,
    VllmQwen3527B,
    VllmQwen3627B,
    VllmQwen3627BFP8,
)
from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION
from cosmos_curator.pipelines.video.utils.data_model import (
    TokenCounts,
    VllmCaptionRequest,
    VllmConfig,
    VllmSamplingConfig,
    WindowConfig,
)

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM

# Add new vLLM plugins to _VLLM_PLUGINS
_VLLM_PLUGINS = {
    VllmCosmosReason1VL.model_variant(): VllmCosmosReason1VL,
    VllmCosmosReason2VL.model_variant(): VllmCosmosReason2VL,
    VllmCosmos3NanoReasonerVL.model_variant(): VllmCosmos3NanoReasonerVL,
    VllmCosmos3SuperReasonerVL.model_variant(): VllmCosmos3SuperReasonerVL,
    VllmNemotronNano12Bv2VL.model_variant(): VllmNemotronNano12Bv2VL,
    VllmQwen3527B.model_variant(): VllmQwen3527B,
    VllmQwen3627B.model_variant(): VllmQwen3627B,
    VllmQwen3627BFP8.model_variant(): VllmQwen3627BFP8,
    VllmQwen3VL235B.model_variant(): VllmQwen3VL235B,
    VllmQwen3VL235BFP8.model_variant(): VllmQwen3VL235BFP8,
    VllmQwen3VL30B.model_variant(): VllmQwen3VL30B,
    VllmQwen3VL30BFP8.model_variant(): VllmQwen3VL30BFP8,
    VllmQwen7B.model_variant(): VllmQwen7B,
}

T = TypeVar("T")


@attrs.define
class VllmWindowResult:
    """Final vLLM caption result for a single window."""

    text: str
    finish_reason: str | None
    token_counts: TokenCounts


def _make_window_result(request: VllmCaptionRequest, token_counts: TokenCounts) -> VllmWindowResult:
    """Assemble the terminal per-window vLLM result."""
    return VllmWindowResult(
        text=request.caption or VLLM_UNKNOWN_CAPTION,
        finish_reason=request.finish_reason,
        token_counts=token_counts,
    )


def _get_vllm_plugin(variant: str) -> VllmPlugin:
    """Get the vLLM plugin for the model variant.

    Args:
        variant: The variant of the model.

    Returns:
        The vLLM plugin.

    Raises:
        ValueError: If the model variant is not supported.

    """
    plugin = _VLLM_PLUGINS.get(variant)
    if plugin is None:
        msg = f"vLLM model variant {variant} not supported"
        raise ValueError(msg)
    return cast("VllmPlugin", plugin)


def _save_frames_as_pngs(
    frames: torch.Tensor,
    output_dir: Path,
    prefix: str,
) -> None:
    """Save video frames as PNG files for debugging.

    Args:
        frames: Tensor of shape [num_frames, C, H, W] containing video frames.
        output_dir: Directory to save the PNG files.
        prefix: Prefix for the PNG filenames (e.g., "frame" or "window_0_frame").

    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # frames shape: [num_frames, C, H, W]
    num_frames = frames.shape[0]
    logger.info(f"Saving {num_frames} frames to {output_dir} with prefix '{prefix}' {frames.shape=}")

    RGB_CHANNELS = 3
    GRAYSCALE_CHANNELS = 1
    MAX_NORMALIZED_VALUE = 1.0
    PNG_MAX_VALUE = 255

    for frame_idx in range(num_frames):
        frame = frames[frame_idx]  # shape: [C, H, W]

        # Convert from torch tensor to numpy
        # Assuming frame is in [C, H, W] format with values in [0, 255] or [0, 1]
        frame_np = frame.cpu().numpy()

        # Convert from [C, H, W] to [H, W, C]
        frame_np = np.transpose(frame_np, (1, 2, 0))

        # Ensure values are in [0, 255] range
        frame_np = (
            (frame_np * PNG_MAX_VALUE).astype(np.uint8)
            if frame_np.max() <= MAX_NORMALIZED_VALUE
            else frame_np.astype(np.uint8)
        )

        # Handle grayscale (single channel) or RGB
        if frame_np.shape[2] == GRAYSCALE_CHANNELS:
            frame_np = frame_np.squeeze(2)
            img = Image.fromarray(frame_np, mode="L")
        elif frame_np.shape[2] == RGB_CHANNELS:
            img = Image.fromarray(frame_np, mode="RGB")
        else:
            logger.warning(f"Unexpected number of channels: {frame_np.shape[2]}, skipping frame {frame_idx}")
            continue

        # Save as PNG - if prefix already ends with "_frame", don't add it again
        if prefix.endswith("_frame") or prefix == "frame":
            filename = f"{prefix}_{frame_idx:04d}.png"
        else:
            filename = f"{prefix}_frame_{frame_idx:04d}.png"
        output_path = output_dir / filename
        img.save(output_path, "PNG")

    logger.info(f"Saved {num_frames} frames to {output_dir}")


def vllm_model(config: VllmConfig) -> LLM:
    """Create a vLLM model instance.

    Args:
       config: Configuration for the vLLM model.

    Returns:
        A vLLM model instance.

    """
    return _get_vllm_plugin(config.model_variant).model(config)


def sampling_params(config: VllmSamplingConfig) -> SamplingParams:
    """Create a sampling parameters object for the vLLM model.

    Args:
        config: Configuration for the vLLM model.

    Returns:
        A sampling parameters object.

    """
    # Performance: consider adding skip_clone=True here once the sync
    # VllmCaptionStage path has been verified safe for shallow copies.
    # With skip_clone=True, vLLM's InputProcessor.process_inputs()
    # uses copy.copy() instead of copy.deepcopy() per request.
    return SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        min_p=config.min_p,
        min_tokens=config.min_tokens,
        max_tokens=config.max_tokens,
        stop_token_ids=[],
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


def auto_processor(config: VllmConfig) -> AutoProcessor:
    """Get the auto process for the model.

    Args:
        config: The configuration of the model.

    Returns:
        The auto processor for the model.

    """
    return _get_vllm_plugin(config.model_variant).processor(config)


def make_metadata(frames: Iterable[torch.Tensor], window_config: WindowConfig) -> list[dict[str, Any]]:
    """Make metadata for a iterable of frames.

    Args:
        frames: The frames to make metadata for.
        window_config: The window configuration to use for the metadata.

    Returns:
        The metadata for the frames.

    """
    # Verify that all tensors are 4D
    NUM_EXPECTED_DIMS = 4
    for i, f in enumerate(frames):
        if f.ndim != NUM_EXPECTED_DIMS:
            msg = (
                f"Expected all frames to have 4 dimensions (batch of videos of shape [num_frames, C, H, W]), "
                f"but frames[{i}] has shape {getattr(f, 'shape', None)}"
            )
            raise ValueError(msg)

    def _make_metadata(frames: torch.Tensor) -> dict[str, Any]:
        fps = window_config.sampling_fps
        num_frames = frames.shape[0]

        return {
            "fps": fps,
            "duration": num_frames / fps,
            "width": frames.shape[3],
            "height": frames.shape[2],
            "total_num_frames": num_frames,
            "frames_indices": list(range(num_frames)),
            "video_backend": "opencv",
            "do_sample_frames": False,
        }

    return [_make_metadata(frames) for frames in frames]


def make_model_inputs(  # noqa: PLR0913
    videos: list[torch.Tensor],
    metadata: list[dict[str, Any]],
    config: VllmConfig,
    processor: AutoProcessor,
    prompt: str,
    *,
    debug_window_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Make model inputs for a list of videos.

    Args:
        videos: list of decoded videos
        metadata: The metadata for each video
        config: The configuration for the vLLM model.
        processor: The processor to use for the vLLM model.
        prompt: The prompt to use for the vLLM model.
        debug_window_ids: Optional list of clip UUIDs for organizing debug frames.
            Frames will be saved to {debug_frames_output_dir}/{clip_uuid}/frame_NNNN.png

    Returns:
        A list of LLM inputs for each video

    """
    vllm_plugin = _get_vllm_plugin(config.model_variant)

    # Debug: Save frames as PNGs if enabled
    if config.debug_save_frames:
        if config.debug_frames_output_dir is None:
            logger.warning("debug_save_frames is True but debug_frames_output_dir is None, skipping frame saving")
        else:
            for idx, frames in enumerate(videos):
                # Use clip UUID as the subdirectory if provided
                if debug_window_ids and idx < len(debug_window_ids):
                    clip_uuid = debug_window_ids[idx]
                    output_dir = config.debug_frames_output_dir / clip_uuid
                else:
                    # Fallback to simple numeric naming
                    output_dir = config.debug_frames_output_dir / f"window_{idx:04d}"

                _save_frames_as_pngs(
                    frames,
                    output_dir,
                    prefix="frame",
                )

    return [
        vllm_plugin.make_llm_input(prompt, frames, md, processor, config)
        for frames, md in zip(videos, metadata, strict=True)
    ]


def vllm_generate(
    llm: LLM,
    sampling_params: SamplingParams,
    requests: list[VllmCaptionRequest],
    batch_size: int,
) -> list[RequestOutput]:
    """Generate captions for the data using the vLLM model.

    Args:
        llm: The vLLM model.
        sampling_params: The sampling parameters.
        requests: The captioning requests for the llm
        batch_size: The batch size.

    Returns:
        A list of captions.

    """
    inputs = [r.inputs for r in requests]
    all_outputs: list[RequestOutput] = []

    for batch_data in grouping.split_by_chunk_size(inputs, batch_size):
        # llm.generate can take a list of dicts, but does not advertize this in its type hints
        outputs = llm.generate(cast("Any", batch_data), sampling_params=sampling_params, use_tqdm=False)
        all_outputs.extend(outputs)

    # Change request ids from integer strings to the vllm_interface unique request ids.
    # Zip is safe because the requests and outputs are in the same order
    for out, req in zip(all_outputs, requests, strict=True):
        out.request_id = req.request_id

    return all_outputs


def process_vllm_output(
    engine_output: list[RequestOutput] | list[PoolingRequestOutput[PoolingOutput]],
    in_flight_requests: dict[str, VllmCaptionRequest],
    vllm_config: VllmConfig,
) -> list[VllmCaptionRequest]:
    """Process vLLM engine output, updating the in-flight requests with the decoded text.

    The output comes from vLLM engine.step().

    Args:
        engine_output: The output from the engine.
        in_flight_requests: The in-flight requests, keyed by request_id.
        vllm_config: The configuration for the VLLM model.

    Returns:
        A list of finished requests.

    """
    vllm_plugin = _get_vllm_plugin(vllm_config.model_variant)
    finished: list[VllmCaptionRequest] = []

    for out in engine_output:
        if not isinstance(out, RequestOutput):
            msg = f"Expected RequestOutput, got {type(out)}. If you are using a pooling model, this is not supported."
            raise TypeError(msg)

        if out.finished:
            request = in_flight_requests[out.request_id]
            request.caption = vllm_plugin.decode(out)
            request.prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids else 0
            request.output_tokens = len(out.outputs[0].token_ids) if out.outputs else 0
            request.finish_reason = out.outputs[0].finish_reason if out.outputs else None
            finished.append(request)

    return finished


def _caption_no_inflight_batching(  # noqa: PLR0913
    model_inputs: list[dict[str, Any]],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    stage2_prompts: list[str | None],
) -> list[VllmWindowResult]:
    """Caption the videos without inflight batching.

    Assumption:
       len(model_inputs) == len(stage2_prompts)

    Args:
        model_inputs: The model inputs for each video.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        vllm_config: The configuration for the vLLM model.
        stage2_prompts: A list of second-stage prompts to use for the
           captioning. If None, no second-stage captioning will be performed.
           Assumed to be the same length as model_inputs.

    Returns:
        Final per-window results in input order.

    """
    vllm_plugin = _get_vllm_plugin(vllm_config.model_variant)

    requests = [
        VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=model_input,
            stage2_prompt=stage2_prompt,
        )
        for model_input, stage2_prompt in zip(model_inputs, stage2_prompts, strict=True)
    ]

    # Map request_id -> original input index so we can return captions in input order
    request_id_to_index = {r.request_id: i for i, r in enumerate(requests)}

    def _process_requests(requests: list[VllmCaptionRequest]) -> list[VllmCaptionRequest]:
        in_flight_requests: dict[str, VllmCaptionRequest] = {r.request_id: r for r in requests}
        outputs = vllm_generate(llm, sampling_params, requests, vllm_config.batch_size)
        finished_requests = process_vllm_output(outputs, in_flight_requests, vllm_config)

        # Sanity check
        if len(finished_requests) != len(requests):
            msg = f"Expected {len(requests)} finished requests, got {len(finished_requests)}, this is a bug"
            raise RuntimeError(msg)

        return finished_requests

    # stage 1 captioning
    finished_s1 = _process_requests(requests)

    results: dict[int, VllmWindowResult] = {}
    token_counts: dict[int, TokenCounts] = {}
    needs_stage2 = []
    for r in finished_s1:
        idx = request_id_to_index[r.request_id]
        token_counts[idx] = TokenCounts(r.prompt_tokens, r.output_tokens)
        if r.stage2_prompt is None:
            results[idx] = _make_window_result(r, token_counts[idx])
        else:
            needs_stage2.append(r)

    # stage 2 captioning
    refine_requests = [vllm_plugin.make_refined_llm_request(r, processor, r.stage2_prompt) for r in needs_stage2]
    for orig_req, refined_req in zip(needs_stage2, refine_requests, strict=True):
        request_id_to_index[refined_req.request_id] = request_id_to_index[orig_req.request_id]

    for r in _process_requests(refine_requests):
        idx = request_id_to_index[r.request_id]
        prev = token_counts.get(idx, TokenCounts())
        token_counts[idx] = TokenCounts(prev.prompt_tokens + r.prompt_tokens, prev.output_tokens + r.output_tokens)
        results[idx] = _make_window_result(r, token_counts[idx])

    n = len(requests)
    return [results[i] for i in range(n)]


def _caption_inflight_batching(  # noqa: PLR0913
    model_inputs: list[dict[str, Any]],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    max_inflight_requests: int,
    stage2_prompts: list[str | None],
) -> list[VllmWindowResult]:
    """Caption the videos using inflight batching.

    Assumption:
       len(model_inputs) == len(stage2_prompts)

    Args:
        model_inputs: The model inputs for each video.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        vllm_config: The configuration for the VLLM model.
        max_inflight_requests: Maximum number of inflight requests to vLLM
           engine. Set to 0 for unlimited inflight requests.
        stage2_prompts: A list of second-stage prompts to use for the
           captioning. If None, no second-stage captioning will be performed.
           Assumed to be the same length as model_inputs.

    Returns:
        Final per-window results in input order.

    """
    vllm_plugin = _get_vllm_plugin(vllm_config.model_variant)
    request_q: Deque[VllmCaptionRequest] = deque()  # noqa: UP006, remove noqa when python 3.10 support is dropped
    in_flight_requests: dict[str, VllmCaptionRequest] = {}
    results: dict[int, VllmWindowResult] = {}
    token_counts: dict[int, TokenCounts] = {}

    # Map request_id -> original input index so we can return captions in input order
    request_id_to_index: dict[str, int] = {}

    for idx, (model_input, stage2_prompt) in enumerate(zip(model_inputs, stage2_prompts, strict=True)):
        request_id = secrets.token_hex(8)
        request_q.append(
            VllmCaptionRequest(
                request_id=request_id,
                inputs=model_input,
                stage2_prompt=stage2_prompt,
            )
        )
        request_id_to_index[request_id] = idx

    total_requests = len(request_q)
    engine = llm.llm_engine

    while len(results) < total_requests:
        if request_q and (max_inflight_requests == 0 or len(in_flight_requests) < max_inflight_requests):
            request = request_q.popleft()
            # engine.add_request can accept a dictionary, but does not advertise this in its type hints
            engine.add_request(request.request_id, cast("Any", request.inputs), sampling_params)
            in_flight_requests[request.request_id] = request

        engine_output = cast("list[RequestOutput] | list[PoolingRequestOutput[PoolingOutput]]", engine.step())

        # Finished requests are requests that have been completed by the vLLM engine and have a caption.
        # These requests may still need stage2 refinement.
        # engine.step() returns list[RequestOutput | PoolingRequestOutput], but process_vllm_output
        # expects either list[RequestOutput] or list[PoolingRequestOutput] - at runtime, the list
        # will contain only one type based on the engine configuration
        finished = process_vllm_output(engine_output, in_flight_requests, vllm_config)

        for request in finished:
            del in_flight_requests[request.request_id]

        for r in finished:
            original_idx = request_id_to_index[r.request_id]
            # Accumulate token counts (stage-1 + stage-2)
            prev = token_counts.get(original_idx, TokenCounts())
            token_counts[original_idx] = TokenCounts(
                prev.prompt_tokens + r.prompt_tokens, prev.output_tokens + r.output_tokens
            )
            if r.stage2_prompt is None:
                results[original_idx] = _make_window_result(r, token_counts[original_idx])

        needs_stage2 = [r for r in finished if r.stage2_prompt is not None]

        for request in needs_stage2:
            original_idx = request_id_to_index[request.request_id]
            refined_request = vllm_plugin.make_refined_llm_request(request, processor, request.stage2_prompt)
            request_q.append(refined_request)
            # Propagate the original index to the refined request's new request_id
            request_id_to_index[refined_request.request_id] = original_idx

    return [results[i] for i in range(total_requests)]


def vllm_caption(  # noqa: PLR0913
    model_inputs: list[dict[str, Any]],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    max_inflight_requests: int,
    *,
    inflight_batching: bool,
    stage2_prompts: list[str | None] | None = None,
) -> list[VllmWindowResult]:
    """Caption the videos using the vLLM model.

    This is the main entry point for video captioning. It handles:
    1. Creating VllmCaptionRequest objects (each with unique ID)
    2. Batching and generating captions via vLLM
    3. Two-stage captioning if stage2_prompts provided
    4. Returning final caption strings with token counts

    Flow: This function → _caption_[no_]inflight_batching() → vllm_generate()
          → process_vllm_output() → plugin.decode() → captions

    Args:
        model_inputs: The model inputs for each video.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        vllm_config: The configuration for the VLLM model.
        max_inflight_requests: Maximum number of inflight requests to vLLM
           engine. Set to 0 for unlimited inflight requests.
        inflight_batching: Whether to use inflight batching.
        stage2_prompts: A list of second-stage prompts to use for the
           captioning. If None, no second-stage captioning will be performed.
           Must be the same length as model_inputs.

    Returns:
        Final per-window results in input order.

    Raises:
        ValueError: If max_inflight_requests is negative.
        ValueError: If stage2_prompts is not None and not the same length as model_inputs.

    """
    if max_inflight_requests < 0:
        msg = f"{max_inflight_requests=} must be >= 0"
        raise ValueError(msg)

    stage2_prompts = _resolve_stage2_prompts(stage2_prompts, len(model_inputs))

    if inflight_batching:
        return _caption_inflight_batching(
            model_inputs, llm, processor, sampling_params, vllm_config, max_inflight_requests, stage2_prompts
        )

    return _caption_no_inflight_batching(
        model_inputs,
        llm,
        processor,
        sampling_params,
        vllm_config,
        stage2_prompts,
    )


def _resolve_stage2_prompts(
    stage2_prompts: list[str | None] | None,
    n: int,
) -> list[str | None]:
    """Return per-input stage-2 prompts, padding with ``None`` when omitted."""
    if stage2_prompts is None:
        return [None] * n
    if len(stage2_prompts) != n:
        msg = f"{len(stage2_prompts)=} != {n=}, must be same length"
        raise ValueError(msg)
    return stage2_prompts


def _fresh_prompt_payload(inputs: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the outer mutable shell of a vLLM prompt payload.

    vLLM/HF/Transformers may mutate the outer dict (e.g. resolve relative
    paths, attach derived fields).  Across retries or stage-2 refinement the
    same ``inputs`` dict can be reused, so we rebuild the outer dict and any
    ``multi_modal_data`` lists here while preserving the underlying tensor
    references (zero-copy by reference) so the payload itself remains
    immutable from the caller's perspective.
    """
    fresh: dict[str, Any] = dict(inputs)
    mm_data = fresh.get("multi_modal_data")
    if isinstance(mm_data, dict):
        fresh["multi_modal_data"] = {
            key: (list(value) if isinstance(value, list) else value) for key, value in mm_data.items()
        }
    return fresh


@attrs.define
class _AsyncCaptioner:
    """Concurrent captioning runner for an in-process ``AsyncLLM`` engine.

    Mirrors sync's ``_caption_inflight_batching`` semantics: ordered
    results, accumulated stage-1 + stage-2 token counts, two-stage
    refinement loop, and ``EngineDeadError`` propagation.

    Concurrency is gated by a caller-owned ``asyncio.Semaphore``.
    The semaphore is held only while ``engine.generate`` iterates,
    not during retry backoff or result decoding, so a slow request
    (or one in tenacity backoff) never starves healthy siblings.
    """

    engine: "AsyncLLM"
    processor: AutoProcessor
    plugin: VllmPlugin
    semaphore: asyncio.Semaphore
    sampling_params: SamplingParams
    max_retries: int
    request_id_factory: Callable[[], str]
    on_window_done: Callable[[int, VllmWindowResult], None] | None = None
    on_window_error: Callable[[int, str, Exception], None] | None = None
    _request_id_to_index: dict[str, int] = attrs.field(factory=dict, init=False)
    _results: dict[int, VllmWindowResult] = attrs.field(factory=dict, init=False)
    _token_counts: dict[int, TokenCounts] = attrs.field(factory=dict, init=False)
    _pending: dict[asyncio.Task[VllmCaptionRequest], VllmCaptionRequest] = attrs.field(factory=dict, init=False)
    # Tracks which dispatch generation each in-flight task belongs to so
    # ``on_window_error`` reports an accurate phase tag.
    _phase: dict[asyncio.Task[VllmCaptionRequest], str] = attrs.field(factory=dict, init=False)

    async def run(self, requests: Iterable[VllmCaptionRequest]) -> list[VllmWindowResult]:
        """Dispatch every request concurrently and return results in input order.

        Raises:
            EngineDeadError: Re-raised so the caller can restart the actor.

        """
        # Track the total count locally so the final result list can be
        # ordered without re-materializing the iterable.  ``idx`` is bound
        # only when ``requests`` is non-empty; ``count`` stays at ``0``
        # otherwise, which short-circuits the trailing range correctly.
        count = 0
        try:
            for idx, request in enumerate(requests):
                self._request_id_to_index[request.request_id] = idx
                if not request.inputs:
                    # Per-slot sentinel: an empty ``inputs`` dict means the
                    # request builder (e.g. ``VllmAsyncCaptionStage._iter_requests``)
                    # detected a missing model input upstream and already
                    # logged the cause.  Skip ``engine.generate`` and emit
                    # ``VLLM_UNKNOWN_CAPTION`` so siblings continue running.
                    self._emit_unknown(idx)
                else:
                    self._spawn(request)
                count = idx + 1
            while self._pending:
                done, _ = await asyncio.wait(self._pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    self._handle_completed(task)
        finally:
            await self._cancel_pending()
        return [self._results[i] for i in range(count)]

    def _spawn(self, request: VllmCaptionRequest, *, phase: str = "stage1") -> None:
        """Create the asyncio.Task driving ``request`` and register it as pending.

        Args:
            request: The captioning request to dispatch.
            phase: Dispatch generation tag (``"stage1"`` for initial
                requests, ``"stage2"`` for refined requests).  Recorded
                on the task so ``on_window_error`` can attribute
                failures accurately.

        """
        task = asyncio.create_task(self._drive_request(request))
        self._pending[task] = request
        self._phase[task] = phase

    def _handle_completed(self, task: asyncio.Task[VllmCaptionRequest]) -> None:
        """Route one completed task to stage-2 dispatch, final emit, or sentinel on failure."""
        req = self._pending.pop(task)
        phase = self._phase.pop(task)
        # task.result() re-raises EngineDeadError so the caller restarts the actor.
        try:
            req = task.result()
        except EngineDeadError:
            raise
        except Exception as exc:  # noqa: BLE001 - per-window containment
            idx = self._request_id_to_index[req.request_id]
            self._emit_unknown(idx)
            if self.on_window_error is not None:
                self.on_window_error(idx, phase, exc)
            req.inputs = None  # type: ignore[assignment]
            return

        idx = self._request_id_to_index[req.request_id]
        # Accumulate stage-1 + stage-2 tokens, matching sync.
        prev = self._token_counts.get(idx, TokenCounts())
        self._token_counts[idx] = TokenCounts(
            prev.prompt_tokens + req.prompt_tokens,
            prev.output_tokens + req.output_tokens,
        )

        if req.stage2_prompt is None:
            self._emit_final(idx, req)
            req.inputs = None  # type: ignore[assignment]
            return

        # Build refined request for stage-2 refinement.
        try:
            refined = self.plugin.make_refined_llm_request(req, self.processor, req.stage2_prompt)
        except Exception as exc:  # noqa: BLE001 - per-window containment
            self._emit_unknown(idx)
            if self.on_window_error is not None:
                self.on_window_error(idx, "stage2_build", exc)
            # stage-2 build failed, no further reads.
            req.inputs = None  # type: ignore[assignment]
            return

        self._request_id_to_index[refined.request_id] = idx
        self._spawn(refined, phase="stage2")
        req.inputs = None  # type: ignore[assignment]

    async def _drive_request(self, request: VllmCaptionRequest) -> VllmCaptionRequest:
        """Drive one request through ``engine.generate`` and populate output fields.

        Returns the same ``VllmCaptionRequest`` with ``caption``,
        ``prompt_tokens``, ``output_tokens`` and ``finish_reason``
        populated.  Raises :class:`EngineDeadError` unchanged (never
        retried).  Other exceptions are retried up to ``max_retries``
        attempts (mirroring sync's ``stop_after_attempt`` shape) and
        then re-raised.
        """

        @tenacity.retry(  # type: ignore[misc]
            stop=tenacity.stop_after_attempt(self.max_retries),
            reraise=True,
            retry=tenacity.retry_if_not_exception_type(EngineDeadError),
        )
        async def _attempt() -> RequestOutput:
            # vLLM rejects reusing a request_id whose state is still pending in
            # the engine if the previous attempt failed mid-stream.
            attempt_id = self.request_id_factory()
            payload = _fresh_prompt_payload(request.inputs)
            async with self.semaphore:
                with traced_span(
                    "VllmAsyncCaptionStage.generate",
                    attributes={"vllm.request_id": attempt_id},
                ) as gen_span:
                    final_output: RequestOutput | None = None
                    async for output in self.engine.generate(
                        prompt=cast("Any", payload),
                        sampling_params=self.sampling_params,
                        request_id=attempt_id,
                    ):
                        final_output = output
                    if final_output is None or not final_output.outputs:
                        msg = f"AsyncLLM engine returned no outputs for request_id={attempt_id!r}"
                        raise RuntimeError(msg)

                    gen_out0 = final_output.outputs[0]
                    gen_span.set_attributes(
                        {
                            "vllm.prompt_tokens": (
                                len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0
                            ),
                            "vllm.output_tokens": (len(gen_out0.token_ids) if gen_out0.token_ids else 0),
                            "vllm.finish_reason": gen_out0.finish_reason or "",
                        },
                    )
                    return final_output

        final_output = await _attempt()
        out0 = final_output.outputs[0]
        request.caption = self.plugin.decode(final_output)
        request.prompt_tokens = len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0
        request.output_tokens = len(out0.token_ids) if out0.token_ids else 0
        request.finish_reason = out0.finish_reason
        return request

    def _emit_final(self, idx: int, request: VllmCaptionRequest) -> None:
        """Record the terminal result for one window and fire the per-window callback."""
        result = _make_window_result(request, self._token_counts[idx])
        self._results[idx] = result
        if self.on_window_done is not None:
            self.on_window_done(idx, result)

    def _emit_unknown(self, idx: int) -> None:
        """Record a sentinel ``VLLM_UNKNOWN_CAPTION`` result for a failed window."""
        tc = self._token_counts.get(idx, TokenCounts())
        result = VllmWindowResult(text=VLLM_UNKNOWN_CAPTION, finish_reason=None, token_counts=tc)
        self._results[idx] = result
        if self.on_window_done is not None:
            self.on_window_done(idx, result)

    async def _cancel_pending(self) -> None:
        """Cancel any in-flight tasks and consume their exceptions on teardown.

        Called from :meth:`run` ``finally`` so that on early exit
        (cancellation, engine death, etc.) every outstanding task is
        cancelled and its exception is consumed.  Without this drain
        the event loop logs "Task exception was never retrieved" at GC.
        """
        if not self._pending:
            return
        for task in self._pending:
            task.cancel()
        for task in self._pending:
            # Drain-only: we already cancelled, and we just need to consume
            # the exception so asyncio doesn't log it at GC.  Both regular
            # exceptions and CancelledError are expected and discarded.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task

        self._pending.clear()
        self._phase.clear()


async def vllm_caption_async(  # noqa: PLR0913
    requests: Iterable[VllmCaptionRequest],
    engine: "AsyncLLM",
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    *,
    semaphore: asyncio.Semaphore,
    max_retries: int = 1,
    request_id_factory: Callable[[], str] = lambda: secrets.token_hex(8),
    on_window_done: Callable[[int, VllmWindowResult], None] | None = None,
    on_window_error: Callable[[int, str, Exception], None] | None = None,
) -> list[VllmWindowResult]:
    """Caption an iterable of pre-built ``VllmCaptionRequest`` objects via ``AsyncLLM``.

    Async sibling of :func:`vllm_caption`.  Thin module-level entry
    point: resolves the plugin and delegates orchestration (concurrent
    ``engine.generate``, stage-2 refinement, retry, per-window
    callbacks) to :class:`_AsyncCaptioner`.

    Args:
        requests: Iterable of pre-built per-window caption requests
            (each carrying its own LLM input dict and optional stage-2
            prompt).  A request whose ``inputs`` is an empty mapping
            ``{}`` is treated as the per-slot "missing model input"
            sentinel defined on :class:`VllmCaptionRequest`: the slot
            is short-circuited to ``VLLM_UNKNOWN_CAPTION`` without
            invoking ``engine.generate`` and without firing
            ``on_window_error`` (the caller is expected to have already
            logged the underlying cause when constructing the
            sentinel).
        engine: In-process ``AsyncLLM`` engine owned by the caller stage.
        processor: HuggingFace processor used for stage-2 refinement.
        sampling_params: Shared sampling configuration applied to every
            request.
        vllm_config: ``VllmConfig`` describing the model variant and
            stage-2 settings.
        semaphore: Concurrency budget shared across all in-flight
            invocations on the same actor.
        max_retries: Per-request retry bound for transient
            ``engine.generate`` failures.  ``EngineDeadError`` is never
            retried.
        request_id_factory: Builds the request_id for each engine call.
            Callers that need monotonic IDs inject a counter-backed factory.
        on_window_done: Optional callback fired as each window's final
            caption is known, enabling per-window scatter and cleanup.
        on_window_error: Optional callback fired when a request fails
            before it can be represented as a normal result.

    Returns:
        Final per-window results in the order in which ``requests``
        yielded its elements.

    Raises:
        EngineDeadError: Re-raised so the caller can restart the actor.

    """
    plugin = _get_vllm_plugin(vllm_config.model_variant)
    captioner = _AsyncCaptioner(
        engine=engine,
        processor=processor,
        plugin=plugin,
        semaphore=semaphore,
        sampling_params=sampling_params,
        max_retries=max_retries,
        request_id_factory=request_id_factory,
        on_window_done=on_window_done,
        on_window_error=on_window_error,
    )
    return await captioner.run(requests)
