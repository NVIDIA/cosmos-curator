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

"""CLI argument utilities for vLLM async captioning."""

import argparse
from typing import Any

import attrs

from cosmos_curator.pipelines.common.model_constraints import PreprocessMode, resolve_preprocess_mode
from cosmos_curator.pipelines.video.utils.data_model import VllmAsyncConfig, VllmSamplingConfig

# Matches cosmos_curator.pipelines.video.utils.vision_process.FPS.
_DEFAULT_VIDEO_SAMPLE_FPS: float = 2.0


@attrs.define(frozen=True)
class VllmAsyncPrepConfig:
    """Configuration for ``VllmAsyncPrepStage`` (CPU-only windowing, decode, and prompt build)."""

    model_variant: str
    sampling_config: VllmSamplingConfig = attrs.Factory(VllmSamplingConfig)
    prompt_variant: str = "default"
    prompt_text: str | None = None
    sample_fps: float = _DEFAULT_VIDEO_SAMPLE_FPS
    window_size: int = 256
    remainder_threshold: int = 128
    keep_mp4: bool = False
    use_input_bit_rate: bool = False
    decode_workers: int = 0


@attrs.define(frozen=True)
class _VllmArgSpec:
    """Declarative spec mapping a ``VllmAsyncConfig`` field to a ``--vllm-async-*`` CLI argument."""

    field: str
    help: str
    arg_type: type | None = None
    choices: tuple[str, ...] | None = None


_VLLM_ARG_SPECS: tuple[_VllmArgSpec, ...] = (
    _VllmArgSpec(
        field="num_gpus",
        # vLLM's ``tensor_parallel_size`` is integer-only; mirror sync's
        # ``--qwen-num-gpus-per-worker`` (also ``int``) instead of silently
        # truncating fractional CLI inputs in ``plugin.model_async()``.
        arg_type=int,
        help="Number of GPUs per engine replica for tensor parallelism (maps to --tensor-parallel-size). Default: 1.",
    ),
    _VllmArgSpec(
        field="data_parallel_size",
        arg_type=int,
        help=(
            "Number of data-parallel engine replicas inside the vllm async engine process. "
            "Each replica owns --vllm-async-num-gpus GPUs (tensor parallelism); "
            "total GPU usage is num_gpus * data_parallel_size. Default: 1. "
            "See https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/ ."
        ),
    ),
    _VllmArgSpec(
        field="fp8",
        help=(
            "Request FP8 quantization for the model weights and KV cache. "
            "Honored by plugins whose model_async() maps fp8=True to "
            "vLLM's quantization='fp8' (e.g. Qwen2.x, Cosmos R1). "
            "Ignored by plugins that omit quantization on the async path "
            "(e.g. Qwen3-VL, Nemotron) - see each plugin's model_async() "
            "docstring. Mirrors sync's --vllm-fp8 flag. Default: False."
        ),
    ),
    _VllmArgSpec(
        field="disable_mmcache",
        help=(
            "Disable the multimodal processor cache (sets --mm-processor-cache-gb=0.0). "
            "Mirrors sync's --vllm-disable-mmcache flag. Default: False (4 GiB cache)."
        ),
    ),
    _VllmArgSpec(
        field="max_num_seqs",
        arg_type=int,
        help=(
            "Maximum concurrent sequences for vllm async engine (maps to --max-num-seqs). "
            "0 = auto-detect (vLLM chooses based on model and hardware). Default: 0."
        ),
    ),
    _VllmArgSpec(
        field="enforce_eager",
        help="Disable CUDA graphs (maps to --enforce-eager). Useful for debugging. Default: False.",
    ),
    _VllmArgSpec(
        field="mm_encoder_tp_mode",
        arg_type=str,
        help=(
            "Multimodal encoder tensor parallelism mode (maps to --mm-encoder-tp-mode). "
            "'data' = batch-level DP for vision encoder, ~10-40%% throughput gain at TP>1. "
            "Empty string = use vLLM default. Default: data."
        ),
    ),
    _VllmArgSpec(
        field="kv_cache_dtype",
        arg_type=str,
        help=(
            "KV cache precision for vllm async engine (maps to --kv-cache-dtype). "
            "Default: auto. Values not pre-validated - vLLM rejects unknown values at engine init."
        ),
    ),
    _VllmArgSpec(
        field="mm_processor_cache_type",
        arg_type=str,
        help=(
            "Multimodal processor cache type (maps to --mm-processor-cache-type). "
            "e.g. 'shm' for TP>1. Empty = vLLM default. Default: empty string."
        ),
    ),
    _VllmArgSpec(
        field="disable_log_stats",
        help=(
            "Suppress vLLM's internal per-iteration stat logging "
            "(Prometheus counters, throughput metrics). Avoids overhead in batch "
            "pipelines where no metrics scraper is running. Default: True."
        ),
    ),
    _VllmArgSpec(
        field="async_scheduling",
        help=(
            "Enable async scheduling (double-buffering). Eliminates idle gaps between steps. "
            "Only supported with mp/uni executor, NOT ray. "
            "Default: None (auto-detect - vLLM auto-enables with mp backend). "
            "Use --vllm-async-async-scheduling to force enable, "
            "--no-vllm-async-async-scheduling to force disable."
        ),
    ),
    _VllmArgSpec(
        field="enable_chunked_prefill",
        help=(
            "Enable chunked prefill (split long prefills across scheduling steps). "
            "Default: None (auto-detect - vLLM enables for supported models like Qwen2-VL). "
            "Use --vllm-async-enable-chunked-prefill to force enable, "
            "--no-vllm-async-enable-chunked-prefill to force disable."
        ),
    ),
    _VllmArgSpec(
        field="disable_chunked_mm_input",
        help=(
            "Prevent splitting multimodal input tokens across chunked-prefill steps. "
            "Requires max_num_batched_tokens >= max_tokens_per_mm_item (128K for "
            "Qwen video). Default: False (allow chunking)."
        ),
    ),
    _VllmArgSpec(
        field="long_prefill_token_threshold",
        arg_type=int,
        help=(
            "Maximum prefill tokens a single request can consume per scheduling step. "
            "Caps long-prompt prefills so decode tokens share the budget. "
            "0 = disabled (no clamping). >0 = explicit threshold. Default: 0."
        ),
    ),
    _VllmArgSpec(
        field="stream_interval",
        arg_type=int,
        help=(
            "Number of tokens to buffer in EngineCore before sending intermediate "
            "results via ZMQ. With output_kind=FINAL_ONLY (batch captioning), a high "
            "value suppresses wasteful intermediate IPC. Default: 9999."
        ),
    ),
    _VllmArgSpec(
        field="distributed_executor_backend",
        arg_type=str,
        choices=("ray", "mp", "uni"),
        help=(
            "Backend for distributed model workers (maps to --distributed-executor-backend). "
            "'ray' is recommended when running inside a Ray cluster (proper GPU isolation). "
            "'mp' uses Python multiprocessing. Default: ray."
        ),
    ),
    _VllmArgSpec(
        field="skip_mm_profiling",
        help=(
            "Skip multimodal encoder profiling during engine init. "
            "Eliminates 30-60s startup profiling delay. vLLM uses a conservative "
            "memory estimate instead of measured GPU memory for the encoder cache. "
            "Negligible impact for video captioning. Default: True."
        ),
    ),
    _VllmArgSpec(
        field="extra_env_vars",
        arg_type=str,
        help=(
            "JSON dict of extra environment variables to set before vLLM engine init. "
            "Keys and values must be strings. These propagate to forked EngineCore "
            "subprocesses. Useful for troubleshooting "
            '(e.g. \'{"CUDA_LAUNCH_BLOCKING": "1", "NCCL_DEBUG": "TRACE"}\'). '
            "See https://docs.vllm.ai/en/latest/usage/troubleshooting/ for common vars. "
            "Empty string = no extra vars."
        ),
    ),
    _VllmArgSpec(
        field="preprocess_mode",
        arg_type=str,
        choices=tuple(mode.value for mode in PreprocessMode),
        help=(
            "Owner of visual resize/rescale/normalize before vLLM inference. "
            "'curator' (default) makes the CPU prep stage emit normalized float16 frames "
            "and tells vLLM to skip those processor steps. 'model' hands resized uint8 "
            "frames to vLLM/HF and enables its multimodal processor."
        ),
    ),
    _VllmArgSpec(
        field="max_retries",
        arg_type=int,
        help=(
            "Per-request retry budget for transient engine.generate() failures. "
            "EngineDeadError is never retried; "
            "it propagates so Xenna can restart the actor. Default: 3 (min: 1)."
        ),
    ),
    _VllmArgSpec(
        field="gpu_memory_utilization",
        arg_type=float,
        help=(
            "Fraction of GPU memory the vllm async engine may use (0.0 < value <= 1.0). "
            "Unset = plugin per-variant default. "
            "Default: unset."
        ),
    ),
)

# Validate at import time that every _VllmArgSpec.field corresponds to an
# actual VllmAsyncConfig attrs field.  Catches typos immediately rather
# than at pipeline-launch time when build_vllm_async_config() is called.
_VALID_FIELDS: frozenset[str] = frozenset(f.name for f in attrs.fields(VllmAsyncConfig))
for _spec in _VLLM_ARG_SPECS:
    if _spec.field not in _VALID_FIELDS:
        msg = (
            f"_VllmArgSpec references unknown VllmAsyncConfig field {_spec.field!r}. "
            f"Valid fields: {sorted(_VALID_FIELDS)}"
        )
        raise AttributeError(msg)


def _normalize_prefix(prefix: str) -> tuple[str, str]:
    """Return ``(flag_prefix, dest_prefix)`` for ``add_vllm_async_cli_args``.

    ``flag_prefix`` is appended to ``--`` (e.g. ``"event-caption-"``);
    ``dest_prefix`` is the matching argparse dest prefix (``event_caption_``).
    Empty input yields empty outputs so default callers stay byte-identical.
    """
    if not prefix:
        return "", ""
    flag_prefix = prefix if prefix.endswith("-") else f"{prefix}-"
    dest_prefix = flag_prefix.replace("-", "_")
    return flag_prefix, dest_prefix


def build_vllm_async_config(
    args: argparse.Namespace,
    sampling_config: VllmSamplingConfig,
    *,
    prefix: str = "",
) -> VllmAsyncConfig | None:
    """Build ``VllmAsyncConfig`` from CLI args when ``caption_algo="vllm_async"``.

    Per-variant numeric tuning is NOT injected here - it lives entirely on
    the plugin and is read inside ``plugin.model_async()``.  This builder
    only collects user knobs; "CLI value or attrs default" is the full merge.

    When called without a ``prefix`` the per-window splitting-pipeline call
    site is byte-identical with the previous signature.  ``prefix="event-caption-"``
    looks up ``args.event_caption_vllm_async_*`` for the per-event captioner
    instead; in that case the ``captioning_algorithm`` guard is bypassed
    because the per-event backend is selected via its own
    ``--event-caption-backend``.
    """
    _, dest_prefix = _normalize_prefix(prefix)
    if not dest_prefix:
        if not getattr(args, "generate_captions", True):
            return None
        if args.captioning_algorithm.lower() != "vllm_async":
            return None

    model_variant = getattr(args, f"{dest_prefix}vllm_async_model_name")
    kwargs: dict[str, Any] = {"model_variant": model_variant}
    for spec in _VLLM_ARG_SPECS:
        arg_name = f"{dest_prefix}vllm_async_{spec.field}"
        cli_val = getattr(args, arg_name, None)
        if cli_val is None:
            continue
        kwargs[spec.field] = cli_val

    kwargs["preprocess_mode"] = resolve_preprocess_mode(
        model_variant,
        kwargs.get("preprocess_mode", PreprocessMode.CURATOR),
    )

    # enable_log_requests is wired from --verbose, not a --vllm-async-* arg
    kwargs["enable_log_requests"] = getattr(args, "verbose", False)
    kwargs["sampling_config"] = sampling_config

    return VllmAsyncConfig(**kwargs)


def add_vllm_async_cli_args(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    """Register all ``--vllm-async-*`` CLI arguments on *parser*.

    ``prefix`` (e.g. ``"event-caption-"``) lets a second consumer (the
    per-event captioner) register a parallel set of flags
    ``--event-caption-vllm-async-*`` without colliding with the
    per-window registration. Empty prefix preserves the existing call
    site byte-for-byte.
    """
    flag_prefix, _ = _normalize_prefix(prefix)
    parser.add_argument(
        f"--{flag_prefix}vllm-async-model-name",
        type=str,
        default="qwen",
        help=(
            "Model name for vllm async engine. Can be a variant key from the vLLM model registry "
            "(e.g. 'qwen', 'nemotron', 'cosmos_r1'). Used when --captioning-algorithm is 'vllm_async'."
        ),
    )
    parser.add_argument(
        f"--{flag_prefix}vllm-async-stage-batch-size",
        type=int,
        default=0,
        help=(
            "Number of tasks to process per stage for vllm_async captioning. "
            "0 = auto-derive as max(2, total_gpus) to keep GPU pipeline fed. "
            "Assumes ~20 windows per task. See stage_batch_size property for sizing rationale."
        ),
    )
    parser.add_argument(
        f"--{flag_prefix}vllm-async-max-concurrent-requests",
        type=int,
        default=0,
        help=(
            "Maximum concurrent async generate requests the VllmAsyncCaptionStage "
            "submits to the in-process AsyncLLM engine. Higher values let the engine "
            "batch more requests in a single GPU forward pass. "
            "0 = auto-derive as 256 (N-actors) or 256 * total_gpus (DP). "
            "Explicit positive value overrides auto-derivation."
        ),
    )
    parser.add_argument(
        f"--{flag_prefix}vllm-async-num-workers-per-node",
        type=int,
        default=0,
        help=(
            "Number of vLLM captioning workers per node in N-actors mode (dp <= 1). "
            "0 = Xenna autoscale (default): dynamically allocates workers based on "
            "available GPUs. Positive = explicit fixed count (e.g. 7 for 8xH100 "
            "with 1 GPU shared by other stages). "
            "Ignored in DP mode (always 1 worker per node)."
        ),
    )
    parser.add_argument(
        f"--{flag_prefix}vllm-async-stage2-caption",
        action="store_true",
        default=False,
        help=(
            "Enable stage-2 caption refinement for vllm_async captioning. "
            "The stage-1 caption is fed back with a refinement prompt for "
            "a second inference pass, producing a more detailed description."
        ),
    )
    parser.add_argument(
        f"--{flag_prefix}vllm-async-stage2-prompt-text",
        type=str,
        default=None,
        help=(
            "Custom prompt text for stage-2 refinement. If not set, the default "
            "refinement prompt is used. Only effective when --vllm-async-stage2-caption is set."
        ),
    )

    # all VllmAsyncConfig-mapped args from specs
    for spec in _VLLM_ARG_SPECS:
        flag = f"--{flag_prefix}vllm-async-{spec.field.replace('_', '-')}"
        kwargs: dict[str, Any] = {"default": None, "help": spec.help}
        if spec.arg_type is None:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = spec.arg_type
        if spec.choices is not None:
            kwargs["choices"] = list(spec.choices)
        parser.add_argument(flag, **kwargs)
