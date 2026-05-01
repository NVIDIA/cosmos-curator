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

"""Configuration classes and CLI argument utilities for vLLM async captioning."""

import argparse
import json
from typing import Any, Literal

import attrs

from cosmos_curator.pipelines.video.utils.data_model import VllmSamplingConfig
from cosmos_curator.pipelines.video.utils.vision_process import VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS

# Mirrors vllm.config.model.ModelDType without a heavyweight vllm import.
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]

# Matches cosmos_curator.pipelines.video.utils.vision_process.FPS.
_DEFAULT_VIDEO_SAMPLE_FPS: float = 2.0

# Model-specific overrides are applied only by build_vllm_async_config()
# (CLI > _MODEL_DEFAULTS > attrs defaults). Direct VllmAsyncConfig(...)
# construction does not inject these values automatically.
_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    # Qwen-specific overrides
    "qwen": {
        "max_model_len": 32768,
        "max_num_batched_tokens": 32768,
        # Qwen-family-compatible nested video kwargs; currently applied only to "qwen".
        "mm_processor_kwargs": json.dumps(
            {
                "videos_kwargs": {
                    "size": {
                        "shortest_edge": VIDEO_MIN_PIXELS,
                        "longest_edge": VIDEO_MAX_PIXELS,
                    }
                }
            }
        ),
    },
}


@attrs.define(frozen=True)
class VllmAsyncConfig:
    """Configuration for the in-process ``AsyncLLM`` engine."""

    model_variant: str
    num_gpus: float = 1.0
    data_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 0
    dtype: ModelDType = "auto"
    quantization: str | None = None
    max_num_batched_tokens: int = 0
    max_num_seqs: int = 0
    enforce_eager: bool = False
    cudagraph_mode: str = "piecewise"
    limit_mm_per_prompt: str = json.dumps({"image": 0, "video": 1})
    mm_encoder_tp_mode: str = "data"
    kv_cache_dtype: str = "auto"
    mm_processor_cache_gb: float = 4.0
    mm_processor_cache_type: str = ""
    trust_remote_code: bool = True
    disable_log_stats: bool = True
    enable_log_requests: bool = False
    sampling_config: VllmSamplingConfig = attrs.Factory(VllmSamplingConfig)
    async_scheduling: bool | None = None
    enable_chunked_prefill: bool | None = None
    disable_chunked_mm_input: bool = False
    long_prefill_token_threshold: int = 0
    stream_interval: int = 9999
    distributed_executor_backend: str = "ray"
    skip_mm_profiling: bool = True
    mm_processor_kwargs: str = json.dumps({"max_pixels": VIDEO_MAX_PIXELS})
    extra_env_vars: str = ""

    @property
    def total_gpus(self) -> float:
        """Total GPU footprint: ``num_gpus * data_parallel_size``."""
        return self.num_gpus * self.data_parallel_size

    @staticmethod
    def _validate_json_field(value: str, field_name: str) -> None:
        """Validate that a string field contains valid JSON (when non-empty)."""
        if not value:
            return
        try:
            json.loads(value)
        except json.JSONDecodeError as exc:
            msg = f"{field_name} must be valid JSON, got {value!r}"
            raise ValueError(msg) from exc

    @staticmethod
    def _validate_extra_env_vars(value: str) -> None:
        """Validate ``extra_env_vars`` is a JSON ``dict[str, str]`` (when non-empty)."""
        if not value:
            return
        VllmAsyncConfig._validate_json_field(value, "extra_env_vars")
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            msg = f"extra_env_vars must be a JSON object (dict), got {type(parsed).__name__}"
            raise TypeError(msg)
        for k, v in parsed.items():
            if not isinstance(v, str):
                msg = f"extra_env_vars values must be strings, got key={k!r}, value={v!r} ({type(v).__name__})"
                raise TypeError(msg)

    def __attrs_post_init__(self) -> None:
        """Validate cross-field constraints at construction time."""
        _ASYNC_SCHED_BACKENDS = frozenset({"mp", "uni"})
        if self.async_scheduling is True and self.distributed_executor_backend not in _ASYNC_SCHED_BACKENDS:
            msg = (
                f"async_scheduling=True requires distributed_executor_backend "
                f"in {sorted(_ASYNC_SCHED_BACKENDS)}, got "
                f"{self.distributed_executor_backend!r}. The 'ray' backend does "
                f"not support async scheduling in vLLM v0.17. Either set "
                f"async_scheduling=False or change the backend."
            )
            raise ValueError(msg)

        if self.num_gpus < 1.0:
            msg = f"num_gpus must be >= 1.0, got {self.num_gpus}"
            raise ValueError(msg)

        self._validate_json_field(self.limit_mm_per_prompt, "limit_mm_per_prompt")
        self._validate_json_field(self.mm_processor_kwargs, "mm_processor_kwargs")
        self._validate_extra_env_vars(self.extra_env_vars)


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


# Declarative specifications for all --vllm-async-* CLI arguments that map
# 1:1 to VllmAsyncConfig fields.  Drives both add_vllm_async_cli_args()
# (argparse registration) and build_vllm_async_config() (CLI > model-defaults
# > attrs-defaults merge), eliminating duplication between the argument
# definitions and the config-building loop.
#
#     _VLLM_ARG_SPECS
#         |
#         +---> add_vllm_async_cli_args()   (register --vllm-async-* flags)
#         |
#         +---> build_vllm_async_config()   (merge CLI > model > attrs defaults)
#
_VLLM_ARG_SPECS: tuple[_VllmArgSpec, ...] = (
    _VllmArgSpec(
        field="num_gpus",
        arg_type=float,
        help="Number of GPUs per engine replica for tensor parallelism (maps to --tensor-parallel-size). Default: 1.0.",
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
        field="gpu_memory_utilization",
        arg_type=float,
        help="Fraction of GPU memory vllm async engine may use (0.0 to 1.0). Default: 0.85.",
    ),
    _VllmArgSpec(
        field="max_model_len",
        arg_type=int,
        help="Maximum sequence length for vllm async engine. 0 means auto-detect from model config. Default: 0.",
    ),
    _VllmArgSpec(
        field="dtype",
        arg_type=str,
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Weight/activation precision for vllm async engine (maps to --dtype). Default: auto.",
    ),
    _VllmArgSpec(
        field="quantization",
        arg_type=str,
        help="Quantization method for vllm async engine (e.g. awq, gptq, fp8). Default: None (auto-detect).",
    ),
    _VllmArgSpec(
        field="max_num_batched_tokens",
        arg_type=int,
        help=(
            "Chunked prefill budget for vllm async engine (maps to --max-num-batched-tokens). "
            "Also controls encoder cache budget and persistent CUDA buffer sizes. "
            "Default: 0 (vLLM engine default). Explicit positive values override."
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
        field="cudagraph_mode",
        arg_type=str,
        help=(
            "CUDA graph compilation mode for vllm async engine (emitted via -O flag). "
            "Default: 'piecewise' (matches all in-process vLLM plugins). "
            "Set to empty string '' to disable (use vLLM built-in default)."
        ),
    ),
    _VllmArgSpec(
        field="limit_mm_per_prompt",
        arg_type=str,
        help=(
            "Per-modality item count limits as JSON (maps to --limit-mm-per-prompt). "
            'Default: \'{"image": 0, "video": 1}\'. '
            "The simple count format lets vLLM auto-size the encoder cache. "
            "Resolution-constrained format is also accepted but may cause undersized "
            "encoder caches if declared dimensions exceed model intrinsic limits."
        ),
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
        help="KV cache precision for vllm async engine (maps to --kv-cache-dtype). Default: auto.",
    ),
    _VllmArgSpec(
        field="mm_processor_cache_gb",
        arg_type=float,
        help="Multimodal processor cache size in GiB (maps to --mm-processor-cache-gb). 0 = disable. Default: 4.0.",
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
        field="trust_remote_code",
        help="Allow custom model code (maps to --trust-remote-code). Default: True.",
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
        field="mm_processor_kwargs",
        arg_type=str,
        help=(
            "JSON dict of multimodal processor kwargs (maps to --mm-processor-kwargs). "
            f"Default: '{{\"max_pixels\": {VIDEO_MAX_PIXELS}}}' (= VIDEO_MAX_PIXELS). "
            "Some model variants override this - see _MODEL_DEFAULTS. "
            "Empty string disables any model-specific override and falls back to the field default."
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


def build_vllm_async_config(
    args: argparse.Namespace,
    sampling_config: VllmSamplingConfig,
) -> VllmAsyncConfig | None:
    """Build ``VllmAsyncConfig`` from CLI args when ``caption_algo="vllm_async"``."""
    if not getattr(args, "generate_captions", True):
        return None
    if args.captioning_algorithm.lower() != "vllm_async":
        return None

    model_variant = args.vllm_async_model_name
    model_overrides = _MODEL_DEFAULTS.get(model_variant, {})

    kwargs: dict[str, Any] = {"model_variant": model_variant}
    for spec in _VLLM_ARG_SPECS:
        arg_name = f"vllm_async_{spec.field}"
        cli_val = getattr(args, arg_name, None)
        # Empty strings on string-typed args mean "explicitly unset":
        # skip model overrides and let the attrs field default apply.
        # Example: --vllm-async-quantization="" overrides qwen's
        # default "fp8" with the attrs default None (no quantization).
        # For mm_processor_kwargs, the attrs field default is the
        # legacy flat {"max_pixels": VIDEO_MAX_PIXELS} value.
        if cli_val is not None and not (spec.arg_type is str and cli_val == ""):
            kwargs[spec.field] = cli_val
        elif cli_val is None and spec.field in model_overrides:
            kwargs[spec.field] = model_overrides[spec.field]
        # Otherwise attrs field default applies (empty string or absent).

    # enable_log_requests is wired from --verbose, not a --vllm-async-* arg
    kwargs["enable_log_requests"] = getattr(args, "verbose", False)
    kwargs["sampling_config"] = sampling_config

    return VllmAsyncConfig(**kwargs)


def add_vllm_async_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register all ``--vllm-async-*`` CLI arguments on *parser*."""
    parser.add_argument(
        "--vllm-async-model-name",
        type=str,
        default="qwen",
        help=(
            "Model name for vllm async engine. Can be a variant key from the vLLM model registry "
            "(e.g. 'qwen', 'nemotron', 'cosmos_r1') or a direct HuggingFace model ID "
            "(e.g. 'Qwen/Qwen2.5-VL-7B-Instruct'). Used when --captioning-algorithm is 'vllm_async'."
        ),
    )
    parser.add_argument(
        "--vllm-async-stage-batch-size",
        type=int,
        default=0,
        help=(
            "Number of tasks to process per process_data() call for vllm_async captioning. "
            "0 = auto-derive as max(2, total_gpus) to keep GPU pipeline fed. "
            "Assumes ~20 windows per task. See stage_batch_size property for sizing rationale."
        ),
    )
    parser.add_argument(
        "--vllm-async-max-concurrent-requests",
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
        "--vllm-async-num-workers-per-node",
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
        "--vllm-async-stage2-caption",
        action="store_true",
        default=False,
        help=(
            "Enable stage-2 caption refinement for vllm_async captioning. "
            "The stage-1 caption is fed back with a refinement prompt for "
            "a second inference pass, producing a more detailed description."
        ),
    )
    parser.add_argument(
        "--vllm-async-stage2-prompt-text",
        type=str,
        default=None,
        help=(
            "Custom prompt text for stage-2 refinement. If not set, the default "
            "refinement prompt is used. Only effective when --vllm-async-stage2-caption is set."
        ),
    )

    # all VllmAsyncConfig-mapped args from specs
    for spec in _VLLM_ARG_SPECS:
        flag = f"--vllm-async-{spec.field.replace('_', '-')}"
        kwargs: dict[str, Any] = {"default": None, "help": spec.help}
        if spec.arg_type is None:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = spec.arg_type
        if spec.choices is not None:
            kwargs["choices"] = list(spec.choices)
        parser.add_argument(flag, **kwargs)
