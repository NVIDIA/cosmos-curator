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

"""Shared argparse wiring for the per-event VLM captioning stage.

The full splitting pipeline and the standalone ``sam3_event_pipeline.py``
example both surface the same VLM knobs to users; this helper keeps them in
lock-step so help text and defaults can't drift.
"""

import argparse
import pathlib

from loguru import logger


def add_event_caption_args(
    parser: argparse.ArgumentParser,
    *,
    include_enable_flag: bool = True,
) -> None:
    """Register ``--event-caption-*`` arguments on ``parser``.

    Args:
        parser: Argparse parser to register the arguments on.
        include_enable_flag: If True, register ``--enable-event-captioning``.
            Both pipelines currently use this flag; included for symmetry with
            ``add_sam3_args``.

    """
    if include_enable_flag:
        parser.add_argument(
            "--enable-event-captioning",
            action="store_true",
            default=False,
            help=(
                "Enable per-event VLM captioning stage after SAM3. Produces per-clip "
                "sam3_events JSON. Requires --enable-sam3 when used in the splitting pipeline."
            ),
        )

    parser.add_argument(
        "--event-caption-prompt",
        type=str,
        default=None,
        help=(
            "Prompt template for the per-event VLM stage as a literal string. A compact "
            "SAM3 object summary is appended automatically. Mutually exclusive with "
            "--event-caption-prompt-file. Leave unset for the packaged "
            "traffic-surveillance default (prompts/traffic_surveillance.md)."
        ),
    )
    parser.add_argument(
        "--event-caption-prompt-file",
        type=str,
        default=None,
        help=(
            "Path to a file (e.g. .md) whose contents become the VLM prompt template. "
            "Read client-side at submit time so you can iterate without rebuilding the "
            "image. Mutually exclusive with --event-caption-prompt."
        ),
    )
    parser.add_argument(
        "--event-caption-backend",
        type=str,
        default="qwen",
        choices=["qwen", "gemini"],
        help=(
            "VLM backend: 'qwen' runs Qwen-VL locally on a GPU (default, no API key); "
            "'gemini' calls the remote Gemini API and requires a gemini.api_key entry "
            "in the project config."
        ),
    )
    parser.add_argument(
        "--event-caption-qwen-variant",
        type=str,
        default="qwen",
        choices=["qwen", "qwen3_vl_30b", "qwen3_vl_30b_fp8"],
        help=(
            "Qwen model variant. 'qwen' = Qwen2.5-VL-7B-Instruct (BF16, fits on 24 GB+, "
            "default). 'qwen3_vl_30b' = Qwen3-VL-30B-A3B-Instruct (MoE, 3B active, BF16 "
            "~60 GB weights, recommended on 80 GB GPUs). 'qwen3_vl_30b_fp8' = same "
            "checkpoint FP8-quantised (~30 GB weights, ~2x throughput on Hopper/Ada "
            "via native FP8 tensor cores). Pair 'qwen3_vl_30b_fp8' with "
            "--event-caption-qwen-fp8 on H100/L40; do NOT enable FP8 on Ampere "
            "(A6000/A100) - vLLM's Marlin fallback crashes on Qwen's vision encoder."
        ),
    )
    parser.add_argument(
        "--event-caption-qwen-sampling-fps",
        type=float,
        default=2.0,
        help="Sampling fps for the Qwen backend when decoding a clip. Higher = more tokens.",
    )
    parser.add_argument(
        "--event-caption-qwen-fp8",
        action="store_true",
        default=False,
        help=(
            "Enable FP8 quantization for Qwen. Only safe on GPUs with native FP8 "
            "(Hopper/Ada). On Ampere (A6000/A100) leave this off - vLLM falls back to "
            "Marlin FP8 and fails on Qwen's vision encoder."
        ),
    )
    parser.add_argument(
        "--event-caption-qwen-temperature",
        type=float,
        default=None,
        help=(
            "Override Qwen sampling temperature (default from QwenVL is ~0.1, effectively "
            "greedy). Try 0.4-0.7 if Qwen collapses to a single cookie-cutter event "
            "under heavy vision load."
        ),
    )
    parser.add_argument(
        "--event-caption-qwen-top-p",
        type=float,
        default=None,
        help=(
            "Override Qwen top_p (default from QwenVL is ~0.001, near-greedy). Pair with "
            "--event-caption-qwen-temperature; e.g. 0.9 with 0.5."
        ),
    )
    parser.add_argument(
        "--event-caption-qwen-top-k",
        type=int,
        default=None,
        help="Override Qwen top_k.",
    )
    parser.add_argument(
        "--event-caption-gemini-model-name",
        type=str,
        default="models/gemini-2.5-flash",
        help=(
            "Gemini model name. Default 'models/gemini-2.5-flash' is free-tier friendly. "
            "Use 'models/gemini-2.5-pro' only with a billed project."
        ),
    )
    parser.add_argument(
        "--event-caption-gemini-fps",
        type=float,
        default=4.0,
        help=(
            "fps at which Gemini ingests the annotated clip. Default 4.0 - Gemini's own "
            "default (~1 fps) misses sub-second impacts and close-call events."
        ),
    )
    parser.add_argument(
        "--event-caption-gemini-media-resolution",
        type=str,
        choices=("low", "medium", "high"),
        default="high",
        help=(
            "Gemini per-frame media resolution. 'high' (default) is needed for the SAM3 "
            "#id overlay labels and thin mask contours to survive per-frame downscaling."
        ),
    )
    parser.add_argument(
        "--event-caption-gemini-thinking-budget",
        type=int,
        default=-1,
        help=(
            "Gemini 2.5 thinking-tokens budget. -1 = dynamic (recommended), 0 = disabled, "
            "N = hard cap. Flash with thinking_budget=0 cannot reliably distinguish queued "
            "vehicles from colliding vehicles."
        ),
    )
    parser.add_argument(
        "--event-caption-gemini-max-output-tokens",
        type=int,
        default=16384,
        help=(
            "Max Gemini response tokens (shared with thinking on 2.5 Flash). Too small "
            "(old default 4096) caused mid-event truncation; raise to 24000+ for dense clips."
        ),
    )


def resolve_event_caption_prompt(args: argparse.Namespace) -> str | None:
    """Resolve the VLM prompt into a string (or ``None`` for packaged default).

    ``--event-caption-prompt-file`` is read client-side at submit time so the
    file can live in the user's edit tree and iterate without a rebuild; the
    resolved string is then forwarded to every worker via stage constructor
    args.
    """
    prompt = getattr(args, "event_caption_prompt", None)
    prompt_file = getattr(args, "event_caption_prompt_file", None)
    if prompt is not None and prompt_file is not None:
        msg = "Pass only one of --event-caption-prompt or --event-caption-prompt-file."
        raise SystemExit(msg)
    if prompt_file is not None:
        path = pathlib.Path(prompt_file).expanduser().resolve()
        if not path.is_file():
            msg = f"--event-caption-prompt-file not found: {path}"
            raise SystemExit(msg)
        text = path.read_text(encoding="utf-8")
        logger.info(f"Loaded per-event VLM prompt from {path} ({len(text)} chars)")
        return text
    return prompt
