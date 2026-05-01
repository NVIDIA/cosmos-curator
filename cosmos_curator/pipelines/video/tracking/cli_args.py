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

"""Shared argparse wiring for the SAM3 tracking stage.

The full splitting pipeline and the standalone ``sam3_event_pipeline.py``
example both surface the same SAM3 knobs to users; this helper keeps them in
lock-step so help text and defaults can't drift.
"""

import argparse


def add_sam3_args(
    parser: argparse.ArgumentParser,
    *,
    include_enable_flag: bool = True,
    include_write_annotated_flag: bool = True,
    sam3_prompts_required: bool = False,
) -> None:
    """Register ``--sam3-*`` arguments on ``parser``.

    Args:
        parser: Argparse parser to register the arguments on.
        include_enable_flag: If True, register ``--enable-sam3``. The example
            pipeline always runs SAM3 and does not need it.
        include_write_annotated_flag: If True, register
            ``--sam3-write-annotated-video``. The example pipeline always
            writes the annotated mp4 and does not need it.
        sam3_prompts_required: If True, ``--sam3-prompts`` is required at the
            argparse level (example pipeline). Otherwise it defaults to
            ``None`` and the caller validates non-emptiness at runtime
            (splitting pipeline).

    """
    if include_enable_flag:
        parser.add_argument(
            "--enable-sam3",
            action="store_true",
            default=False,
            help=(
                "Enable SAM3 object tracking stage after the default captioning stage. "
                "Produces per-clip sam3_instances/sam3_objects JSON + optional tracked.mp4."
            ),
        )

    parser.add_argument(
        "--sam3-prompts",
        nargs="+",
        required=sam3_prompts_required,
        default=None,
        help=(
            "Text descriptions of objects to track with SAM3, space-separated. "
            "Example: --sam3-prompts 'a car' 'a pedestrian' 'a traffic light'."
        ),
    )
    parser.add_argument(
        "--sam3-target-fps",
        type=float,
        default=10.0,
        help="Frames-per-second SAM3 is run at (source frames subsampled to this rate).",
    )
    parser.add_argument(
        "--sam3-max-clip-duration-s",
        type=float,
        default=30.0,
        help=(
            "Skip SAM3 on clips longer than this (seconds). VRAM scales with clip "
            "length * target_fps; the default is safe on a single 48 GB GPU at 16 fps."
        ),
    )
    parser.add_argument(
        "--sam3-session-reset-s",
        type=float,
        default=10.0,
        help=(
            "Chunk length in seconds. The SAM3 session is re-initialised between chunks "
            "to bound GPU memory; lower = safer but loses ID consistency across chunks."
        ),
    )

    # Quality knobs (None = SAM3 default from Sam3VideoConfig).
    parser.add_argument(
        "--sam3-score-threshold-detection",
        type=float,
        default=None,
        help=(
            "Minimum confidence for a frame-level detection to count as an object. "
            "Range [0.0, 1.0]; raise (0.6-0.8) to suppress weak/false detections, "
            "lower (0.3-0.4) to recall more objects in cluttered scenes."
        ),
    )
    parser.add_argument(
        "--sam3-det-nms-thresh",
        type=float,
        default=None,
        help=(
            "IoU above which two overlapping detections are merged into one (NMS). "
            "Range [0.0, 1.0]; lower (0.3-0.4) means fewer near-duplicate boxes, "
            "higher (0.6-0.7) keeps more overlapping candidates."
        ),
    )
    parser.add_argument(
        "--sam3-new-det-thresh",
        type=float,
        default=None,
        help=(
            "Confidence required to spawn a *new* tracked instance mid-clip "
            "(stricter than the per-frame detection threshold). Range [0.0, 1.0]; "
            "raise (0.7-0.9) to avoid spurious tracks, lower to start tracks earlier."
        ),
    )
    parser.add_argument(
        "--sam3-fill-hole-area",
        type=int,
        default=None,
        help=(
            "Pixel area threshold below which mask holes are filled. Typical 0-1024 px; "
            "raising smooths jittering masks at the cost of swallowing small genuine gaps."
        ),
    )
    parser.add_argument(
        "--sam3-recondition-every-nth-frame",
        type=int,
        default=None,
        help=(
            "How often to re-run the detector during tracking (in subsampled frames). "
            "Higher (8-16) gives smoother temporal consistency and fewer ID switches; "
            "very low (1-2) updates more aggressively but can cause box flicker."
        ),
    )
    parser.add_argument(
        "--sam3-recondition-on-trk-masks",
        type=lambda v: str(v).lower() in {"1", "true", "yes"},
        default=None,
        help=(
            "Feed current tracker masks back into the detector during reconditioning. "
            "Helps temporal stability for long-lived objects; off is more responsive to "
            "fast appearance changes."
        ),
    )
    parser.add_argument(
        "--sam3-high-conf-thresh",
        type=float,
        default=None,
        help=(
            "Confidence above which a detection is treated as a 'definite match' to an "
            "existing track during association. Range [0.0, 1.0]; lower makes the tracker "
            "more eager to attach detections, higher is stricter."
        ),
    )
    parser.add_argument(
        "--sam3-high-iou-thresh",
        type=float,
        default=None,
        help=(
            "IoU above which a detection is treated as a 'definite match' to an existing "
            "track during association. Range [0.0, 1.0]; same direction as "
            "--sam3-high-conf-thresh but on geometric overlap instead of confidence."
        ),
    )

    if include_write_annotated_flag:
        parser.add_argument(
            "--sam3-write-annotated-video",
            action="store_true",
            default=False,
            help=(
                "Emit a per-clip tracked.mp4 with boxes/masks/ids drawn on top. "
                "Automatically enabled when --enable-event-captioning is set "
                "(the captioner needs it as VLM input)."
            ),
        )
    parser.add_argument(
        "--sam3-annotated-video-trails",
        action="store_true",
        default=False,
        help=(
            "Draw trajectory trails for each tracked object in the annotated video. "
            "Off by default — trails clutter the frame and confuse the per-event VLM; "
            "only enable for offline visual inspection."
        ),
    )
