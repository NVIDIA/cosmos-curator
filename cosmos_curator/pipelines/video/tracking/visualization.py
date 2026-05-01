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

"""Shared drawing helpers for SAM3 annotated video output.

Used by both ``tools/sam3_bbox_demo.py`` and ``AnnotatedVideoWriterStage`` so
that any styling change is picked up in both places.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt

# Distinct BGR colours per prompt (OpenCV uses BGR).
COLOURS: list[tuple[int, int, int]] = [
    (0, 200, 255),  # amber
    (255, 80, 80),  # blue
    (80, 255, 80),  # green
    (255, 80, 255),  # magenta
    (80, 255, 255),  # cyan
]
# Double-stroke sans-serif; more robust to VLM downscale than ``SIMPLEX``.
FONT = cv2.FONT_HERSHEY_DUPLEX
# Mask outline is drawn as a contour polyline (no fill) so the original object
# pixels survive for a VLM fed the annotated video. Thickness 2 is the minimum
# that survives Gemini's ``high``-res downscale and H.264 motion blur; bump to
# 3 on 2K+ sources if contours alias in the VLM's view.
MASK_CONTOUR_THICKNESS = 2
# ID label rendering. 0.6 is a balance between human legibility at 1080p and
# staying unobtrusive in dense-traffic clips. Raise to ~0.85 if you need a VLM
# to OCR labels reliably off the annotated video.
LABEL_SCALE = 0.6
LABEL_THICKNESS = 1
LABEL_OUTLINE_THICKNESS = 3
LABEL_OUTLINE_COLOUR: tuple[int, int, int] = (0, 0, 0)
LABEL_TEXT_COLOUR: tuple[int, int, int] = (255, 255, 255)
TRAIL_THICKNESS = 2
TRAIL_MIN_POINTS = 2

# Burnt-in wall-clock timestamp ("t=12.34s", top-left). Gives downstream VLMs a
# literal on-frame time to OCR so ``start_time`` / ``end_time`` don't drift.
# Styled distinctly from ``#id`` labels (larger scale, thicker stroke, solid
# background) so the VLM can cleanly separate frame-time from object-id.
TIMESTAMP_SCALE = 1.1
TIMESTAMP_THICKNESS = 2
TIMESTAMP_TEXT_COLOUR: tuple[int, int, int] = (255, 255, 255)
TIMESTAMP_BG_COLOUR: tuple[int, int, int] = (0, 0, 0)
TIMESTAMP_MARGIN = 8
TIMESTAMP_PAD = 6


def draw_timestamp(frame: npt.NDArray[np.uint8], current_time_s: float) -> None:
    """Burn a ``t=X.XXs`` badge into the top-left corner (mutates ``frame``).

    The prompt instructs the VLM to key off this exact ``t=X.XXs`` format.
    """
    text = f"t={current_time_s:.2f}s"
    (tw, th), baseline = cv2.getTextSize(text, FONT, TIMESTAMP_SCALE, TIMESTAMP_THICKNESS)
    x0, y0 = TIMESTAMP_MARGIN, TIMESTAMP_MARGIN
    pad = TIMESTAMP_PAD
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + tw + 2 * pad, y0 + th + baseline + 2 * pad),
        TIMESTAMP_BG_COLOUR,
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (x0 + pad, y0 + pad + th),
        FONT,
        TIMESTAMP_SCALE,
        TIMESTAMP_TEXT_COLOUR,
        TIMESTAMP_THICKNESS,
        cv2.LINE_AA,
    )


@dataclass
class Detection:
    """A single object detection for one frame."""

    prompt: str
    object_id: int
    box_xyxy: list[float]
    # Boolean mask of shape (H, W); ``repr=False`` to keep logs readable.
    mask: npt.NDArray[np.bool_] = field(repr=False)

    @property
    def center(self) -> tuple[int, int]:
        """Return the integer pixel centre of the bounding box."""
        x1, y1, x2, y2 = self.box_xyxy
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def to_json_dict(self) -> dict[str, object]:
        """Serializable view of the detection.

        Boolean masks are too large to ship per-frame, so we emit COCO-style
        polygons under ``contours_xy`` instead. Format matches ``draw_frame``
        (``RETR_EXTERNAL`` + ``CHAIN_APPROX_SIMPLE``) so the polygons in
        ``objects.json`` trace the same silhouettes as ``tracked.mp4``.

        ``contours_xy`` is ``list[list[int]]``: outer list is one entry per
        disconnected polygon (usually 1), inner list is flat
        ``[x0, y0, x1, y1, ...]`` pixel coordinates.
        """
        mask_u8 = self.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_xy = [c.flatten().tolist() for c in contours]
        return {
            "prompt": self.prompt,
            "object_id": self.object_id,
            "box_xyxy": self.box_xyxy,
            "contours_xy": contours_xy,
        }


def draw_frame(  # noqa: PLR0913 — each arg controls an orthogonal overlay aspect; bundling into a config object would hurt call-site readability
    frame: npt.NDArray[np.uint8],
    detections: list[Detection],
    prompts: list[str],
    trails: dict[int, list[tuple[int, int]]],
    *,
    draw_trails: bool = False,
    current_time_s: float | None = None,
) -> npt.NDArray[np.uint8]:
    """Draw mask outlines, object IDs, and optional trajectory trails.

    Style: mask contour (outline only, no fill) + short ``#id`` label with a
    dark outline. Mask pixels are NOT re-coloured — this preserves the object's
    original appearance for a VLM fed the annotated video. No bbox is drawn;
    the contour already localises more precisely.

    Args:
        frame: ``H x W x 3`` BGR uint8. Not mutated; a copy is returned.
        detections: flat list of ``Detection`` objects for this frame.
        prompts: ordered list of all prompts — used to assign stable colours.
        trails: mapping of ``object_id`` to accumulated ``(cx, cy)`` centre
            points; updated in place by the caller.
        draw_trails: if ``True``, draw polyline trails for each object.
        current_time_s: if provided, burn a ``t=X.XXs`` timestamp so downstream
            VLMs can anchor event times to the clip timeline.

    Returns:
        Annotated BGR frame.

    """
    prompt_colour = {p: COLOURS[i % len(COLOURS)] for i, p in enumerate(prompts)}
    out = frame.copy()

    for det in detections:
        colour = prompt_colour.get(det.prompt, (200, 200, 200))

        mask_u8 = det.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x1, y1, _, _ = (int(v) for v in det.box_xyxy)

        # Two-pass text: thick black stroke, then thinner white fill — the
        # standard video-caption technique for legibility over arbitrary
        # backgrounds without a filled label box occluding the scene.
        label = f"#{det.object_id}"
        # Anchor to the top of the mask silhouette rather than the bbox corner;
        # for large/diagonal objects the bbox corner often sits in empty space
        # and makes the label look detached. Falls back to the bbox corner on
        # the rare empty-contour case.
        if contours:
            cv2.drawContours(out, contours, -1, colour, thickness=MASK_CONTOUR_THICKNESS)
            all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
            anchor_y = int(all_pts[:, 1].min())
            anchor_x = int(np.median(all_pts[all_pts[:, 1] == anchor_y, 0]))
        else:
            anchor_x, anchor_y = x1, y1
        (tw, th), _ = cv2.getTextSize(label, FONT, LABEL_SCALE, LABEL_THICKNESS)
        # Clamp inside the frame on the top/left edges.
        text_origin = (max(0, anchor_x - tw // 2), max(th + 2, anchor_y - 6))
        cv2.putText(
            out,
            label,
            text_origin,
            FONT,
            LABEL_SCALE,
            LABEL_OUTLINE_COLOUR,
            LABEL_OUTLINE_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            text_origin,
            FONT,
            LABEL_SCALE,
            LABEL_TEXT_COLOUR,
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )

        if draw_trails:
            trail = trails.get(det.object_id, [])
            if len(trail) >= TRAIL_MIN_POINTS:
                pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], isClosed=False, color=colour, thickness=TRAIL_THICKNESS)

    # Draw last so no contour/label can occlude the timestamp badge.
    if current_time_s is not None:
        draw_timestamp(out, current_time_s)

    return out
