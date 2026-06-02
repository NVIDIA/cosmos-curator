# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PaddleOCR model for post-production / artificial text detection.

Detection-only pipeline: runs OCR text detection on video frames, then heuristics
(stable text tracks, corner text) to classify overlay/artificial text.
"""

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import av
import cv2
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.model import pixi_utils

if TYPE_CHECKING:
    import paddle  # type: ignore[import-not-found]
    import paddle.inference  # type: ignore[import-not-found]
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

if pixi_utils.is_running_in_env("paddle-ocr") or pixi_utils.is_running_in_env("unified"):
    import paddle  # type: ignore[import-not-found]
    import paddle.inference  # type: ignore[import-not-found]
    from paddleocr import PaddleOCR


WEIGHTS_NAME_PREFIX: Final[str] = "/config/models/"
PADDLE_OCR_DET_MODEL_ID: Final[str] = "PaddlePaddle/PP-OCRv4_mobile_det"
PADDLE_OCR_REC_MODEL_ID: Final[str] = "PaddlePaddle/PP-OCRv4_mobile_rec"
PADDLE_OCR_CLS_MODEL_ID: Final[str] = "PaddlePaddle/paddle_ocr_cls"

# Heuristic parameters for artificial text detection (stable text + corner text).
IOU_MATCH_THRESHOLD = 0.9
MAX_FRAME_GAP_FOR_TRACK = 5
MIN_DURATION_FRAMES = 10
MIN_DURATION_FRAMES_CORNER_RATIO = 0.1
STABILITY_IOU_CONSECUTIVE_THRESHOLD = 0.9
CORNER_X_MARGIN_NORM = 0.1
CORNER_Y_MARGIN_NORM = 0.1

# Default inference settings.
_TARGET_LONGEST_SIDE_DEFAULT = 640
_FRAME_INTERVAL_DEFAULT = 3

# Box and point shape constants (OCR boxes are quadrilaterals; points are [x, y]).
_NUM_BOX_POINTS = 4
_COORDS_PER_POINT = 2
_MIN_HISTORY_FOR_IOU = 2


def _get_corners_from_points(points: list[list[float]]) -> tuple[int, int, int, int]:
    """Convert a list of 4 [x,y] points to (min_x, min_y, max_x, max_y)."""
    if not points or len(points) != _NUM_BOX_POINTS:
        return 0, 0, 0, 0
    min_x = int(min(p[0] for p in points))
    max_x = int(max(p[0] for p in points))
    min_y = int(min(p[1] for p in points))
    max_y = int(max(p[1] for p in points))
    return min_x, min_y, max_x, max_y


def _calculate_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute IoU for boxes (min_x, min_y, max_x, max_y)."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    denom = float(area_a + area_b - inter)
    return inter / denom if denom > 0 else 0.0


def _is_bbox_in_corner_zone(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    corner_x_norm: float,
    corner_y_norm: float,
) -> bool:
    """Return True if box center is near one of the four frame corners."""
    if frame_width <= 0 or frame_height <= 0:
        return False
    min_x, min_y, max_x, max_y = box
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    mx = corner_x_norm * frame_width
    my = corner_y_norm * frame_height
    return (
        (cx <= mx and cy <= my)
        or (cx >= (frame_width - mx) and cy <= my)
        or (cx <= mx and cy >= (frame_height - my))
        or (cx >= (frame_width - mx) and cy >= (frame_height - my))
    )


class _TextTrack:
    """Single text track across frames for stability analysis."""

    def __init__(self, bbox_points: list[list[float]], frame_num: int, timestamp: float) -> None:
        self.bbox_corners_history = [_get_corners_from_points(bbox_points)]
        self.frame_numbers = [frame_num]
        self.timestamps = [timestamp]
        self.last_seen_frame = frame_num

    def update(self, bbox_points: list[list[float]], frame_num: int, timestamp: float) -> None:
        self.bbox_corners_history.append(_get_corners_from_points(bbox_points))
        self.frame_numbers.append(frame_num)
        self.timestamps.append(timestamp)
        self.last_seen_frame = frame_num

    def get_duration_frames(self) -> int:
        return len(self.frame_numbers)

    def get_avg_bbox_stability_iou(self) -> float:
        if len(self.bbox_corners_history) < _MIN_HISTORY_FOR_IOU:
            return 1.0
        ious = [
            _calculate_iou(self.bbox_corners_history[i - 1], self.bbox_corners_history[i])
            for i in range(1, len(self.bbox_corners_history))
        ]
        return sum(ious) / len(ious) if ious else 0.0

    def check_if_stable_text(
        self,
        _frame_height: int,
        _frame_width: int,
        frame_interval: int,
        min_duration_frames: int,
        stability_iou_threshold: float,
    ) -> tuple[bool, str]:
        adjusted_min = max(1, min_duration_frames // frame_interval)
        if self.get_duration_frames() < adjusted_min:
            return False, "too_short_for_stable"
        if self.get_avg_bbox_stability_iou() >= stability_iou_threshold:
            return True, "stable_location_text"
        return False, "duration_met_not_stable_enough"

    def to_dict_segment(
        self,
        reason: str,
        _frame_height: int | None,
        _frame_width: int | None,
    ) -> dict[str, Any]:
        duration_sec = (
            round(self.timestamps[-1] - self.timestamps[0], 3) if len(self.timestamps) >= _MIN_HISTORY_FOR_IOU else 0.0
        )
        return {
            "start_frame": self.frame_numbers[0],
            "end_frame": self.frame_numbers[-1],
            "start_time_sec": self.timestamps[0],
            "end_time_sec": self.timestamps[-1],
            "duration_frames": self.get_duration_frames(),
            "duration_seconds": duration_sec,
            "avg_bbox_stability_iou": round(self.get_avg_bbox_stability_iou(), 3),
            "classification_reason": reason,
        }


class _StableTextDetector:
    """Detects text that remains in a stable position across frames (overlay-like)."""

    def __init__(
        self,
        frame_height: int,
        frame_width: int,
        frame_interval: int = _FRAME_INTERVAL_DEFAULT,
        *,
        min_duration_frames: int = MIN_DURATION_FRAMES,
        stability_iou_threshold: float = STABILITY_IOU_CONSECUTIVE_THRESHOLD,
    ) -> None:
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_interval = frame_interval
        self.min_duration_frames = min_duration_frames
        self.stability_iou_threshold = stability_iou_threshold
        self.max_frame_gap = MAX_FRAME_GAP_FOR_TRACK * frame_interval
        self.active_tracks: list[_TextTrack] = []
        self.confirmed_segments: list[dict[str, Any]] = []

    def process_frame(self, frame_content: dict[str, Any]) -> None:
        frame_num = frame_content["frame_number"]
        timestamp = frame_content["timestamp_sec"]
        updated: list[_TextTrack] = []
        for track in self.active_tracks:
            if frame_num - track.last_seen_frame > self.max_frame_gap:
                is_artificial, reason = track.check_if_stable_text(
                    self.frame_height,
                    self.frame_width,
                    self.frame_interval,
                    self.min_duration_frames,
                    self.stability_iou_threshold,
                )
                if is_artificial:
                    self.confirmed_segments.append(track.to_dict_segment(reason, self.frame_height, self.frame_width))
            else:
                updated.append(track)
        self.active_tracks = updated
        for det in frame_content["detections"]:
            bbox_corners = _get_corners_from_points(det["bbox"])
            matched = False
            for track in self.active_tracks:
                if (
                    track.bbox_corners_history
                    and _calculate_iou(bbox_corners, track.bbox_corners_history[-1]) >= IOU_MATCH_THRESHOLD
                ):
                    track.update(det["bbox"], frame_num, timestamp)
                    matched = True
                    break
            if not matched:
                self.active_tracks.append(_TextTrack(det["bbox"], frame_num, timestamp))

    def finalize(self) -> list[dict[str, Any]]:
        for track in self.active_tracks:
            is_artificial, reason = track.check_if_stable_text(
                self.frame_height,
                self.frame_width,
                self.frame_interval,
                self.min_duration_frames,
                self.stability_iou_threshold,
            )
            if is_artificial:
                self.confirmed_segments.append(track.to_dict_segment(reason, self.frame_height, self.frame_width))
        return self.confirmed_segments

    def analyze(self, frames_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.active_tracks = []
        self.confirmed_segments = []
        for fc in frames_data:
            self.process_frame(fc)
        return self.finalize()


class _CornerTextDetector:
    """Detects text that appears in frame corners (e.g. channel logos, watermarks)."""

    def __init__(
        self,
        frame_height: int,
        frame_width: int,
        *,
        min_duration_frames_corner_ratio: float = MIN_DURATION_FRAMES_CORNER_RATIO,
        corner_x_margin_norm: float = CORNER_X_MARGIN_NORM,
        corner_y_margin_norm: float = CORNER_Y_MARGIN_NORM,
    ) -> None:
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.min_duration_frames_corner_ratio = min_duration_frames_corner_ratio
        self.corner_x_margin_norm = corner_x_margin_norm
        self.corner_y_margin_norm = corner_y_margin_norm

    def detect(
        self,
        frames_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not frames_data:
            return []
        num_sampled = 0
        total_corner = 0
        segments: list[dict[str, Any]] = []
        for fc in frames_data:
            if not fc.get("detections"):
                continue
            num_sampled += 1
            has_corner = any(
                _is_bbox_in_corner_zone(
                    _get_corners_from_points(det["bbox"]),
                    self.frame_width,
                    self.frame_height,
                    self.corner_x_margin_norm,
                    self.corner_y_margin_norm,
                )
                for det in fc["detections"]
            )
            if has_corner:
                total_corner += 1
                frame_num = fc["frame_number"]
                ts = fc["timestamp_sec"]
                segments.append(
                    {
                        "start_frame": frame_num,
                        "end_frame": frame_num,
                        "start_time_sec": ts,
                        "end_time_sec": ts,
                        "duration_frames": 1,
                        "duration_seconds": 0.0,
                        "classification_reason": "corner_text",
                    }
                )
        if num_sampled > 0 and total_corner / num_sampled > self.min_duration_frames_corner_ratio:
            return segments
        return []


class ArtificialTextDetector:
    """Maps OCR box results to artificial/overlay text segments via stable + corner heuristics."""

    def __init__(  # noqa: PLR0913
        self,
        frame_height: int,
        frame_width: int,
        fps: float = 30.0,
        *,
        use_corner_detection: bool = True,
        frame_interval: int = _FRAME_INTERVAL_DEFAULT,
        min_duration_frames: int = MIN_DURATION_FRAMES,
        min_duration_frames_corner_ratio: float = MIN_DURATION_FRAMES_CORNER_RATIO,
        stability_iou_threshold: float = STABILITY_IOU_CONSECUTIVE_THRESHOLD,
        ignore_corner_region: bool = False,
        corner_x_margin_norm: float = CORNER_X_MARGIN_NORM,
        corner_y_margin_norm: float = CORNER_Y_MARGIN_NORM,
    ) -> None:
        """Initialize detector with frame dimensions, FPS, and optional corner detection."""
        _err_dim = "Frame height and width must be positive."
        _err_fps = "FPS and frame_interval must be positive."
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError(_err_dim)
        if fps <= 0 or frame_interval <= 0:
            raise ValueError(_err_fps)
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.fps = fps
        self.frame_interval = frame_interval
        self._ignore_corner_region = ignore_corner_region
        self._corner_x_margin_norm = corner_x_margin_norm
        self._corner_y_margin_norm = corner_y_margin_norm
        self._stable = _StableTextDetector(
            frame_height,
            frame_width,
            frame_interval,
            min_duration_frames=min_duration_frames,
            stability_iou_threshold=stability_iou_threshold,
        )
        self._corner = (
            _CornerTextDetector(
                frame_height,
                frame_width,
                min_duration_frames_corner_ratio=min_duration_frames_corner_ratio,
                corner_x_margin_norm=corner_x_margin_norm,
                corner_y_margin_norm=corner_y_margin_norm,
            )
            if use_corner_detection
            else None
        )

    def _transform_ocr_results(
        self,
        ocr_results_per_frame: list[list[list[list[float]]]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for frame_idx, frame_boxes in enumerate(ocr_results_per_frame):
            timestamp_sec = round(frame_idx / self.fps, 3)
            if frame_boxes:
                detections = []
                for box in frame_boxes:
                    if not (
                        isinstance(box, list) and all(isinstance(p, list) and len(p) == _COORDS_PER_POINT for p in box)
                    ):
                        continue
                    if self._ignore_corner_region:
                        bbox_rect = _get_corners_from_points(box)
                        if _is_bbox_in_corner_zone(
                            bbox_rect,
                            self.frame_width,
                            self.frame_height,
                            self._corner_x_margin_norm,
                            self._corner_y_margin_norm,
                        ):
                            continue
                    detections.append({"bbox": box})
            else:
                detections = []
            out.append(
                {
                    "frame_number": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "detections": detections,
                }
            )
        return out

    def detect(
        self,
        ocr_results_per_frame: list[list[list[list[float]]]],
    ) -> list[dict[str, Any]]:
        """Return list of artificial text segment dicts from OCR per-frame box list."""
        frames_data = self._transform_ocr_results(ocr_results_per_frame)
        all_segments: list[dict[str, Any]] = []
        all_segments.extend(self._stable.analyze(frames_data))
        if self._corner and frames_data:
            all_segments.extend(self._corner.detect(frames_data))
        return all_segments


_PADDLE_CPU_PATCHED: bool = False


def _patch_paddle_cpu_inference() -> None:
    """Apply workarounds needed for CPU paddlepaddle inference.

    1. paddleocr unconditionally calls config.enable_memory_optim(True); wrap
       create_predictor to force it False — the pass isn't present in CPU builds.
    2. PIR executor (new in paddle 3.x) + OneDNN raises ConvertPirAttribute2RuntimeAttribute
       even with enable_mkldnn=False; disable PIR entirely for CPU inference.

    Guarded by a module-level flag so repeated calls (e.g. retries) don't stack wrappers.
    """
    global _PADDLE_CPU_PATCHED  # noqa: PLW0603
    if _PADDLE_CPU_PATCHED:
        return
    paddle.set_flags({"FLAGS_enable_pir_api": False})
    _orig = paddle.inference.create_predictor

    def _patched(config: object) -> object:
        if hasattr(config, "enable_memory_optim"):
            config.enable_memory_optim(False)  # type: ignore[union-attr]  # noqa: FBT003
        return _orig(config)

    paddle.inference.create_predictor = _patched
    _PADDLE_CPU_PATCHED = True


class PaddleOCRModel(ModelInterface):
    """PaddleOCR detection-only model for post-production text detection.

    Uses the detection model only (no recognition or angle classifier). Run OCR with
    det=True, rec=False, cls=False and feed results to ArtificialTextDetector.
    """

    def __init__(
        self,
        target_longest_side: int | None = _TARGET_LONGEST_SIDE_DEFAULT,
        frame_interval: int = _FRAME_INTERVAL_DEFAULT,
        *,
        use_gpu: bool = True,
    ) -> None:
        """Store target longest side, frame interval, and GPU flag for detection."""
        super().__init__()
        self._target_longest_side = target_longest_side
        self._frame_interval = frame_interval
        self._use_gpu = use_gpu
        self._model: Any = None

    @property
    def conda_env_name(self) -> str:
        """Return the conda environment name for this model.

        GPU mode (default) runs in paddle-ocr (paddlepaddle-gpu via post_install).
        CPU mode runs in unified (paddlepaddle CPU, no torch conflict).
        """
        return "paddle-ocr" if self._use_gpu else "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Return model IDs for pipeline weight download (det, rec).

        Cls is not on Hugging Face; see setup() warning if missing.
        """
        return [PADDLE_OCR_DET_MODEL_ID, PADDLE_OCR_REC_MODEL_ID]

    def setup(self) -> None:
        """Load PaddleOCR detection model."""
        cls_model_dir = Path(WEIGHTS_NAME_PREFIX) / PADDLE_OCR_CLS_MODEL_ID
        if not cls_model_dir.is_dir() or not (cls_model_dir / "inference.pdmodel").exists():
            logger.warning(
                "PaddleOCR cls (angle classifier) weights not found at {}. "
                "Pre-download per cosmos_curator/models/README.md to avoid runtime failures "
                "when multiple workers trigger PaddleOCR's download.",
                cls_model_dir,
            )
        model_root = Path(WEIGHTS_NAME_PREFIX)
        if not self._use_gpu:
            _patch_paddle_cpu_inference()
        self._model = PaddleOCR(
            use_gpu=self._use_gpu,
            det_model_dir=os.fspath(model_root / PADDLE_OCR_DET_MODEL_ID),
            rec_model_dir=os.fspath(model_root / PADDLE_OCR_REC_MODEL_ID),
            cls_model_dir=os.fspath(model_root / PADDLE_OCR_CLS_MODEL_ID),
            det_db_thresh=0.4,
            det_db_box_thresh=0.7,
            lang="en",
        )

    def generate_single(self, video_bytes: bytes) -> list[list[list[list[float]]]]:  # noqa: C901
        """Run detection on video bytes; return per-frame list of boxes (each box = list of 4 [x,y] points)."""
        _err_not_init = "PaddleOCR model not initialized. Call setup() first."
        if self._model is None:
            raise RuntimeError(_err_not_init)
        if len(video_bytes) == 0:
            logger.warning("PaddleOCR generate_single called with empty video_bytes")
            return []
        all_detections: list[list[list[list[float]]]] = []
        stream = io.BytesIO(video_bytes)
        with av.open(stream, format="mp4") as container:
            input_container = cast("av.container.InputContainer", container)
            for frame_idx, frame in enumerate(input_container.decode(video=0)):
                if frame_idx % self._frame_interval != 0 and frame_idx != 0:
                    all_detections.append([])
                    continue
                frame_rgb = frame.to_ndarray(format="rgb24")
                scale_ratio = 1.0
                if self._target_longest_side and self._target_longest_side > 0:
                    h, w = frame_rgb.shape[:2]
                    if max(w, h) == 0:
                        all_detections.append([])
                        continue
                    scale_ratio = float(self._target_longest_side) / max(w, h)
                    if scale_ratio != 1.0:
                        new_w = max(1, int(w * scale_ratio))
                        new_h = max(1, int(h * scale_ratio))
                        frame_rgb = cv2.resize(  # type: ignore[assignment]
                            frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
                        )
                raw = self._model.ocr(frame_rgb, det=True, rec=False, cls=False)
                boxes_this_frame: list[list[list[float]]] = []
                # With rec=False, PaddleOCR returns List[box] where each box is 4 [x,y] points.
                if raw and raw[0]:
                    for box_candidate in raw[0]:
                        if not isinstance(box_candidate, (list, tuple)) or len(box_candidate) != _NUM_BOX_POINTS:
                            continue
                        box_points = [
                            list(p)
                            for p in box_candidate
                            if isinstance(p, (list, tuple)) and len(p) == _COORDS_PER_POINT
                        ]
                        if len(box_points) != _NUM_BOX_POINTS:
                            continue
                        if scale_ratio != 1.0:
                            box_points = [[p[0] / scale_ratio, p[1] / scale_ratio] for p in box_points]
                        boxes_this_frame.append(box_points)
                all_detections.append(boxes_this_frame)
        return all_detections
