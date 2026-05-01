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

"""SAM3 bounding-box tracking stage for the splitting pipeline.

Runs SAM3 in pre-loaded chunked mode (enables hotstart heuristics:
phantom/duplicate removal, occlusion handling) on each clip's transcoded
mp4 bytes and populates per-clip outputs:

- ``clip.sam3_instances``        — per-``object_id`` summary across the clip
- ``clip.sam3_objects_by_frame`` — ``{frame_idx: [{object_id, prompt, box_xyxy}, ...]}``
- ``clip.sam3_annotated_video``  — optional re-encoded mp4 with overlays drawn

This stage runs in the ``sam3`` pixi environment (isolated from vLLM) and
requires one full GPU. Annotation is folded into the stage rather than being
a separate CPU stage because SAM3 masks are too large to transport across the
Ray task boundary (roughly ``frames * objects * H * W`` boolean bytes per clip).
"""

import collections
import pathlib
import tempfile
from typing import Any

import attrs
import cv2
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_startup
from cosmos_curator.core.utils.misc.memfd import buffer_as_memfd_path
from cosmos_curator.models.sam3 import SAM3Model
from cosmos_curator.pipelines.video.tracking.visualization import Detection, draw_frame
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask


@attrs.define
class SAM3QualityConfig:
    """Optional ``Sam3VideoConfig`` attribute overrides; ``None`` = SAM3 default."""

    score_threshold_detection: float | None = None
    det_nms_thresh: float | None = None
    new_det_thresh: float | None = None
    fill_hole_area: int | None = None
    recondition_every_nth_frame: int | None = None
    recondition_on_trk_masks: bool | None = None
    high_conf_thresh: float | None = None
    high_iou_thresh: float | None = None

    def to_overrides(self) -> dict[str, Any] | None:
        """Return a dict of non-``None`` overrides or ``None`` if all are default."""
        overrides = {k: v for k, v in attrs.asdict(self).items() if v is not None}
        return overrides or None


def _read_frames_from_bytes(
    mp4_bytes: bytes,
    target_fps: float,
) -> tuple[
    list[npt.NDArray[np.uint8]],
    list[npt.NDArray[np.uint8]],
    list[int],
    float,
    float,
    int,
    int,
    int,
]:
    """Decode ``mp4_bytes`` and return frames subsampled to ``target_fps``.

    Returns:
        ``(rgb_frames, bgr_frames, source_indices, src_fps, out_fps, step, width, height)``.

    """
    with buffer_as_memfd_path(mp4_bytes, name="sam3-bbox-clip") as path:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            msg = "SAM3BBoxStage: cannot open clip mp4 bytes via memfd"
            raise RuntimeError(msg)
        src_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, round(src_fps / target_fps))
        out_fps = src_fps / step

        rgb_frames: list[npt.NDArray[np.uint8]] = []
        bgr_frames: list[npt.NDArray[np.uint8]] = []
        source_indices: list[int] = []
        idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb.astype(np.uint8, copy=False))
                bgr_frames.append(np.asarray(bgr, dtype=np.uint8))
                source_indices.append(idx)
            idx += 1
        cap.release()

    return rgb_frames, bgr_frames, source_indices, src_fps, out_fps, step, w, h


def _postprocess_to_detections(processed: dict[str, Any], prompts: list[str]) -> list[Detection]:
    """Convert SAM3 ``postprocess_outputs`` dict to a flat list of ``Detection``."""
    obj_ids: list[int] = processed["object_ids"].tolist()
    masks = processed["masks"]
    boxes = processed["boxes"]
    p2o: dict[str, list[int]] = processed["prompt_to_obj_ids"]

    detections: list[Detection] = []
    for prompt in prompts:
        for oid in p2o.get(prompt, []):
            if oid not in obj_ids:
                continue
            idx = obj_ids.index(oid)
            mask_np = masks[idx].cpu().numpy()
            if mask_np.any():
                detections.append(
                    Detection(
                        prompt=prompt,
                        object_id=oid,
                        box_xyxy=boxes[idx].tolist(),
                        mask=mask_np,
                    )
                )
    return detections


def _encode_annotated_video(
    annotated_bgr_frames: list[npt.NDArray[np.uint8]],
    out_fps: float,
    width: int,
    height: int,
) -> bytes | None:
    """Encode a list of BGR frames to an mp4 byte buffer via a temp file.

    ``cv2.VideoWriter`` needs a filesystem path, so we write to a temp file and
    read the bytes back. ``delete=False`` + explicit ``unlink`` avoids racing
    the ``VideoWriter``'s own handle on the same path.
    """
    if not annotated_bgr_frames:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_path = tf.name
    try:
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            out_fps,
            (width, height),
        )
        if not writer.isOpened():
            logger.warning("SAM3BBoxStage: cv2.VideoWriter failed to open — skipping annotated video")
            return None
        for frame in annotated_bgr_frames:
            writer.write(frame)
        writer.release()
        return pathlib.Path(tmp_path).read_bytes()
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


def _run_sam3_on_clip(  # noqa: PLR0913  # inference helper, parameters mirror SAM3Model API
    sam3: SAM3Model,
    clip_mp4_bytes: bytes,
    prompts: list[str],
    target_fps: float,
    session_reset_s: float,
    *,
    write_annotated: bool,
    draw_trails: bool,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]], bytes | None]:
    """Run SAM3 pre-loaded chunked inference on a single clip.

    Returns:
        ``(objects_by_frame, instances, annotated_mp4_bytes_or_none)``.

    """
    rgb_frames, bgr_frames, source_indices, src_fps, out_fps, _step, width, height = _read_frames_from_bytes(
        clip_mp4_bytes,
        target_fps=target_fps,
    )

    if not rgb_frames:
        return {}, [], None

    chunk_size = max(1, int(session_reset_s * target_fps) if session_reset_s else len(rgb_frames))
    n_chunks = (len(rgb_frames) + chunk_size - 1) // chunk_size

    objects_by_frame: dict[int, list[dict[str, Any]]] = {}
    # SAM3 assigns object_ids fresh per session, so we namespace by
    # ``(chunk_idx, object_id)`` to avoid cross-chunk collisions.
    instances_map: dict[tuple[int, int], dict[str, Any]] = {}
    annotated_bgr: list[npt.NDArray[np.uint8]] = []
    trails: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)

    with torch.no_grad():
        session = None
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(rgb_frames))
            chunk_rgb = rgb_frames[start:end]
            chunk_bgr = bgr_frames[start:end]
            chunk_src_indices = source_indices[start:end]

            if session is not None:
                del session
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            session = sam3.processor.init_video_session(
                video=chunk_rgb,
                inference_device="cuda",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )
            for prompt in prompts:
                sam3.processor.add_text_prompt(session, prompt)

            for model_outputs in sam3.model.propagate_in_video_iterator(
                inference_session=session,
                show_progress_bar=False,
            ):
                processed = sam3.processor.postprocess_outputs(session, model_outputs)
                local_idx = model_outputs.frame_idx
                if local_idx >= len(chunk_bgr):
                    continue
                src_idx = chunk_src_indices[local_idx]

                detections = _postprocess_to_detections(processed, prompts)

                objects_by_frame[src_idx] = [det.to_json_dict() for det in detections]

                # Seconds since clip start (``src_idx`` indexes the ORIGINAL,
                # pre-subsampled video); rounded to ms for compact JSON.
                src_time_s = round(src_idx / src_fps, 3) if src_fps > 0 else 0.0
                for det in detections:
                    key = (chunk_idx, det.object_id)
                    entry = instances_map.setdefault(
                        key,
                        {
                            "object_id": det.object_id,
                            "prompt": det.prompt,
                            "start_time_s": src_time_s,
                            "end_time_s": src_time_s,
                            "num_frames": 0,
                        },
                    )
                    entry["end_time_s"] = src_time_s
                    entry["num_frames"] += 1
                    trails[det.object_id].append(det.center)

                if write_annotated:
                    annotated_bgr.append(
                        draw_frame(
                            chunk_bgr[local_idx],
                            detections,
                            prompts,
                            trails,
                            draw_trails=draw_trails,
                            current_time_s=src_time_s,
                        )
                    )

            trails.clear()

    # Chronological by start time, then object_id for stable output.
    instances = sorted(instances_map.values(), key=lambda e: (e["start_time_s"], e["object_id"]))
    annotated_bytes = _encode_annotated_video(annotated_bgr, out_fps, width, height) if write_annotated else None
    return objects_by_frame, instances, annotated_bytes


class SAM3BBoxStage(CuratorStage):
    """SAM3 object tracking stage producing per-clip bbox/instance metadata.

    Uses pre-loaded chunked inference (enables SAM3's hotstart heuristics for
    higher-quality tracks) and optionally draws annotated mp4 output. Consumes
    ``clip.encoded_data`` (post-transcode mp4 bytes) and populates ``Clip``'s
    SAM3 output fields.
    """

    def __init__(  # noqa: PLR0913  # flat config surface keeps CLI wiring straightforward
        self,
        prompts: list[str],
        *,
        target_fps: float = 10.0,
        max_clip_duration_s: float = 30.0,
        session_reset_s: float = 10.0,
        quality_config: SAM3QualityConfig | None = None,
        write_annotated_video: bool = False,
        draw_trails: bool = False,
        gpus_per_worker: float = 1.0,
        verbose: bool = False,
    ) -> None:
        """Initialise the stage.

        Args:
            prompts: Text descriptions of objects to track.
            target_fps: Sub-sampling rate applied to clip frames before inference.
            max_clip_duration_s: Safety rail — clips longer than this are skipped
                (GPU memory scales with clip length; the memory-bank grows per
                frame inside a session).
            session_reset_s: Chunk length in seconds. The SAM3 session is re-init'd
                between chunks to bound GPU memory.
            quality_config: Optional ``Sam3VideoConfig`` tuning knobs.
            write_annotated_video: If ``True``, emit an annotated mp4 per clip
                (boxes + masks + ids + optional trails) into ``clip.sam3_annotated_video``.
            draw_trails: If ``True`` and ``write_annotated_video`` is on, draw
                trajectory trails.
            gpus_per_worker: GPU fraction (default: one full GPU).
            verbose: Extra per-clip logging.

        """
        if not prompts:
            msg = "SAM3BBoxStage requires at least one prompt"
            raise ValueError(msg)
        self._prompts = prompts
        self._target_fps = target_fps
        self._max_clip_duration_s = max_clip_duration_s
        self._session_reset_s = session_reset_s
        self._quality_config = quality_config or SAM3QualityConfig()
        self._write_annotated_video = write_annotated_video
        self._draw_trails = draw_trails
        self._gpus_per_worker = gpus_per_worker
        self._verbose = verbose
        # Eager construct so ``self.model`` resolves when the pipeline builder
        # probes it; weights are loaded later in ``stage_setup``.
        self._sam3_model: SAM3Model = SAM3Model()

    @property
    def conda_env_name(self) -> str:
        """Return the pixi environment name for this stage."""
        return "sam3"

    @property
    def resources(self) -> CuratorStageResource:
        """Return resource requirements."""
        return CuratorStageResource(gpus=self._gpus_per_worker)

    @property
    def model(self) -> ModelInterface:
        """Return the underlying SAM3 model wrapper (weights loaded by ``stage_setup``)."""
        return self._sam3_model

    def stage_setup(self) -> None:
        """Load SAM3 with any configured overrides and log GPU memory."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._sam3_model.setup(config_overrides=self._quality_config.to_overrides())
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def _process_clip(self, clip: Clip) -> None:
        # Release allocator reservations between clips; without this,
        # fragmentation accumulates and clip N+1 can OOM even when clip N fit.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if clip.duration > self._max_clip_duration_s:
            logger.warning(
                f"[SAM3BBoxStage] clip {clip.uuid}: duration {clip.duration:.1f}s exceeds "
                f"max_clip_duration_s={self._max_clip_duration_s}s — skipping"
            )
            clip.errors["sam3_bbox"] = "clip_too_long"
            return

        mp4_data = clip.encoded_data.resolve()
        if mp4_data is None:
            logger.warning(f"[SAM3BBoxStage] clip {clip.uuid}: encoded_data missing — skipping")
            clip.errors["sam3_bbox"] = "missing_encoded_data"
            return

        mp4_bytes = mp4_data.tobytes()

        try:
            objects_by_frame, instances, annotated_bytes = _run_sam3_on_clip(
                self._sam3_model,
                mp4_bytes,
                self._prompts,
                target_fps=self._target_fps,
                session_reset_s=self._session_reset_s,
                write_annotated=self._write_annotated_video,
                draw_trails=self._draw_trails,
            )
        except Exception:  # noqa: BLE001
            clip.errors["sam3_bbox"] = "inference_error"
            logger.exception(f"[SAM3BBoxStage] clip {clip.uuid}: SAM3 inference failed")
            return

        clip.sam3_objects_by_frame = objects_by_frame
        clip.sam3_instances = instances
        if annotated_bytes is not None:
            clip.sam3_annotated_video = bytes_to_numpy(annotated_bytes)  # type: ignore[assignment]

        if self._verbose:
            num_frames = len(objects_by_frame)
            num_instances = len(instances)
            logger.info(
                f"[SAM3BBoxStage] clip {clip.uuid}: {num_frames} annotated frames, "
                f"{num_instances} instances, prompts={self._prompts}"
            )

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Run SAM3 on every clip of every video in ``tasks``."""
        for task in tasks:
            for video in task.videos:
                for clip in video.clips:
                    self._process_clip(clip)
        return tasks
