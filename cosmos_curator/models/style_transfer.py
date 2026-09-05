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

"""Pluggable style-transfer model backends.

``StyleTransferModel`` is the seam the pipeline stage talks to. It extends
``ModelInterface`` (weight download + env handling) with a single ``generate``
call that maps a source clip + text prompt (+ optional spatial control) to
restyled mp4 bytes. New backends (a hosted API, a diffusers pipeline) implement
this same interface; the stage never changes.

The first backend is ``Cosmos3OmniTransferModel``, which drives **Cosmos3
Generator Transfer** in-process via vLLM-Omni's offline ``Omni`` API. The source
clip is handed to the Cosmos3 pipeline as ``multi_modal_data["video"]`` and the
transfer control (edge / blur) is selected through ``extra_args`` transfer
hints; vLLM-Omni derives edge/blur on-the-fly (or consumes a pre-computed
control video). Multi-GPU (Super) runs through vLLM-Omni's parallel config in
the same process -- no subprocess, no torchrun, no separate framework venv.

See the Cosmos3 recipe (transfer section) and
``vllm_omni/diffusion/models/cosmos3/transfer.py`` for the hint schema.
"""

import abc
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import attrs
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.model import model_utils

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

# Style-transfer variant -> HuggingFace weights id. Unlike the older subprocess
# backend, this is a normal HF model consumed in-process, so it is prefetched
# into the curator weight cache (see model_id_names) and loaded from there.
_VARIANT_TO_MODEL: dict[str, str] = {
    "cosmos3_nano": "nvidia/Cosmos3-Nano",
    "cosmos3_super": "nvidia/Cosmos3-Super",
}

# Cosmos3-Super is a 32B multi-GPU model; the cookbook runs it on 4+ GPUs. Nano
# is single-GPU.
_COSMOS3_SUPER_MIN_GPUS = 4


def style_transfer_variants() -> list[str]:
    """Return the style-transfer model variant keys usable as ``--style-transfer-model``."""
    return list(_VARIANT_TO_MODEL)


def clamp_num_gpus_for_variant(variant: str, num_gpus: int) -> int:
    """Clamp ``num_gpus`` up to the minimum a variant requires (Super needs 4+)."""
    if variant == "cosmos3_super" and num_gpus < _COSMOS3_SUPER_MIN_GPUS:
        logger.warning(
            f"cosmos3_super transfer requires at least {_COSMOS3_SUPER_MIN_GPUS} GPUs; "
            f"setting num_gpus to {_COSMOS3_SUPER_MIN_GPUS}"
        )
        return _COSMOS3_SUPER_MIN_GPUS
    return max(1, num_gpus)


@attrs.define(frozen=True)
class StyleTransferParams:
    """Backend-agnostic generation knobs for a single transfer call.

    These map onto vLLM-Omni's Cosmos3 transfer request but are named generically
    so other backends can consume the subset they support. Source geometry
    (``num_frames`` / ``fps``) is filled in per clip by the stage.
    """

    prompt: str
    negative_prompt: str
    control: str  # "edge" | "blur" (host- or on-the-fly computable)
    control_guidance: float
    guidance: float
    seed: int
    resolution: str
    fps: int
    num_frames: int
    num_video_frames_per_chunk: int
    num_conditional_frames: int
    edge_preset: str
    blur_preset: str


class TransferResult(NamedTuple):
    """Result of a single style-transfer generate() call.

    ``num_frames`` is the number of frames the backend actually generated (not the
    number of conditioning frames fed in), so callers can record true output length.
    """

    mp4_bytes: bytes
    num_frames: int


class StyleTransferModel(ModelInterface, abc.ABC):
    """Abstract backend for video style transfer: clip + prompt -> restyled mp4 bytes."""

    @abc.abstractmethod
    def generate(
        self,
        *,
        vision_frames: "npt.NDArray[np.uint8]",
        control_paths: dict[str, Path] | None,
        params: StyleTransferParams,
        work_dir: Path,
    ) -> TransferResult:
        """Run style transfer for one clip and return the restyled mp4 + frame count.

        Args:
            vision_frames: Decoded source frames as a ``(T, H, W, 3)`` uint8 RGB
                array (always provided; used as the video conditioning input and for
                on-the-fly control derivation).
            control_paths: Pre-computed control videos keyed by control type. When
                supplied for the active control, it is used instead of on-the-fly.
            params: Generation parameters.
            work_dir: Scratch directory for intermediate files and outputs.

        Returns:
            A ``TransferResult`` with the restyled mp4 bytes and the generated frame count.

        """


class Cosmos3OmniTransferModel(StyleTransferModel):
    """Cosmos3 Generator Transfer backend running in-process via vLLM-Omni.

    The constructor only stores config. ``setup()`` runs on a worker actor in the
    dedicated ``style-transfer`` environment and builds the in-process ``Omni``
    engine. ``generate()`` issues one transfer request per clip and returns the
    restyled mp4 bytes.
    """

    def __init__(self, variant: str = "cosmos3_nano", num_gpus: int = 1, *, guardrails: bool = False) -> None:
        """Initialize the Cosmos3 transfer backend.

        Args:
            variant: Style-transfer variant ('cosmos3_nano' or 'cosmos3_super').
            num_gpus: GPUs for the in-process engine (Super is clamped up to 4).
            guardrails: Enable Cosmos3 text/video safety guardrails (default False).
                vLLM-Omni loads them at engine build time and requires the
                ``cosmos-guardrail`` package (not in any env), so enabling them on the
                current build fails; opt in once that package is available (the NVIDIA
                Open Model License expects guardrails on).

        Raises:
            ValueError: If ``variant`` is unknown.

        """
        super().__init__()
        if variant not in _VARIANT_TO_MODEL:
            msg = f"Unknown style-transfer variant: {variant}. Choose from: {', '.join(_VARIANT_TO_MODEL)}"
            raise ValueError(msg)
        self._variant = variant
        self._num_gpus = clamp_num_gpus_for_variant(variant, num_gpus)
        self._guardrails = guardrails
        self._omni: Any | None = None

    @property
    def variant(self) -> str:
        """Return the style-transfer variant key."""
        return self._variant

    @property
    def model_hf_id(self) -> str:
        """Return the HuggingFace weights id for this variant."""
        return _VARIANT_TO_MODEL[self._variant]

    @property
    def conda_env_name(self) -> str:
        """Run in the dedicated environment that carries vLLM-Omni."""
        return "style-transfer"

    @property
    def model_id_names(self) -> list[str]:
        """Return HuggingFace ids for the curator prefetch (the Cosmos3 variant)."""
        return [self.model_hf_id]

    def setup(self) -> None:
        """Build the in-process vLLM-Omni engine on the worker actor.

        Loads the Cosmos3 weights from the curator cache and constructs an
        ``Omni`` engine with a tensor-parallel size matching the GPU count.
        """
        from vllm_omni.diffusion.data import DiffusionParallelConfig  # noqa: PLC0415
        from vllm_omni.entrypoints.omni import Omni  # noqa: PLC0415

        model_dir = model_utils.get_local_dir_for_weights_name(self.model_hf_id)
        if not model_dir.exists():
            msg = (
                f"Cosmos3 transfer weights not found at {model_dir}. Prefetch them with "
                f"`cosmos-curator model download` or let the framework auto-download via the model cache."
            )
            raise RuntimeError(msg)

        parallel_config = DiffusionParallelConfig(tensor_parallel_size=self._num_gpus)
        logger.info(
            f"Cosmos3OmniTransferModel setup: variant={self._variant} model={model_dir} "
            f"tensor_parallel_size={self._num_gpus} guardrails={self._guardrails}"
        )
        # ``model_config['guardrails']`` gates vLLM-Omni's Cosmos3 safety checker,
        # loaded eagerly at engine build (mirrors the serve CLI's --no-guardrails).
        # Enabled requires the ``cosmos-guardrail`` package; disabled skips it.
        self._omni = Omni(
            model=str(model_dir),
            parallel_config=parallel_config,
            model_config={"guardrails": self._guardrails},
            enforce_eager=False,
        )

    def generate(
        self,
        *,
        vision_frames: "npt.NDArray[np.uint8]",
        control_paths: dict[str, Path] | None,
        params: StyleTransferParams,
        work_dir: Path,  # noqa: ARG002 -- part of StyleTransferModel.generate contract; this in-process backend encodes mp4 in-memory and needs no scratch
    ) -> TransferResult:
        """Run Cosmos3 transfer for one clip via the in-process Omni engine."""
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: PLC0415
        from vllm_omni.outputs import OmniRequestOutput  # noqa: PLC0415

        if self._omni is None:
            msg = "Cosmos3OmniTransferModel.setup() must be called before generate()."
            raise RuntimeError(msg)

        extra_args = build_transfer_extra_args(params, control_paths)

        # Cosmos3 wants the conditioning video as decoded frames (THWC uint8 array),
        # not a path. Wrap it in a mapping so vLLM-Omni also reads the source fps
        # (it unwraps ``data`` and reads ``fps``). height/width are left unset:
        # Cosmos3 transfer re-derives them from the source aspect scaled to
        # extra_args["resolution"].
        prompt: dict[str, Any] = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "multi_modal_data": {"video": {"data": vision_frames, "fps": float(params.fps)}},
        }
        # ``fps`` (int) and ``frame_rate`` (float) are intentionally both set to the
        # same value: vLLM-Omni 0.28 accepts both aliases, and keeping both preserves
        # compatibility with older releases. Neither is dead code.
        sampling_params = OmniDiffusionSamplingParams(
            num_frames=params.num_frames,
            fps=params.fps,
            frame_rate=float(params.fps),
            guidance_scale=params.guidance,
            guidance_scale_provided=True,
            seed=params.seed,
            extra_args=extra_args,
        )

        outputs = self._omni.generate(prompt, sampling_params)
        output = outputs[0] if isinstance(outputs, list) else outputs
        if not isinstance(output, OmniRequestOutput):
            msg = f"Unexpected Omni output type: {type(output)!r}"
            raise TypeError(msg)

        frames = _extract_uint8_frames(output)
        mp4_bytes = _encode_mp4(frames, fps=params.fps)
        return TransferResult(mp4_bytes=mp4_bytes, num_frames=int(frames.shape[0]))

    def shutdown(self) -> None:
        """Gracefully shut down the in-process Omni engine (idempotent).

        Without this, vLLM-Omni's diffusion worker subprocess is reaped during
        interpreter teardown and logs a spurious "worker(s) died unexpectedly".
        """
        omni = self._omni
        if omni is None:
            return
        self._omni = None
        try:
            omni.shutdown()
        except Exception:  # noqa: BLE001 -- best-effort teardown; never fail the stage on cleanup
            logger.exception("Cosmos3OmniTransferModel shutdown failed")


def build_transfer_extra_args(
    params: StyleTransferParams,
    control_paths: dict[str, Path] | None,
) -> dict[str, Any]:
    """Build the vLLM-Omni Cosmos3 transfer ``extra_args`` dict for one control hint.

    The active control becomes a hint object; a pre-computed control video is
    passed via ``control_path`` (otherwise vLLM-Omni derives edge/blur from the
    source video, using the matching preset). Top-level keys mirror
    ``vllm_omni.diffusion.models.cosmos3.transfer.resolve_transfer_config``.

    Args:
        params: Generation parameters (control type, presets, guidance, chunking).
        control_paths: Optional pre-computed control videos keyed by control type.

    Returns:
        The ``extra_args`` mapping passed on ``OmniDiffusionSamplingParams``.

    """
    hint: dict[str, Any] = {}
    if control_paths and params.control in control_paths:
        hint["control_path"] = str(control_paths[params.control])
    if params.control == "edge":
        hint["preset_edge_threshold"] = params.edge_preset
    elif params.control == "blur":
        hint["preset_blur_strength"] = params.blur_preset

    return {
        params.control: hint,
        "control_guidance": params.control_guidance,
        "num_video_frames_per_chunk": params.num_video_frames_per_chunk,
        "num_conditional_frames": params.num_conditional_frames,
        "max_frames": params.num_frames,
        # Cosmos3 transfer sizes the output from the source aspect ratio scaled to
        # this resolution bucket (one of 256/480/704/720); it is read both at
        # preprocess and in the worker, so it must be present or a stray value leaks in.
        "resolution": params.resolution,
    }


# Array ranks used when normalizing vLLM-Omni diffusion output to THWC video.
_FRAME_NDIM = 3
_VIDEO_NDIM = 4
_BATCHED_VIDEO_NDIM = 5
# RGB / RGBA channel counts (a trailing/leading axis of this size is the color axis).
_COLOR_CHANNELS = (3, 4)


def _extract_uint8_frames(output: Any) -> "npt.NDArray[np.uint8]":  # noqa: ANN401 -- OmniRequestOutput imported lazily
    """Extract the generated video from an ``OmniRequestOutput`` as a (T, H, W, 3) uint8 array.

    vLLM-Omni diffusion outputs carry the video in ``images`` as a tensor / ndarray
    / list of frames; this normalizes the common shapes to uint8 RGB THWC.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    payload: Any = output.images
    # Unwrap pipeline wrappers / single-element containers down to the raw video.
    while isinstance(payload, list) and len(payload) == 1 and not _is_frame(payload[0]):
        payload = payload[0]
    if isinstance(payload, dict):
        payload = payload.get("frames") or payload.get("video") or payload.get("data")
    if payload is None:
        msg = "No video frames found in Omni transfer output."
        raise ValueError(msg)

    if isinstance(payload, torch.Tensor):
        arr = payload.detach().cpu().float().numpy()
    elif isinstance(payload, np.ndarray):
        arr = payload
    elif isinstance(payload, list):
        return np.stack([_frame_to_hwc_uint8(f) for f in payload], axis=0)
    else:
        msg = f"Unsupported Omni transfer video payload: {type(payload)!r}"
        raise TypeError(msg)

    if arr.ndim == _BATCHED_VIDEO_NDIM:  # (B, ...) -> drop batch
        arr = arr[0]
    if arr.ndim != _VIDEO_NDIM:
        msg = f"Expected a 4D video tensor, got shape {arr.shape}"
        raise ValueError(msg)
    arr = _normalize_to_thwc(arr)
    arr = arr[..., :3]
    return _to_uint8(arr)


def _normalize_to_thwc(arr: "npt.NDArray[Any]") -> "npt.NDArray[Any]":
    """Normalize a 4D video array to (T, H, W, C), transposing channel-first layouts."""
    import numpy as np  # noqa: PLC0415

    if arr.shape[0] in _COLOR_CHANNELS and arr.shape[-1] not in _COLOR_CHANNELS:
        return np.transpose(arr, (1, 2, 3, 0))
    if arr.shape[0] in _COLOR_CHANNELS and arr.shape[-1] in _COLOR_CHANNELS:
        # Both first and last dims look like a channel count (e.g. a width-3/4 video),
        # so the layout is ambiguous. Assume already-THWC; warn to surface it in debugging.
        logger.warning(f"Ambiguous channel axis for video tensor {arr.shape}; assuming (T, H, W, C) layout.")
    return arr


def _is_frame(value: Any) -> bool:  # noqa: ANN401 -- accepts arbitrary Omni payload items
    """Return True if ``value`` looks like a single frame (PIL / HWC array / CHW tensor)."""
    import numpy as np  # noqa: PLC0415
    import PIL.Image  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if isinstance(value, PIL.Image.Image):
        return True
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.ndim == _FRAME_NDIM
    return False


def _frame_to_hwc_uint8(frame: Any) -> "npt.NDArray[np.uint8]":  # noqa: ANN401 -- accepts arbitrary Omni frame item
    """Convert a single PIL/ndarray/tensor frame to an (H, W, 3) uint8 array."""
    import numpy as np  # noqa: PLC0415
    import PIL.Image  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if isinstance(frame, PIL.Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().float().numpy()
    if not isinstance(frame, np.ndarray):
        msg = f"Unsupported frame type: {type(frame)!r}"
        raise TypeError(msg)
    if frame.ndim == _FRAME_NDIM and frame.shape[0] in _COLOR_CHANNELS and frame.shape[-1] not in _COLOR_CHANNELS:
        frame = np.transpose(frame, (1, 2, 0))
    return _to_uint8(frame[..., :3])


def _to_uint8(arr: "npt.NDArray[Any]") -> "npt.NDArray[np.uint8]":
    """Scale a float [0,1]/[-1,1] or already-uint8/[0,255] array to uint8, preserving shape."""
    import numpy as np  # noqa: PLC0415

    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)
    # Float payloads may be [0,1], [-1,1], or already [0,255]. A max above ~1.5 can
    # only be the [0,255] range, so pass it through instead of scaling it to white.
    if arr.size and float(arr.max()) > 1.5:  # noqa: PLR2004 -- >1 means values are already in [0,255]
        return np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
    scaled = arr
    if scaled.size and float(scaled.min()) < 0.0:  # [-1, 1] -> [0, 1]
        scaled = scaled * 0.5 + 0.5
    return (np.clip(scaled, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _encode_mp4(frames: "npt.NDArray[np.uint8]", *, fps: int) -> bytes:
    """Encode a (T, H, W, 3) uint8 RGB array to mp4 bytes in-memory via PyAV.

    Uses GPU H.264 (``h264_nvenc``); the software ``libx264`` encoder isn't built
    into the image's ffmpeg, so nvenc is our H.264 path. Encoding here (rather than
    via ``imageio`` / ``export_to_video``) also means the image doesn't need the
    ``imageio-ffmpeg`` package, whose wheel bundles a prebuilt ffmpeg binary we don't
    ship. Falls back to software ``mpeg4`` if NVENC is unavailable at runtime (matches
    the SeedVR encoder in ``super_resolution/inference_seedvr2_window.py``).
    """
    import io  # noqa: PLC0415
    from fractions import Fraction  # noqa: PLC0415

    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if frames.ndim != _VIDEO_NDIM or int(frames.shape[-1]) != 3:  # noqa: PLR2004 -- (T, H, W, 3) RGB
        msg = f"_encode_mp4 expects (T, H, W, 3) uint8 RGB frames, got shape {frames.shape}"
        raise ValueError(msg)

    height, width = int(frames.shape[1]), int(frames.shape[2])
    rate = Fraction(fps).limit_denominator(1000)

    def _encode_with(codec_name: str) -> bytes:
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="mp4") as container:
            # add_stream widens to a Video|Audio|Subtitle union for a non-literal codec
            # name; we only ever request video codecs here.
            stream = cast("av.video.stream.VideoStream", container.add_stream(codec_name, rate=rate))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "p4", "tune": "hq"} if codec_name == "h264_nvenc" else {"qscale": "5"}
            for frame in frames:
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame, dtype=np.uint8), format="rgb24")
                for pkt in stream.encode(vf):
                    container.mux(pkt)
            for pkt in stream.encode(None):
                container.mux(pkt)
        return buf.getvalue()

    try:
        return _encode_with("h264_nvenc")
    except Exception as exc:  # noqa: BLE001 -- NVENC may be unavailable; fall back to software mpeg4
        logger.warning(f"h264_nvenc encode failed ({exc}); falling back to software mpeg4")
        return _encode_with("mpeg4")
