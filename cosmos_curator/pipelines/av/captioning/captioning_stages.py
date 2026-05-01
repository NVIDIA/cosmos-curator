# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Captioning stage."""

import re
from collections.abc import Collection
from typing import cast

from loguru import logger
from nvtx import nvtx  # type: ignore[import-untyped]

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.config.operation_context import is_running_on_the_cloud
from cosmos_curator.core.utils.data.ref_resolver import prefetch, resolve_as_ready
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.models import (
    qwen_vl,
    t5_encoder,
)
from cosmos_curator.models.chat_lm import ChatLM, make_chat_lm_input
from cosmos_curator.pipelines.av.utils.av_data_info import CAMERA_MAPPING
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
    CaptionWindow,
    ClipForAnnotation,
    append_captions_to_clips,
    get_clip_window_mappings,
    get_last_captions,
)
from cosmos_curator.pipelines.video.utils import windowing_utils

PROMPT_TYPES_FOR_T5_EMBEDDING = ["default"]
PROMPT_TYPES_FOR_ENHANCE_CAPTIONS = ["visibility", "road_conditions", "illumination"]
_PROMPT_TYPES_FOR_STAGE2_CAPTIONS = ["default"]

_DEFAULT_PROMPT = """
You are describing the driving video. Ensure the description is concise, precise, and focuses only on visible and verifiable details.

The description should include as many of the following elements as applicable:
1. Objects in the scene: Identify visible objects (e.g., cars, pedestrians, traffic lights, trees, houses).
2. Actions of objects: Highlight actions of objects in the scene (e.g., a car turning left, a pedestrian crossing the road).
3. Scene setting: Provide a general description of the environment (e.g., a busy street, a quiet neighborhood).
4. Weather conditions: Mention the visible weather (e.g., sunny, cloudy, rainy).
5. Time of day: Note the time of day based on the lighting (e.g., day, night).

Output:
[Your concise and informative description]

Please respond only in English and do not include Chinese words.
"""  # noqa: E501

_VRI_PROMPT = """
First, what is the visibility condition of the video?
        1. Clear
        2. Foggy
        3. Rainy
        4. Snowy
        5. Other
If you are unsure about the visibility condition, please classify it as Other.

Then, what is the road surface condition of the video?
        1. Dry
        2. Wet
        3. Snow-Ice
        4. Other
If you are unsure about the road surface condition, please classify it as Other.

Finally, what is the illumination condition of the video?
       1. Artificial: Scene primarily illuminated by artificial lighting sources such as streetlights, traffic signals, or indoor lights. Typically occurring during nighttime or within tunnels and parking structures. Please note that vehicle headlights are not considered as artificial lighting sources;
       2. Low: Naturally lit environment (e.g., dawn, dusk, heavy overcast) with limited natural lighting, resulting in dim visibility, muted colors, and reduced contrast, but without prominent artificial lighting;
       3. Bright: Daytime conditions with ample natural daylight;
       4. Dark: Very limited visibility with only car headlights, little or no illumination present (natural or artificial), making it challenging to identify objects or surroundings clearly. Usually associated with nighttime environments lacking sufficient artificial lighting;
       5. Other
If you are unsure about the illumination condition, please classify it as Other.

Please output your result in the following format:
       **Visibility Condition:** [Clear, Foggy, Rainy, Snowy, Other].
       **Road Surface Condition:** [Dry, Wet, Snow-Ice, Other].
       **Illumination Condition:** [Artificial, Low, Bright, Dark, Other].
"""  # noqa: E501

_VISIBILITY_PROMPT = "Describe the visibility condition in this video in detail."
_ROAD_CONDITIONS_PROMPT = "Describe the road slippery condition in this video in detail."
_ILLUMINATION_PROMPT = "Describe the illumination condition in this video in detail."

_ENHANCE_VISIBILITY_PROMPT = "Based on the visibility condition summary, what is the most likely visibility condition in the following categories: 1. Clear, 2. Rain, 3. Fog, 4. Snow, 5. Other? If you are unsure about the visibility condition, please classify it as Other. Please output your result in the following format: - **Visibility Condition:** [Clear, Rain, Fog, Snow, Other]."  # noqa: E501
_ENHANCE_ROAD_CONDITIONS_PROMPT = "Based on the summary, classify the road surface condition into the following categories: 1. Dry, 2. Wet, 3. Snow-Ice, 4. Other. If you are unsure about the road surface condition, please classify it as Other. Please output your result in the following format: - **Road Surface Condition:** [Dry, Wet, Snow-Ice, Other]."  # noqa: E501
_ENHANCE_ILLUMINATION_PROMPT = "Based on the illumination condition summary, what is the most likely illumination condition in the following categories: 1. Artificial, 2. Dark, 3. Low, 4. Bright, 5. Other? If you are unsure about the illumination condition, please classify it as Other. Please output your result in the following format: - **Illumination Condition:** [Artificial, Dark, Low, Bright, Other]."  # noqa: E501

_PROMPTS = {
    "default": _DEFAULT_PROMPT,
    "visibility": _VISIBILITY_PROMPT,
    "road_conditions": _ROAD_CONDITIONS_PROMPT,
    "illumination": _ILLUMINATION_PROMPT,
    "vri": _VRI_PROMPT,
}

_ENHANCE_PROMPTS = {
    "default": "",
    "visibility": _ENHANCE_VISIBILITY_PROMPT,
    "road_conditions": _ENHANCE_ROAD_CONDITIONS_PROMPT,
    "illumination": _ENHANCE_ILLUMINATION_PROMPT,
}

_ENHANCE_CAPTION_PREFIXES = {
    "visibility": "Here is video's visibility condition summary: ",
    "road_conditions": "Here is video's road slippery condition summary: ",
    "illumination": "Here is video's illumination condition summary: ",
}

# VRI: Visibility, Road Conditions, and Illumination
VRI_PROMPTS = {"visibility", "road_conditions", "illumination", "vri"}
VRI_PROMPTS_TO_DECODE = {"vri"}


def _decode_vri_text(caption_text: str, prompt_variant: str) -> dict[str, str]:
    vri_tags: dict[str, str] = {}
    if prompt_variant == "vri":
        text = " ".join(caption_text.splitlines()).lower().replace("*", "").replace(":", "")
        match = re.search(
            r"visibility condition(.+)road surface condition(.+)illumination condition(.+)",
            text,
        )

        if match is None:
            error = "Failed to decode VRI tags"
            raise ValueError(error)

        vri_tags["visibility"] = match.group(1).strip().rstrip(".")
        vri_tags["road_condition"] = match.group(2).strip().rstrip(".")
        vri_tags["illumination"] = match.group(3).strip().rstrip(".")

    return vri_tags


def _get_prompt(prompt_variant: str, prompt_text: str | None) -> str:
    prompt = ""
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _PROMPTS:
            error = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error)
        prompt = _PROMPTS[prompt_variant]
    return prompt


def _get_prompts(prompt_variants: list[str], prompt_text: str | None) -> dict[str, str]:
    if prompt_text is not None:
        return {"custom": prompt_text}

    return {prompt_variant: _get_prompt(prompt_variant, None) for prompt_variant in prompt_variants}


def is_vri_prompt(prompt_variant: str) -> bool:
    """Check if a prompt variant is a VRI prompt.

    Args:
        prompt_variant: The prompt variant to check.

    Returns:
        True if the prompt variant is a VRI prompt, False otherwise.

    """
    return prompt_variant in VRI_PROMPTS


def _get_frame_counts(
    prompt_variants: Collection[str],
    target_clip_size: int,
    front_window_size: int,
) -> list[int]:
    """Get frame counts based on prompt variants.

    Args:
        prompt_variants: Collection of prompt variant strings
            (e.g. "default", "visibility")
        target_clip_size: Target size for the clip (e.g. 256)
        front_window_size: Size of the front window (e.g. 57)

    Returns:
        List of frame counts to use for video processing. Return:
        - Empty list for empty input
        - [target_clip_size, front_window_size] for prompts including "default"
        - [target_clip_size] for non-default prompts only

    """
    if not prompt_variants:
        return []
    if "default" in prompt_variants:
        return [target_clip_size, front_window_size]
    return [target_clip_size]


def _filter_prompts(
    frame_count: int,
    prompts: dict[str, str],
    target_clip_size: int,
    front_window_size: int,
) -> dict[str, str]:
    """Filter prompts based on frame count and window size.

    This function determines which prompts to use based on the frame count.

    * If frame_count is equal to target_clip_size, all prompts are used.
    * If frame_count is equal to front_window_size, only the default prompt is
      used, if it is in the prompts dictionary.
    * Any other frame count is considered invalid.

    Args:
        frame_count: Number of frames in the current video clip
        prompts: Dictionary mapping prompt variants to their text
            (e.g. {"default": "prompt1", "visibility": "prompt2"})
        target_clip_size: Target size for the main clip (e.g. 256)
        front_window_size: Size of the front window (e.g. 57)

    Returns:
        dict[str, str]: Filtered dictionary of prompt variants -> prompt text

    Raises:
        ValueError: If frame_count is neither target_clip_size nor front_window_size

    """
    if frame_count == target_clip_size:
        _prompts = prompts
    elif frame_count == front_window_size:
        _prompts = {"default": prompts["default"]} if "default" in prompts else {}
    else:
        error = f"Bug: {frame_count=} must be one of {target_clip_size} or {front_window_size}"
        raise ValueError(error)
    return _prompts


class QwenInputPreparationStage(CuratorStage):
    """QwenInputPreparationStage class that prepares input for the Qwen language model.

    This class prepares input for the Qwen language model.
    """

    def __init__(  # noqa: PLR0913
        self,
        camera_format_id: str,
        model_variant: str = "qwen",
        target_clip_size: int = 256,
        front_window_size: int = 0,
        prompt_variants: list[str] | None = None,
        prompt_text: str | None = None,
        sampling_fps: float = 2.0,
        num_cpus_per_actor: float = 4.0,
        preprocess_dtype: str = "float16",
        *,
        model_does_preprocess: bool = False,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the QwenInputPreparationStage.

        Args:
            camera_format_id: The camera format ID.
            model_variant: The model variant.
            target_clip_size: The target clip size.
            front_window_size: The front window size.
            prompt_variants: The prompt variants.
            prompt_text: The prompt text.
            sampling_fps: The sampling FPS.
            num_cpus_per_actor: The number of CPUs per actor.
            preprocess_dtype: The preprocess dtype.
            model_does_preprocess: Whether the model does preprocess.
            keep_mp4: If True, preserve clip.encoded_data after processing.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        prompt_variants = ["default"] if prompt_variants is None else prompt_variants
        self._timer = StageTimer(self)
        self._target_clip_size = target_clip_size
        self._front_window_size = front_window_size
        self._prompts = _get_prompts(prompt_variants, prompt_text)
        self._sampling_fps = sampling_fps
        self._num_cpus_per_actor = num_cpus_per_actor
        self._preprocess_dtype = preprocess_dtype
        self._model_does_preprocess = model_does_preprocess
        self._keep_mp4 = keep_mp4
        self._verbose = verbose
        self._log_stats = log_stats
        self._qwen_utils = qwen_vl.QwenUtils(model_variant)
        self._flip_input = CAMERA_MAPPING[camera_format_id].get("flip_caption_input", {})

        if not self._prompts:
            error = "No prompts found"
            raise ValueError(error)

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(
            cpus=self._num_cpus_per_actor,
        )

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    def stage_setup(self) -> None:
        """Set up the QwenInputPreparationStage."""
        self._qwen_utils.setup()

    @nvtx.annotate("QwenInputPreparationStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[AvClipAnnotationTask]) -> list[AvClipAnnotationTask]:
        """Process the data.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed task.

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        prefetch([clip.encoded_data for clip in task.clips])
        for clip, data in resolve_as_ready([(clip, clip.encoded_data) for clip in task.clips]):
            if data is None:
                logger.warning(f"Clip {clip.uuid} has no buffer.")
                continue
            with self._timer.time_process():
                # Get full list of frame counts based on the list of prompt variants
                frame_counts = _get_frame_counts(
                    self._prompts.keys(),
                    self._target_clip_size,
                    self._front_window_size,
                )
                flip_input = self._flip_input.get(clip.camera_id, False)

                # Cache decoded video for reuse
                video_frames = {
                    frame_count: windowing_utils.split_video_into_windows(
                        data,
                        sampling_fps=self._sampling_fps,
                        model_does_preprocess=self._model_does_preprocess,
                        preprocess_dtype=self._preprocess_dtype,
                        flip_input=flip_input,
                        num_frames_to_use=frame_count,
                        return_bytes=False,
                    )[1][0]
                    for frame_count in frame_counts
                }

                for frame_count in frame_counts:
                    # Filter prompts based on frame count.
                    # When frame_count == front_window_size, only the default prompt is used.
                    _prompts = _filter_prompts(
                        frame_count,
                        self._prompts,
                        self._target_clip_size,
                        self._front_window_size,
                    )

                    if not _prompts:
                        logger.error(f"No prompts found for frame count {frame_count}?")
                        continue

                    caption_window = CaptionWindow(start_frame=0, end_frame=frame_count)

                    for prompt_variant, prompt in _prompts.items():
                        if self._verbose:
                            logger.debug(f"Generate vlm inputs for {clip.uuid} {prompt_variant=}: {prompt}")
                        qwen_llm_inputs = self._qwen_utils.generate_llm_inputs(
                            prompt=prompt,
                            video_inputs=video_frames[frame_count],
                        )

                        caption_window.model_input[prompt_variant] = qwen_llm_inputs

                    clip.caption_windows.append(caption_window)

                # Only clear encoded_data if not keeping it for dataset generation
                if not self._keep_mp4:
                    clip.encoded_data.drop()

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task


def qwen_vl_caption(
    clips: list[ClipForAnnotation],
    model: qwen_vl.QwenVL,
    prompt_variant: str,
    batch_size: int,
) -> list[str]:
    """Generate captions for a batch of video clips using the Qwen language model.

    Args:
        clips: List of video clips to generate captions for
        model: Qwen language model instance used for caption generation
        prompt_variant: Type of prompt to use (e.g. 'default', 'visibility', etc.)
        batch_size: Number of clips to process in each batch

    Returns:
        list[str]: Generated captions for the input clips

    Note:
        The function also updates the clips in-place by appending the generated
        captions to their caption_windows.

    """
    mappings = get_clip_window_mappings(clips, prompt_variant, skip_missing="model_input")
    inputs = [
        clips[clip_idx].caption_windows[window_idx].model_input[prompt_variant] for clip_idx, window_idx in mappings
    ]
    generate_stage2_caption = prompt_variant in _PROMPT_TYPES_FOR_STAGE2_CAPTIONS
    captions: list[str] = model.generate(
        inputs,
        generate_stage2_caption=generate_stage2_caption,
        batch_size=batch_size,
    )
    if prompt_variant not in VRI_PROMPTS:
        logger.info(f"Qwen captioned {len(captions)} {prompt_variant} windows; {generate_stage2_caption=}")

    append_captions_to_clips(clips, prompt_variant, captions, mappings)

    def _set_vri_tags(clip: ClipForAnnotation) -> None:
        try:
            clip.vri_tags = _decode_vri_text(clip.get_vri_caption_text(), prompt_variant)
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Failed to decode VRI text for clip {clip.uuid}, {prompt_variant=}, "
                f"{clip.get_vri_caption_text()=}: {e}"
            )
            return

    if prompt_variant in VRI_PROMPTS_TO_DECODE:
        for clip in clips:
            _set_vri_tags(clip)

    return captions


class QwenCaptionStage(CuratorStage):
    """QwenCaptionStage class that generates captions for a batch of video clips using the Qwen language model.

    This class generates captions for a batch of video clips using the Qwen language model.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen",
        prompt_variants: list[str] | None = None,
        batch_size: int = 16,
        fp8_enable: bool = False,  # noqa: FBT001, FBT002
        max_output_tokens: int = 8192,
        model_does_preprocess: bool = False,  # noqa: FBT001, FBT002
        disable_mmcache: bool = False,  # noqa: FBT001, FBT002
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the QwenCaptionStage.

        Args:
            model_variant: The model variant.
            prompt_variants: The prompt variants.
            batch_size: The batch size.
            fp8_enable: Whether to use FP8.
            max_output_tokens: The maximum output tokens.
            model_does_preprocess: Whether the model does preprocess.
            disable_mmcache: Whether to disable MMCache.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        self._timer = StageTimer(self)
        self._prompt_variants = ["default"] if prompt_variants is None else prompt_variants
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_stats = log_stats
        self._disable_mmcache = disable_mmcache
        self._raw_model = qwen_vl.QwenVL(
            model_variant,
            fp8=fp8_enable,
            max_output_tokens=max_output_tokens,
            model_does_preprocess=model_does_preprocess,
            disable_mmcache=self._disable_mmcache,
        )

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(
            gpus=1,
            cpus=1,
        )

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._raw_model

    @nvtx.annotate("QwenCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[AvClipAnnotationTask]) -> list[AvClipAnnotationTask]:
        """Process the data.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed task.

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        self._timer.reinit(self, task.get_major_size())

        with self._timer.time_process(len(task.clips)):
            try:
                for prompt_variant in self._prompt_variants:
                    qwen_vl_caption(task.clips, self._raw_model, prompt_variant, self._batch_size)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Qwen captioning failed: {e}")

            # Clean up the model inputs, which includes the decoded video frames
            for clip in task.clips:
                for window in clip.caption_windows:
                    window.model_input.clear()

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task


class T5Stage(CuratorStage):
    """T5Stage class that encodes captions using the T5 encoder.

    This class encodes captions using the T5 encoder.
    """

    def __init__(
        self,
        prompt_variants: list[str] | None = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the T5Stage.

        Args:
            prompt_variants: The prompt variants.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        self._timer = StageTimer(self)
        self._prompt_variants = ["default"] if prompt_variants is None else prompt_variants
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = t5_encoder.T5Encoder(t5_encoder.ModelVariant.T5_XXL)

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(
            gpus=1,
            cpus=1,
        )

    @nvtx.annotate("T5Stage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[AvClipAnnotationTask]) -> list[AvClipAnnotationTask]:
        """Process the data.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed task.

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        with self._timer.time_process(len(task.clips)):
            for prompt_variant in self._prompt_variants:
                if prompt_variant not in PROMPT_TYPES_FOR_T5_EMBEDDING:
                    continue

                all_prompts = []
                mapping: list[tuple[int, int]] = []

                for clip_idx, clip in enumerate(task.clips):
                    for window_idx, window in enumerate(clip.caption_windows):
                        captions = window.captions.get(prompt_variant, [])
                        if len(captions) == 0:
                            logger.error(f"Clip {clip.uuid} window-{window_idx} has no default caption.")
                            continue
                        mapping.append((clip_idx, window_idx))
                        all_prompts.append(captions[-1])

                batch_size = 16 if is_running_on_the_cloud() else 4

                if len(all_prompts) > 0:
                    # Encode all prompts at once
                    encoded_results = self._model.encode(all_prompts, batch_size=batch_size)

                    for idx, result in enumerate(encoded_results):
                        clip_idx, window_idx = mapping[idx]
                        window = task.clips[clip_idx].caption_windows[window_idx]
                        window.t5_xxl_embeddings[prompt_variant] = result.encoded_text

                    logger.info(f"T5 encoded {len(encoded_results)} {prompt_variant} captions.")
                else:
                    logger.warning(f"No {prompt_variant} captions to encode.")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return task


def _add_prefix_to_captions(
    captions: list[str],
    prompt_variant_key: str,
    prompt_prefixes: dict[str, str],
) -> list[str]:
    """Add a prefix to captions.

    Args:
        captions: List of captions to add a prefix to
        prompt_variant_key: Type of prompt, for example: (visibility,
            road_conditions,illumination, default)
        prompt_prefixes: Dictionary of prefixes to add to the captions,
            prompt_variant_key is used to choose the prefix

    Returns:
        List of modified captions

    """
    modified_captions: list[str] = []
    for caption in captions:
        modified_last_caption = f"{prompt_prefixes[prompt_variant_key]}{caption}"
        modified_captions.append(modified_last_caption)
    return modified_captions


def enhance_captions(  # noqa: PLR0913
    clips: list[ClipForAnnotation],
    model: ChatLM,
    caption_prefixes: dict[str, str],
    prompt_variant_key: str,
    prompt_variants: dict[str, str],
    prompt_text: str | None,
    batch_size: int | None = None,
) -> None:
    """Enhance captions for a list of clips using a chat language model.

    Modifies the list of clips by:

    1. Extracting the last caption in the chain
    2. Adding the chosen prefix based on the prompt variant
    3. Picks a new prompt based on the prompt variant or prompt text
    4. Passes the batch of clips to the chat LM to generate new captions
    5. Appends the new caption to each clip's caption chain

    Args:
        clips: List of clips containing caption windows
        model: Language model instance to generate enhanced captions
        caption_prefixes: Dictionary of prefixes to add to the captions.
            prompt_variant is used to choose the prefix
        prompt_variant_key: Type of prompt, for example: (visibility,
            road_conditions,illumination, default)
        prompt_variants: Dictionary of prompts to send to the model.
            prompt_variant_key is used to choose the prompt
        prompt_text: Text of the prompt to send to the model
        batch_size: Optional batch size hint passed to the model backend.

    Returns:
        None. The enhanced captions are appended directly to the clips'
            caption chains.

    """
    mappings = get_clip_window_mappings(clips, prompt_variant_key, skip_missing="captions")
    last_captions = get_last_captions(clips, prompt_variant_key, mappings)
    prefixed_captions = _add_prefix_to_captions(
        captions=last_captions,
        prompt_variant_key=prompt_variant_key,
        prompt_prefixes=caption_prefixes,
    )
    next_prompts = make_chat_lm_input(
        user_content=prefixed_captions,
        prompt_variant_key=prompt_variant_key,
        prompt_variants=prompt_variants,
        prompt_text=prompt_text,
    )
    enhanced_captions = model.generate(next_prompts, batch_size=batch_size)
    append_captions_to_clips(clips, prompt_variant_key, enhanced_captions, mappings)
    logger.info(f"Enhanced {len(enhanced_captions)} {prompt_variant_key} captions.")


class EnhanceCaptionStage(CuratorStage):
    """A CuratorStage wrapper class to the enhance_captions function.

    This stores the needed state and passes it to the function when
    process_data is called.

    This class is intentionally lightweight so that pipelines that do
    not use a AvClipAnnotationTask can still use the enhance_captions
    function.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen_lm",
        prompt_variants: list[str] | None = None,
        prompt_text: str | None = None,
        batch_size: int = 128,
        openai_model: str = "auto",
        fp8_enable: bool = False,  # noqa: FBT001, FBT002
        max_output_tokens: int = 2048,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the EnhanceCaptionStage.

        Args:
            model_variant: Which language model backend to use.
            prompt_variants: The prompt variants.
            prompt_text: The prompt text.
            batch_size: The batch size.
            openai_model: OpenAI model name (only used when model_variant is "openai").
            fp8_enable: Whether to use FP8 (only for local models).
            max_output_tokens: The maximum output tokens.
            verbose: If True, log verbose information.
            log_stats: If True, log statistics.

        """
        self._timer = StageTimer(self)
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_stats = log_stats

        quantization = "fp8" if fp8_enable else None
        self._raw_model = ChatLM(
            model_variant,
            max_output_tokens=max_output_tokens,
            quantization=quantization,
            openai_model=openai_model,
            verbose=verbose,
        )
        self._prompt_variants = ["default"] if prompt_variants is None else prompt_variants
        self._prompt_text = prompt_text

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return cast("ModelInterface", self._raw_model)

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        gpus = 1.0 if self._raw_model.requires_gpu else 0.0
        return CuratorStageResource(cpus=1.0, gpus=gpus)

    @nvtx.annotate("EnhanceCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[AvClipAnnotationTask]) -> list[AvClipAnnotationTask]:
        """Process the data.

        Args:
            tasks: The task to process.

        Returns:
            The processed task.

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        self._timer.reinit(self, task.get_major_size())
        with self._timer.time_process(len(task.clips)):
            for prompt_variant in self._prompt_variants:
                if prompt_variant not in PROMPT_TYPES_FOR_ENHANCE_CAPTIONS:
                    continue
                enhance_captions(
                    clips=task.clips,
                    model=self._raw_model,
                    caption_prefixes=_ENHANCE_CAPTION_PREFIXES,
                    prompt_variant_key=prompt_variant,
                    prompt_variants=_ENHANCE_PROMPTS,
                    prompt_text=self._prompt_text,
                    batch_size=self._batch_size,
                )
        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task
