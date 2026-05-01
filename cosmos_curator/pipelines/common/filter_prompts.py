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

"""Shared prompts, categories, and filter criteria for media filtering and classification."""

from loguru import logger

# 27 shared media type categories (underscores, no spaces) for CLI-safe single-token arguments.
MEDIA_TYPE_LABELS: tuple[str, ...] = (
    "movie/film_scene",
    "tv_drama/tv_series",
    "animation",
    "cartoon",
    "video_game",
    "news_reporting",
    "talk_show",
    "product_review/unboxing",
    "how_to/demo",
    "digital_illustration/digital_text",
    "visual_pattern",
    "egocentric_video/walking_POV",
    "vehicle_POV",
    "aerial_footage",
    "travel/luggage/plane",
    "food/drink",
    "cooking",
    "sports/exercise",
    "clothing/fashion/makeup",
    "exterior_building/cityscape",
    "interior_spaces",
    "nature_environment",
    "animal/insect/wildlife",
    "pets",
    "person/crowd",
    "talking_head",
    "other",
)

VIDEO_TYPE_LABELS: tuple[str, ...] = MEDIA_TYPE_LABELS
IMAGE_TYPE_LABELS: tuple[str, ...] = MEDIA_TYPE_LABELS

# Default filter criteria per prompt variant. For "type", use the shared media type labels.
FILTER_CRITERIA: dict[str, list[str]] = {
    "default": [
        "slideshow",
        "synthetic video",
        "visual filter",
        "post-production text",
        "video in video",
        "photographic artifacts",
        "distorted view",
        "rotated view",
        "low resolution",
        "blurred region",
        "abrupt camera motion",
        "fast zoom",
        "unnatural speed",
    ],
    "type": list(MEDIA_TYPE_LABELS),
}

IMAGE_FILTER_CRITERIA: dict[str, list[str]] = {
    "default": [
        "synthetic image",
        "visual filter",
        "post-production text",
        "image in image",
        "photographic artifacts",
        "distorted view",
        "rotated view",
        "low resolution",
        "blurred region",
    ],
    "type": list(MEDIA_TYPE_LABELS),
}

_VIDEO_DEFAULT_PROMPT = """Can you answer the following questions about this video:
    1. Is this a slideshow (e.g., only showing static images, animated text / image, etc.)
    or a video with slide transitions?
    2. Is this a synthetic video (e.g., screen recording, motion graphics, AI-generated video,
    stop motion, slideshow, etc.)
    as opposed to a video captured by an optical camera sensor?
    3. Does this video have visual filters (e.g., editing properties
    like grain, noise, saturation, color, aliasing, simulated weather effect, etc.)?
    4. Does this video have text overlaid in post-production (e.g., watermark, subtitles, logo, graphics, etc.)?
    Text that is part of the original video content is not considered as post-production text.
    5. Is this video a video in video (e.g., video overlay/collage, etc.)?
    6. Does this video have bad photographic artifacts(e.g., over/under exposure, lens flare, poor focus, etc.)?
    7. Does this video have distorted view (e.g., fisheye effect form wide field of view)?
    8. Is the video rotated to an uncommon view?
    9. Does this video have low resolution or is it blurry for all or some of the frames?
    10. Does this video contain any blurred or pixelated region
    (e.g., on specific human faces or objects, background, logo region, etc.)?
    11. Does this video mainly involve camera movement and little scene dynamics?
    12. Does this video contain abrupt / very large camera motion or camera shake (e.g., in some
    frames the camera is moving or rotated so fast that you cannot see clearly what's happening).
    13. Does this video involve fast zoom in or zoom out of the camera?
    14. Does this video have unnatural speed (e.g., slow motion, time lapse, frame skipping, etc.)?
    Answer format:
    {
    "slideshow": "yes" or "no",
    "synthetic video": "yes" or "no",
    "visual filter": "yes" or "no",
    "post-production text": "yes" or "no",
    "video in video": "yes" or "no",
    "photographic artifacts": "yes" or "no",
    "distorted view": "yes" or "no",
    "rotated view": "yes" or "no",
    "low resolution": "yes" or "no",
    "blurred region": "yes" or "no",
    "little scene dynamics": "yes" or "no",
    "abrupt camera motion": "yes" or "no",
    "fast zoom": "yes" or "no",
    "unnatural speed": "yes" or "no"
}"""


def _format_answer_schema(categories: list[str]) -> str:
    lines = [f'"{category}": "yes" or "no"' for category in categories]
    return "{\n" + ",\n".join(lines) + "\n}"


def _build_media_questions_prompt(categories: list[str]) -> str:
    prompt = "Can you answer the following questions about this media:\n"
    for index, category in enumerate(categories, start=1):
        prompt += f"{index}. Is there {category} in this media?\n"
    prompt += f"\nAnswer format:\n{_format_answer_schema(categories)}"
    return prompt


def _build_media_type_prompt(categories: list[str]) -> str:
    prompt = (
        "Classify this media into the following categories. For each category, answer yes or no "
        "depending on whether the media content matches that category. "
        "Please provide text in the specified json format.\n\n"
    )
    prompt += "\n".join(f"Is this media: {category}?" for category in categories)
    prompt += f"\n\nAnswer format:\n{_format_answer_schema(categories)}"
    prompt += "\n\nPlease provide text in the specified json format.\n"
    return prompt


def _parse_filter_categories(filter_categories: str | None) -> tuple[list[str] | None, bool]:
    if filter_categories is None:
        return None, True
    try:
        categories = [category.strip() for category in filter_categories.split(",") if category.strip()]
    except AttributeError:
        return None, False
    return (categories or None), True


def get_media_filter_prompt(
    prompt_variant: str,
    filter_categories: str | None,
    *,
    criteria_by_variant: dict[str, list[str]],
    verbose: bool = False,
    log_label: str = "Filtering",
) -> str:
    """Return a shared media filtering/classification prompt."""
    categories, categories_are_valid = _parse_filter_categories(filter_categories)
    if categories is not None:
        prompt = _build_media_questions_prompt(categories)
    else:
        if prompt_variant not in criteria_by_variant:
            msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(msg)
        categories = criteria_by_variant[prompt_variant]
        prompt = (
            _build_media_type_prompt(categories)
            if prompt_variant == "type"
            else _build_media_questions_prompt(categories)
        )
    if filter_categories is not None and not categories_are_valid:
        logger.warning(f"Prompt text is not a comma separated list: {filter_categories}")
        prompt = _build_media_questions_prompt(criteria_by_variant["default"])
    if verbose:
        logger.debug(f"{log_label} prompt: {prompt}")
    return prompt


def get_qwen_filter_prompt(
    prompt_variant: str,
    filter_categories: str | None,
    *,
    verbose: bool = False,
) -> str:
    """Return the video-facing wrapper around the shared media prompt builder.

    Args:
        prompt_variant: One of the supported prompt variants (for example ``"default"`` or ``"type"``).
        filter_categories: Optional comma-separated categories for custom prompt.
        verbose: Log the chosen prompt.

    Returns:
        The prompt string.

    """
    if filter_categories is None and prompt_variant == "default":
        if verbose:
            logger.debug(f"Video filtering prompt: {_VIDEO_DEFAULT_PROMPT}")
        return _VIDEO_DEFAULT_PROMPT
    return get_media_filter_prompt(
        prompt_variant,
        filter_categories,
        criteria_by_variant=FILTER_CRITERIA,
        verbose=verbose,
        log_label="Video filtering",
    )


def get_image_filter_prompt(
    prompt_variant: str,
    filter_categories: str | None,
    *,
    verbose: bool = False,
) -> str:
    """Return the image-facing wrapper around the shared media prompt builder.

    Args:
        prompt_variant: One of the supported prompt variants (for example ``"default"`` or ``"type"``).
        filter_categories: Optional comma-separated categories for custom prompt.
        verbose: Log the chosen prompt.

    Returns:
        The prompt string.

    """
    return get_media_filter_prompt(
        prompt_variant,
        filter_categories,
        criteria_by_variant=IMAGE_FILTER_CRITERIA,
        verbose=verbose,
        log_label="Image filtering",
    )
