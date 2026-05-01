# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Prompts for captioning."""

from loguru import logger

_PROMPTS = {
    "default": """
        Elaborate on the visual and narrative elements of the video in detail.
    """,
    "av": """
        The video depicts the view from a camera mounted on a car as it is driving.
        Pay special attention to the motion of the cars, including the primary car
        whose point-of-view we observe in the video. Also note important factors
        that would relate to driving safety like the relative positions of pedestrians,
        lane markers, road signs, traffic signals, and any aggressive driving behavior
        of other vehicles. Also pay attention to interesting landmarks and describe
        them in detail.
    """,
    "av-surveillance": """
        The video depicts the view from a surveillance camera. Pay special attention
        to the motion of the cars and other important factors that would relate to
        driving safety like the relative positions of pedestrians, lane markers,
        road signs, traffic signals, and any aggressive driving behavior of vehicles.
        Also pay attention to interesting landmarks and describe them in detail.
    """,
    "image": """
        Describe the image in detail.
    """,
}


_ENHANCE_PROMPTS = {
    "default": """
        You are a chatbot that enhances video caption inputs, adding more color and details to the text.
        The output should be longer than the provided input caption.
        Respond only with the enhanced caption; do not ask follow-up questions or offer additional assistance.
    """,
    "av-surveillance": """
        You are a chatbot that enhances video captions from vehicle dashboard cameras or surveillance cameras.
        Add more details and generate a summary from the original text.
        The output should be longer than the provided input caption.
    """,
}


_DEFAULT_STAGE2_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.
"""


def get_prompt(
    prompt_variant: str,
    prompt_text: str | None,
    *,
    verbose: bool = False,
) -> str:
    """Get the captioning prompt.

    Args:
        prompt_variant: The variant of the prompt.
        prompt_text: The text of the prompt.
        verbose: Whether to print the prompt.

    Returns:
        The captioning prompt.

    Raises:
        ValueError: If the prompt variant is invalid.

    """
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Captioning prompt: {prompt}")
    return prompt


def get_enhance_prompt(prompt_variant: str, prompt_text: str | None, *, verbose: bool = False) -> str:
    """Get the enhance captioning prompt.

    Args:
        prompt_variant: The variant of the prompt.
        prompt_text: The text of the prompt.
        verbose: Whether to print the prompt.

    Returns:
        The enhance captioning prompt.

    Raises:
        ValueError: If the prompt variant is invalid.

    """
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _ENHANCE_PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _ENHANCE_PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Enhance Captioning prompt: {prompt}")
    return prompt


def get_stage2_prompt(prompt: str | None) -> str:
    """Get the stage 2 prompt.

    Args:
        prompt: The text of the stage 2 prompt. If None, the default stage 2
            prompt will be used.

    Returns:
        The stage 2 prompt.

    """
    if prompt is not None:
        return prompt
    return _DEFAULT_STAGE2_PROMPT.strip() + "\n"


def build_refinement_prompt_text(
    processor: object,
    first_caption: str,
    refinement_instruction: str | None = None,
) -> str:
    """Build rendered prompt text for stage-2 caption refinement via ``apply_chat_template``."""
    instruction = get_stage2_prompt(refinement_instruction)
    final_text = instruction + first_caption
    messages: list[dict[str, str | list[dict[str, str]]]] = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": final_text},
            ],
        },
    ]
    return processor.apply_chat_template(  # type: ignore[attr-defined, no-any-return]
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
