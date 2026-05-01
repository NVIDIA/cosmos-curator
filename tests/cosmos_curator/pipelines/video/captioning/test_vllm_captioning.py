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

"""Test vLLM captioning results."""

import pathlib
import uuid
from collections import Counter

import pytest
from scipy.spatial.distance import cosine

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION
from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import (
    VllmCaptionStage,
    VllmPrepStage,
)
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    Video,
    VllmConfig,
    VllmSamplingConfig,
    WindowConfig,
)  # type: ignore[import-untyped]

_THRESHOLDS = {
    "qwen": 0.9,
    "cosmos_r1": 0.75,
    "cosmos_r2": 0.7,
}
_NUM_CLIPS = 1
_VLLM_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "cosmos_r2": {
        "preprocess": True,
    },
}
_WINDOW_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "qwen": {
        "sampling_fps": 2.0,
        "model_does_preprocess": False,
    },
    "cosmos_r1": {
        "sampling_fps": 4.0,
        "model_does_preprocess": False,
    },
    "cosmos_r2": {
        "sampling_fps": 4.0,
        "model_does_preprocess": True,
    },
}
_EXPECTED_CAPTIONS: dict[str, list[str]] = {
    "qwen": [
        (
            "The video begins with a scene set in a snowy, mountainous environment. The atmosphere is cold and harsh, "
            "with snow covering the ground and mountains in the background. A character, dressed in rugged, "
            "survivalist attire, is seen walking through this treacherous landscape. The character appears to be "
            "equipped for extreme conditions, holding a long stick or staff, which suggests they might be using it "
            "for navigation or defense against potential threats. The overall tone of the scene is one of isolation "
            "and resilience, as the character navigates through the challenging terrain.\n"
            "\n"
            'The scene then transitions to a black screen with white text that reads "THE BLENDER FOUNDATION '
            'presents." This indicates that the video is likely a production by Blender Foundation, a non-profit '
            "organization known for its contributions to open-source software and animation projects. The transition "
            "from the outdoor scene to the black screen with text serves as a clear demarcation between different "
            "segments of the video, possibly signaling the end of one part and the beginning of another.\n"
            "\n"
            "Following the text screen, the video shifts to an indoor setting where a character with a distinctive "
            "appearance is shown. This character has a long beard and mustache, adorned with various accessories such "
            "as earrings and a headband. The character's attire includes a textured garment, suggesting a historical "
            "or fantasy context. The background features intricate designs, adding to the rich and detailed "
            "environment. The lighting in this scene is warm and focused, highlighting the character's facial features "
            "and the texture of their clothing. The overall mood of this segment is more intimate and personal, "
            "contrasting with the earlier outdoor scene's sense of adventure and survival.\n"
            "\n"
            "In summary, the video starts with a character navigating a snowy, mountainous landscape, emphasizing "
            "themes of survival and resilience. It then transitions to a title card by the Blender Foundation, marking "
            "a shift in content. Finally, the video moves indoors, focusing on a character with a detailed and ornate "
            "appearance, set against a backdrop of intricate designs, creating a more intimate and detailed narrative."
        ),
    ],
    "cosmos_r1": [
        (
            "The video unfolds through a series of distinct yet interconnected scenes, each rich in visual and "
            "narrative detail:\n"
            "\n"
            "### **Visual Elements**  \n"
            "1. **Opening Scene: Snowy Mountains**  \n"
            "   - **Cool Tones & Harsh Environment**: The icy blue and white palette dominates, evoking isolation "
            "and danger. Jagged mountain peaks and swirling fog amplify the sense of vastness and "
            "unpredictability.  \n"
            "   - **Character Design**: The young woman's rugged attire (fur-lined jacket, utility belt) and staff "
            "suggest preparedness for survival or adventure. Her determined expression and steady posture contrast "
            "with the chaotic backdrop, emphasizing her resolve.  \n"
            "   - **Camera Work**: A slow zoom-in and upward pedestal movement track toward her, heightening "
            "tension and focus on her as the central figure.  \n"
            "\n"
            "2. **Transition to Dark Environment**  \n"
            "   - **Lighting Shift**: Daylight fades to artificial, dim lighting, creating a sudden tone shift from "
            "hope to uncertainty. Tribal accessories and staff on the second character introduce mystique or "
            "cultural significance.  \n"
            "   - **Mood & Symbolism**: The darkened setting and close-up on the second character's concerned "
            "expression suggest impending conflict or revelation, deepening the narrative's emotional stakes.  \n"
            "\n"
            "3. **Dimly Lit Room Scene**  \n"
            "   - **Warm, Intimate Lighting**: Contrasting cool tones with golden hues, the room's shadows and "
            "sword hint at danger or hidden history. The red-haired woman's tattoos and warrior-like appearance "
            "signal combat readiness or a protective role.  \n"
            "   - **Close-Up Dynamics**: Subtle facial expressions and the camera's focus on her eyes convey inner "
            "turmoil or resolve, while the ornate background adds layers of world-building.  \n"
            "\n"
            "### **Narrative Progression**  \n"
            "- **Character Development**: Each scene introduces a protagonist with distinct traits (determination, "
            "wisdom, combat readiness), suggesting a collective journey or shared mission.  \n"
            "- **Environmental Storytelling**: The snowy wilderness, tribal artifacts, and mysterious room imply a "
            "fantasy or survival narrative, possibly involving magic, survival, or hidden knowledge.  \n"
            "- **Tension & Pacing**: Slow transitions and deliberate framing build suspense, guiding the viewer "
            "through a character-driven story with high stakes.  \n"
            "\n"
            "### **Symbolism & Themes**  \n"
            "- **Staffs & Accessories**: Tools of guidance or power, tying the characters to a larger mythic or "
            "cultural narrative.  \n"
            "- **Light vs. Shadow**: Represents clarity versus mystery, danger versus safety, reflecting the "
            "characters' psychological and physical challenges.  \n"
            "\n"
            "In sum, the video crafts a rich, atmospheric tale centered on resilience and mystery, using visual "
            "storytelling to draw viewers into a world where survival, wisdom, and inner strength converge."
        ),
    ],
    "cosmos_r2": [
        (
            "The video opens with a character traversing a harsh, snowy mountainous landscape under a thick fog. "
            "Dressed in rugged outdoor gear and gripping a long wooden staff, the individual navigates the desolate "
            "terrain, conveying a sense of determination and resilience against the unforgiving environment. As the "
            "camera zooms in, the character's focused expression and partially covered face (via a scarf) highlight "
            "their preparedness for the extreme conditions. The misty backdrop of snow-capped peaks reinforces the "
            "isolation and challenge of the setting.\n"
            "\n"
            'The scene transitions to a black screen displaying the text "THE BLENDER FOUNDATION presents," '
            "signaling a connection to the organization known for its open-source 3D software. This interlude "
            "serves as a title card, likely introducing a project or animation created using Blender.\n"
            "\n"
            "Next, the video shifts to an older man with a distinctive appearance\u2014long beard, facial piercings, "
            "and traditional attire. He appears to be speaking or reacting, possibly narrating a story or offering "
            "commentary. His expressive demeanor adds depth to the narrative, suggesting a cultural or historical "
            "context tied to the preceding snowy scene.\n"
            "\n"
            "Finally, the focus narrows to a woman with red hair, shown in close-up. Her intense gaze and detailed "
            "clothing imply she is deeply engaged in the unfolding events, perhaps as a listener or participant in "
            "the broader story. The sequence collectively builds a narrative of exploration, discovery, and cultural "
            "richness within a visually immersive world crafted with meticulous attention to detail."
        ),
    ],
}


def tf_vector(sentence: str) -> dict[str, float]:
    """Compute term frequency vector for a sentence.

    Args:
        sentence: Sentence to compute term frequency vector for.

    Returns:
        Term frequency vector for the sentence.

    """
    words = sentence.lower().split()
    tf = Counter(words)
    total = sum(tf.values())
    return {word: count / total for word, count in tf.items()}


def cosine_similarity(s1: str, s2: str) -> float:
    """Compute cosine similarity between two sentences.

    Args:
        s1: First sentence.
        s2: Second sentence.

    Returns:
        Cosine similarity between the two sentences.

    """
    tf1 = tf_vector(s1)
    tf2 = tf_vector(s2)
    all_words = list(set(tf1.keys()) | set(tf2.keys()))

    v1 = [tf1.get(w, 0) for w in all_words]
    v2 = [tf2.get(w, 0) for w in all_words]

    return 1 - cosine(v1, v2)  # type: ignore[no-any-return]


@pytest.fixture
def sample_captioning_task(sample_clip_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample captioning task."""
    clip = Clip(
        uuid=uuid.uuid5(uuid.NAMESPACE_URL, "sample_clip.mp4#0.0-10.0"),
        source_video="sample_clip.mp4",
        span=(0.0, 10.0),
        encoded_data=sample_clip_data,
    )

    video = Video(
        input_video=pathlib.Path("sample_clip.mp4"),
        clips=[clip],
    )
    return SplitPipeTask(
        session_id="test-session",
        video=video,
    )


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", ["qwen", "cosmos_r1", "cosmos_r2"])
def test_vllm_caption_generation(
    sample_captioning_task: SplitPipeTask, sequential_runner: RunnerInterface, model_variant: str
) -> None:
    """Test the vLLM captioning result."""
    vllm_config = VllmConfig(
        model_variant=model_variant,
        sampling_config=VllmSamplingConfig(temperature=0.0, max_tokens=2048),
        **_VLLM_CONFIG_OVERRIDES.get(model_variant, {}),
    )
    window_config = WindowConfig(
        **_WINDOW_CONFIG_OVERRIDES.get(model_variant, {}),
    )
    stages = [
        VllmPrepStage(vllm_config=vllm_config, window_config=window_config),
        VllmCaptionStage(vllm_config=vllm_config),
    ]
    tasks = run_pipeline([sample_captioning_task], stages, runner=sequential_runner)

    # Validate that captions were generated
    assert tasks is not None
    assert len(tasks) > 0
    assert len(tasks[0].video.clips) == _NUM_CLIPS

    expected_captions = _EXPECTED_CAPTIONS[model_variant]

    for clip_idx, clip in enumerate(tasks[0].video.clips):
        assert len(clip.windows) > 0, f"Clip {clip_idx} should have at least one window"
        assert len(expected_captions) == len(clip.windows), (
            f"Clip {clip_idx} should have {len(expected_captions)} windows for model {model_variant}"
        )

        for window_idx, window in enumerate(clip.windows):
            assert model_variant in window.caption, (
                f"Clip {clip_idx} window {window_idx} should have {model_variant} caption"
            )

            generated_caption = window.caption[model_variant]
            assert generated_caption.strip(), (
                f"Clip {clip_idx} window {window_idx} should have non-empty {model_variant} caption"
            )

            expected_caption = expected_captions[window_idx]
            similarity = cosine_similarity(generated_caption, expected_caption)
            threshold = _THRESHOLDS[model_variant]
            assert similarity >= threshold, (
                f"Caption similarity {similarity:.3f} below threshold for clip {clip_idx} window {window_idx}: "
                f"[{generated_caption}] vs. [{expected_caption}]"
            )


@pytest.mark.env("unified")
def test_vllm_caption_regression_signals(
    sample_captioning_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
) -> None:
    """Smoke-test rollout caption health signals on one real vLLM inference path."""
    vllm_config = VllmConfig(
        model_variant="qwen",
        prompt_variant="default",
        prompt_text=None,
        fp8=False,
        preprocess=False,
        stage2_caption=False,
        sampling_config=VllmSamplingConfig(
            temperature=0.1,
            top_p=0.001,
            top_k=0,
            repetition_penalty=1.05,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            min_p=0.0,
            min_tokens=16,
            max_tokens=8192,
        ),
    )
    window_config = WindowConfig(
        sampling_fps=2.0,
        window_size=256,
        remainder_threshold=128,
        model_does_preprocess=False,
    )
    stages = [
        VllmPrepStage(vllm_config=vllm_config, window_config=window_config),
        VllmCaptionStage(vllm_config=vllm_config),
    ]
    tasks = run_pipeline([sample_captioning_task], stages, runner=sequential_runner)

    assert tasks is not None
    assert len(tasks) == 1

    clips = tasks[0].video.clips
    assert len(clips) == 1

    windows = clips[0].windows
    assert len(windows) == 1

    window = windows[0]
    assert "qwen" in window.caption

    caption = window.caption["qwen"]
    assert caption.strip()
    assert caption != VLLM_UNKNOWN_CAPTION
    assert window.caption_status == "success"
    assert window.caption_failure_reason is None

    assert "qwen" in window.token_counts
    token_counts = window.token_counts["qwen"]
    assert token_counts.prompt_tokens > 0
    assert token_counts.output_tokens > 0
