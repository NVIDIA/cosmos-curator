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
from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name
from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
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
    "qwen3_5_27b": 0.8,
}
_NUM_CLIPS = 1
_VLLM_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "cosmos_r2": {
        "preprocess": True,
    },
}
# Per-variant cap on generated tokens. Reasoning variants (Qwen3.5+) need the
# larger production-default headroom (8192) so the chain-of-thought plus the
# answer fit; otherwise the window is marked failed.
# We increase this to avoid flaky test failures.
_SAMPLING_MAX_TOKENS: dict[str, int] = {
    "qwen3_5_27b": 8192,
}
_DEFAULT_SAMPLING_MAX_TOKENS = 2048
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
    "qwen3_5_27b": {
        "sampling_fps": 2.0,
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
    "qwen3_5_27b": [
        (
            "The video presents a sequence of animated scenes that appear to be from a fantasy short film, "
            "characterized by high-quality 3D rendering.\n"
            "\n"
            "**Visual Elements:**\n"
            "\n"
            "*   **00:00 - 00:04 (The Journey):** The video opens with a wide, atmospheric shot of a harsh, wintry "
            "landscape. Jagged, snow-capped mountains loom in the background, obscured by thick, grey fog. In the "
            "foreground, a young woman with short, reddish-brown hair walks through the deep snow. She is dressed "
            "in earth-toned clothing—a brown tunic and a dark scarf wrapped around her lower face to protect "
            "against the cold. She carries a long wooden staff over her shoulder, suggesting she is a traveler or "
            "a warrior. Her arms bear visible tattoos. The lighting is flat and cool, emphasizing the freezing "
            "temperature and isolation.\n"
            '*   **00:04 - 00:06 (Title Card):** The screen fades to black, displaying the text "THE BLENDER '
            'FOUNDATION presents" in a classic white serif font. This identifies the production company behind '
            "the animation.\n"
            "*   **00:07 - 00:09 (The Elder):** The scene cuts to a dimly lit interior, likely a cave or a rustic "
            "hut. We see a close-up of an older man with a weathered face, a thick grey beard, and a headband "
            "adorned with metal rings. He has distinctive piercings in his nose and cheeks. The lighting here is "
            "warm and directional, resembling firelight, which contrasts sharply with the cold blue tones of the "
            "opening scene. He appears to be speaking or listening intently.\n"
            "*   **00:09 - 00:10 (The Reaction):** The camera cuts to a close-up of the young woman from the "
            "first scene (Sintel). She is now inside the same warm environment. Her expression is one of concern, "
            "worry, or perhaps sadness. Her mouth is slightly open as if she is about to speak or has just heard "
            "something troubling. The warm light highlights the texture of her skin and hair.\n"
            "\n"
            "**Narrative Elements:**\n"
            "\n"
            "*   **Contrast of Environments:** The video establishes a strong narrative contrast between the "
            "dangerous, cold outside world and the safe, warm interior. This suggests a journey where the "
            "protagonist has sought refuge or is visiting someone important.\n"
            "*   **Character Dynamics:** The juxtaposition of the young, determined traveler and the older, "
            "wise-looking man suggests a mentor-student relationship or a significant encounter. The man's tribal "
            "appearance (jewelry, piercings) hints at a specific culture or setting within the story.\n"
            "*   **Emotional Arc:** The sequence moves from physical endurance (walking in the snow) to an "
            "emotional beat. The final shot of the woman's worried face implies that the conversation or "
            "situation inside is serious, raising questions about what she has found or what she is facing. The "
            "narrative seems to pivot from an external adventure to an internal or interpersonal conflict."
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


_GOLDEN_CAPTION_PLACEHOLDER = "__PLACEHOLDER_CAPTURE_FROM_FIRST_RUN__"

# Variants that ``cosmos_curate.core.managers.model_cli`` excludes from its
# default download set (see ``_get_default_models``). For these variants the
# weights must be downloaded explicitly before this test runs, e.g.::
#
#     pixi run -e model-download python -m cosmos_curate.core.managers.model_cli \
#         download --models <variant>
_VARIANTS_NOT_DOWNLOADED_BY_DEFAULT: frozenset[str] = frozenset({"qwen3_5_27b"})


def _skip_if_weights_missing(model_variant: str) -> None:
    """Skip cleanly when the variant's weights are not present locally.

    ``AutoProcessor.from_pretrained`` falls back to treating a missing local
    path as a Hugging Face repo id, which then fails ``validate_repo_id`` with
    a noisy ``HFValidationError``. Guarding here gives a much clearer signal.
    """
    weights_dir = get_local_dir_for_weights_name(get_vllm_model_id(model_variant))
    if weights_dir.is_dir():
        return
    hint = ""
    if model_variant in _VARIANTS_NOT_DOWNLOADED_BY_DEFAULT:
        hint = (
            f" (variant {model_variant!r} is excluded from the default model-download set; "
            f"run `pixi run -e model-download python -m cosmos_curate.core.managers.model_cli "
            f"download --models {model_variant}`)"
        )
    pytest.skip(f"Model weights for {model_variant} not found at {weights_dir}{hint}")


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", ["qwen", "cosmos_r1", "cosmos_r2", "qwen3_5_27b"])
def test_vllm_caption_generation(
    sample_captioning_task: SplitPipeTask, sequential_runner: RunnerInterface, model_variant: str
) -> None:
    """Test the vLLM captioning result."""
    _skip_if_weights_missing(model_variant)
    vllm_config = VllmConfig(
        model_variant=model_variant,
        sampling_config=VllmSamplingConfig(
            temperature=0.0,
            max_tokens=_SAMPLING_MAX_TOKENS.get(model_variant, _DEFAULT_SAMPLING_MAX_TOKENS),
        ),
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
            assert generated_caption != VLLM_UNKNOWN_CAPTION, (
                f"Clip {clip_idx} window {window_idx} returned the {model_variant} unknown-caption sentinel"
            )
            assert window.caption_status == "success", (
                f"Clip {clip_idx} window {window_idx} caption_status={window.caption_status!r} "
                f"for {model_variant}, expected 'success'"
            )
            assert window.caption_failure_reason is None, (
                f"Clip {clip_idx} window {window_idx} caption_failure_reason="
                f"{window.caption_failure_reason!r} for {model_variant}, expected None"
            )

            assert model_variant in window.token_counts, (
                f"Clip {clip_idx} window {window_idx} should have {model_variant} token_counts"
            )
            token_counts = window.token_counts[model_variant]
            assert token_counts.prompt_tokens > 0, (
                f"Clip {clip_idx} window {window_idx} {model_variant} prompt_tokens should be > 0"
            )
            assert token_counts.output_tokens > 0, (
                f"Clip {clip_idx} window {window_idx} {model_variant} output_tokens should be > 0"
            )

            expected_caption = expected_captions[window_idx]
            if expected_caption == _GOLDEN_CAPTION_PLACEHOLDER:
                # No golden caption recorded yet for this variant: skip the
                # cosine-similarity check but keep the structural assertions above.
                # Print the generated caption so a developer running the test with
                # `-s` can paste it into _EXPECTED_CAPTIONS.
                print(  # noqa: T201
                    f"\n[golden-caption-capture] {model_variant} clip={clip_idx} "
                    f"window={window_idx}:\n{generated_caption}\n"
                )
                continue

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
