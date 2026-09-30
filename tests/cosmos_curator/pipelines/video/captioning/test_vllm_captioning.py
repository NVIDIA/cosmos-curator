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
import re
import uuid
from collections import Counter

import pytest
from scipy.spatial.distance import cosine

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name
from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION
from cosmos_curator.pipelines.common.model_constraints import PreprocessMode
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

_WORD_PATTERN = re.compile(r"[^\W_]+")
_THRESHOLDS = {
    "qwen": 0.8,
    "cosmos_r1": 0.7,
    "cosmos_r2": 0.7,
    "cosmos3_nano": 0.7,
    "qwen3_5_27b": 0.8,
}
_NUM_CLIPS = 1
_VLLM_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "cosmos_r2": {
        "preprocess_mode": PreprocessMode.MODEL,
    },
    "qwen3_5_27b": {
        "preprocess_mode": PreprocessMode.MODEL,
    },
}
_WINDOW_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "qwen": {
        "sampling_fps": 2.0,
    },
    "cosmos_r1": {
        "sampling_fps": 4.0,
    },
    "cosmos_r2": {
        "sampling_fps": 4.0,
    },
    "cosmos3_nano": {
        "sampling_fps": 4.0,
    },
    "qwen3_5_27b": {
        "sampling_fps": 2.0,
    },
}
_GOLDEN_CAPTION_PLACEHOLDER = "__PLACEHOLDER_CAPTURE_FROM_FIRST_RUN__"
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
            "The video opens with a **determined female character** navigating a **harsh, snowy mountainous "
            "landscape**. Her rugged appearance-weathered face, dark hair, and earth-toned clothing-suggests she is "
            "on a **challenging journey**. She holds a **staff** in her right hand, implying a role as a guide, "
            "protector, or seeker. The camera uses **wide shots** to emphasize the vast, treacherous environment, "
            "while her **focused gaze** and steady posture convey resolve. The muted, natural lighting and swirling "
            "snow enhance the **isolation and severity** of the setting.\n"
            "\n"
            "The scene then transitions via a **fade-out/fade-in** to an **indoor setting**, where a **mysterious "
            "male character** appears. He wears elaborate **tribal accessories** (pierced face, feathers, chains) "
            "and a **textured robe**, situating him in a **cultural or spiritual context**. The dim, shadowy lighting "
            "and **ornate carvings** on the walls suggest a **temple, council chamber, or sacred space**. The camera "
            "focuses closely on his **expressive face** and attire, building intrigue about his role and intentions. "
            "His **haunted expression** hints at a **hidden backstory** or internal conflict.\n"
            "\n"
            "The transition between these two scenes underscores a **contrast between external struggle (the snowy "
            "wilderness)** and **internal or communal stakes** (the dimly lit chamber). The **Blender Foundation "
            "logos** appear briefly as production credits, grounding the fantasy visuals in their creative source. "
            "The overall tone blends **epic adventure** with **dark, atmospheric storytelling**, leaving viewers "
            "curious about the characters' motivations and the narrative's direction."
        ),
    ],
    "qwen3_5_27b": [
        (
            "This story unfolds across two distinct scenes, each rich with visual and narrative detail. The first "
            "scene is set in a desolate, snow-covered landscape under an overcast sky. A lone figure stands amidst "
            "the harsh environment, clad in rugged attire suitable for survival in such conditions. This character "
            "holds a staff or weapon, suggesting readiness for potential threats or challenges that lie ahead. "
            "Their determined expression hints at resilience and purpose as they navigate this unforgiving terrain.\n"
            "\n"
            "The second scene transitions to a dimly lit interior space, creating a stark contrast to the cold "
            "exterior. Here, we encounter two characters engaged in what appears to be a significant interaction. "
            "One individual, distinguished by their long beard and traditional headgear adorned with intricate "
            "designs, exudes wisdom and authority. They are dressed in earth-toned garments that suggest cultural "
            "significance or historical context. Opposite them sits another character with striking red hair, "
            "wearing minimal clothing that reveals tattoos on their skin—markings that may symbolize identity or "
            "affiliation within their world.\n"
            "\n"
            "The atmosphere between these two figures is charged with tension or contemplation, underscored by the "
            "warm yet subdued lighting that casts dramatic shadows around them. Decorative elements in the "
            "background hint at a setting steeped in tradition or ritualistic importance. Together, these visuals "
            "weave a tale of exploration, confrontation, or alliance against the backdrop of contrasting "
            "environments—one external and brutal, the other internal and intimate."
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
    "cosmos3_nano": [
        (
            "The video begins with a misty, snowy mountain landscape, setting a cold and desolate atmosphere. A young "
            "woman emerges from the fog, holding a long wooden staff. She is dressed in rugged, practical clothing "
            "suitable for harsh weather conditions, including a scarf covering her lower face. Her expression is "
            "serious and determined as she looks around cautiously, suggesting she is on a mission or searching for "
            "something important.\n"
            "\n"
            "As the scene progresses, the camera focuses more closely on the woman's face, emphasizing her alertness "
            "and resolve. The background remains shrouded in mist, adding to the sense of isolation and mystery. "
            'Suddenly, the screen fades to black, and white text appears, reading "THE BLENDER FOUNDATION presents," '
            "indicating that this is an animation produced by the Blender Foundation.\n"
            "\n"
            "The next scene transitions to a dimly lit interior, where an older man with a long beard and traditional "
            "attire is seen. He has distinctive facial piercings and wears a headband adorned with metal rings. His "
            "expression is one of concern or contemplation as he speaks, possibly delivering important information or "
            "advice. The lighting highlights his features, creating a dramatic effect and drawing attention to his "
            "role in the story.\n"
            "\n"
            "Finally, the scene shifts to another character, a young woman with short red hair. She is shown in "
            "close-up, her expression showing surprise or shock. The background is dark, with some decorative elements "
            "visible, suggesting an indoor setting. Her reaction implies that she has just received unexpected news or "
            "witnessed something surprising, adding tension and intrigue to the narrative."
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
    words = _WORD_PATTERN.findall(sentence.casefold())
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


def test_cosine_similarity_normalizes_word_formatting() -> None:
    """Ignore case, punctuation, and Markdown when comparing caption words."""
    formatted = "**VISUAL ELEMENTS:** __snowy__, _snow-covered_ mountains."
    plain = "visual elements snowy snow covered mountains"

    assert cosine_similarity(formatted, plain) == pytest.approx(1.0)


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


# Variants that ``cosmos_curate.core.managers.model_cli`` excludes from its
# default download set (see ``_get_default_models``). For these variants the
# weights must be downloaded explicitly before this test runs, e.g.::
#
#     pixi run -e model-download python -m cosmos_curate.core.managers.model_cli \
#         download --models <variant>
_VARIANTS_NOT_DOWNLOADED_BY_DEFAULT: frozenset[str] = frozenset({"cosmos3_nano", "qwen3_5_27b"})


def _assert_weights_available(model_variant: str) -> None:
    """Fail clearly when the variant's weights are not present locally.

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
    pytest.fail(f"Model weights for {model_variant} not found at {weights_dir}{hint}")


@pytest.mark.env("default")
@pytest.mark.parametrize(
    "model_variant",
    ["qwen", "cosmos_r1", "cosmos_r2", "cosmos3_nano", "qwen3_5_27b"],
    ids=["qwen", "cosmos_r1", "cosmos_r2", "cosmos3_nano", "qwen3_5_27b"],
)
def test_vllm_caption_generation(
    sample_captioning_task: SplitPipeTask, sequential_runner: RunnerInterface, model_variant: str
) -> None:
    """Test the vLLM captioning result."""
    _assert_weights_available(model_variant)
    vllm_config = VllmConfig(
        model_variant=model_variant,
        sampling_config=VllmSamplingConfig(temperature=0.0),
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


@pytest.mark.env("default")
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
        preprocess_mode=PreprocessMode.CURATOR,
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
