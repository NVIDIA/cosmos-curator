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
"""Test vllm_interface.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.pipelines.video.utils.data_model import (
    TokenCounts,
    VllmCaptionRequest,
    VllmConfig,
    VllmSamplingConfig,
    WindowConfig,
)

if conda_utils.is_running_in_env("unified"):
    import torch
    from vllm import SamplingParams

    from cosmos_curator.models.vllm_interface import (
        _VLLM_PLUGINS,
        VllmWindowResult,
        _caption_inflight_batching,
        _caption_no_inflight_batching,
        _get_vllm_plugin,
        _save_frames_as_pngs,
        auto_processor,
        make_metadata,
        make_model_inputs,
        process_vllm_output,
        sampling_params,
        vllm_caption,
        vllm_generate,
        vllm_model,
    )
    from tests.utils.vllm_mock import MockLLM, MockVllmPlugin, make_request_output

    VALID_VARIANTS = list(_VLLM_PLUGINS.keys())
else:
    VALID_VARIANTS = []


@pytest.fixture(autouse=True)
def patch_vllm_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _VLLM_PLUGINS dict with {"mock": MockVllmPlugin} for every test in this module."""
    if conda_utils.is_running_in_env("unified"):
        monkeypatch.setitem(
            _VLLM_PLUGINS,
            "mock",
            MockVllmPlugin,
        )
        for k in list(_VLLM_PLUGINS.keys()):
            if k != "mock":
                monkeypatch.delitem(_VLLM_PLUGINS, k)


@pytest.mark.env("unified")
def test_get_vllm_plugin_raises() -> None:
    """Test _get_vllm_plugin raises ValueError for invalid variant."""
    with pytest.raises(ValueError, match=r".*"):
        _get_vllm_plugin("invalid")


@pytest.mark.env("unified")
def test_vllm_model() -> None:
    """vllm_model should return "llm"."""
    cfg = VllmConfig(model_variant="mock")
    assert isinstance(vllm_model(cfg), MockLLM)


@pytest.mark.env("unified")
def test_sampling_params() -> None:
    """Test sampling_params."""
    temperature = 0.1
    top_p = 0.2
    repetition_penalty = 1.3
    max_tokens = 1024
    vllm_config = VllmConfig(
        model_variant="mock",
        sampling_config=VllmSamplingConfig(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        ),
    )
    sp = sampling_params(vllm_config.sampling_config)
    assert isinstance(sp, SamplingParams)
    assert sp.temperature == temperature
    assert sp.top_p == top_p
    assert sp.repetition_penalty == repetition_penalty
    assert sp.max_tokens == max_tokens


@pytest.mark.env("unified")
def test_auto_processor() -> None:
    """Test auto_processor.

    This ensures that the plugin's auto_processor is called.
    """
    vllm_config = VllmConfig(model_variant="mock")
    assert auto_processor(vllm_config) is not None


@pytest.mark.env("unified")
def test_make_metadata() -> None:
    """Test make_metadata."""
    width = 32
    height = 64
    num_videos = 5
    num_frames = 2
    fps = 1.0
    frames = [torch.zeros((num_frames, 3, height, width)) for _ in range(num_videos)]
    window_config = WindowConfig(sampling_fps=fps)
    metadata = make_metadata(frames, window_config)
    assert len(metadata) == len(frames)
    for i in range(len(frames)):
        assert metadata[i]["fps"] == fps
        assert metadata[i]["duration"] == num_frames / fps
        assert metadata[i]["width"] == width
        assert metadata[i]["height"] == height
        assert metadata[i]["total_num_frames"] == num_frames
        assert metadata[i]["frames_indices"] == list(range(num_frames))
        assert not metadata[i]["do_sample_frames"]


@pytest.mark.env("unified")
def test_make_metadata_raises() -> None:
    """Test make_metadata raises ValueError for non-4D tensors."""
    with pytest.raises(ValueError, match=r".*"):
        make_metadata([torch.zeros((3, 32, 32))], WindowConfig(sampling_fps=1.0))


@pytest.mark.env("unified")
def test_make_model_inputs() -> None:
    """Test make_model_inputs."""
    videos = [torch.zeros((2, 3, 32, 32)) for _ in range(5)]
    metadata = [{"fps": 1.0}] * len(videos)
    config = VllmConfig(model_variant="mock")
    processor = MagicMock()
    prompt = "p"
    output = make_model_inputs(videos, metadata, config, processor, prompt)
    assert len(output) == len(videos)
    for i in range(len(videos)):
        ot = output[i]
        assert "prompt" in ot
        assert "multi_modal_data" in ot
        assert ot["prompt"] == prompt

        mm_data = ot["multi_modal_data"]
        assert mm_data["video"][0][0].shape == (2, 3, 32, 32)
        assert mm_data["video"][0][1] == metadata[i]


@pytest.mark.env("unified")
def test_vllm_generate() -> None:
    """Test vllm_generate."""
    llm = MockLLM()
    sampling_params = SamplingParams(max_tokens=100)
    requests = [VllmCaptionRequest(request_id=f"id{i}", inputs={"i": i}) for i in range(5)]
    captions = vllm_generate(llm, sampling_params, requests, batch_size=2)  # type: ignore[arg-type]
    assert len(captions) == len(requests)
    assert [c.request_id for c in captions] == [r.request_id for r in requests]
    captions_text = [c.outputs[0].text for c in captions]
    expected_captions = [f"mock-caption-{i}" for i in range(len(requests))]
    assert captions_text == expected_captions


@pytest.mark.env("unified")
def test_process_vllm_output() -> None:
    """Test process_vllm_output."""
    vllm_plugin = _get_vllm_plugin("mock")
    requests = [VllmCaptionRequest(request_id=f"id{i}", inputs={"i": i}) for i in range(5)]
    engine_output = [make_request_output(f"id{i}", f"mock-caption-{i}") for i in range(5)]
    inflight_requests = {r.request_id: r for r in requests}
    vllm_config = VllmConfig(model_variant="mock")
    finished = process_vllm_output(engine_output, inflight_requests, vllm_config)  # type: ignore[arg-type]
    assert len(finished) == len(engine_output)
    assert [r.request_id for r in finished] == [r.request_id for r in engine_output]
    assert [r.caption for r in finished] == [vllm_plugin.decode(r) for r in engine_output]
    assert all(r.finish_reason == "stop" for r in finished)


@pytest.mark.env("unified")
def test_process_vllm_output_raises() -> None:
    """Test process_vllm_output raises TypeError for non-RequestOutput output."""
    request_output = [object()]
    with pytest.raises(TypeError, match=r".*"):
        process_vllm_output(request_output, {}, VllmConfig(model_variant="mock"))  # type: ignore[arg-type]


@pytest.mark.env("unified")
def test_process_vllm_output_not_finished() -> None:
    """Test that no output is returned when no requests are finished."""
    requests = [VllmCaptionRequest(request_id=f"id{i}", inputs={"i": i}) for i in range(5)]
    engine_output = [make_request_output(f"id{i}", f"mock-caption-{i}", finished=False) for i in range(5)]
    inflight_requests = {r.request_id: r for r in requests}
    vllm_config = VllmConfig(model_variant="mock")
    finished = process_vllm_output(engine_output, inflight_requests, vllm_config)  # type: ignore[arg-type]
    assert len(finished) == 0


@pytest.mark.env("unified")
@pytest.mark.parametrize("stage2", [False, True])
def test_caption_no_inflight_batching(*, stage2: bool) -> None:
    """Test _caption_no_inflight_batching."""
    vllm_config = VllmConfig(model_variant="mock")
    model = vllm_model(vllm_config)
    processor = auto_processor(vllm_config)
    sp = sampling_params(vllm_config.sampling_config)

    model_inputs = [{"a": 1}, {"b": 2}]
    stage2_prompts: list[str | None] = [None, None] if not stage2 else ["ref", "ref"]

    results = _caption_no_inflight_batching(
        model_inputs=model_inputs,
        llm=model,
        processor=processor,
        sampling_params=sp,
        vllm_config=vllm_config,
        stage2_prompts=stage2_prompts,
    )

    if stage2:
        off = len(model_inputs)
        expected_captions = [f"mock-caption-{i + off}" for i in range(len(model_inputs))]
    else:
        expected_captions = [f"mock-caption-{i}" for i in range(len(model_inputs))]

    assert [result.text for result in results] == expected_captions
    assert len(results) == len(model_inputs)
    assert all(result.finish_reason == "stop" for result in results)


@pytest.mark.env("unified")
@patch("cosmos_curator.models.vllm_interface.process_vllm_output")
def test_caption_no_inflight_batching_raises(mock_process_vllm_output: MagicMock) -> None:
    """Test _caption_no_inflight_batching raises RuntimeError on length mismatch."""
    mock_process_vllm_output.return_value = []
    vllm_config = VllmConfig(model_variant="mock")
    llm = vllm_model(vllm_config)
    with pytest.raises(RuntimeError, match=r".*"):
        _caption_no_inflight_batching(
            model_inputs=[{"a": 1}],
            llm=llm,
            processor=MagicMock(),
            sampling_params=MagicMock(),
            vllm_config=vllm_config,
            stage2_prompts=[None],
        )


@pytest.mark.env("unified")
@pytest.mark.parametrize("stage2", [False, True])
def test_caption_no_inflight_batching_preserves_order(*, stage2: bool) -> None:
    """Non-inflight batching must return captions in original input order with mixed stage2 prompts.

    When stage2_prompts contains a mix of None and non-None values, the
    filter-and-concatenate pattern previously returned all stage1-only results
    first, then all stage2 results, instead of preserving the original input
    order.  The mixed-prompts variant is the critical case — when all prompts
    are the same (all None or all non-None) no split occurs.
    """
    vllm_config = VllmConfig(model_variant="mock")
    model = vllm_model(vllm_config)
    processor = auto_processor(vllm_config)
    sp = sampling_params(vllm_config.sampling_config)

    num_inputs = 5
    model_inputs = [{"idx": i} for i in range(num_inputs)]
    # Mixed: indices 1 and 3 need stage2, others don't
    if stage2:
        stage2_prompts: list[str | None] = [None, "refine", None, "refine", None]
    else:
        stage2_prompts = [None] * num_inputs

    results = _caption_no_inflight_batching(
        model_inputs=model_inputs,
        llm=model,
        processor=processor,
        sampling_params=sp,
        vllm_config=vllm_config,
        stage2_prompts=stage2_prompts,
    )

    assert len(results) == num_inputs

    # Extract the integer suffix from each "mock-caption-N" string.
    caption_indices = [int(result.text.split("-")[-1]) for result in results]

    # All captions must be unique (no duplicates from misordering).
    assert len(set(caption_indices)) == num_inputs

    if stage2:
        # Stage1 produces captions 0-4 for inputs 0-4.
        # Inputs 0, 2, 4 (no stage2) keep their stage1 captions.
        # Inputs 1, 3 (stage2) get new captions from the refinement pass.
        # Stage1-only captions (indices 0, 2, 4) must appear at positions 0, 2, 4.
        stage1_only_indices = [caption_indices[i] for i in [0, 2, 4]]
        assert stage1_only_indices == sorted(stage1_only_indices)
        # Stage2 captions (positions 1, 3) have higher indices than stage1 captions.
        stage2_indices = [caption_indices[i] for i in [1, 3]]
        assert all(s2 > max(stage1_only_indices) for s2 in stage2_indices)
        assert stage2_indices == sorted(stage2_indices)


@pytest.mark.env("unified")
@pytest.mark.parametrize("stage2", [False, True])
def test_caption_inflight_batching(*, stage2: bool) -> None:
    """Test _caption_inflight_batching."""
    vllm_config = VllmConfig(model_variant="mock")
    model = vllm_model(vllm_config)
    processor = auto_processor(vllm_config)
    sp = sampling_params(vllm_config.sampling_config)

    model_inputs = [{"a": 1}, {"b": 2}]
    stage2_prompts: list[str | None] = [None, None] if not stage2 else ["ref", "ref"]

    results = _caption_inflight_batching(
        model_inputs=model_inputs,
        llm=model,
        processor=processor,
        sampling_params=sp,
        vllm_config=vllm_config,
        stage2_prompts=stage2_prompts,
        max_inflight_requests=0,
    )

    if stage2:
        off = len(model_inputs)
        expected_captions = [f"mock-caption-{i + off}" for i in range(len(model_inputs))]
    else:
        expected_captions = [f"mock-caption-{i}" for i in range(len(model_inputs))]

    assert [result.text for result in results] == expected_captions
    assert len(results) == len(model_inputs)
    assert all(result.finish_reason == "stop" for result in results)


@pytest.mark.env("unified")
def test_vllm_caption_negative_inflight_raises() -> None:
    """vllm_caption should validate non-negative inflight param."""
    with pytest.raises(ValueError, match=r"must be >= 0"):
        vllm_caption(
            model_inputs=[{}],
            llm=MagicMock(),
            processor=MagicMock(),
            sampling_params=MagicMock(),
            vllm_config=VllmConfig(model_variant="qwen"),
            max_inflight_requests=-1,
            inflight_batching=True,
            stage2_prompts=[None],
        )


@pytest.mark.env("unified")
def test_vllm_caption_stage2_prompts_mismatch_raises() -> None:
    """vllm_caption should validate stage2_prompts length matches model_inputs length."""
    with pytest.raises(ValueError, match=r"must be same length"):
        vllm_caption(
            model_inputs=[{}],
            llm=MagicMock(),
            processor=MagicMock(),
            sampling_params=MagicMock(),
            vllm_config=VllmConfig(model_variant="qwen"),
            max_inflight_requests=0,
            inflight_batching=True,
            stage2_prompts=[None, None],
        )


@pytest.mark.env("unified")
@pytest.mark.parametrize("inflight", [False, True])
@patch("cosmos_curator.models.vllm_interface._caption_inflight_batching")
@patch("cosmos_curator.models.vllm_interface._caption_no_inflight_batching")
def test_vllm_caption_dispatch(mock_no_ifb: MagicMock, mock_ifb: MagicMock, *, inflight: bool) -> None:
    """vllm_caption dispatches to correct helper based on inflight flag.

    This is a happy path test that verifies that vllm_caption:
    1. runs without errors
    2. dispatches to the correct helper based on the inflight flag
    """
    mock_no_ifb.return_value = [VllmWindowResult(text="no_ifb", finish_reason="stop", token_counts=TokenCounts())]
    mock_ifb.return_value = [VllmWindowResult(text="ifb", finish_reason="stop", token_counts=TokenCounts())]
    out = vllm_caption(
        model_inputs=[{}],
        llm=MagicMock(),
        processor=MagicMock(),
        sampling_params=MagicMock(),
        vllm_config=VllmConfig(model_variant="any"),
        max_inflight_requests=0,
        inflight_batching=inflight,
    )
    assert out == (
        [VllmWindowResult(text="ifb", finish_reason="stop", token_counts=TokenCounts())]
        if inflight
        else [VllmWindowResult(text="no_ifb", finish_reason="stop", token_counts=TokenCounts())]
    )


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("shape", "values_normalized", "prefix"),
    [
        # RGB frames with normalized values [0, 1]
        ((3, 3, 64, 64), True, "frame"),
        # RGB frames with uint8 values [0, 255]
        ((5, 3, 128, 128), False, "window_0_frame"),
        # Grayscale frames with normalized values
        ((2, 1, 32, 32), True, "test_prefix"),
        # Grayscale frames with uint8 values
        ((4, 1, 96, 96), False, "frame"),
        # Single frame RGB
        ((1, 3, 50, 50), True, "single"),
        # Prefix already ending with "_frame"
        ((2, 3, 40, 40), True, "window_5_frame"),
    ],
)
def test_save_frames_as_pngs(
    shape: tuple[int, int, int, int],
    *,
    values_normalized: bool,
    prefix: str,
    tmp_path: Path,
) -> None:
    """Test _save_frames_as_pngs with various frame configurations.

    Tests:
    - RGB and grayscale frames
    - Normalized [0, 1] and uint8 [0, 255] value ranges
    - Different frame counts, sizes, and prefixes
    - Correct file naming with prefix handling
    - PNG files can be loaded and have correct dimensions/channels
    """
    num_frames, channels, height, width = shape

    # Create test frames with specified shape
    if values_normalized:
        frames = torch.rand((num_frames, channels, height, width))
    else:
        frames = torch.randint(0, 256, (num_frames, channels, height, width), dtype=torch.uint8).float()

    output_dir = tmp_path / "frames_output"

    # Call the function
    _save_frames_as_pngs(frames, output_dir, prefix)

    # Verify output directory was created
    assert output_dir.exists()
    assert output_dir.is_dir()

    # Verify correct number of files were created
    png_files = sorted(output_dir.glob("*.png"))
    assert len(png_files) == num_frames

    # Verify each frame was saved correctly
    for frame_idx, png_file in enumerate(png_files):
        # Check filename format
        if prefix.endswith("_frame") or prefix == "frame":
            expected_filename = f"{prefix}_{frame_idx:04d}.png"
        else:
            expected_filename = f"{prefix}_frame_{frame_idx:04d}.png"
        assert png_file.name == expected_filename

        # Load the saved image and verify properties
        img = Image.open(png_file)
        assert img.size == (width, height)

        # Verify channel count
        if channels == 1:
            assert img.mode == "L"  # Grayscale
        elif channels == 3:
            assert img.mode == "RGB"

        # Convert to numpy and verify value range
        img_array = np.array(img)
        assert img_array.dtype == np.uint8
        assert img_array.min() >= 0
        assert img_array.max() <= 255

        # Verify dimensions match
        if channels == 1:
            assert img_array.shape == (height, width)
        elif channels == 3:
            assert img_array.shape == (height, width, channels)


@pytest.mark.env("unified")
@pytest.mark.parametrize("stage2", [False, True])
def test_caption_inflight_batching_preserves_order(*, stage2: bool) -> None:
    """Inflight batching must return captions in input order even when the engine finishes requests out of order.

    The mock engine assigns captions as "mock-caption-N" where N is an
    auto-incrementing counter.  By making input 0 very slow (10 steps) and
    inputs 1-4 fast (1 step), inputs 1-4 complete before input 0.

    Completion order: input1 (caption-0), input2 (caption-1), input3 (caption-2),
                      input4 (caption-3), input0 (caption-4).

    Without the ordering fix, results would be returned in completion order.
    With the fix, results are reordered to match the original input order.
    """
    vllm_config = VllmConfig(model_variant="mock")
    processor = auto_processor(vllm_config)
    sp = sampling_params(vllm_config.sampling_config)

    num_inputs = 5
    model_inputs = [{"idx": i} for i in range(num_inputs)]
    stage2_prompts: list[str | None] = ["refine"] * num_inputs if stage2 else [None] * num_inputs

    # Input 0 is slow (10 steps), inputs 1-4 are fast (1 step).
    # This guarantees inputs 1-4 complete before input 0.
    model = MockLLM(steps_to_complete=lambda req_idx: 10 if req_idx == 0 else 1)

    results = _caption_inflight_batching(
        model_inputs=model_inputs,
        llm=model,  # type: ignore[arg-type]
        processor=processor,
        sampling_params=sp,
        vllm_config=vllm_config,
        stage2_prompts=stage2_prompts,
        max_inflight_requests=0,
    )

    assert len(results) == num_inputs

    # Extract the integer suffix from each "mock-caption-N" string.
    caption_indices = [int(result.text.split("-")[-1]) for result in results]

    # All captions must be unique (no duplicates from misordering).
    assert len(set(caption_indices)) == num_inputs

    # Input 0 finishes last so it gets the highest caption index.
    assert caption_indices[0] == max(caption_indices)

    # Inputs 1-4 all complete in submission order (each needs 1 step),
    # so their caption indices must be consecutive and ascending.
    assert caption_indices[1:] == sorted(caption_indices[1:])


@pytest.mark.env("unified")
@patch("cosmos_curator.models.vllm_interface.process_vllm_output")
@patch("cosmos_curator.models.vllm_interface.vllm_generate")
def test_caption_no_inflight_batching_terminal_finish_reason_wins(
    mock_vllm_generate: MagicMock, mock_process_vllm_output: MagicMock
) -> None:
    """Stage-2 finish_reason must overwrite stage-1 when assembling the final window result."""
    vllm_config = VllmConfig(model_variant="mock")
    request_batches: list[list[VllmCaptionRequest]] = []

    def _fake_generate(
        _llm: object,
        _sampling_params: object,
        requests: list[VllmCaptionRequest],
        _batch_size: int,
    ) -> list[object]:
        request_batches.append(requests)
        return []

    def _fake_process(
        _engine_output: list[object], in_flight_requests: dict[str, VllmCaptionRequest], _config: VllmConfig
    ) -> list[VllmCaptionRequest]:
        requests = list(in_flight_requests.values())
        request = requests[0]
        if request.stage2_prompt is not None:
            request.caption = "stage1-caption"
            request.prompt_tokens = 2
            request.output_tokens = 3
            request.finish_reason = "length"
            return requests

        request.caption = "stage2-caption"
        request.prompt_tokens = 5
        request.output_tokens = 7
        request.finish_reason = "stop"
        return requests

    mock_vllm_generate.side_effect = _fake_generate
    mock_process_vllm_output.side_effect = _fake_process

    results = _caption_no_inflight_batching(
        model_inputs=[{"idx": 0}],
        llm=MagicMock(),
        processor=MagicMock(),
        sampling_params=MagicMock(),
        vllm_config=vllm_config,
        stage2_prompts=["refine"],
    )

    assert len(request_batches) == 2
    assert results == [
        VllmWindowResult(
            text="stage2-caption",
            finish_reason="stop",
            token_counts=TokenCounts(prompt_tokens=7, output_tokens=10),
        )
    ]
