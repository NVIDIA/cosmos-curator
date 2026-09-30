# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared TransNetV2 inference preserves frame alignment and pads temporal context."""

import numpy as np
import pytest
import torch

from cosmos_curator.models.transnetv2 import predict_transitions


@pytest.mark.parametrize("count", [1, 24, 25, 49, 50, 51, 75, 100, 101])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_inference_preserves_every_frame_and_context(count: int, batch_size: int) -> None:
    """Edge padding, final windows, and batching preserve one prediction per input frame."""
    frames = np.broadcast_to(np.arange(count, dtype=np.uint8)[:, None, None, None], (count, 27, 48, 3))
    seen = []

    def model(inputs: torch.Tensor) -> torch.Tensor:
        assert tuple(inputs.shape[1:]) == (100, 27, 48, 3)
        assert inputs.dtype == torch.uint8
        assert torch.is_inference_mode_enabled()
        seen.extend(inputs[:, :, 0, 0, 0].tolist())
        return (inputs[:, :, 0, 0, 0] % 2).float().unsqueeze(-1)

    actual = predict_transitions(model, frames, threshold=0.4, window_batch_size=batch_size, device="cpu")
    np.testing.assert_array_equal(actual, np.arange(count) % 2 == 1)
    assert actual.dtype == np.bool_
    assert seen[0][:25] == [0] * 25
    assert seen[-1][-25:] == [count - 1] * 25
    assert len(seen) == (count + 49) // 50
