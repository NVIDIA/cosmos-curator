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

"""Tests for the ``--captioning-setup-attempts`` CLI flag and its plumbing.

End-to-end, the flag's job is to let the actor pool re-spawn a vLLM caption
worker if its setup fails on the first GPU placement (typically because the
device hasn't finished CUDA context reclaim). These tests pin the three layers
the flag has to traverse without invoking any of vLLM's GPU machinery:

1. Argparse: the flag is registered, defaults to 1, and accepts integer values.
2. ``CaptioningConfig.caption_setup_attempts``: defaults to 1 and rejects values
   below 1 so a typo can't silently disable the retry.
3. ``_build_captioning_caption_stage``: the configured value flows onto the
   returned ``CuratorStageSpec.num_setup_attempts_python`` (only for the vLLM
   backend; the API backends do not expose this knob).
"""

import argparse

import attrs
import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.pipelines.video.captioning.captioning_builders import (
    CaptioningConfig,
    OpenAIConfig,
    _build_captioning_caption_stage,
)
from cosmos_curator.pipelines.video.splitting_pipeline import _setup_parser
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig, WindowConfig


def _parser() -> argparse.ArgumentParser:
    """Build the splitting-pipeline argparse parser used by the CLI entrypoint."""
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser


# ---------------------------------------------------------------------------
# CLI flag
# ---------------------------------------------------------------------------


class TestCliFlag:
    """Pins the argparse contract for ``--captioning-setup-attempts``."""

    def test_defaults_to_one_when_omitted(self) -> None:
        """Omitting the flag must preserve the historical single-attempt behavior.

        Bumping the default would silently change resource consumption on every
        existing invocation, so the default has to stay at 1 unless explicitly
        opted into.
        """
        args = _parser().parse_args([])

        assert args.captioning_setup_attempts == 1

    def test_accepts_explicit_integer(self) -> None:
        """Operators bump this to allow retries through transient GPU placement issues."""
        args = _parser().parse_args(["--captioning-setup-attempts", "3"])

        assert args.captioning_setup_attempts == 3

    def test_rejects_non_integer_value(self) -> None:
        """Argparse's ``type=int`` must reject strings like ``3.5`` or ``foo``."""
        with pytest.raises(SystemExit):
            _parser().parse_args(["--captioning-setup-attempts", "not-a-number"])


# ---------------------------------------------------------------------------
# CaptioningConfig
# ---------------------------------------------------------------------------


def _minimal_vllm_captioning_config(**overrides: object) -> CaptioningConfig:
    """Return a CaptioningConfig with the minimum fields required for the vLLM path.

    ``CaptioningConfig`` is frozen so this helper keeps each test free of repeated
    boilerplate while still allowing per-test overrides. ``model_variant`` is a
    required positional on ``VllmConfig`` so we pin a known-valid value here -
    the builder never executes the vLLM stage itself, so the variant identity
    is only used to satisfy attrs construction.
    """
    defaults: dict[str, object] = {
        "backend": VllmConfig(model_variant="qwen"),
        "window_config": WindowConfig(),
    }
    defaults.update(overrides)
    return CaptioningConfig(**defaults)  # type: ignore[arg-type]


class TestCaptioningConfigSetupAttempts:
    """Pins the ``CaptioningConfig.caption_setup_attempts`` field contract."""

    def test_defaults_to_one(self) -> None:
        """Default must match the CLI default so the two sources of truth agree."""
        cfg = _minimal_vllm_captioning_config()

        assert cfg.caption_setup_attempts == 1

    def test_accepts_positive_integer(self) -> None:
        """Higher attempt counts are the whole point of the field."""
        cfg = _minimal_vllm_captioning_config(caption_setup_attempts=5)

        assert cfg.caption_setup_attempts == 5

    @pytest.mark.parametrize("invalid_value", [0, -1, -10])
    def test_rejects_values_below_one(self, invalid_value: int) -> None:
        """Zero or negative attempts would disable setup entirely.

        The validator prevents an off-by-one mistake (or an env-driven config
        rewrite) from quietly turning the actor pool's setup loop into a no-op.
        """
        with pytest.raises(ValueError, match=r".*"):
            _minimal_vllm_captioning_config(caption_setup_attempts=invalid_value)


# ---------------------------------------------------------------------------
# _build_captioning_caption_stage plumbing
# ---------------------------------------------------------------------------


class TestBuilderWiresSetupAttempts:
    """The configured attempt count must reach the returned ``CuratorStageSpec``."""

    def test_vllm_backend_propagates_default(self) -> None:
        """Default config produces a spec with ``num_setup_attempts_python == 1``.

        The vLLM caption stage is the only backend that exposes this knob today;
        Xenna's StageSpec also defaults to 1, so a missing wiring would still
        appear correct in a smoke test. Pinning the value explicitly catches
        that regression.
        """
        spec = _build_captioning_caption_stage(_minimal_vllm_captioning_config())

        assert isinstance(spec, CuratorStageSpec)
        assert spec.num_setup_attempts_python == 1

    def test_vllm_backend_propagates_explicit_value(self) -> None:
        """A bumped attempt count must flow through to the spec the runner sees."""
        spec = _build_captioning_caption_stage(
            _minimal_vllm_captioning_config(caption_setup_attempts=4),
        )

        assert isinstance(spec, CuratorStageSpec)
        assert spec.num_setup_attempts_python == 4

    def test_openai_backend_ignores_setup_attempts(self) -> None:
        """The OpenAI backend must not crash when ``caption_setup_attempts`` is set.

        The flag is documented as a vLLM-only knob; API-backed caption stages
        don't suffer from GPU placement issues and therefore ignore it. Pinning
        this contract stops a future "DRY-up" refactor from accidentally
        routing the value into the wrong spec.

        Only the openai backend is exercised here because ``GeminiCaptionStage``
        eagerly calls ``load_config()`` in its constructor and refuses to
        initialize without a populated ``~/.config/cosmos_curator/config.yaml``
        - testing that path would require a config fixture that the assertion
        itself doesn't need. The two backends share the same builder branch
        shape (return a bare ``CuratorStage``, not a ``CuratorStageSpec``), so
        openai is a sufficient representative.
        """
        cfg = CaptioningConfig(
            backend=OpenAIConfig(),
            window_config=WindowConfig(),
            caption_setup_attempts=5,
        )

        # Must not raise. The API backends return a bare CuratorStage rather
        # than a spec, so there is no ``num_setup_attempts_python`` attribute
        # to assert against - the contract is "ignore, don't crash".
        stage = _build_captioning_caption_stage(cfg)
        assert stage is not None
        assert not isinstance(stage, CuratorStageSpec)

    def test_caption_setup_attempts_is_frozen_with_other_config(self) -> None:
        """``CaptioningConfig`` is ``frozen=True`` so the attempts field is immutable.

        This is a guard against well-meaning code mutating the field after
        construction; once the spec is built any later change wouldn't take
        effect anyway, so the type system should reject the attempt.
        """
        cfg = _minimal_vllm_captioning_config(caption_setup_attempts=3)

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            cfg.caption_setup_attempts = 7  # type: ignore[misc]
