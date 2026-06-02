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

"""Tests for ``VllmPlugin.model_async()`` per registered plugin.

Verifies that each plugin's ``model_async`` returns ``AsyncEngineArgs``
populated from:

- per-variant module-scope constants (``MAX_MODEL_LEN``, ``GPU_MEMORY_UTILIZATION``,
  etc.) -- the **same** source of truth as sync ``model()``.
- ``VllmAsyncConfig`` user knobs (passed through unchanged).
- Async-only invariants (``mm_processor_kwargs`` carries ``do_sample_frames=False``).

Drift tripwire: each test asserts that the constants the async path reads
match the constants sync ``model()`` reads, so a future regression that
diverges sync vs async tuning is caught locally.

These tests require the ``unified`` pixi env (vLLM installed).
"""

import importlib
from unittest.mock import patch

import pytest

from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.pipelines.video.utils.data_model import VllmAsyncConfig, VllmConfig

# Heavy vLLM imports are env-gated: the ``unified`` pixi env carries vLLM
# and the plugin modules; in the default CPU collection env we still want
# the module to import (so test selection / discovery works) but the
# tests themselves are skipped via ``pytestmark``.  ``VllmCosmosReason2VL``
# joined this set as part of cosmos_r2 onboarding -- import it from the
# same conditional so test bodies do not need a per-test local import.
if pixi_utils.is_running_in_env("unified"):
    from cosmos_curator.models import vllm_cosmos_reason1_vl, vllm_nemotron, vllm_qwen
    from cosmos_curator.models.vllm_cosmos_reason2_vl import VllmCosmosReason2VL


pytestmark = pytest.mark.env("unified")


def _async_cfg(**overrides: object) -> VllmAsyncConfig:
    """Build a minimal ``VllmAsyncConfig`` for plugin test, defaults for everything else."""
    # ``num_gpus`` is now integer-only (mirrors sync ``VllmConfig``); the
    # ``instance_of(int)`` validator on ``VllmAsyncConfig`` rejects floats
    # outright so tests must construct with an ``int`` value here.
    base: dict[str, object] = {"model_variant": "test-variant", "num_gpus": 1}
    base.update(overrides)
    return VllmAsyncConfig(**base)  # type: ignore[arg-type]


def _sync_cfg(**overrides: object) -> VllmConfig:
    """Build a minimal ``VllmConfig`` for plugin test, defaults for everything else."""
    base: dict[str, object] = {"model_variant": "test-variant", "num_gpus": 1}
    base.update(overrides)
    return VllmConfig(**base)  # type: ignore[arg-type]


def test_qwen_model_async_uses_module_constants() -> None:
    """``VllmQwen.model_async`` reads the SAME constants as ``VllmQwen.model``."""
    with patch.object(vllm_qwen.VllmQwen, "model_path") as mock_path:
        mock_path.return_value = "/cache/qwen"
        args = vllm_qwen.VllmQwen.model_async(_async_cfg())

    assert args.model == "/cache/qwen"
    assert args.served_model_name == ["test-variant"]
    assert args.max_model_len == vllm_qwen.MAX_MODEL_LEN
    assert args.gpu_memory_utilization == vllm_qwen.GPU_MEMORY_UTILIZATION
    assert args.max_num_batched_tokens == vllm_qwen.MAX_NUM_BATCHED_TOKENS
    assert args.trust_remote_code == vllm_qwen.TRUST_REMOTE_CODE
    assert args.limit_mm_per_prompt == vllm_qwen.LIMIT_MM_PER_PROMPT_VIDEO


def test_qwen_model_async_bakes_async_invariants() -> None:
    """``mm_processor_kwargs`` carries ``do_sample_frames=False`` + preprocess gates.

    With ``config.preprocess=False`` (the default), CPU owns resize/rescale/
    normalize so the vLLM processor must skip them.  ``do_sample_frames``
    is always ``False`` on the async path because CPU prep pre-samples
    frames.  The legacy ``"size"`` kwarg was removed when the plugin
    moved to deterministic CPU-side ``smart_resize``.
    """
    with patch.object(vllm_qwen.VllmQwen, "model_path", return_value="/m"):
        args = vllm_qwen.VllmQwen.model_async(_async_cfg())
    assert args.mm_processor_kwargs is not None
    assert args.mm_processor_kwargs["do_sample_frames"] is False
    assert args.mm_processor_kwargs["do_resize"] is False
    assert args.mm_processor_kwargs["do_rescale"] is False
    assert args.mm_processor_kwargs["do_normalize"] is False
    assert "size" not in args.mm_processor_kwargs


def test_qwen_model_async_preprocess_true_enables_vllm_resize() -> None:
    """``config.preprocess=True`` -> vLLM processor owns resize/rescale/normalize."""
    with patch.object(vllm_qwen.VllmQwen, "model_path", return_value="/m"):
        args = vllm_qwen.VllmQwen.model_async(_async_cfg(preprocess=True))
    assert args.mm_processor_kwargs is not None
    assert args.mm_processor_kwargs["do_resize"] is True
    assert args.mm_processor_kwargs["do_rescale"] is True
    assert args.mm_processor_kwargs["do_normalize"] is True
    # Frame sampling is still owned by CPU prep -- ``preprocess`` only
    # gates the in-engine pixel-space ops.
    assert args.mm_processor_kwargs["do_sample_frames"] is False


def test_qwen_model_async_fp8_user_knob_routes_to_quantization() -> None:
    """``config.fp8=True`` -> ``quantization='fp8'`` (matches sync derivation)."""
    with patch.object(vllm_qwen.VllmQwen, "model_path", return_value="/m"):
        args = vllm_qwen.VllmQwen.model_async(_async_cfg(fp8=True))
    assert args.quantization == "fp8"


def test_qwen_model_async_disable_mmcache_routes_to_zero_gb() -> None:
    """``config.disable_mmcache=True`` -> ``mm_processor_cache_gb=0.0`` (matches sync)."""
    with patch.object(vllm_qwen.VllmQwen, "model_path", return_value="/m"):
        args = vllm_qwen.VllmQwen.model_async(_async_cfg(disable_mmcache=True))
    assert args.mm_processor_cache_gb == 0.0


def test_qwen3vl_model_async_omits_quantization_and_batched_tokens() -> None:
    """Qwen3-VL doesn't accept ``quantization`` / ``max_num_batched_tokens``."""
    with patch.object(vllm_qwen.VllmQwen3VL, "model_path", return_value="/m"):
        args = vllm_qwen.VllmQwen3VL.model_async(_async_cfg())
    # Defaults are None when omitted from AsyncEngineArgs constructor.
    assert args.quantization is None
    assert args.max_model_len == vllm_qwen.MAX_MODEL_LEN
    assert args.trust_remote_code == vllm_qwen.TRUST_REMOTE_CODE


def test_qwen3vl_model_async_omits_gpu_memory_utilization_when_unset() -> None:
    """Qwen3-VL must NOT pass ``gpu_memory_utilization`` when CLI is unset.

    Verifies the contract directly by patching ``AsyncEngineArgs`` and
    asserting the kwarg is absent from the constructor call -- robust
    against future tuning changes to per-plugin defaults or vLLM's
    built-in default value.
    """
    with (
        patch.object(vllm_qwen.VllmQwen3VL, "model_path", return_value="/m"),
        patch("cosmos_curator.models.vllm_qwen.AsyncEngineArgs") as mock_args,
    ):
        vllm_qwen.VllmQwen3VL.model_async(_async_cfg())
    assert "gpu_memory_utilization" not in mock_args.call_args.kwargs


def test_nemotron_model_async_uses_module_constants() -> None:
    """Nemotron's ``model_async`` reads the same module-scope constants as ``model``."""
    with patch.object(vllm_nemotron.VllmNemotronNano12Bv2VL, "model_path", return_value="/n"):
        args = vllm_nemotron.VllmNemotronNano12Bv2VL.model_async(_async_cfg())
    assert args.model == "/n"
    assert args.max_model_len == vllm_nemotron.MAX_MODEL_LEN
    assert args.gpu_memory_utilization == vllm_nemotron.GPU_MEMORY_UTILIZATION
    assert args.trust_remote_code is vllm_nemotron.TRUST_REMOTE_CODE
    assert args.limit_mm_per_prompt == vllm_nemotron.LIMIT_MM_PER_PROMPT_VIDEO


def test_nemotron_model_async_emits_preprocess_gates_with_default_preprocess() -> None:
    """Nemotron skips in-engine resize/rescale/normalize when ``config.preprocess=False``.

    Nemotron's custom processor accepts the generic ``do_resize``/``do_rescale``/
    ``do_normalize`` toggles plus ``do_sample_frames``; ``preprocess=False``
    (default) means CPU prep owns those steps so vLLM must skip them.
    The legacy assertion that ``mm_processor_kwargs`` contained ONLY
    ``do_sample_frames`` is stale -- the plugin now also forwards the
    three preprocess toggles.
    """
    with patch.object(vllm_nemotron.VllmNemotronNano12Bv2VL, "model_path", return_value="/n"):
        args = vllm_nemotron.VllmNemotronNano12Bv2VL.model_async(_async_cfg())
    assert args.mm_processor_kwargs == {
        "do_sample_frames": False,
        "do_resize": False,
        "do_rescale": False,
        "do_normalize": False,
    }


def test_nemotron_model_async_preprocess_true_enables_vllm_resize() -> None:
    """``config.preprocess=True`` -> Nemotron lets vLLM run resize/rescale/normalize."""
    with patch.object(vllm_nemotron.VllmNemotronNano12Bv2VL, "model_path", return_value="/n"):
        args = vllm_nemotron.VllmNemotronNano12Bv2VL.model_async(_async_cfg(preprocess=True))
    assert args.mm_processor_kwargs == {
        "do_sample_frames": False,
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
    }


def test_cosmos_reason1_model_async_uses_module_constants() -> None:
    """Cosmos-Reason1's ``model_async`` reads the same module-scope constants as ``model``."""
    with patch.object(vllm_cosmos_reason1_vl.VllmCosmosReason1VL, "model_path", return_value="/c"):
        args = vllm_cosmos_reason1_vl.VllmCosmosReason1VL.model_async(_async_cfg())
    assert args.max_model_len == vllm_cosmos_reason1_vl.MAX_MODEL_LEN
    assert args.gpu_memory_utilization == vllm_cosmos_reason1_vl.GPU_MEMORY_UTILIZATION
    assert args.max_num_batched_tokens == vllm_cosmos_reason1_vl.MAX_NUM_BATCHED_TOKENS
    assert args.trust_remote_code is vllm_cosmos_reason1_vl.TRUST_REMOTE_CODE
    assert args.limit_mm_per_prompt == vllm_cosmos_reason1_vl.LIMIT_MM_PER_PROMPT_VIDEO


def test_cosmos_reason1_model_async_preprocess_true_enables_vllm_resize() -> None:
    """``config.preprocess=True`` -> Cosmos-Reason1 lets vLLM run resize/rescale/normalize."""
    with patch.object(vllm_cosmos_reason1_vl.VllmCosmosReason1VL, "model_path", return_value="/c"):
        args = vllm_cosmos_reason1_vl.VllmCosmosReason1VL.model_async(_async_cfg(preprocess=True))
    assert args.mm_processor_kwargs is not None
    assert args.mm_processor_kwargs["do_resize"] is True
    assert args.mm_processor_kwargs["do_rescale"] is True
    assert args.mm_processor_kwargs["do_normalize"] is True
    assert args.mm_processor_kwargs["do_sample_frames"] is False


def test_cosmos_reason1_model_async_default_preprocess_disables_vllm_resize() -> None:
    """Default ``config.preprocess=False`` -> CPU owns resize/rescale/normalize."""
    with patch.object(vllm_cosmos_reason1_vl.VllmCosmosReason1VL, "model_path", return_value="/c"):
        args = vllm_cosmos_reason1_vl.VllmCosmosReason1VL.model_async(_async_cfg())
    assert args.mm_processor_kwargs is not None
    assert args.mm_processor_kwargs["do_resize"] is False
    assert args.mm_processor_kwargs["do_rescale"] is False
    assert args.mm_processor_kwargs["do_normalize"] is False
    assert args.mm_processor_kwargs["do_sample_frames"] is False


def test_cosmos_reason2_inherits_model_async() -> None:
    """``VllmCosmosReason2VL`` extends ``VllmCosmosReason1VL`` and inherits ``model_async``."""
    with patch.object(VllmCosmosReason2VL, "model_path", return_value="/c2"):
        args = VllmCosmosReason2VL.model_async(_async_cfg(model_variant="cosmos_r2"))
    assert args.served_model_name == ["cosmos_r2"]
    assert args.max_model_len == vllm_cosmos_reason1_vl.MAX_MODEL_LEN  # constants come from base


@pytest.mark.parametrize(
    ("plugin_module", "plugin_cls_name", "compared_keys"),
    [
        (
            "cosmos_curator.models.vllm_qwen",
            "VllmQwen",
            (
                "max_model_len",
                "gpu_memory_utilization",
                "max_num_batched_tokens",
                "trust_remote_code",
                "limit_mm_per_prompt",
            ),
        ),
        (
            "cosmos_curator.models.vllm_nemotron",
            "VllmNemotronNano12Bv2VL",
            (
                "max_model_len",
                "gpu_memory_utilization",
                "trust_remote_code",
                "limit_mm_per_prompt",
            ),
        ),
        (
            "cosmos_curator.models.vllm_cosmos_reason1_vl",
            "VllmCosmosReason1VL",
            (
                "max_model_len",
                "gpu_memory_utilization",
                "max_num_batched_tokens",
                "trust_remote_code",
                "limit_mm_per_prompt",
            ),
        ),
    ],
)
def test_model_async_matches_model_for_shared_engine_knobs(
    plugin_module: str,
    plugin_cls_name: str,
    compared_keys: tuple[str, ...],
) -> None:
    """Drift tripwire: per-variant engine knobs match between sync and async paths.

    Captures the kwargs ``model()`` passes to ``vllm.LLM`` and the corresponding
    fields on the ``AsyncEngineArgs`` returned by ``model_async()``, then asserts
    every shared engine-init knob has the same value.  Catches regressions where
    someone retunes one path but forgets the other -- without coupling to source
    text or constant naming.
    """
    module = importlib.import_module(plugin_module)
    plugin_cls = getattr(module, plugin_cls_name)

    with (
        patch.object(module, "LLM") as mock_llm,
        patch.object(plugin_cls, "model_path", return_value="/m"),
    ):
        plugin_cls.model(_sync_cfg())
    sync_kwargs = mock_llm.call_args.kwargs

    with patch.object(plugin_cls, "model_path", return_value="/m"):
        async_args = plugin_cls.model_async(_async_cfg())

    for key in compared_keys:
        sync_value = sync_kwargs[key]
        async_value = getattr(async_args, key)
        assert sync_value == async_value, (
            f"{plugin_cls_name}: '{key}' diverged between sync model() "
            f"({sync_value!r}) and async model_async() ({async_value!r}) -- "
            f"sync and async must agree on per-variant engine tuning."
        )


@pytest.mark.parametrize(
    ("plugin_module", "plugin_cls_name"),
    [
        ("cosmos_curator.models.vllm_qwen", "VllmQwen"),
        ("cosmos_curator.models.vllm_qwen", "VllmQwen3VL"),
        ("cosmos_curator.models.vllm_nemotron", "VllmNemotronNano12Bv2VL"),
        ("cosmos_curator.models.vllm_cosmos_reason1_vl", "VllmCosmosReason1VL"),
    ],
)
def test_model_async_propagates_cli_gpu_memory_utilization(
    plugin_module: str,
    plugin_cls_name: str,
) -> None:
    """Explicit ``config.gpu_memory_utilization`` overrides the per-plugin default."""
    import importlib  # noqa: PLC0415

    module = importlib.import_module(plugin_module)
    plugin_cls = getattr(module, plugin_cls_name)

    with patch.object(plugin_cls, "model_path", return_value="/m"):
        async_args = plugin_cls.model_async(_async_cfg(gpu_memory_utilization=0.5))

    assert async_args.gpu_memory_utilization == 0.5
