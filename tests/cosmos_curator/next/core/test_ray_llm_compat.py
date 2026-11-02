# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray's multimodal preparation stage resolves vLLM's relocated chat parser."""

import asyncio
import sys
import types
from collections.abc import AsyncIterator, Iterator
from typing import Any

import attrs
import pytest
from ray.llm._internal.batch.stages import prepare_multimodal_stage

from cosmos_curator.next.core.ray_llm_compat import (
    PREPARE_MULTIMODAL_STAGE,
    RAY_CHAT_UTILS_MODULE,
    VLLM_CHAT_UTILS_MODULE,
    PrepareMultimodalUDF,
    install_prepare_multimodal_shim,
)


@attrs.define
class _StubProcessor:
    """Minimal stand-in exposing the two accessors the shim relies on."""

    stages: dict[str, Any]

    def list_stage_names(self) -> list[str]:
        """Return the stage names in insertion order."""
        return list(self.stages)

    def get_stage_by_name(self, name: str) -> Any:  # noqa: ANN401 -- stubs vary by test
        """Return one stage by name."""
        return self.stages[name]


def _skip_ray_constructor(self: object, *args: object, **kwargs: object) -> None:
    """Stand in for Ray's constructor, which builds a vLLM ModelConfig from model files."""


async def _drain(rows: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row async for row in rows]


@pytest.fixture
def unaliased_chat_utils(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start without the old module path and drop any alias a test installs."""
    monkeypatch.delitem(sys.modules, RAY_CHAT_UTILS_MODULE, raising=False)
    monkeypatch.setattr(prepare_multimodal_stage.PrepareMultimodalUDF, "__init__", _skip_ray_constructor)
    yield
    sys.modules.pop(RAY_CHAT_UTILS_MODULE, None)


def test_shim_replaces_only_the_multimodal_stage_udf() -> None:
    """The preparation stage runs the subclass under Ray's operator name; other stages are untouched."""
    stage = prepare_multimodal_stage.PrepareMultimodalStage(fn_constructor_kwargs={}, map_batches_kwargs={})
    engine_stage = object()
    processor = _StubProcessor({PREPARE_MULTIMODAL_STAGE: stage, "vLLMEngineStage": engine_stage})

    install_prepare_multimodal_shim(processor)

    assert stage.fn is PrepareMultimodalUDF
    assert PrepareMultimodalUDF.__name__ == prepare_multimodal_stage.PrepareMultimodalUDF.__name__
    assert processor.get_stage_by_name("vLLMEngineStage") is engine_stage


def test_shim_ignores_a_processor_without_multimodal_preparation() -> None:
    """Text-only processors have no preparation stage to replace."""
    processor = _StubProcessor({"vLLMEngineStage": object()})

    install_prepare_multimodal_shim(processor)

    assert processor.list_stage_names() == ["vLLMEngineStage"]


def test_shim_refuses_to_replace_an_unexpected_udf() -> None:
    """A UDF other than Ray's pinned one is not silently overridden."""

    class _OtherUDF(prepare_multimodal_stage.PrepareMultimodalUDF):
        pass

    stage = types.SimpleNamespace(fn=_OtherUDF)

    with pytest.raises(TypeError, match="unexpected UDF type"):
        install_prepare_multimodal_shim(_StubProcessor({PREPARE_MULTIMODAL_STAGE: stage}))
    assert stage.fn is _OtherUDF


@pytest.mark.usefixtures("unaliased_chat_utils")
def test_ray_batch_handler_imports_the_relocated_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the UDF aliases the module, so Ray's own per-batch import resolves."""
    relocated = types.ModuleType(VLLM_CHAT_UTILS_MODULE)
    relocated.parse_chat_messages_async = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, VLLM_CHAT_UTILS_MODULE, relocated)

    udf = PrepareMultimodalUDF()

    assert sys.modules[RAY_CHAT_UTILS_MODULE] is relocated
    assert asyncio.run(_drain(udf.udf([]))) == []


@pytest.mark.env("default")
@pytest.mark.usefixtures("unaliased_chat_utils")
def test_ray_batch_handler_imports_the_pinned_vllm_parser() -> None:
    """Under the pinned vLLM, the alias points Ray's import at vLLM's real parser."""
    from vllm.renderers import chat_utils  # noqa: PLC0415 -- default environment only

    udf = PrepareMultimodalUDF()

    assert sys.modules[RAY_CHAT_UTILS_MODULE] is chat_utils
    assert asyncio.run(_drain(udf.udf([]))) == []
