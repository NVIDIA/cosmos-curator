"""Test-only mock vLLM plugin and engine.

This module provides lightweight stand-ins for the vLLM types used by
`cosmos_curator.models.vllm_interface` so unit tests can run without
pulling heavy dependencies or models.
"""

import secrets
from collections.abc import Callable
from typing import cast

from transformers import AutoProcessor
from vllm import CompletionOutput, RequestOutput

from cosmos_curator.models.vllm_plugin import VllmPlugin
from cosmos_curator.pipelines.video.utils.data_model import VllmCaptionRequest, VllmConfig

# ruff: noqa: ARG002, D102, D107


def make_request_output(
    request_id: str, text: str, *, finished: bool = True, finish_reason: str | None = "stop"
) -> RequestOutput:
    """Make an instance of vLLM's RequestOutput with a single CompletionOutput.

    Args:
        request_id: The request id.
        text: The text of the completion.
        finished: Whether the request is finished.
        finish_reason: The completion finish reason to attach to the first output.

    Returns:
        An instance of vLLM's RequestOutput.

    """
    output = CompletionOutput(index=0, token_ids=[], cumulative_logprob=0.0, logprobs=[], text=text)
    output.finish_reason = finish_reason
    return RequestOutput(
        request_id=request_id,
        outputs=[output],
        finished=finished,
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=[],
    )


class MockLLMEngine:
    """Simple engine with `add_request` and `step` methods.

    Args:
        steps_to_complete: Either a fixed int (default 3) applied to all requests,
            or a callable ``(request_index) -> int`` that returns the number of
            ``step()`` calls needed before a given request completes.  Use a
            callable to make requests finish out-of-order.

    """

    def __init__(self, steps_to_complete: int | Callable[[int], int] = 3) -> None:
        self._pending: list[tuple[str, dict[str, object], int]] = []
        self._steps_to_complete = steps_to_complete
        self._caption_idx = 0
        self._request_count = 0

    def add_request(self, request_id: str, inputs: dict[str, object], _sampling_params: object) -> None:
        if callable(self._steps_to_complete):
            steps = self._steps_to_complete(self._request_count)
        else:
            steps = self._steps_to_complete
        self._request_count += 1
        self._pending.append((request_id, inputs, steps))

    def step(self) -> list[RequestOutput]:
        finished: list[RequestOutput] = []
        still_pending: list[tuple[str, dict[str, object], int]] = []

        for req_id, _inputs, rem in self._pending:
            new_rem = rem - 1
            if new_rem <= 0:
                finished.append(make_request_output(req_id, f"mock-caption-{self._caption_idx}"))
                self._caption_idx += 1
            else:
                still_pending.append((req_id, _inputs, new_rem))

        self._pending = still_pending
        return finished


class MockLLM:
    """Minimal `LLM` replacement with `generate` and `llm_engine`."""

    def __init__(self, steps_to_complete: int | Callable[[int], int] = 3) -> None:
        self.llm_engine = MockLLMEngine(steps_to_complete)
        self._caption_idx = 0

    def generate(
        self,
        batch: list[dict[str, object]],
        *,
        sampling_params: object | None = None,
        use_tqdm: bool | None = None,
    ) -> list[RequestOutput]:
        captions = [f"mock-caption-{i + self._caption_idx}" for i in range(len(batch))]
        ids = [f"id{i + self._caption_idx}" for i in range(len(batch))]
        results = [make_request_output(i, caption) for i, caption in zip(ids, captions, strict=True)]
        self._caption_idx += len(batch)
        return results


class MockVllmPlugin(VllmPlugin):
    """Mock plugin implementing the `VllmPlugin` interface for tests."""

    @staticmethod
    def model_variant() -> str:  # pragma: no cover - constant
        return "mock"

    @classmethod
    def processor(cls, _config: VllmConfig) -> AutoProcessor:
        # Avoid heavyweight initialization in tests.
        return cast("AutoProcessor", object())

    @classmethod
    def model(cls, _config: VllmConfig) -> MockLLM:  # type: ignore[override]
        return MockLLM()

    @staticmethod
    def make_llm_input(
        prompt: str,
        frames: object,
        metadata: dict[str, object],
        _processor: AutoProcessor,
        _config: VllmConfig,
    ) -> dict[str, object]:
        return {"prompt": prompt, "multi_modal_data": {"video": [(frames, metadata)]}}

    @staticmethod
    def make_refined_llm_input(
        caption: str,
        prev_input: dict[str, object],
        _processor: AutoProcessor,
        refine_prompt: str | None = None,
    ) -> dict[str, object]:
        refine_prefix = refine_prompt or "Refine: "
        return {"prompt": f"{refine_prefix}{caption}", "multi_modal_data": prev_input.get("multi_modal_data", {})}

    @staticmethod
    def make_refined_llm_request(
        request: VllmCaptionRequest,
        processor: AutoProcessor,
        refine_prompt: str | None = None,
    ) -> VllmCaptionRequest:
        if request.caption is None:
            msg = "Request caption is None"
            raise ValueError(msg)
        inputs = MockVllmPlugin.make_refined_llm_input(request.caption, request.inputs, processor, refine_prompt)
        return VllmCaptionRequest(request_id=secrets.token_hex(8), inputs=inputs)

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        return vllm_output.outputs[0].text
