# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-focused LM wrapper supporting both local (vLLM) and remote (OpenAI API) inference.

This module provides a unified implementation for chat-style text generation that can use:
- Local models via vLLM for on-premises inference
- OpenAI API for hosted inference

Key capabilities:
- Unified interface for both local and remote models via variant selection
- For local: Resolves weights/tokenizer and constructs a vLLM `LLM` with optional quantization
- For remote: Uses OpenAI API with model selection
- Formats prompts with chat templates for consistent behavior across backends
- Supports batching and provides `make_chat_lm_input` helper
- Designed for the "unified" conda environment and emits NVTX ranges
"""

from typing import TYPE_CHECKING, cast

from loguru import logger
from nvtx import nvtx  # type: ignore[import-untyped]

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.config.config import maybe_load_config, resolve_model_name_auto
from cosmos_curator.core.utils.misc import grouping
from cosmos_curator.core.utils.model import model_utils, pixi_utils

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
    from vllm.model_executor.layers.quantization import QuantizationMethods

if pixi_utils.is_running_in_env("unified"):
    from openai import OpenAI as OpenAIClient
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams


_VARIANT_TO_MODEL_ID: dict[str, str] = {
    "qwen_lm": "Qwen/Qwen2.5-14B-Instruct",
    "gpt_oss_20b": "openai/gpt-oss-20b",
}

_LOCAL_VARIANTS = {"qwen_lm", "gpt_oss_20b"}
_REMOTE_VARIANTS = {"openai"}


class ChatLM(ModelInterface):
    """Unified chat LM supporting both local (vLLM) and remote (OpenAI API) backends."""

    def __init__(
        self,
        model_variant: str,
        *,
        max_output_tokens: int = 2048,
        quantization: str | None = None,
        openai_model: str = "auto",
        verbose: bool = False,
    ) -> None:
        """Initialize the ChatLM.

        Args:
            model_variant: Short variant key (e.g., "qwen_lm", "openai").
            max_output_tokens: Maximum tokens to generate per prompt.
            quantization: Optional quantization override for vLLM (e.g., "fp8").
                Only applies to local variants. If None, vLLM uses the model's config-defined quantization.
            openai_model: OpenAI API model name (only used when model_variant is "openai").
            verbose: Whether to emit verbose debug logs (e.g., OpenAI request metadata).

        """
        super().__init__()
        self._model_variant = model_variant
        self._is_local = model_variant in _LOCAL_VARIANTS
        self._is_remote = model_variant in _REMOTE_VARIANTS

        if not self._is_local and not self._is_remote:
            error = f"Unsupported chat LM variant: {model_variant}"
            raise ValueError(error)

        self._model_id = _VARIANT_TO_MODEL_ID.get(model_variant) if self._is_local else None
        self.max_output_tokens = max_output_tokens
        self._quantization = cast("QuantizationMethods | None", quantization)
        self._openai_model = openai_model
        self._openai_base_url: str | None = None
        self._temperature = 0.1
        self._top_p = 0.001
        self._verbose = verbose

        # Warn about ignored parameters for remote variants
        if self._is_remote and quantization is not None:
            logger.warning(
                f"quantization parameter ('{quantization}') is ignored for remote model variant '{model_variant}'"
            )

        # Early validation for OpenAI to fail fast
        if self._is_remote:
            self._resolve_openai_settings()

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name used for this model."""
        return "unified"

    @property
    def requires_gpu(self) -> bool:
        """Check if this model requires GPU resources.

        Returns:
            True for local models (vLLM), False for remote API models.

        """
        return self._is_local

    @property
    def model_id_names(self) -> list[str]:
        """Return the underlying model identifiers."""
        if self._is_local:
            assert self._model_id is not None
            return [self._model_id]
        # Remote API models don't require local weight downloads
        return []

    def _resolve_openai_settings(self) -> tuple[str, str | None]:
        """Load OpenAI settings from the cosmos-curator config file.

        Returns:
            Tuple of (api_key, base_url).

        """
        config = maybe_load_config()
        endpoint = config.openai.enhance if config is not None and config.openai is not None else None

        if endpoint is None or not endpoint.api_key:
            error_msg = (
                "OpenAI enhance configuration not found. "
                "Provide openai.enhance.api_key in ~/.config/cosmos_curator/config.yaml"
            )
            raise RuntimeError(error_msg)

        return endpoint.api_key, endpoint.base_url

    @nvtx.annotate("Setup Chat LM model")  # type: ignore[untyped-decorator]
    def setup(self) -> None:
        """Set up the model and tokenizer, and sampling parameters."""
        if self._is_local:
            assert self._model_id is not None
            self.weight_file = str(model_utils.get_local_dir_for_weights_name(self._model_id))

            # Construct vLLM LLM. Avoid forcing quantization unless explicitly requested,
            # so that model-config (e.g., mxfp4) is honored by default.
            if self._quantization is not None:
                self.llm = LLM(
                    model=self.weight_file,
                    quantization=self._quantization,
                    enforce_eager=False,
                )
            else:
                self.llm = LLM(model=self.weight_file, enforce_eager=False)

            self.sampling_params = SamplingParams(
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=1.05,
                max_tokens=self.max_output_tokens,
                stop_token_ids=[],
            )

            # Prefer local tokenizer to avoid Hub lookups when weights are local.
            self.tokenizer = AutoTokenizer.from_pretrained(self.weight_file)  # type: ignore[no-untyped-call]

        elif self._is_remote:
            # Set up OpenAI client
            api_key, base_url = self._resolve_openai_settings()
            self._openai_base_url = base_url
            client_kwargs: dict[str, object] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.openai_client: OpenAIClient = OpenAIClient(**client_kwargs)  # type: ignore[arg-type]
            self._openai_model = resolve_model_name_auto(
                self.openai_client, self._openai_model, endpoint_label="OpenAI enhance"
            )

    @nvtx.annotate("Chat LM Generate tokens")  # type: ignore[untyped-decorator]
    def generate(
        self,
        prompts: list[list[dict[str, str]]],
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate text given chat prompts.

        Args:
            prompts: Batched chat prompts, each as a list of role/content dicts.
            batch_size: Batch size for generation. Defaults to 32 for local models,
                full batch for remote models.

        Returns:
            List of generated strings, one per prompt.

        """
        if self._is_local:
            return self._generate_local(prompts, batch_size or 32)
        if self._is_remote:
            return self._generate_remote(prompts, batch_size)
        error = f"Unknown model variant: {self._model_variant}"
        raise RuntimeError(error)

    def _generate_local(self, prompts: list[list[dict[str, str]]], batch_size: int) -> list[str]:
        """Generate text using local vLLM backend."""
        generated_text: list[str] = []
        for batch_prompts in grouping.split_by_chunk_size(prompts, batch_size):
            formatted_prompts = self.tokenizer.apply_chat_template(
                list(batch_prompts), tokenize=False, add_generation_prompt=True
            )
            outputs = self.llm.generate(formatted_prompts, sampling_params=self.sampling_params, use_tqdm=False)
            generated_text.extend([out.outputs[0].text for out in outputs])

        return generated_text

    def _generate_remote(self, prompts: list[list[dict[str, str]]], batch_size: int | None) -> list[str]:
        """Generate text using OpenAI API."""
        if not prompts:
            return []
        # batch_size retained for interface parity; OpenAI API processes one conversation at a time.
        _ = batch_size

        outputs: list[str] = []

        for message_bundle in prompts:
            messages: list[dict[str, str]] = [
                {"role": message["role"], "content": str(message["content"])} for message in message_bundle
            ]
            try:
                if self._verbose:
                    logger.info(
                        "OpenAI request (model='{}', messages={}, roles={}, max_output_tokens={}, base_url={})",
                        self._openai_model,
                        len(messages),
                        [msg["role"] for msg in messages],
                        self.max_output_tokens,
                        self._openai_base_url,
                    )
                response = self.openai_client.responses.create(
                    model=self._openai_model,
                    input=messages,  # type: ignore[arg-type]
                    max_output_tokens=self.max_output_tokens,
                )
                usage_info = getattr(response, "usage", None)
                if self._verbose:
                    if usage_info:
                        logger.info(
                            ("OpenAI response usage (model='{}', input_tokens={}, output_tokens={}, total_tokens={})"),
                            self._openai_model,
                            getattr(usage_info, "input_tokens", None),
                            getattr(usage_info, "output_tokens", None),
                            getattr(usage_info, "total_tokens", None),
                        )
                    else:
                        logger.info(
                            "OpenAI response (model='{}') returned without usage metadata",
                            self._openai_model,
                        )
            except Exception as exc:  # noqa: BLE001  # pragma: no cover
                logger.error(
                    "OpenAI API call failed for model {}: {}",
                    self._openai_model,
                    exc,
                )
                outputs.append("")
                continue
            content = getattr(response, "output_text", "")
            if not content:
                logger.warning("OpenAI API returned empty output for model {}", self._openai_model)
            outputs.append(content)

        return outputs


def make_chat_lm_input(
    user_content: list[str],
    *,
    prompt_variant_key: str | None = None,
    prompt_variants: dict[str, str] | None = None,
    prompt_text: str | None = None,
) -> list[list[dict[str, str]]]:
    """Generate chat-style inputs given user content and a prompt source.

    Exactly one of (prompt_text) or (prompt_variant_key+prompt_variants) must be provided.

    Args:
        user_content: List of user messages to send to the model
        prompt_variant_key: Key to select prompt from prompt_variants
        prompt_variants: Mapping of prompt variants to prompt text
        prompt_text: Direct prompt text

    Returns:
        A list of chat messages (system+user) per input content.

    """
    if prompt_variant_key is not None and prompt_variants is None:
        error = "prompt_variant_key provided but no prompt_variants"
        raise ValueError(error)
    if prompt_variant_key is not None and prompt_text is not None:
        error = "Cannot provide both prompt_variant_key and prompt_text"
        raise ValueError(error)
    if prompt_variant_key is None and prompt_variants is None and prompt_text is None:
        error = "Must provide either prompt_variant_key+prompt_variants or prompt_text"
        raise ValueError(error)

    if prompt_text is not None:
        prompt = prompt_text
    else:
        assert prompt_variants is not None
        assert prompt_variant_key is not None
        prompt = prompt_variants[prompt_variant_key]

    return [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
        for content in user_content
    ]
