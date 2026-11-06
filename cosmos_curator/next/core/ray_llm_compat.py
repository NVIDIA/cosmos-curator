# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Ray 2.60's multimodal preparation stage against vLLM 0.32 and later.

Ray Data LLM's ``PrepareMultimodalStage`` imports ``parse_chat_messages_async``
from ``vllm.entrypoints.chat_utils`` on every batch. vLLM 0.32 moved that module,
unchanged, to ``vllm.renderers.chat_utils``. Ray master imports the new path
(ray-project/ray#67197), but the Ray 2.60.0 release branch predates the change.
Until the Ray pin includes it, swap the stage's UDF for a subclass that aliases
the old module path in its own actor process. Delete this module with that Ray
upgrade. This workaround depends on the pinned Ray processor and stage interfaces.
"""

import importlib
import sys
from typing import Any

from ray.llm._internal.batch.stages import prepare_multimodal_stage

PREPARE_MULTIMODAL_STAGE = "PrepareMultimodalStage"
RAY_CHAT_UTILS_MODULE = "vllm.entrypoints.chat_utils"
VLLM_CHAT_UTILS_MODULE = "vllm.renderers.chat_utils"


class PrepareMultimodalUDF(prepare_multimodal_stage.PrepareMultimodalUDF):
    """Ray's multimodal preparation UDF, with vLLM's chat utilities at the path Ray imports.

    Keeps Ray's class name so the operator still reports as ``MapBatches(PrepareMultimodalUDF)``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 -- forwards Ray's constructor arguments
        """Alias the relocated module, then run Ray's constructor."""
        # The alias must exist in the actor process that runs udf(), not on the driver.
        sys.modules.setdefault(RAY_CHAT_UTILS_MODULE, importlib.import_module(VLLM_CHAT_UTILS_MODULE))
        super().__init__(*args, **kwargs)


def install_prepare_multimodal_shim(processor: Any) -> None:  # noqa: ANN401 -- public Ray Processor facade
    """Run the processor's multimodal preparation stage, if it has one, through :class:`PrepareMultimodalUDF`."""
    if PREPARE_MULTIMODAL_STAGE not in processor.list_stage_names():
        return
    stage = processor.get_stage_by_name(PREPARE_MULTIMODAL_STAGE)
    if stage.fn is not prepare_multimodal_stage.PrepareMultimodalUDF:
        msg = f"Ray {PREPARE_MULTIMODAL_STAGE} uses unexpected UDF type {stage.fn!r}"
        raise TypeError(msg)
    # Ray annotates this stage's fn as a UDF instance rather than a class, so pydantic's assignment
    # validation rejects every class. Ray's own default passes only because pydantic skips defaults.
    object.__setattr__(stage, "fn", PrepareMultimodalUDF)
