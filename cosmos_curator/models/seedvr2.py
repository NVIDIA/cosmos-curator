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

"""SeedVR2 super-resolution model interface.

SeedVR2 is a diffusion-based video super-resolution model from ByteDance.
It uses a VAE encoder/decoder with a DiT (Diffusion Transformer) backbone
to upscale video frames while preserving temporal consistency.

See: https://github.com/ByteDance-Seed/SeedVR
"""

import gc
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.model import model_utils

_VARIANT_TO_MODEL_ID: dict[str, str] = {
    "seedvr2_3b": "ByteDance-Seed/SeedVR2-3B",
    "seedvr2_7b": "ByteDance-Seed/SeedVR2-7B",
    "seedvr2_7b_sharp": "ByteDance-Seed/SeedVR2-7B",
}


def _ensure_seedvr_importable() -> str:
    """Ensure the SeedVR repo modules are importable via the ``SEEDVR_ROOT`` env var.

    In the Docker image, the SeedVR tarball is installed to a known path and
    ``SEEDVR_ROOT`` is set by the Dockerfile (see ``default.dockerfile.jinja2``).
    For local development, set ``SEEDVR_ROOT`` to a local SeedVR checkout.

    Returns:
        The resolved SEEDVR_ROOT path.

    Raises:
        RuntimeError: If ``SEEDVR_ROOT`` is not set or points to a missing directory.

    """
    if importlib.util.find_spec("common") is not None:
        return os.environ.get("SEEDVR_ROOT", "").strip()

    seedvr_root = os.environ.get("SEEDVR_ROOT", "").strip()
    if not seedvr_root:
        msg = (
            "SEEDVR_ROOT environment variable is not set. "
            "In the Docker image this is set automatically by the Dockerfile. "
            "For local development, set it to the path of a SeedVR checkout."
        )
        raise RuntimeError(msg)

    if not Path(seedvr_root).is_dir():
        msg = f"SEEDVR_ROOT={seedvr_root} does not exist or is not a directory."
        raise RuntimeError(msg)

    if seedvr_root not in sys.path:
        sys.path.insert(0, seedvr_root)
    return seedvr_root


def _install_apex_shim() -> None:
    """Register a lightweight ``apex`` shim so SeedVR can import ``apex.normalization``.

    SeedVR configs use ``fusedrms`` / ``fusedln`` norm types which import
    ``apex.normalization.FusedRMSNorm`` / ``FusedLayerNorm``.  The SeedVR
    README provides pre-built apex wheels, but only for Python 3.9/3.10 +
    torch 2.4.0 + CUDA 12.1/12.4 — incompatible with our environment
    (Python 3.12 + torch 2.10 + CUDA 13.0).  Building apex from source
    requires CUDA compilation that is fragile and slow.  Since modern
    PyTorch already ships functionally equivalent fused kernels
    (``torch.nn.RMSNorm``, ``torch.nn.LayerNorm``), we inject a shim
    module that aliases them, avoiding the compilation entirely.
    """
    if "apex" in sys.modules:
        return

    import importlib.machinery  # noqa: PLC0415

    import torch  # noqa: PLC0415

    apex_mod = types.ModuleType("apex")
    apex_mod.__path__ = []  # type: ignore[attr-defined]
    apex_mod.__spec__ = importlib.machinery.ModuleSpec("apex", None, is_package=True)

    norm_mod = types.ModuleType("apex.normalization")
    norm_mod.FusedRMSNorm = torch.nn.RMSNorm  # type: ignore[attr-defined]
    norm_mod.FusedLayerNorm = torch.nn.LayerNorm  # type: ignore[attr-defined]
    norm_mod.__spec__ = importlib.machinery.ModuleSpec("apex.normalization", None)

    apex_mod.normalization = norm_mod  # type: ignore[attr-defined]
    sys.modules["apex"] = apex_mod
    sys.modules["apex.normalization"] = norm_mod
    logger.info("Installed apex shim (FusedRMSNorm -> torch.nn.RMSNorm)")


def _ensure_torch_distributed_env() -> None:
    """Set torch.distributed env vars for single-process execution if not already set.

    SeedVR's ``configure_runner`` calls ``init_torch`` which initialises a
    ``torch.distributed`` process group.  When running inside a Ray actor
    (not launched via ``torchrun``), the required env vars are absent.

    In Curator, each SuperResolutionStage worker is a single Ray actor with
    one GPU (CuratorStageResource gpus=1).  Multi-clip parallelism comes from
    running many 1-GPU actors, not from splitting one clip across GPUs.
    This differs from SeedVR's native ``torchrun --nproc-per-node=N`` path
    where ``sp_size=N`` distributes one video across N GPUs via sequence
    parallelism (and ``torchrun`` sets these env vars automatically).

    If ``sp_size > 1`` multi-GPU-per-clip support is needed in the future,
    this shim would need to be replaced with proper per-actor distributed
    rank assignment.
    """
    defaults = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value


class SeedVR2(ModelInterface):
    """Interface for the SeedVR2 video super-resolution model.

    The constructor runs in the default conda environment and only stores
    configuration. The actual model loading happens in ``setup()`` which
    runs inside the target conda environment on the worker actor.
    """

    def __init__(self, variant: str = "seedvr2_7b", sp_size: int = 1) -> None:
        """Initialize SeedVR2 model configuration.

        Args:
            variant: Model variant ('seedvr2_3b', 'seedvr2_7b', 'seedvr2_7b_sharp').
            sp_size: Sequence parallelism size.

        """
        super().__init__()
        if variant not in _VARIANT_TO_MODEL_ID:
            msg = f"Unknown SeedVR2 variant: {variant}. Choose from: {', '.join(_VARIANT_TO_MODEL_ID)}"
            raise ValueError(msg)
        self._variant = variant
        self._sp_size = sp_size
        self._runner: Any = None

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "seedvr"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of HuggingFace model IDs for weight download.

        """
        return [_VARIANT_TO_MODEL_ID[self._variant]]

    def setup(self) -> None:
        """Load the SeedVR2 model weights and configure the runner.

        This runs inside the target conda environment on the worker actor.
        It calls the vendored ``configure_runner`` which imports the SeedVR2
        variant module and builds the diffusion runner with VAE and DiT components.
        """
        import torch  # noqa: PLC0415

        from cosmos_curator.pipelines.video.super_resolution.inference_seedvr2_window import (  # noqa: PLC0415
            configure_runner,
        )

        logger.info(f"Setting up SeedVR2 model variant={self._variant} sp_size={self._sp_size}")
        seedvr_root = _ensure_seedvr_importable()

        model_dir = model_utils.get_local_dir_for_weights_name(self.model_id_names[0])
        logger.info(f"SeedVR2 model weights directory: {model_dir}")

        _install_apex_shim()

        # SeedVR source code and model weights live in different locations:
        #   - Source: $SEEDVR_ROOT (e.g. /opt/cosmos-curator/SeedVR from the tarball)
        #   - Weights: /config/models/ByteDance-Seed/SeedVR2-* (downloaded via model_cli)
        # SeedVR inference scripts hardcode checkpoint paths like ./ckpts/seedvr2_ema_3b.pth
        # relative to the repo root. This symlink bridges the two locations.
        ckpts_link = Path(seedvr_root) / "ckpts"
        if not ckpts_link.exists():
            ckpts_link.symlink_to(model_dir)
            logger.info(f"Symlinked {ckpts_link} -> {model_dir}")

        _ensure_torch_distributed_env()

        self._runner = configure_runner(self._sp_size, variant=self._variant)

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("SeedVR2 model setup complete")

    @property
    def runner(self) -> Any:  # noqa: ANN401
        """Get the configured SeedVR2 runner.

        Returns:
            The SeedVR2 diffusion runner instance.

        Raises:
            RuntimeError: If setup() has not been called.

        """
        if self._runner is None:
            msg = "SeedVR2 runner not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._runner

    @property
    def variant_module(self) -> types.ModuleType:
        """Get the SeedVR2 variant inference module.

        Returns:
            The variant module containing generation_step and other utilities.

        Raises:
            RuntimeError: If setup() has not been called.

        """
        if self._runner is None:
            msg = "SeedVR2 variant module not loaded. Call setup() first."
            raise RuntimeError(msg)
        module = getattr(self._runner, "_seedvr_window_variant", None)
        if module is None:
            msg = "Runner is missing '_seedvr_window_variant' attribute."
            raise RuntimeError(msg)
        return module  # type: ignore[no-any-return, return-value]
