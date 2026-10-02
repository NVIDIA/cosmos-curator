# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve typed model identities and validate staged model directories."""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import attrs

from cosmos_curator.core.utils import environment

_REGISTRY_PACKAGE = "cosmos_curator"
_REGISTRY_PARTS = ("configs", "all_models.json")
_MODEL_WEIGHT_SUFFIXES = frozenset({".bin", ".pt", ".safetensors"})


@attrs.frozen
class RegisteredModel:
    """Represent one pinned model identity from the shipped registry.

    Attributes:
        key: Registry key that selected this entry.
        model_id: Hugging Face repository id.
        revision: Immutable commit pinning the weights.
        precision: Weight precision recorded in the registry.

    """

    key: str
    model_id: str
    revision: str
    precision: str

    @property
    def runtime_model_dir(self) -> Path:
        """Return the weight directory mounted identically on every worker."""
        return environment.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR / self.model_id


def resolve_registered_model(key: str) -> RegisteredModel:
    """Resolve one registry key's pinned model id, revision, and precision.

    Read through ``importlib.resources`` rather than the legacy raw accessor,
    which resolves against a container-only source path.

    Args:
        key: Registry key to resolve.

    Returns:
        The resolved model identity.

    Raises:
        KeyError: If the key is absent from the registry.
        TypeError: If the registry or the key's entry is malformed, or the entry
            is incomplete.

    """
    registry_path = f"{_REGISTRY_PACKAGE}/{'/'.join(_REGISTRY_PARTS)}"
    registry_text = files(_REGISTRY_PACKAGE).joinpath(*_REGISTRY_PARTS).read_text(encoding="utf-8")
    try:
        raw: Any = json.loads(registry_text)
    except json.JSONDecodeError as error:
        msg = f"Model registry {registry_path} must contain valid JSON"
        raise TypeError(msg) from error
    if not isinstance(raw, dict):
        msg = f"Model registry {registry_path} must hold an object at its root"
        raise TypeError(msg)
    registry: dict[str, Any] = raw

    # Two refusals with different owners, so two types. An unregistered key is
    # the caller's typo, actionable by naming a key the message lists, and a
    # caller guarding it writes `except KeyError`; TypeError means "wrong type
    # of argument" and would escape that guard. A malformed entry is a fault in
    # the shipped registry, which no caller can act on, so it stays a TypeError
    # alongside the incomplete-entry case below.
    if key not in registry:
        msg = f"Model {key!r} is absent from {registry_path}. Registered: {sorted(registry)}"
        raise KeyError(msg)
    entry = registry[key]
    if not isinstance(entry, dict):
        msg = f"Model {key!r} must hold an object in {registry_path}"
        raise TypeError(msg)
    # ``version`` in the registry IS the revision; the rename is the whole reason
    # a second hand-rolled parser is a liability rather than duplication.
    model_id, revision, precision = entry.get("model_id"), entry.get("version"), entry.get("precision")
    if not all(isinstance(value, str) and value.strip() for value in (model_id, revision, precision)):
        msg = f"Model {key!r} must have nonblank model_id, version, and precision"
        raise TypeError(msg)
    return RegisteredModel(
        key=key,
        model_id=str(model_id),
        revision=str(revision),
        precision=str(precision),
    )


def validate_model_directory(model_dir: Path, *, model_key: str, require_config_json: bool = True) -> Path:
    """Require staged weights and, for models that ship one, a config.json.

    Args:
        model_dir: Model directory to validate.
        model_key: Download key to name in actionable errors.
        require_config_json: Whether the checkpoint supplies its own config.json.

    Returns:
        The validated directory.

    Raises:
        FileNotFoundError: If the directory or required model files are absent.

    """
    download_command = f"pixi run --as-is model-download --models {model_key}"
    if not model_dir.is_dir():
        msg = f"Model weights for {model_key!r} are not staged at {model_dir}. Run: {download_command}"
        raise FileNotFoundError(msg)
    config_path = model_dir / "config.json"
    has_model_file = any(path.is_file() and path.suffix in _MODEL_WEIGHT_SUFFIXES for path in model_dir.rglob("*"))
    if (require_config_json and not config_path.is_file()) or not has_model_file:
        expected = "config.json and staged model files" if require_config_json else "staged model files"
        msg = (
            f"Model directory {model_dir} for {model_key!r} is incomplete: expected {expected}. Run: {download_command}"
        )
        raise FileNotFoundError(msg)
    return model_dir
