# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lookup that turns a registry key into a pinned model identity.

``version`` in the shipped JSON is what every caller means by "revision", so a
second hand-rolled parser reading ``revision`` finds nothing and stamps a
caption with an identity that never produced it. The typed accessor validates
that identity; the legacy raw catalog and model-download path retain ownership
of file-selection data.

The registry itself is not asserted against: which models ship is data that
changes, and pinning it here would make adding a model fail an unrelated test.
Every case below builds its own registry.
"""

import json
import pathlib
from typing import Any

import attrs
import pytest

from cosmos_curator.models import registry as model_registry
from cosmos_curator.models.registry import (
    RegisteredModel,
    resolve_registered_model,
)

_ENTRY: dict[str, Any] = {
    "model_id": "vendor/model-name",
    "version": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    "precision": "FP8",
}


@pytest.fixture
def registry_file(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the synthetic registry path the lookup reads."""
    path = tmp_path / "all_models.json"
    monkeypatch.setattr(model_registry, "_REGISTRY_PACKAGE", "cosmos_curator")
    monkeypatch.setattr(model_registry, "_REGISTRY_PARTS", ())
    monkeypatch.setattr(model_registry, "files", lambda _package: path)
    return path


@pytest.fixture
def registry(registry_file: pathlib.Path) -> Any:  # noqa: ANN401 -- returns a writer
    """Return a writer that installs one synthetic registry for the lookup to read."""

    def _install(entries: object) -> None:
        registry_file.write_text(json.dumps(entries), encoding="utf-8")

    return _install


def test_malformed_json_is_a_registry_type_error(registry_file: pathlib.Path) -> None:
    """Malformed package data is a registry fault rather than a caller lookup fault."""
    registry_file.write_text("{", encoding="utf-8")

    with pytest.raises(TypeError, match="valid JSON"):
        resolve_registered_model("pinned")


def test_a_non_mapping_registry_root_is_refused_as_malformed(registry: Any) -> None:  # noqa: ANN401
    """A valid JSON value still cannot serve as a key-to-entry registry unless it is an object."""
    registry(["not", "a", "mapping"])

    with pytest.raises(TypeError, match="must hold an object at its root"):
        resolve_registered_model("pinned")


def test_a_present_null_entry_is_refused_as_malformed(registry: Any) -> None:  # noqa: ANN401
    """A present null value is a broken entry, not an absent caller-supplied key."""
    registry({"pinned": None})

    with pytest.raises(TypeError, match="Model 'pinned' must hold an object"):
        resolve_registered_model("pinned")


def test_a_present_non_mapping_entry_is_refused_as_malformed(registry: Any) -> None:  # noqa: ANN401
    """Every registered key must resolve to an object containing its pinned identity."""
    registry({"pinned": ["not", "an", "entry"]})

    with pytest.raises(TypeError, match="Model 'pinned' must hold an object"):
        resolve_registered_model("pinned")


def test_the_registrys_version_key_is_what_callers_read_as_the_revision(registry: Any) -> None:  # noqa: ANN401
    """The JSON says ``version``; every consumer of this identity means ``revision``.

    A caller that parses the file itself and looks for ``revision`` finds nothing,
    which is the concrete failure this shared lookup exists to make impossible.
    """
    registry({"pinned": _ENTRY})

    assert resolve_registered_model("pinned").revision == _ENTRY["version"]


def test_registered_model_exposes_only_identity_fields() -> None:
    """The typed boundary excludes download-policy data from its identity."""
    model = RegisteredModel(
        key="pinned",
        model_id="vendor/model-name",
        revision="abc",
        precision="FP8",
    )

    assert attrs.asdict(model) == {
        "key": "pinned",
        "model_id": "vendor/model-name",
        "revision": "abc",
        "precision": "FP8",
    }


def test_an_absent_key_is_refused_rather_than_defaulted(registry: Any) -> None:  # noqa: ANN401
    """A typo in a config's model name must fail before a GPU is allocated.

    A ``KeyError`` naming the registered keys, because this is the one refusal
    here a caller can act on: the message says what to write instead, and the
    type is what a caller guarding a lookup catches.
    """
    registry({"pinned": _ENTRY})

    with pytest.raises(KeyError, match=r"absent from .*Registered: \['pinned'\]"):
        resolve_registered_model("not-a-model")


@pytest.mark.parametrize("missing", ["model_id", "version", "precision"], ids=["model_id", "version", "precision"])
def test_an_entry_missing_any_identity_field_is_refused(registry: Any, missing: str) -> None:  # noqa: ANN401
    """A partially filled entry would produce an identity with a blank component.

    All three are checked because each one alone reaches a persisted contract
    digest, where a blank is indistinguishable from a deliberate value.
    """
    registry({"pinned": {key: value for key, value in _ENTRY.items() if key != missing}})

    with pytest.raises(TypeError, match="nonblank"):
        resolve_registered_model("pinned")


@pytest.mark.parametrize("blank_value", ["", "   "], ids=["empty", "whitespace"])
def test_an_entry_with_a_blank_identity_field_is_refused(
    registry: Any,  # noqa: ANN401
    blank_value: str,
) -> None:
    """Empty and whitespace-only values both lack a usable identity."""
    registry({"pinned": {**_ENTRY, "version": blank_value}})

    with pytest.raises(TypeError, match="nonblank"):
        resolve_registered_model("pinned")


def test_an_absent_model_directory_names_the_download_key(tmp_path: pathlib.Path) -> None:
    """An absent model directory reports the command that stages its key."""
    with pytest.raises(FileNotFoundError, match="model-download --models pinned"):
        model_registry.validate_model_directory(tmp_path / "absent", model_key="pinned")


def test_a_model_directory_without_weights_is_incomplete(tmp_path: pathlib.Path) -> None:
    """A config alone is not a loadable model directory."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        model_registry.validate_model_directory(model_dir, model_key="pinned")


def test_a_model_directory_without_config_is_incomplete(tmp_path: pathlib.Path) -> None:
    """A recognized weight file alone is not a loadable model directory."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        model_registry.validate_model_directory(model_dir, model_key="pinned")


def test_a_model_with_code_owned_config_accepts_staged_weights_without_config_json(tmp_path: pathlib.Path) -> None:
    """InternVideo2 ships its config in the source tree, while its download contains only a .pt file."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"weights")

    assert (
        model_registry.validate_model_directory(model_dir, model_key="internvideo2_mm", require_config_json=False)
        == model_dir
    )


def test_an_incomplete_model_directory_names_the_download_key(tmp_path: pathlib.Path) -> None:
    """An incomplete directory reports the command that stages its key."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="model-download --models pinned"):
        model_registry.validate_model_directory(model_dir, model_key="pinned")


@pytest.mark.parametrize("suffix", [".bin", ".pt", ".safetensors"])
def test_a_complete_model_directory_is_accepted(tmp_path: pathlib.Path, suffix: str) -> None:
    """A config and any recognized nested weight file satisfy the registry gate."""
    model_dir = tmp_path / "model"
    (model_dir / "nested").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "nested" / f"model{suffix}").write_bytes(b"weights")

    assert model_registry.validate_model_directory(model_dir, model_key="pinned") == model_dir
