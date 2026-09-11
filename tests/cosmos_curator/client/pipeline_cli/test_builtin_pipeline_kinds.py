# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in pipeline-kind composition root."""

import pathlib
import subprocess
import sys

import pytest

from cosmos_curator.client.pipeline_cli.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS


def test_registered_kind_names_are_derived_from_concrete_objects() -> None:
    """The composition root never restates discriminator strings as mapping keys."""
    assert BUILTIN_PIPELINE_KINDS.names() == (
        "caption_judge",
        "curate",
        "data-integrity",
        "embeddings",
        "multimodal-split",
        "robot-action-caption",
        "robot-action-split",
        "video-caption",
        "video-cut",
        "video-split",
        "video_split",
    )
    assert tuple(kind.name for kind in BUILTIN_PIPELINE_KINDS) == BUILTIN_PIPELINE_KINDS.names()


def test_importing_composition_root_pulls_in_no_third_party_dependency(repo_root: pathlib.Path) -> None:
    """CLI startup may import adapters, but nothing outside the stdlib and this package.

    Asserted as default-deny rather than against a list of known-heavy names.
    A denylist only fails for the dependencies someone thought to enumerate,
    and the ones that would hurt most here -- pyarrow via a recipe's column
    module, or lance, cuml and cupy via a runtime -- arrive through imports no
    such list anticipated.
    """
    probe = (
        "import sys;"
        "before = set(sys.modules);"
        "from cosmos_curator.client.pipeline_cli.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS;"
        "BUILTIN_PIPELINE_KINDS.names();"
        "added = {m.split('.')[0] for m in set(sys.modules) - before};"
        "print(sorted(added - sys.stdlib_module_names))"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
    )

    assert result.stdout.strip() == "['cosmos_curator']"


def test_robot_action_split_underscore_spelling_is_not_supported() -> None:
    """Only the recipe's canonical hyphenated discriminator is registered."""
    with pytest.raises(ValueError, match="Unknown pipeline kind 'robot_action_split'"):
        BUILTIN_PIPELINE_KINDS.get("robot_action_split")


def test_unknown_kind_names_the_valid_ones() -> None:
    """Lookup failures advertise names derived from the registered objects."""
    with pytest.raises(ValueError, match="Unknown pipeline kind 'nope'") as excinfo:
        BUILTIN_PIPELINE_KINDS.get("nope")

    assert "robot-action-split" in str(excinfo.value)
