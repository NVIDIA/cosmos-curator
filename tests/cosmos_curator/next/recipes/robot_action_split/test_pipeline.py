# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray setup for robot-action-split, with Ray itself replaced."""

from functools import partial
from typing import Any

import pytest

from cosmos_curator.next.recipes.robot_action_split import pipeline
from cosmos_curator.next.recipes.robot_action_split.bindings import resolve_bindings
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig


def _record(calls: dict[str, Any], name: str, **kwargs: object) -> None:
    calls[name] = kwargs


def test_start_ray_serves_split_stores_and_widens_the_no_progress_window(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers get each profile's storage environment, and Ray Data waits as long as the config allows."""
    split = config.model_copy(
        update={
            "input": config.input.model_copy(
                update={"uris": ("s3://source-bucket/dataset/",), "storage_profile": "source-store"}
            ),
            "execution": config.execution.model_copy(
                update={"storage_profile": "output-store", "no_progress_timeout_s": 1234.0}
            ),
        }
    )
    calls: dict[str, Any] = {}

    def env_for(*locations: str, storage_profile: str) -> dict[str, str]:
        return {f"ENV_{storage_profile}": ",".join(locations)}

    monkeypatch.setattr(pipeline, "s3_compatibility_env_vars", env_for)
    monkeypatch.setattr(
        type(resolve_bindings(split)),
        "storage_env_vars",
        lambda self, *locations: env_for(self.uri, *locations, storage_profile=self.execution.storage_profile),
    )
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", partial(_record, calls, "init"))
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", partial(_record, calls, "progress"))
    monkeypatch.setattr(pipeline, "configure_ray_data_stability", partial(_record, calls, "stability"))

    pipeline.start_ray(resolve_bindings(split), split)

    env = calls["init"]["runtime_env_vars"]
    assert env["ENV_source-store"] == "s3://source-bucket/dataset/"
    assert split.output.media_root in env["ENV_output-store"]
    assert calls["stability"] == {"no_progress_timeout_s": 1234.0}
