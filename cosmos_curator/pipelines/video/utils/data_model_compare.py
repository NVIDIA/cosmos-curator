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
"""Task comparators for video pipeline data models."""

from cosmos_curator.core.interfaces.stage_interface import PipelineTask
from cosmos_curator.core.utils.misc.stage_compare import (
    FieldDiff,
    TaskComparator,
    collect_checked_attrs_fields,
    compare_attrs_fields,
    register_comparator,
)
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask


class SplitPipeTaskComparator(TaskComparator):
    """Comparator for SplitPipeTask that ignores non-semantic perf metadata."""

    _FIELD_NAMES = ("session_id", "videos", "errors")

    def checked_fields(self, golden: PipelineTask) -> tuple[str, ...]:
        """Return the semantic SplitPipeTask fields covered by this comparator."""
        return collect_checked_attrs_fields(golden, field_names=self._FIELD_NAMES)

    def compare(
        self,
        golden: PipelineTask,
        candidate: PipelineTask,
        *,
        atol: float,
    ) -> list[FieldDiff]:
        """Compare semantic SplitPipeTask fields."""
        return compare_attrs_fields(
            golden,
            candidate,
            field_names=self._FIELD_NAMES,
            atol=atol,
        )


register_comparator(SplitPipeTask, SplitPipeTaskComparator())
