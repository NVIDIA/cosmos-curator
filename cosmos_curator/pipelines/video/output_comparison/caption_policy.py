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
"""Policy knobs for split output caption structure comparison."""

import attrs

from cosmos_curator.pipelines.video.utils.data_model import CAPTION_OK_STATUSES

_CAPTION_OK_STATUS_VALUES = frozenset(status.value for status in CAPTION_OK_STATUSES)


@attrs.define(frozen=True)
class CaptionComparisonPolicy:
    """Configurable choices for caption metadata loading."""

    metadata_version: str = "v0"
    caption_field_suffix: str = "_caption"
    enhanced_caption_field_suffix: str = "_enhanced_caption"
    caption_ok_statuses: frozenset[str] = _CAPTION_OK_STATUS_VALUES

    def is_regular_caption_field(self, field: str) -> bool:
        """Return whether a window field contains regular caption text."""
        return field.endswith(self.caption_field_suffix) and not field.endswith(self.enhanced_caption_field_suffix)

    def is_caption_ok_status(self, value: object) -> bool:
        """Return whether a ``caption_status`` value is counted as captioned."""
        return isinstance(value, str) and value in self.caption_ok_statuses


DEFAULT_CAPTION_POLICY = CaptionComparisonPolicy()
