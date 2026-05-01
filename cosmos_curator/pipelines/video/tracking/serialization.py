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

"""SAM3 output JSON envelope helpers.

Single source of truth for the per-clip JSON shapes written by both the
production splitting pipeline (``ClipWriterStage._write_clip_sam3``) and the
``sam3_event_pipeline.py`` example. Keeping the envelope shape in one place
prevents drift when the schema evolves.
"""

from typing import Any


def sam3_instances_envelope(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap SAM3 per-clip instance summaries for ``instances.json``."""
    return {"instances": instances}


def sam3_objects_envelope(objects_by_frame: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Wrap SAM3 per-frame object detections for ``objects.json``.

    JSON keys must be strings, so frame indices are coerced.
    """
    return {"objects": {str(k): v for k, v in objects_by_frame.items()}}


def sam3_events_envelope(events: list[Any]) -> dict[str, Any]:
    """Wrap VLM per-clip events for ``events.json``.

    No ``version`` field: the events schema is defined entirely by the VLM
    prompt, so the writer doesn't own a versioned contract.
    """
    return {"events": events}
