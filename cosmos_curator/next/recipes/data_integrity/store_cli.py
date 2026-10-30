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

"""Writing a store from the Ray-free ``di`` tool, without importing one to do it.

Kept apart from :mod:`.cli_support` for one reason: the store pulls in ``lance`` and
``pyarrow``, and a plain check should not pay for them. The import is deferred into
the call below, so that cost lands only on a run whose ``output.store_root`` is set.

Where a store root is *shaped* is :class:`~.config.DataIntegrityOutputConfig`, which
validates it when the config is resolved -- before any source is read, since the store
is written last and a typo would otherwise cost a whole session.
"""

from collections.abc import Mapping

from cosmos_curator.core.sensors.data_integrity.instruments import Thresholds
from cosmos_curator.core.sensors.data_integrity.results import CheckResult, StreamResult
from cosmos_curator.core.utils.storage.storage_client import StorageStat


def persist_run(  # noqa: PLR0913 -- provenance plus credentials, all independent
    store_path: str,
    streams: list[StreamResult],
    *,
    session_path: str | None,
    thresholds: Thresholds,
    tool: str,
    session_metrics: list[CheckResult] | None = None,
    storage_stats: Mapping[str, StorageStat] | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> str:
    """Append one run to the store at ``store_path`` and return its ``run_id``.

    ``session_metrics`` carries the session's own verdicts, which only a whole-session
    run has; see :func:`~.store.write_run`.

    Errors propagate: the operator asked for the results to be saved, so a store that
    could not be written is a failed run even when every check passed, and the tool
    turns that into exit code 2.
    """
    # Imported here, not at module scope, so `lance` / `pyarrow` load only for a run
    # whose config set output.store_root.
    from cosmos_curator.next.recipes.data_integrity import store  # noqa: PLC0415

    return store.write_run(
        store_path,
        streams,
        session_path=session_path,
        thresholds=thresholds,
        tool=tool,
        session_metrics=session_metrics,
        storage_stats=storage_stats,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
    )
