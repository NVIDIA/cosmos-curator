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

"""Curator Next ``data-integrity`` recipe: the CLIs, discovery and store around the metrics.

The reusable half -- the metrics themselves, their pass/fail policy and the
per-stream engine that runs them -- lives in
:mod:`cosmos_curator.core.sensors.data_integrity`. What lives here is everything
that makes those metrics a Cosmos Curator workflow: the ``di`` tool and the
``data-integrity`` pipeline, their shared config, cloud/local stream discovery, the
concurrent session runner, report rendering, the Lance-backed result store, and the
``check_video_index`` diagnostic.

See ``docs/curator/design/data-integrity-design.md`` for the architecture and the
metric catalog. Callers import from concrete module paths; this package does not
re-export symbols.
"""
