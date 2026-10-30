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

"""Generic data-integrity framework for the sensor library.

The reusable half of data integrity: the metrics, the pass/fail policy that judges
them, the vocabulary their results are expressed in, and the per-stream engine
(:mod:`.engine`) that runs the lot over one already-open sensor. Backend-agnostic
throughout -- nothing here accepts a URI, imports a cloud client, or parses an
argument, which is what lets the sensor library use it directly.

The workflow built on top -- the ``di`` tool, stream discovery, report rendering
and the Lance result store -- lives in
:mod:`cosmos_curator.next.recipes.data_integrity`.

See ``docs/curator/design/data-integrity-design.md`` for the architecture and the
metric catalog. Callers import from concrete module paths; this package does not
re-export symbols.
"""
