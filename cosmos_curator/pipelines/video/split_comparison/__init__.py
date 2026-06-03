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
"""Split-pipeline output comparison.

Two-stage clip-level comparison: metadata (Stage 1, CPU caption embeddings via
BGE) and clip MP4 video index (Stage 2, mixed CPU + IO). Public entry point is
``driver.compare_split_outputs``; the CLI lives in ``cli`` (``--config PATH``).
See ``docs/curator/design/split-comparison.md`` for the topology + rationale.
"""
