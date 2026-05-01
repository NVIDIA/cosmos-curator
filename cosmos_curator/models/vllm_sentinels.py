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

"""Sentinel values returned by the vLLM interface to signal out-of-band conditions."""

# Returned by vllm_interface when a request completes but produces no usable output
# (e.g. decode failure, rejected output). Classified as caption_failure_reason="exception"
# by _scatter_captions. Keep this as the single source of truth — do not duplicate the
# literal elsewhere.
VLLM_UNKNOWN_CAPTION = "Unknown caption"
