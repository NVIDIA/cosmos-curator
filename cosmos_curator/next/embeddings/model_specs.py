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

"""Frozen model specifications for the vision and text embedders.

A checkpoint is data, not code: the only model-specific facts a generic embedder
needs are which staged weights to load, which HuggingFace id identifies them, the
output width to assert, and - for a raw vision backbone - how to reduce its output
to one vector. Bundling those into a frozen spec lets one embedder class host any
model that satisfies the same library contract (``AutoModel`` for vision,
``SentenceTransformer`` for text). A checkpoint swap is then an explicit code
change, never a silent runtime override, which is what lets the recipe layer
derive a trustworthy identity for the vectors a spec produces.

What lands in a row is NOT ``model_id``. The recipe hashes the facts that decide
what a vector contains - the registry key, the registered model id and revision,
the pooling reduction, and ``result_version`` - into one model result
fingerprint, because a checkpoint id alone cannot distinguish two different
computations over the same weights. See
``recipes/embeddings/modalities._model_result_fingerprint``.

This module is intentionally dependency-light (only ``enum`` + ``attrs``: no
torch, no Ray, no transformers) so the pure embeddings package can re-export it
without breaking the import-layer purity guard; see
tests/cosmos_curator/next/embeddings/test_import_layers.py. ``result_version`` is
an ``int``, so it does not widen that dependency surface.
"""

import enum

import attrs


class VisionPooling(enum.StrEnum):
    """How to reduce a vision backbone's token output to one vector per image.

    ``pooler_output`` exists on every HF vision backbone but means different
    things per architecture (DINOv2 is the CLS token with no learned layer,
    SigLIP is attention-pooled, a bare ViT has a randomly-initialised pooler when
    the checkpoint ships none), so the reduction is a per-model choice with no
    safe default across architectures.
    """

    POOLER_OUTPUT = "pooler_output"
    CLS_TOKEN = "cls_token"  # noqa: S105 - an enum value, not a credential (the name ends in TOKEN)
    MEAN_TOKENS = "mean_tokens"


@attrs.frozen
class VisionModelSpec:
    """The model-specific surface of an ``AutoModel`` vision embedder.

    Attributes:
        weights_name: ``all_models.json`` key; resolves to the staged local dir.
        model_id: HuggingFace id. One INPUT to the recorded result fingerprint,
            not the recorded value itself.
        dim: Expected output width, asserted against the first batch's real width.
        pooling: How ``_pool`` reduces the backbone output to one vector.
        result_version: Bump when this model's computed vectors change for a
            reason none of the other fields shows. It sits beside ``pooling`` so
            the two cannot be edited in separate diff hunks; the exhaustive
            trigger list is the comment at :data:`DEFAULT_IMAGE_MODEL`.

    """

    weights_name: str
    model_id: str
    dim: int
    pooling: VisionPooling = VisionPooling.POOLER_OUTPUT
    result_version: int = 1


@attrs.frozen
class TextModelSpec:
    """The model-specific surface of a ``SentenceTransformer`` text embedder.

    No pooling field: sentence-transformers bakes pooling into the checkpoint's
    module config, so unlike a raw vision backbone there is no per-call choice.

    Attributes:
        weights_name: ``all_models.json`` key; resolves to the staged local dir.
        model_id: HuggingFace id. One INPUT to the recorded result fingerprint,
            not the recorded value itself.
        dim: Expected output width, asserted against the first batch's real width.
        result_version: Bump when this model's computed vectors change for a
            reason none of the other fields shows. The exhaustive trigger list is
            the comment at :data:`DEFAULT_TEXT_MODEL`.

    """

    weights_name: str
    model_id: str
    dim: int
    result_version: int = 1


# A staged checkpoint the driver's weight-staging path needs by name. Both spec
# shapes expose ``weights_name`` / ``model_id`` / ``dim`` / ``result_version``; the
# union lets a GPU modality carry either without the recipe layer matching on which.
type ModelSpec = VisionModelSpec | TextModelSpec


# The default image / text checkpoints the embedding recipe ships with today,
# named by the modality they serve rather than the model so a checkpoint swap is a
# one-line edit here with no rename churn at the call sites. Adding a model (here
# and in all_models.json) is the whole cost of hosting a new one; test_model_specs.py
# asserts each weights_name resolves in the registry and its model_id matches.
#
# result_version is the MANUAL half of the result fingerprint, and this is its
# trigger list. Bump the affected model's result_version, in the same change set,
# when any of the following changes - each alters the stored vectors while leaving
# every automatic input (weights_name, model id, revision, pooling) identical, so
# nothing else can detect it and the corpus would silently hold two computations
# under one value.
#
# Either model:
#   - editing this key's all_models.json ``filelist`` or ``sha256`` while leaving
#     its ``version`` alone. Those decide which files are staged into
#     ``<cache>/<weights_name>``, which is the directory both embedders load, and
#     the fingerprint reads the entry's version rather than its contents.
#
# Image (DEFAULT_IMAGE_MODEL):
#   - which frame represents a clip (``ClipFrameReader.read_many`` selection);
#   - the ``backend="torchvision"`` decode pin;
#   - the AutoImageProcessor preprocessing transform, if it is ever passed
#     explicitly rather than taken from the checkpoint;
#   - dropping or adding the final L2 normalization;
#   - the SOURCE_COLUMNS the leg projects.
#
# Text (DEFAULT_TEXT_MODEL):
#   - ``format_task`` / ``format_subtask`` output, including the deliberate
#     no-retrieval-prefix choice;
#   - what ``task_name`` / ``subtask_name`` are taken to MEAN by the formatter;
#   - dropping or adding ``normalize_embeddings=True``;
#   - the SOURCE_COLUMNS the leg projects.
#
# NOT triggers, because they change cost or scheduling and never a vector value:
# batch sizes, read concurrency, actor counts, GPU/CPU/memory reservations,
# parquet_rows_per_file, max_fragments, progress display, and the storage profile.
# A weights re-pin in all_models.json is not a trigger either - the registry
# revision is an automatic input and moves the fingerprint on its own. Nor is
# pointing a spec at a different registry key, even a sibling naming the same
# model_id and version: weights_name is an automatic input for exactly that case.
# The price of that is symmetric - RENAMING a key whose weights are byte-identical
# also moves the fingerprint and makes the whole corpus owed - and it is the right
# price, because the key is the directory the weights are loaded from.
DEFAULT_IMAGE_MODEL = VisionModelSpec("dinov2_small", "facebook/dinov2-small", 384)
DEFAULT_TEXT_MODEL = TextModelSpec("bge_small_en_v1_5", "BAAI/bge-small-en-v1.5", 384)
