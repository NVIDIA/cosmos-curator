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
"""Caption embedding helpers shared by the measure stage and the (transitional) metadata stage.

Loading the sentence-transformers model and the batched cosine-similarity
computation live here so both the v3 ``measure_stage`` and the v2
``metadata_stage`` use one implementation. See
``docs/curator/design/split-comparison.md`` ("Caption model").
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cosmos_curator.core.utils.model import model_utils

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]


def load_caption_model(model_id: str, device: str | None = None) -> "SentenceTransformer":
    """Load the caption embedding model from the project's local weights cache.

    Default device is cpu. ``sentence_transformers``/``torch`` are imported here
    rather than at module top so the heavy deps are only loaded when captions
    are actually compared.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]  # noqa: PLC0415

    device = device or "cpu"
    model_dir = model_utils.get_local_dir_for_weights_name(model_id)
    if not model_dir.exists():
        msg = (
            f"Caption model weights not found at {model_dir}. Download via: "
            "cosmos-curator local launch --image-name cosmos-curator -- "
            "pixi run --as-is python -m cosmos_curator.core.managers.model_cli download "
            "--models bge_small_en_v1_5"
        )
        raise FileNotFoundError(msg)
    return SentenceTransformer(str(model_dir), local_files_only=True, device=device)  # type: ignore[no-any-return]


def cosine_similarity_batch(
    model: "SentenceTransformer",
    texts_a: list[str],
    texts_b: list[str],
    *,
    batch_size: int,
) -> NDArray[np.float32]:
    """Embed both lists in one ``encode()`` call; return paired cosine similarities, shape (N,).

    ``texts_a`` and ``texts_b`` must be equal-length: result ``i`` is the
    similarity of ``texts_a[i]`` vs ``texts_b[i]``. The two are concatenated,
    encoded together, then split back in half -- so a length mismatch would
    silently misalign every pair (or raise an opaque split error on an odd
    total). Guarded explicitly below.

    ``sentence_transformers`` chunks the input internally at ``batch_size``; the
    Python-level call overhead is paid once. Embeddings are normalized, so the
    cosine similarity reduces to a dot product.

    Duplicate caption strings recur across windows and clips within a batch (both
    sides are often near-identical runs); each distinct string is encoded once and
    gathered back by index, since the per-string transformer forward pass is the
    stage's dominant cost. (Identical caption *pairs* are short-circuited upstream,
    so the win here is cross-pair / cross-clip repeats.)
    """
    if len(texts_a) != len(texts_b):
        msg = f"cosine_similarity_batch needs element-wise paired inputs; got {len(texts_a)} and {len(texts_b)}"
        raise ValueError(msg)
    if not texts_a:
        return np.zeros(0, dtype=np.float32)
    combined = texts_a + texts_b
    unique_index: dict[str, int] = {}
    for text in combined:
        if text not in unique_index:
            unique_index[text] = len(unique_index)
    encoded = np.asarray(
        model.encode(
            list(unique_index),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    embeddings = encoded[[unique_index[text] for text in combined]]
    embs_a, embs_b = np.vsplit(embeddings, 2)
    return np.asarray((embs_a * embs_b).sum(axis=1))
