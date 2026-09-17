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

"""Ray Data actor for the text-embedding leg.

``SentenceTransformerTextEmbedder`` is a Ray Data callable class hosting any
model describable by a ``TextModelSpec``: the ``SentenceTransformer`` model loads
once per actor (in ``__init__``, from the spec's staged weights, on the target
device) and each ``__call__`` embeds one Arrow batch. Nothing here is
model-specific - ``SentenceTransformer(dir)`` plus ``encode(normalize_embeddings=
True)`` is the library contract for every checkpoint it hosts. The heavy
``sentence_transformers`` import is deferred into the constructor so importing
this module (e.g. in a CPU-only test) does not pull torch.

Each source clip yields two vectors - the subtask instruction and its parent
task - both L2-normalized (``normalize_embeddings=True``) so downstream cosine
distance is a plain dot product. This dual task/subtask output is the one
domain-specific part of the leg. The leg never drops a row: task / subtask are
non-null on the source contract, and an empty instruction embeds to a valid
(content-free) vector.

The embedder is pure compute and carries no identity column: it reads the text
columns and returns only the text group's columns, one row per input row in input
order. Attaching ``clip_id`` is the storage layer's job.

The retrieval instruction prefix that BGE-style models document (e.g.
"Represent this sentence for searching relevant passages:") is deliberately NOT
prepended: this leg embeds for symmetric clustering, where both sides are the
same kind of text, not asymmetric query-vs-passage retrieval. Do not "fix" it.

See docs/curator/design/curator-next-embeddings.md.
"""

from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from loguru import logger

from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name
from cosmos_curator.next.embeddings.model_specs import TextModelSpec
from cosmos_curator.next.embeddings.schemas import TEXT_DIM, text_columns_batch
from cosmos_curator.next.embeddings.text.formatter import format_subtask, format_task
from cosmos_curator.next.embeddings.torch_device import resolve_torch_device


class SentenceTransformerTextEmbedder:
    """Embed the task / subtask instruction of each clip with a SentenceTransformer.

    Attributes:
        SOURCE_COLUMNS: Columns the leg's scan must project. It includes
            ``clip_id`` even though ``__call__`` never reads it: the caller that
            owns storage joins the computed group back on ``clip_id``, so the
            scan has to carry it. Held on the class (not module-level) so the name
            does not collide with the other legs' differing tuples, and pinned by
            test to be a subset of ``EMBED_SOURCE_COLUMNS``.

    """

    SOURCE_COLUMNS: ClassVar[tuple[str, ...]] = ("clip_id", "task_name", "subtask_name")

    def __init__(
        self,
        spec: TextModelSpec,
        *,
        result_fingerprint: str,
        encode_batch_size: int = 256,
        device: str | None = None,
    ) -> None:
        """Load the SentenceTransformer once for this actor.

        Args:
            spec: The checkpoint identity and output width. Weights resolve only
                from ``spec.weights_name`` - there is no runtime override, so what
                this actor computes really is what the caller's fingerprint names.
            result_fingerprint: Identity of the computation these vectors come
                from, written into every row's provenance column. Supplied rather
                than derived here so the value the caller's selection predicate
                compares and the value the rows carry cannot diverge.
            encode_batch_size: Inner batch size for ``model.encode``.
            device: Torch device override; ``None`` auto-selects cuda/cpu.

        """
        from sentence_transformers import (  # type: ignore[import-not-found]  # noqa: PLC0415 - deferred so import stays torch-free
            SentenceTransformer,
        )

        weights_dir = str(get_local_dir_for_weights_name(spec.weights_name))
        self._model = SentenceTransformer(weights_dir, local_files_only=True, device=resolve_torch_device(device))
        self._spec = spec
        self._result_fingerprint = result_fingerprint
        self._encode_batch_size = encode_batch_size

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Embed the task + subtask text of every input row into the text group batch.

        Pure compute: reads only the two text columns, returns only the text
        group's columns, and never touches an identity column - the caller that
        owns storage attaches ``clip_id`` positionally. Cardinality-preserving and
        order-preserving, which is what makes that positional attachment correct:
        the text leg never fails per row (task / subtask are non-null on the source
        contract and an empty instruction embeds to a valid content-free vector),
        so it returns exactly one complete group row for every input row.
        """
        rows = batch.num_rows
        if not rows:
            # A zero-row batch returns an empty group batch.
            # sentence_transformers.encode([]) yields shape (0,) not (0, TEXT_DIM),
            # which _as_matrix would reject as a bad single-row vector;
            # special-casing here removes that dependency on a Ray batcher detail
            # (empty blocks are not currently delivered).
            empty = np.zeros((0, TEXT_DIM), dtype=np.float32)
            return text_columns_batch(0, empty, empty, self._result_fingerprint)
        subtask_texts = [format_subtask(name) for name in batch.column("subtask_name").to_pylist()]
        task_texts = [format_task(name) for name in batch.column("task_name").to_pylist()]
        blanks = sum(not text for text in subtask_texts) + sum(not text for text in task_texts)
        if blanks:
            # Blank instructions all embed to the same point (a false "exact
            # duplicate" signal downstream); the row is kept by design, so make
            # the count observable rather than silent.
            logger.warning(f"text leg: {blanks} blank instruction(s) in a {rows}-row batch")
        subtask_vectors = self._encode(subtask_texts)
        task_vectors = self._encode(task_texts)
        return text_columns_batch(rows, subtask_vectors, task_vectors, self._result_fingerprint)

    def _encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Encode a list of strings into an L2-normalized ``(len(texts), spec.dim)`` matrix.

        Raises:
            ValueError: If the model's real output width disagrees with
                ``spec.dim`` - caught on the first batch, naming the spec.

        """
        vectors = np.asarray(
            self._model.encode(
                texts,
                batch_size=self._encode_batch_size,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if vectors.ndim == 2 and vectors.shape[1] != self._spec.dim:  # noqa: PLR2004 - 2 == a (rows, dim) matrix
            msg = (
                f"text model {self._spec.model_id} (spec {self._spec.weights_name!r}) emitted width "
                f"{vectors.shape[1]}, but spec.dim is {self._spec.dim}"
            )
            raise ValueError(msg)
        return vectors
