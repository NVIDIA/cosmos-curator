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

"""Ray Data actor for the vision image-embedding leg.

``HfVisionImageEmbedder`` is a Ray Data callable class hosting any HuggingFace
``AutoModel`` vision backbone described by a ``VisionModelSpec``::

    clip_uri --> ClipFrameReader --> AutoImageProcessor --> backbone --> _pool --> L2
                 .read_many(uris)                                                   |
                        |                                             embedding_image
                        +-- unreadable clip: all-NULL group row at the same position

The model + ``AutoImageProcessor`` load once per actor (in ``__init__``, from the
spec's staged weights). Heavy imports (``transformers`` / ``torch``) are deferred
into the constructor so importing this module stays torch-free.

Only the spec is model-specific - the staged checkpoint, its output width, and how
to reduce the backbone output to one vector (``_pool``). Everything else is
generic: the ``backend="torchvision"`` pin makes preprocessing a fixed function of
the checkpoint rather than of whichever backend happens to be importable, and
``ClipFrameReader`` owns the frame acquisition, the concurrency of it, and the
drop-vs-systemic-failure split. Why the reads are concurrent while the backbone
stays one batched call on the calling thread is argued in
``docs/curator/design/curator-next-embeddings.md`` section 4.2.

The leg is cardinality- and order-preserving: a clip whose media is missing or
undecodable emits an all-NULL group row rather than being dropped, so a rebuild
replaces a failed row with NULL instead of leaving a stale prior vector, and the
survivors' vectors are scattered back into their input row positions. The reader
returns each frame under the row it was read from, so that scatter never depends
on the order the reads finished in. The embedder carries no identity column -
attaching ``clip_id`` is the storage layer's job, which is why preserving row
count and order matters here.
"""

from typing import Any, ClassVar, assert_never

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name
from cosmos_curator.next.embeddings.image.frame_reader import ClipFrameReader
from cosmos_curator.next.embeddings.model_specs import VisionModelSpec, VisionPooling
from cosmos_curator.next.embeddings.schemas import image_columns_batch
from cosmos_curator.next.embeddings.torch_device import resolve_torch_device


class HfVisionImageEmbedder:
    """Embed one representative frame per clip with an ``AutoModel`` vision backbone.

    Attributes:
        SOURCE_COLUMNS: Columns the leg's scan must project. It includes
            ``clip_id`` even though ``__call__`` never reads it: the caller that
            owns storage joins the computed group back on ``clip_id``, so the
            scan has to carry it. Held on the class (not module-level) so the name
            does not collide with the other legs' differing tuples, and pinned by
            test to be a subset of ``EMBED_SOURCE_COLUMNS``.

    """

    SOURCE_COLUMNS: ClassVar[tuple[str, ...]] = ("clip_id", "clip_uri")

    def __init__(
        self,
        spec: VisionModelSpec,
        *,
        result_fingerprint: str,
        storage_profile: str = "default",
        device: str | None = None,
        read_concurrency: int = 1,
    ) -> None:
        """Load the vision model + processor once for this actor.

        Args:
            spec: The checkpoint identity, output width, and pooling strategy.
                Weights resolve only from ``spec.weights_name`` - there is no
                runtime override, so what this actor computes really is what the
                caller's fingerprint names.
            result_fingerprint: Identity of the computation these vectors come
                from, written into every row's provenance column. Supplied rather
                than derived here so the value the caller's selection predicate
                compares and the value the rows carry cannot diverge.
            storage_profile: Storage profile for reading clip media.
            device: Torch device override; ``None`` auto-selects cuda/cpu.
            read_concurrency: Width the frame reader is built with; see
                ``ClipFrameReader`` for what it does and does not widen. Passed
                through rather than used here, because the reader owns its own
                I/O concurrency.

        """
        import torch  # noqa: PLC0415 - deferred so importing this module stays torch-free
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415 - deferred so import stays torch-free

        self._torch = torch
        self._spec = spec
        self._device = resolve_torch_device(device)
        weights_dir = str(get_local_dir_for_weights_name(spec.weights_name))
        # backend pinned so preprocessing is a function of the checkpoint, not of
        # which image-processor backend happens to be importable in this env.
        self._processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            weights_dir,
            local_files_only=True,
            backend="torchvision",
        )
        self._model = AutoModel.from_pretrained(weights_dir, local_files_only=True).to(self._device).eval()
        self._result_fingerprint = result_fingerprint
        self._reader = ClipFrameReader(storage_profile=storage_profile, read_concurrency=read_concurrency)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Embed the first frame of every readable clip into the image group batch.

        Cardinality- and order-preserving: every input row yields one output row at
        the same position, which is what lets the caller attach ``clip_id``
        positionally. A clip with missing / undecodable media contributes an
        all-NULL group (``valid`` False); only the readable clips run through the
        backbone, and their vectors are scattered back into their input row
        positions.
        """
        rows = batch.num_rows
        frame_by_row = self._reader.read_many(batch.column("clip_uri").to_pylist())
        valid = np.zeros(rows, dtype=np.bool_)
        frames: list[npt.NDArray[np.uint8]] = []
        frame_rows: list[int] = []
        # Assemble in ascending input-row order, independent of the order the reads
        # finished in: positional alignment of frames/frame_rows is the precondition
        # of the scatter below. Popping leaves ``frames`` the sole owner of each
        # frame, so _embed's clear() really does release them before the forward
        # pass; sorted() materializes the keys first, so mutating here is safe.
        for index in sorted(frame_by_row):
            valid[index] = True
            frames.append(frame_by_row.pop(index))
            frame_rows.append(index)
        matrix = np.zeros((rows, self._spec.dim), dtype=np.float32)
        if frames:
            # _embed clears ``frames`` (frees decoded RGB before the forward pass);
            # scatter the survivors' vectors back to their original row positions.
            matrix[frame_rows] = self._embed(frames)
        return image_columns_batch(rows, matrix, self._result_fingerprint, valid)

    def _embed(self, frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        """Run the backbone on RGB frames, returning an L2-normalized ``(N, spec.dim)`` matrix.

        Empties the caller-owned ``frames`` list after preprocessing: the decoded
        uint8 frames are the leg's larger allocation at the default batch size, so
        releasing them before the forward pass roughly halves peak memory. The
        mutation is deliberate; a caller must not reuse ``frames`` afterwards.

        Raises:
            ValueError: If the backbone's real output width disagrees with
                ``spec.dim`` - a same-width model swap slips past this, but a
                wrong-width spec is caught on the first batch, naming the spec.

        """
        inputs = self._processor(images=frames, return_tensors="pt").to(self._device)
        frames.clear()
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        normalized = self._torch.nn.functional.normalize(self._pool(outputs), dim=1)
        matrix = normalized.detach().cpu().numpy().astype(np.float32)
        if matrix.shape[1] != self._spec.dim:
            msg = (
                f"vision backbone {self._spec.model_id} (spec {self._spec.weights_name!r}) emitted width "
                f"{matrix.shape[1]}, but spec.dim is {self._spec.dim}"
            )
            raise ValueError(msg)
        return matrix

    def _pool(self, outputs: Any) -> Any:  # noqa: ANN401 - a transformers model output is untyped
        """Reduce the backbone output to one vector per image per ``spec.pooling``.

        Raises:
            ValueError: If ``POOLER_OUTPUT`` is requested but the backbone returns
                ``None`` (it ships no pooler head), naming the spec and suggesting
                ``CLS_TOKEN``.

        """
        match self._spec.pooling:
            case VisionPooling.POOLER_OUTPUT:
                pooled = outputs.pooler_output
                if pooled is None:
                    msg = (
                        f"pooler_output is None for spec {self._spec.weights_name!r} ({self._spec.model_id}); "
                        "this backbone ships no pooler head - use VisionPooling.CLS_TOKEN"
                    )
                    raise ValueError(msg)
                return pooled
            case VisionPooling.CLS_TOKEN:
                # The CLS token is the first position of the token sequence.
                return outputs.last_hidden_state[:, 0]
            case VisionPooling.MEAN_TOKENS:
                # Mean over the token axis (dim 1 of (batch, tokens, width)).
                return outputs.last_hidden_state.mean(dim=1)
            case _:
                assert_never(self._spec.pooling)
