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

"""Boundary conversion between raw bytes and NumPy arrays for Ray transport.

Encapsulates the bytes-to-numpy and numpy-to-bytes conversions at
pipeline boundaries.  These functions are called at PRODUCER and
CONSUMER stages only -- intermediate stages pass numpy arrays through
unchanged, benefiting from PEP 574 PickleBuffer zero-copy transport.

::

    Serialization tier comparison for Ray Object Store:

    Tier 1 -- npt.NDArray[np.uint8] (this module's output):
        Producer: bytes_to_numpy(raw_bytes) -> np.frombuffer().copy()
        Serialize: PickleBuffer -> direct to Plasma (zero-copy)
        Same-node ray.get(): mmap view (zero-copy)
        Cross-node: 1 network copy + local Plasma
        Cost per stage transition: 0 copies (same-node)

    Tier 2 -- bytes (before this migration):
        Serialize: in-band pickle stream -> memcpy to Plasma
        ray.get(): memcpy from Plasma -> new Python bytes object
        Cost per stage transition: 2 copies
"""

import numpy as np
import numpy.typing as npt
from loguru import logger


def bytes_to_numpy(data: bytes, *, copy: bool = True) -> npt.NDArray[np.uint8]:
    """Convert raw bytes to a numpy array for zero-copy Ray transport.

    Creates a contiguous uint8 array from raw bytes.  By default, copies
    the data so the array owns its memory (safe when the source bytes
    may be garbage collected).  Pass ``copy=False`` when the source is
    guaranteed to outlive the array -- e.g., when the result is
    immediately passed to ``ray.put()`` which serializes synchronously.

    ::

        copy=True (Phase 1 default, safe):
            bytes obj --> np.frombuffer() --> .copy() --> owned array
                          (view, 0 bytes)    (1 memcpy)

        copy=False (Phase 2 optimization, immediate detach):
            bytes obj --> np.frombuffer() --> view array
                          (0 bytes)          (no memcpy, source must stay alive)
                               |
                               v
                          ray.put(view)  <-- synchronous, copies to Plasma
                          source bytes can be GC'd after ray.put returns

    Args:
        data: Raw bytes to convert.
        copy: If True (default), the returned array owns its memory.
            If False, returns a read-only view -- caller must ensure
            the source bytes outlives the array.

    Returns:
        Contiguous uint8 array.  Read-only if ``copy=False``.

    """
    # TODO(perf): ``np.frombuffer()`` returns a zero-cost read-only view --
    # the ``.copy()`` call is the sole source of the O(n) allocation + memcpy.
    # Switching callers to ``copy=False`` where the source ``bytes`` object is
    # guaranteed to outlive the array (e.g., the result is immediately passed
    # to ``ray.put()`` which serializes synchronously) would eliminate this
    # copy entirely.  Requires a per-caller lifetime safety audit; track in a
    # separate PR.
    view = np.frombuffer(data, dtype=np.uint8)
    return view.copy() if copy else view


def numpy_to_bytes(data: npt.NDArray[np.uint8]) -> bytes:
    """Convert numpy array back to bytes at the CONSUMER boundary.

    Used only by final consumers that strictly require a ``bytes``
    object -- e.g., external APIs with ``isinstance(data, bytes)``
    type checks, or protocol serializers that reject buffer objects.

    Most I/O APIs accept buffer-protocol objects directly:
    ``io.BytesIO(array)``, ``file.write(array)``,
    ``Path.write_bytes(array)`` -- prefer passing the array directly
    over calling this function.

    .. warning:: ``subprocess.run(input=array)`` does NOT work

        CPython's ``subprocess._communicate()`` checks ``if not input:``
        which triggers numpy's ambiguous truth-value error for multi-element
        arrays.  Convert to ``bytes(array)`` before passing to subprocess.

    .. note:: C-contiguity

        All arrays produced by ``np.frombuffer()`` and ``np.empty()``
        are C-contiguous, so current callers are safe.  However, arrays
        received from external code (e.g., model outputs with
        non-standard strides, transposed views, fancy-indexed results)
        may be non-contiguous, which causes ``.tobytes()`` to do an
        element-by-element traversal instead of a single ``memcpy``
        (10-100x slower).  A defensive guard below auto-corrects
        non-contiguous input and logs a warning so the caller can be
        fixed upstream.

    Args:
        data: NumPy uint8 array to convert.

    Returns:
        Raw bytes copy of the array data.

    """
    if not data.flags.c_contiguous:
        logger.warning(
            f"numpy_to_bytes received non-contiguous array "
            f"(shape={data.shape}, strides={data.strides}); "
            f"forcing contiguous copy -- fix the upstream caller "
            f"to produce C-contiguous arrays for optimal performance",
        )
        data = np.ascontiguousarray(data)
    return data.tobytes()
