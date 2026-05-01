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

"""Zero-disk-I/O buffer-to-path utility with automatic tmpfile fallback."""

import contextlib
import ctypes
import errno
import os
import pathlib
import tempfile
from collections.abc import Buffer, Generator

from loguru import logger

from cosmos_curator.core.utils.infra import libc as _libc

# memfd_create detection cascade (see module docstring)
#
#   Tier 1: os.memfd_create  -- CPython native, fastest, but absent when
#           the Python build was compiled without HAVE_MEMFD_CREATE.
#
#   Tier 2: _libc.memfd_create -- ctypes call into the runtime glibc.
#           Works even when os.memfd_create is missing, as long as the
#           host glibc (>= 2.27 / kernel >= 3.17) exports the symbol.
#
#   Tier 3: tempfile fallback  -- always available, incurs disk I/O.
_OS_MEMFD_AVAILABLE: bool = hasattr(os, "memfd_create")

# MFD_CLOEXEC -- close fd on exec(), matches <linux/memfd.h>
_MFD_CLOEXEC = 0x0001

# Probe ctypes memfd_create once at import time.
_CTYPES_MEMFD_AVAILABLE: bool = False
if not _OS_MEMFD_AVAILABLE and _libc is not None:
    _CTYPES_MEMFD_AVAILABLE = hasattr(_libc, "memfd_create")
    if _CTYPES_MEMFD_AVAILABLE:
        _libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        _libc.memfd_create.restype = ctypes.c_int
        logger.debug("[memfd] os.memfd_create absent; using ctypes libc.memfd_create fallback")

# Track whether we have already warned about memfd unavailability so we
# do not spam the log on every call from the same worker process.
_memfd_unavailable_warned: bool = False

# Track whether we have already warned about tmpfile fallback usage so we
# log at WARNING only once, then demote to DEBUG for subsequent calls.
_tmpfile_fallback_warned: bool = False


def _try_memfd_create(name: str) -> tuple[int | None, str | None]:
    """Attempt ``memfd_create`` via a three-tier cascade and return ``(fd, fallback_reason)``."""
    global _memfd_unavailable_warned  # noqa: PLW0603

    # -- Tier 1: os.memfd_create (CPython native) --
    if _OS_MEMFD_AVAILABLE:
        try:
            # Linux-only; runtime-guarded via ``_OS_MEMFD_AVAILABLE``.
            return os.memfd_create(name), None  # type: ignore[attr-defined]
        except OSError as e:
            return None, _format_oserror_reason(e, name, source="os.memfd_create")

    # -- Tier 2: ctypes libc.memfd_create (conda-forge workaround) --
    if _CTYPES_MEMFD_AVAILABLE:
        try:
            fd = _libc.memfd_create(name.encode(), _MFD_CLOEXEC)  # type: ignore[union-attr]
            if fd < 0:
                # ctypes.get_errno() requires the _libc handle to be loaded
                # with use_errno=True (set in cosmos_curator.core.utils.infra).
                err = ctypes.get_errno()
                errno_name = errno.errorcode.get(err, "?")
                reason = f"ctypes libc.memfd_create returned {fd} (errno={err}/{errno_name}, name={name!r})"
                logger.warning(reason)
                return None, reason
            return fd, None  # noqa: TRY300
        except OSError as e:
            return None, _format_oserror_reason(e, name, source="ctypes libc.memfd_create")

    # -- Tier 3: neither available --
    if not _memfd_unavailable_warned:
        _memfd_unavailable_warned = True
        logger.warning(
            "memfd_create not available via os or ctypes"
            " - all buffer_as_memfd_path calls will use tempfile fallback (disk I/O)",
        )
    return None, "memfd_create not available"


def _format_oserror_reason(e: OSError, name: str, *, source: str) -> str:
    """Build a diagnostic string for a failed ``memfd_create`` OSError."""
    errno_name = errno.errorcode.get(e.errno, "?") if e.errno else "?"
    reason = f"{source} failed: {e} (errno={e.errno}/{errno_name}, name={name!r})"
    if e.errno == errno.EPERM:
        reason += (
            " -- EPERM typically means seccomp blocks memfd_create (common in NVCF / restricted container environments)"
        )
    elif e.errno == errno.ENOSYS:
        reason += " -- ENOSYS means the syscall is not available in this kernel"
    elif e.errno == errno.ENOMEM:
        reason += " -- ENOMEM means insufficient memory to create the fd"
    logger.warning(reason, exc_info=True)
    return reason


@contextlib.contextmanager
def _yield_memfd_path(fd: int, raw: bytes | bytearray | memoryview, name: str) -> Generator[str]:
    """Write *raw* to an already-opened memfd and yield its ``/proc`` path."""
    memfd_path = f"/proc/self/fd/{fd}"
    try:
        try:
            with os.fdopen(fd, "wb", closefd=False) as fp:
                fp.write(raw)
                fp.flush()
        except OSError as e:
            buf_mb = len(raw) / (1024 * 1024)
            logger.warning(
                f"memfd write failed: {e} (name={name!r}, buf={buf_mb:.1f} MB, fd={fd})",
                exc_info=True,
            )
            raise
        os.lseek(fd, 0, os.SEEK_SET)
        yield memfd_path
    finally:
        os.close(fd)


@contextlib.contextmanager
def _yield_tmpfile_path(
    raw: bytes | bytearray | memoryview,
    name: str,
    fallback_reason: str | None,
) -> Generator[str]:
    """Write *raw* to a ``NamedTemporaryFile`` and yield its path."""
    global _tmpfile_fallback_warned  # noqa: PLW0603
    buf_mb = len(raw) / (1024 * 1024)
    msg = (
        f"[memfd] Using tempfile fallback for name={name!r} "
        f"(buf={buf_mb:.1f} MB). "
        f"Reason: {fallback_reason or 'unknown'}. "
        f"This incurs disk I/O; check seccomp policy if running in a container."
    )
    if not _tmpfile_fallback_warned:
        _tmpfile_fallback_warned = True
        logger.warning(msg)
    else:
        logger.debug(msg)
    tmp_path: str | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(prefix=f"{name}_", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
        except OSError as e:
            logger.warning(
                f"tempfile write failed: {e} (name={name!r}, buf={buf_mb:.1f} MB)",
                exc_info=True,
            )
            raise
        logger.debug(f"[memfd] Fallback tmpfile created: {tmp_path} (name={name!r}, buf={buf_mb:.1f} MB)")
        yield tmp_path
    finally:
        if tmp_path is not None:
            pathlib.Path(tmp_path).unlink(missing_ok=True)


@contextlib.contextmanager
def buffer_as_memfd_path(data: Buffer, *, name: str = "memfd") -> Generator[str]:
    """Write *data* to a memory-backed fd and yield a readable path."""
    raw = data if isinstance(data, (bytes, bytearray, memoryview)) else bytes(data)

    fd, fallback_reason = _try_memfd_create(name)

    if fd is not None:
        with _yield_memfd_path(fd, raw, name) as path:
            yield path
        return

    with _yield_tmpfile_path(raw, name, fallback_reason) as path:
        yield path
