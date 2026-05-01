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

"""Tests for memfd.py: three-tier memfd_create cascade and buffer_as_memfd_path."""

import errno
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from cosmos_curator.core.utils.misc.memfd import (
    _format_oserror_reason,
    _try_memfd_create,
    buffer_as_memfd_path,
)

_MODULE = "cosmos_curator.core.utils.misc.memfd"


class TestTryMemfdCreateTier1:
    """Tests for Tier 1: os.memfd_create (CPython native)."""

    def test_tier1_success(self) -> None:
        """When _OS_MEMFD_AVAILABLE is True, os.memfd_create should be called."""
        with (
            patch(f"{_MODULE}._OS_MEMFD_AVAILABLE", new=True),
            patch("os.memfd_create", create=True, return_value=42) as mock_os_memfd,
        ):
            fd, reason = _try_memfd_create("test-name")

        assert fd == 42
        assert reason is None
        mock_os_memfd.assert_called_once_with("test-name")

    def test_tier1_oserror_returns_none_with_reason(self) -> None:
        """When os.memfd_create raises OSError, should return (None, reason)."""
        oserr = OSError(errno.EPERM, "Operation not permitted")
        oserr.errno = errno.EPERM
        with (
            patch(f"{_MODULE}._OS_MEMFD_AVAILABLE", new=True),
            patch("os.memfd_create", create=True, side_effect=oserr),
        ):
            fd, reason = _try_memfd_create("test-name")

        assert fd is None
        assert reason is not None
        assert "EPERM" in reason


class TestTryMemfdCreateTier2:
    """Tests for Tier 2: ctypes libc.memfd_create fallback."""

    def test_tier2_success(self) -> None:
        """When os.memfd_create absent but ctypes available, libc call should succeed."""
        mock_libc = MagicMock()
        mock_libc.memfd_create.return_value = 7

        with (
            patch(f"{_MODULE}._OS_MEMFD_AVAILABLE", new=False),
            patch(f"{_MODULE}._CTYPES_MEMFD_AVAILABLE", new=True),
            patch(f"{_MODULE}._libc", mock_libc),
        ):
            fd, reason = _try_memfd_create("ctypes-test")

        assert fd == 7
        assert reason is None
        mock_libc.memfd_create.assert_called_once_with(b"ctypes-test", 0x0001)

    def test_tier2_negative_fd_returns_none(self) -> None:
        """When libc.memfd_create returns -1, should return (None, reason)."""
        mock_libc = MagicMock()
        mock_libc.memfd_create.return_value = -1

        with (
            patch(f"{_MODULE}._OS_MEMFD_AVAILABLE", new=False),
            patch(f"{_MODULE}._CTYPES_MEMFD_AVAILABLE", new=True),
            patch(f"{_MODULE}._libc", mock_libc),
            patch("ctypes.get_errno", return_value=errno.EPERM),
        ):
            fd, reason = _try_memfd_create("ctypes-fail")

        assert fd is None
        assert reason is not None
        assert "ctypes libc.memfd_create returned -1" in reason

    def test_tier2_oserror_returns_none(self) -> None:
        """When libc call raises OSError, should return (None, reason)."""
        mock_libc = MagicMock()
        oserr = OSError(errno.ENOSYS, "Function not implemented")
        oserr.errno = errno.ENOSYS
        mock_libc.memfd_create.side_effect = oserr

        with (
            patch(f"{_MODULE}._OS_MEMFD_AVAILABLE", new=False),
            patch(f"{_MODULE}._CTYPES_MEMFD_AVAILABLE", new=True),
            patch(f"{_MODULE}._libc", mock_libc),
        ):
            fd, reason = _try_memfd_create("ctypes-oserr")

        assert fd is None
        assert reason is not None
        assert "ENOSYS" in reason


class TestTryMemfdCreateTier3:
    """Tests for Tier 3: neither os nor ctypes memfd available."""

    def test_tier3_warns_once(self) -> None:
        """First call should warn; subsequent calls should not."""
        with (
            patch(f"{_MODULE}._OS_MEMFD_AVAILABLE", new=False),
            patch(f"{_MODULE}._CTYPES_MEMFD_AVAILABLE", new=False),
            patch(f"{_MODULE}._memfd_unavailable_warned", new=False),
            patch(f"{_MODULE}.logger") as mock_logger,
        ):
            fd, reason = _try_memfd_create("no-memfd")

        assert fd is None
        assert reason is not None
        assert "not available" in reason
        mock_logger.warning.assert_called_once()

    def test_tier3_subsequent_call_silent(self) -> None:
        """After warning once, subsequent calls should not warn again."""
        with (
            patch(f"{_MODULE}._OS_MEMFD_AVAILABLE", new=False),
            patch(f"{_MODULE}._CTYPES_MEMFD_AVAILABLE", new=False),
            patch(f"{_MODULE}._memfd_unavailable_warned", new=True),
        ):
            fd, reason = _try_memfd_create("no-memfd-2")

        assert fd is None
        assert reason == "memfd_create not available"


class TestFormatOserrorReason:
    """Tests for _format_oserror_reason helper."""

    def test_eperm_includes_seccomp_hint(self) -> None:
        """EPERM errors should include seccomp hint."""
        oserr = OSError(errno.EPERM, "Operation not permitted")
        oserr.errno = errno.EPERM
        reason = _format_oserror_reason(oserr, "test", source="os.memfd_create")
        assert "seccomp" in reason

    def test_enosys_includes_kernel_hint(self) -> None:
        """ENOSYS errors should include kernel hint."""
        oserr = OSError(errno.ENOSYS, "Function not implemented")
        oserr.errno = errno.ENOSYS
        reason = _format_oserror_reason(oserr, "test", source="ctypes libc.memfd_create")
        assert "not available in this kernel" in reason

    def test_enomem_includes_memory_hint(self) -> None:
        """ENOMEM errors should include memory hint."""
        oserr = OSError(errno.ENOMEM, "Cannot allocate memory")
        oserr.errno = errno.ENOMEM
        reason = _format_oserror_reason(oserr, "test", source="os.memfd_create")
        assert "insufficient memory" in reason


class TestBufferAsMemfdPath:
    """Tests for the public buffer_as_memfd_path context manager."""

    @pytest.mark.skipif(sys.platform != "linux", reason="memfd requires Linux")
    def test_memfd_path_readable(self) -> None:
        """On Linux, buffer_as_memfd_path should yield a /proc path with correct data."""
        payload = b"hello memfd"
        with buffer_as_memfd_path(payload, name="test-read") as path:
            assert "/proc/self/fd/" in path or "/tmp" in path  # noqa: S108
            with pathlib.Path(path).open("rb") as f:
                assert f.read() == payload

    def test_tmpfile_fallback_yields_readable_path(self) -> None:
        """When memfd is unavailable, tmpfile fallback should still yield readable data."""
        payload = b"fallback content"
        with (
            patch(f"{_MODULE}._try_memfd_create", return_value=(None, "forced fallback")),
            buffer_as_memfd_path(payload, name="test-fallback") as path,
        ):
            assert "/tmp" in path or "test-fallback" in path  # noqa: S108
            with pathlib.Path(path).open("rb") as f:
                assert f.read() == payload

    def test_tmpfile_cleaned_up_after_exit(self) -> None:
        """Temp file should be deleted after the context manager exits."""
        payload = b"cleanup test"
        with patch(f"{_MODULE}._try_memfd_create", return_value=(None, "forced fallback")):
            with buffer_as_memfd_path(payload, name="test-cleanup") as path:
                assert pathlib.Path(path).exists()
            assert not pathlib.Path(path).exists()
