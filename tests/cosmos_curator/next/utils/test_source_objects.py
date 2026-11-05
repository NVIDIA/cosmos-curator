# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only a missing or archived source object is recognized as stably unreadable."""

import pytest
from botocore.exceptions import ClientError

from cosmos_curator.next.utils import source_objects


@pytest.mark.parametrize(
    ("exc", "bucket_exists", "terminal"),
    [
        (FileNotFoundError("gone"), False, True),
        (ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject"), False, True),
        (ClientError({"Error": {"Code": "InvalidObjectState"}}, "GetObject"), False, True),
        (ClientError({"Error": {"Code": "404"}}, "HeadObject"), True, True),
        (ClientError({"Error": {"Code": "404"}}, "HeadObject"), False, False),
        (ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject"), True, False),
        (ClientError({"Error": {"Code": "SlowDown"}}, "GetObject"), True, False),
        (ConnectionError("reset"), True, False),
    ],
)
def test_only_stably_unreadable_sources_are_terminal(
    exc: BaseException, *, bucket_exists: bool, terminal: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare 404 is a missing key only when the bucket exists; environment failures never are."""
    monkeypatch.setattr(source_objects, "_bucket_exists", lambda *_args, **_kwargs: bucket_exists)
    assert source_objects.is_unreadable_source(exc, "s3://bucket/key", storage_profile="default") is terminal
