# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media failure contract shared by the FFprobe/FFmpeg utilities and the recipes that call them.

Media utilities identify a failure; recipes decide how a confirmed one becomes a
row. A :class:`MediaError` is a permanent property of one source or clip, raised
only on positive evidence. Anything else, including :class:`ProbeError` and
:class:`FFmpegRunError` for runs that failed without establishing a media
problem, describes the environment or a transient condition: it must propagate
so the work stays retryable, and the shared Ray runtime retries both by name.
"""


class MediaError(RuntimeError):
    """Base for media failures attributable to one source or clip.

    Callers distinguish these from every other exception: a ``MediaError`` is a
    data outcome for one item, while anything else is a broken environment or a
    programming error and should fail the task it happened in.
    """


class InvalidMediaError(MediaError):
    """FFprobe explicitly reported invalid data in a complete local source.

    Callers must finish staging immutable input before probing a local path.
    The same diagnostic over HTTP remains a :class:`ProbeError`, since a
    transport failure can make otherwise valid media appear truncated.
    """


class UnsupportedMediaError(MediaError):
    """Raised when media is readable but does not satisfy the requested operation.

    FFprobe returned metadata, but it cannot support the requested operation.
    Remote callers should verify this against a complete local copy before
    recording a terminal outcome.
    """


class CutError(MediaError):
    """A readable source cannot yield one planned cut's exact frames.

    Raised for a cut the source's frame index rules out, a decode that selects
    other frames than planned, or a completed cut with the wrong frame count.
    """


class ProbeError(RuntimeError):
    """FFprobe failed without establishing a permanent media problem.

    Timeouts, process termination, IO failures, and unrecognized responses may
    be retried. Exhausting retries must fail the task rather than turn this
    exception into a terminal media outcome.
    """


class FFmpegRunError(RuntimeError):
    """An FFmpeg run failed without establishing a permanent media problem.

    A nonzero exit without a recognized cause, a timeout, a signal, an OS error,
    or an output that could not be counted says more about the machine than the
    media, so it is retried like :class:`ProbeError` and never becomes a terminal row.
    """
