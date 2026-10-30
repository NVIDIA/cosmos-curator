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

"""Turn the ``input`` config block into the session paths one run covers.

Two resolvers over the same block, because the two executors cover different ground:
:func:`single_session` for the Ray-free tool, which measures one session, and
:func:`expand_sessions` for a pipeline run, which measures a dataset.

Three forms -- an explicit list, a file of paths, and dataset roots to expand --
unioned into one deduplicated, sorted tuple. Deduplication is not cosmetic: two
spellings of one session would be measured twice and then collide on ``stream_id``
with nothing to break the tie, quietly dropping one session's provenance (see the
"One source reached twice in one run" section of the design doc).

Expansion happens on the driver, before Ray, because an input that resolves to
nothing is a configuration error and should be reported as one rather than as a
successful run that measured zero streams.
"""

import json
from pathlib import Path

from loguru import logger

from cosmos_curator.core.sensors.data_integrity import identity
from cosmos_curator.core.utils.storage.azure_client import is_azure_path
from cosmos_curator.core.utils.storage.s3_client import S3Prefix, is_s3path
from cosmos_curator.core.utils.storage.storage_utils import list_child_directories
from cosmos_curator.core.utils.storage_cli import make_s3_client
from cosmos_curator.next.recipes.data_integrity import storage_io
from cosmos_curator.next.recipes.data_integrity.config import (
    DataIntegrityExecutionConfig,
    DataIntegrityInputConfig,
)


def _join(root: str, child: str) -> str:
    """Append one path component to a root, for local paths and URIs alike."""
    return f"{root.rstrip('/')}/{child}"


def read_session_list(uri: str, *, s3_profile_name: str | None = None, endpoint_url: str | None = None) -> list[str]:
    """Read a session list from a local file or an ``s3://`` object.

    Accepts either a JSON array of strings or one path per line, because both are
    natural outputs for whatever produced the list: a query writes JSON, a shell
    pipeline writes lines. Blank lines and ``#`` comments are ignored in the line
    form.

    Args:
        uri: local path or ``s3://`` object holding the list.
        s3_profile_name: AWS profile for an ``s3://`` list.
        endpoint_url: S3 endpoint override.

    Returns:
        The session paths, in file order.

    Raises:
        ValueError: if ``uri`` carries an unsupported scheme, or if the content is
            JSON but not an array of strings.

    """
    if is_s3path(uri):
        payload = storage_io.read_text(uri, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    elif "://" in uri:
        msg = f"session_list_uri must be a local path or an s3:// object, got {uri!r}"
        raise ValueError(msg)
    else:
        payload = Path(uri).expanduser().read_text(encoding="utf-8")

    stripped = payload.lstrip()
    if stripped.startswith("["):
        loaded = json.loads(payload)
        if not isinstance(loaded, list) or any(not isinstance(entry, str) for entry in loaded):
            msg = f"session_list_uri {uri!r} holds JSON that is not an array of strings"
            raise ValueError(msg)
        return [entry.strip() for entry in loaded if entry.strip()]

    lines = (line.strip() for line in payload.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def list_sessions_under_root(
    root: str,
    *,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> list[str]:
    """List the session paths one level below ``root``.

    A session is a direct child of the root -- the layout the ``di`` tool documents,
    ``clips/<uuid>/`` under ``clips/``. Both branches delegate to the shared
    storage-level listers, which ``multimodal-split`` also uses, so the two recipes
    cannot disagree about what a child prefix is.

    The S3 client is built here with :func:`~cosmos_curator.core.utils.storage_cli.make_s3_client`
    rather than left to the lister, so enumeration uses the same AWS profile and
    endpoint override as the reads that follow: the credential chain
    ``storage_utils.get_storage_client`` would pick is a different one, and a run that
    enumerated a bucket it could not read would fail one stream at a time.

    Args:
        root: local directory or ``s3://`` prefix holding sessions.
        s3_profile_name: AWS profile for an ``s3://`` root.
        endpoint_url: S3 endpoint override.

    Returns:
        Full session paths, in listing order.

    Raises:
        ValueError: if ``root`` is an ``az://`` URI or carries another unsupported
            scheme. Sessions themselves may live on ``az://``; only expanding a root
            needs a delimited listing, which the shared listers do not offer there.
        FileNotFoundError: if the root does not exist or holds nothing.

    """
    if is_s3path(root):
        prefix = S3Prefix(root)
        children = make_s3_client(root, s3_profile_name, endpoint_url).list_child_prefixes(prefix)
        base = f"s3://{prefix.bucket}/{prefix.prefix}"
        return [_join(base, child) for child in children]
    if is_azure_path(root):
        msg = f"session_roots cannot be expanded on az:// yet: {root!r}; list the sessions explicitly instead"
        raise ValueError(msg)
    if "://" in root:
        msg = f"unsupported session root {root!r}; use a local directory or an s3:// prefix"
        raise ValueError(msg)
    local_root = str(Path(root).expanduser())
    return [_join(local_root, child) for child in list_child_directories(local_root)]


def _enclosing_session(session: str, kept: set[str]) -> str | None:
    """Return the kept session ``session`` lies under, if any.

    Walks the candidate's own parent paths rather than scanning what is kept, which
    keeps this linear in path depth instead of quadratic in session count, and only
    matches whole segments: ``clips/a`` encloses ``clips/a/b`` but not ``clips/ab``.
    """
    head = session
    while "/" in head:
        head = head.rpartition("/")[0]
        if head in kept:
            return head
    return None


def _drop_nested(sessions: list[str]) -> list[str]:
    """Drop any session that lies under another, keeping the enclosing one.

    ``discover_streams`` recurses, so a session nested under another is measured
    twice: once on its own and once as part of the session enclosing it. Both times
    the source string is identical, so both row sets carry the same ``stream_id``
    under this run's single ``run_id``, and nothing is left to break the tie. The
    enclosing session is the one kept, because it already covers the nested one's
    streams -- dropping it would lose coverage, where dropping the nested one loses
    nothing.

    Sorting the input first is what makes the choice deterministic rather than
    dependent on which form of the input named which path: an enclosing session
    sorts before everything under it.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for session in sessions:
        enclosing = _enclosing_session(session, seen)
        if enclosing is not None:
            logger.warning(
                "session {} lies under {} and would be measured twice; dropping the nested one",
                session,
                enclosing,
            )
            continue
        kept.append(session)
        seen.add(session)
    return kept


def single_session(input_config: DataIntegrityInputConfig) -> str:
    """Resolve the ``input`` block to the one session the Ray-free ``di`` tool measures.

    Deliberately not :func:`expand_sessions` with a count check afterwards. The two
    expanding forms are rejected on the config rather than on what they expand to, so
    the error costs no listing and no manifest read, and is the same error whether the
    root holds one session or ten thousand -- a run that happened to pass today because
    a root held exactly one session would start failing the day a second one landed.

    Canonicalized with the same :func:`identity.normalize_source` the expanding path
    uses, so one session measured by this tool and by a pipeline run carries one
    ``session_id`` in the store rather than two.

    Args:
        input_config: the resolved ``input`` block.

    Returns:
        The session path: a directory, a cloud prefix, or a single video file, all of
        which :func:`~.discovery.discover_streams` handles.

    Raises:
        ValueError: if the config names the expanding forms, or names anything other
            than exactly one session. Both messages point at ``run-pipeline``, which is
            the tool that does cover many sessions.

    """
    multi = [
        name
        for name, is_set in (
            ("session_list_uri", input_config.session_list_uri is not None),
            ("session_roots", bool(input_config.session_roots)),
        )
        if is_set
    ]
    if multi:
        msg = (
            f"{' and '.join(multi)} describe more than one session, which this tool does not run; "
            "name the one session in input.sessions, or use run-pipeline to cover a dataset"
        )
        raise ValueError(msg)
    if len(input_config.sessions) != 1:
        msg = (
            f"this tool measures exactly one session, but input.sessions names {len(input_config.sessions)}; "
            "use run-pipeline to cover more than one"
        )
        raise ValueError(msg)
    return identity.normalize_source(input_config.sessions[0])


def expand_sessions(
    input_config: DataIntegrityInputConfig,
    *,
    execution: DataIntegrityExecutionConfig,
) -> tuple[str, ...]:
    """Resolve the ``input`` block into the distinct session paths of one run.

    Paths are canonicalized with :func:`identity.normalize_source` -- the same
    function the store uses to derive ``session_id`` -- so two spellings of one
    session collapse here rather than becoming two sessions whose rows fight over one
    ``stream_id``. Local paths therefore come back absolute, without following
    symlinks, and cloud URIs keep their key verbatim minus a trailing slash.

    Sessions nested inside other sessions are then dropped by :func:`_drop_nested`,
    for the same reason: this is the whole of the run's protection against one source
    being measured twice, since each session is measured independently and no later
    stage sees them together.

    Args:
        input_config: the resolved ``input`` block.
        execution: the resolved ``execution`` block, for the listing credentials.

    Returns:
        Distinct session paths, sorted, so a re-run of one config covers the same
        sessions in the same order.

    Raises:
        ValueError: if the whole input expands to no sessions at all. A run that
            measured nothing is a mistake in the config, not a passing run.

    """
    collected: list[str] = list(input_config.sessions)

    if input_config.session_list_uri is not None:
        listed = read_session_list(
            input_config.session_list_uri,
            s3_profile_name=execution.s3_profile_name,
            endpoint_url=execution.endpoint_url,
        )
        logger.info("read {} session paths from {}", len(listed), input_config.session_list_uri)
        collected.extend(listed)

    for root in input_config.session_roots:
        expanded = list_sessions_under_root(
            root,
            s3_profile_name=execution.s3_profile_name,
            endpoint_url=execution.endpoint_url,
        )
        logger.info("expanded {} into {} sessions", root, len(expanded))
        collected.extend(expanded)

    distinct = sorted({identity.normalize_source(path) for path in collected})
    if not distinct:
        msg = (
            "input expanded to zero sessions; check 'sessions', 'session_list_uri' and 'session_roots' "
            "name sessions that exist"
        )
        raise ValueError(msg)
    sessions = tuple(_drop_nested(distinct))
    if len(sessions) != len(collected):
        logger.info("input named {} session paths, {} distinct", len(collected), len(sessions))
    return sessions
