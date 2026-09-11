# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable workspace whose PATH is the compatibility check for staged inference.

Two runs share completed inference if and only if they resolve to one directory,
a digest over the five inputs that decide whether one run's staged rows are
INTERCHANGEABLE with another's, so runs that any of those inputs separate cannot
see each other's rows and no per-row compatibility test exists downstream.

::

    <staging_root>/inference/<identity digest>/
        results/          staged terminal rows (Parquet)
        checkpoints/      completed row ids

The path IS the whole mechanism - no manifest, no lock, no claim. A manifest
could only restate what the path already pins, and being a second copy of that
state it could disagree with it. Interchangeability is all it pins, so two
further guarantees a reader might expect of a namespace are operator discipline
here rather than a runtime check:

- ONE WRITER AT A TIME per namespace, a precondition the caller enforces where
  runs are launched. A resume satisfies it: the earlier attempt has stopped and
  the checkpoint leaves the two clip sets disjoint. Two concurrent runs instead
  stage differing captions for one clip, publication refuses that pair, and
  cleanup follows only a commit, so the namespace must be deleted before the leg
  can proceed.
- TABLE IDENTITY. ``source_uri`` is a path rather than an identity, so a table
  dropped and recreated there inherits its predecessor's namespace once the read
  version and the contract digest coincide too. Give a recreated table a fresh
  path, or a staging root the previous table's namespaces do not live under.

Both are stated as limitations in
``docs/curator/design/curator-next-robot-action-caption.md``.

No completion marker either: the single Lance commit is the completion signal.

The workspace carries its own filesystem because the staged rows, the checkpoint
that indexes them, and post-commit cleanup must use ONE credential set; two
independently resolved ones would let the sink write where the checkpoint cannot
read or let cleanup run under different credentials. It carries the storage
profile those credentials came from for the same reason: cleanup deletes through
boto3 rather than through the Arrow filesystem (see ``cleanup_workspace``), and
resolving that client from the profile the filesystem was built from is what
keeps the two clients one credential set rather than two.
"""

from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import attrs
import pyarrow.fs as pafs
from loguru import logger

from cosmos_curator.core.utils.storage.s3_client import S3Prefix, create_s3_client, is_s3path
from cosmos_curator.next.recipes.robot_action_caption.request import staged_schema_fingerprint
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, filesystem_path
from cosmos_curator.next.utils.identity import canonical_digest

_NAMESPACE_SEGMENT: Final = "inference"
_RESULTS_SEGMENT: Final = "results"
_CHECKPOINTS_SEGMENT: Final = "checkpoints"

# Version of HOW staged rows are written, checkpointed, and read back, as opposed
# to WHAT fields they carry - `staged_schema_fingerprint` already covers the
# fields, so adding, removing, retyping, or reordering a staged field needs no
# bump here. Bump it for any change that leaves STAGED_SCHEMA byte-identical
# while changing the meaning or the recovery of a staged row. The list is meant
# to be exhaustive over what the current write and read paths depend on:
#
#   - checkpoint-id semantics: the `__checkpoint_id` column name, or the value it
#     is derived from (today Lance's `_rowaddr`), because a resume matches new
#     ids against the ids an earlier attempt recorded;
#   - the checkpoint surviving a successful Phase A
#     (`delete_checkpoint_on_success=False`), without which a resume re-infers
#     rows already staged and appends a second answer for each of them;
#   - the `results` and `checkpoints` segment names, because moving one without
#     the other separates the staged rows from the checkpoint indexing them;
#   - the partition columns and their Hive `key=value` directory scheme, which is
#     how publication addresses one fragment's staged partition;
#   - the Parquet file format;
#   - the append-versus-overwrite write mode;
#   - Ray's file-sink transaction assumptions - that a shard which became durable
#     without its checkpoint entry is discarded and replayed - including a Ray
#     upgrade that changes the sink's commit protocol or the checkpoint index
#     format.
#
# Throughput choices are NOT triggers: batch sizes and the rows-per-file hints
# change how many staged files a phase leaves, never whether one can be read.
#
# It is not a caption-contract input, and that is the whole reason it exists
# separately: none of those changes alters a caption's text, so a bump must
# isolate old workspaces without recaptioning a single canonical row.
STAGING_PROTOCOL_VERSION: Final = 1


def _normalize_staging_root(staging_root: str) -> str:
    """Remove redundant trailing separators without erasing filesystem roots."""
    normalized = staging_root.strip()
    parsed = urlsplit(normalized)
    # POSIX "/" and the path in "file:///" need their final slash to remain
    # absolute roots; non-root locations retain the existing slash-free form.
    if normalized == "/" or (parsed.scheme.lower() == "file" and parsed.path == "/"):
        return normalized
    return normalized.rstrip("/")


@attrs.frozen
class CaptionWorkspace:
    """Resolved locations for one (source snapshot, caption contract) pair.

    Attributes:
        root_uri: The namespace directory; equal for two compatible runs.
        filesystem: Credentialed filesystem the staged rows and the checkpoint
            are both written through.
        storage_profile: The profile ``filesystem`` was built from, so cleanup
            can resolve its own client against the same credentials.

    """

    root_uri: str
    filesystem: pafs.FileSystem
    storage_profile: str = "default"

    @property
    def results_uri(self) -> str:
        """Return the root of the staged terminal rows written by Phase A."""
        return f"{self.root_uri}/{_RESULTS_SEGMENT}"

    @property
    def checkpoints_uri(self) -> str:
        """Return the cluster-visible directory recording completed row ids."""
        return f"{self.root_uri}/{_CHECKPOINTS_SEGMENT}"


def resolve_workspace(
    *,
    staging_root: str,
    source_uri: str,
    read_version: int,
    digest: str,
    storage_profile: str = "default",
) -> CaptionWorkspace:
    """Return the namespace this run stages into, creating nothing.

    Pure path arithmetic plus a filesystem handle: nothing is written and nothing
    is claimed, so a run that turns out to have no pending work leaves no trace.

    Args:
        staging_root: Absolute local path, absolute local ``file://`` URI, or
            supported remote URI that contains the namespace.
        source_uri: Canonical (already normalized) URI of the clip table. Two
            spellings of one table must normalize to one value here, or the same
            work lands in two namespaces that cannot see each other.
        read_version: Lance version pinned AFTER the caption columns exist, so a
            first run and its resume pin the same one.
        digest: This run's caption contract digest.
        storage_profile: Credential profile for the staged writes and reads.

    Returns:
        The resolved workspace.

    Raises:
        ValueError: If ``staging_root`` is blank, unsupported, or a relative
            local path.

    """
    # A blank root would resolve the namespace against whichever directory the
    # process happens to be running in, which for a Ray worker is not the
    # directory the driver would resolve.
    base = _normalize_staging_root(staging_root)
    if not base:
        msg = "staging_root must be a non-empty absolute URI or path"
        raise ValueError(msg)
    parsed_root = urlsplit(base)
    # Raises on an unsupported scheme and on a non-local ``file://`` authority:
    # the shared Arrow seam owns the URI contract, so the only local fault left to
    # decide here is whether the path names one physical directory.
    root_path = filesystem_path(base)
    # Ray workers need not share the driver's working directory. A relative root
    # would make one namespace string name a different physical directory in each
    # process, splitting staged rows from the checkpoint that indexes them.
    if parsed_root.scheme.lower() != "s3" and not Path(root_path).is_absolute():
        msg = f"local staging_root must be absolute, got {staging_root!r}"
        raise ValueError(msg)
    # The last two inputs are in the namespace identity but NOT in the contract
    # digest, and the difference is deliberate. Neither changes any caption's text,
    # so neither may invalidate canonical captions, but both change how a staged row
    # is read back, so work across either boundary is incompatible with a resume.
    # The namespace is the only place that distinction belongs.
    #
    # Selection scope is deliberately NOT here, though it decides which rows an
    # attempt owes. Two attempts differ in scope only while pinning one read
    # version, which means the first never committed, and the two directions are
    # not alike: a WIDENED resume finds a subset of what it owes, which the
    # checkpoint and publication's coverage check already handle, while a NARROWED
    # one finds a superset and is refused before the commit with its staged rows
    # intact. Digesting the flag would buy that refusal by stranding every
    # completed caption whenever an attempt widens - unrecoverably, because
    # generation is unseeded and cleanup follows only a commit.
    #
    # The digest is a sha256, so the segment below is 64 hex characters: it can
    # carry neither a path separator nor a parent reference, and needs no
    # sanitizing.
    namespace = canonical_digest(
        {
            "source_uri": source_uri,
            "read_version": read_version,
            "contract_digest": digest,
            "staging_protocol_version": STAGING_PROTOCOL_VERSION,
            "staged_schema_fingerprint": staged_schema_fingerprint(),
        }
    )
    separator = "" if base.endswith("/") else "/"
    root_uri = f"{base}{separator}{_NAMESPACE_SEGMENT}/{namespace}"
    logger.info(f"Caption workspace for {source_uri} at v{read_version} resolves to {root_uri}.")
    return CaptionWorkspace(
        root_uri=root_uri,
        filesystem=arrow_filesystem(root_uri, storage_profile=storage_profile),
        storage_profile=storage_profile,
    )


def cleanup_workspace(workspace: CaptionWorkspace) -> None:
    """Remove the namespace after its rows have been published.

    Safe without a manifest check because the namespace is derived, not claimed:
    anything under this path was staged by a run whose semantics are identical to
    this one's, and this run has just published those very rows. Which rests on
    the one-writer precondition (module docstring) rather than on anything the
    path pins - a second concurrent attempt's staged rows would be deleted here
    too.

    Args:
        workspace: The workspace this run published from.

    Raises:
        ValueError: If an S3 namespace resolves no S3 client, which would mean
            the URI and the profile disagree about the backend.

    """
    if not is_s3path(workspace.root_uri):
        workspace.filesystem.delete_dir(filesystem_path(workspace.root_uri))
        return
    # Deleted through boto3 rather than through the workspace's own Arrow
    # filesystem because the two clients disagree about which checksum a
    # multi-object delete carries, and only one of the two answers is one that
    # Google Cloud Storage accepts. DeleteObjects is the single S3 delete the
    # service model marks requestChecksumRequired, so a checksum goes out no
    # matter what AWS_REQUEST_CHECKSUM_CALCULATION says - "when_required" is
    # satisfied here, not suppressed. PyArrow's embedded AWS SDK for C++ picks
    # CRC64-NVME, which GCS does not recognize and rejects as a missing header;
    # botocore picks CRC32, which GCS names as accepted. PyArrow exposes no
    # knob to change its choice, so the client has to change instead. Both
    # resolve the same storage profile, so this stays one credential set.
    client = create_s3_client(workspace.root_uri, workspace.storage_profile, can_delete=True)
    if client is None:
        msg = f"no S3 client for S3 namespace {workspace.root_uri} under profile {workspace.storage_profile!r}"
        raise ValueError(msg)
    deleted = client.delete_prefix(S3Prefix(workspace.root_uri))
    logger.info(f"Removed {deleted} staged object(s) under {workspace.root_uri}.")
