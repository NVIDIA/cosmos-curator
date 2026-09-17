# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable workspace whose PATH is the compatibility check for staged embeddings.

Two runs share computed vectors if and only if they resolve to one directory, a
digest over the inputs that decide whether one run's staged rows are
INTERCHANGEABLE with another's. Runs that any of those inputs separate cannot see
each other's rows, so no per-row compatibility test exists downstream.

::

    <staging_root>/<modality>/<identity digest>/
        results/          staged computed rows (Parquet)
        checkpoints/      completed row ids

The modality is a path SEGMENT rather than a digest input. Being unconditional it
separates the modalities on its own, and it leaves a listing legible to an
operator, which 64 hex characters do not. Digesting it as well would separate
nothing further - each modality's staged schema and recovery contract are derived
from its own column names, so both already differ per modality.

The path IS the whole mechanism - no manifest, no lock, no claim, no completion
marker. A manifest could only restate what the path already pins, and being a
second copy of that state it could disagree with it. The single Lance commit is
the completion signal.

Interchangeability is all the path pins, so two further guarantees a reader might
expect of a namespace are operator discipline rather than a runtime check:

- ONE WRITER AT A TIME per namespace, enforced where runs are launched. A resume
  satisfies it: the earlier attempt has stopped, and the checkpoint leaves the
  two row sets disjoint. Two concurrent runs instead stage differing vectors for
  one clip, and cleanup follows only a commit, so the namespace must be deleted
  before the leg can proceed.
- TABLE IDENTITY. ``source_uri`` is a path rather than an identity, so a table
  dropped and recreated there inherits its predecessor's namespaces once the read
  version and the contract coincide too. Give a recreated table a fresh path, or
  a staging root the previous table's namespaces do not live under.

The workspace carries its own filesystem because the staged rows, the checkpoint
indexing them, and post-commit cleanup must use ONE credential set; two
independently resolved ones would let the sink write where the checkpoint cannot
read. It carries the storage profile those credentials came from for the same
reason: cleanup deletes through boto3 rather than through the Arrow filesystem
(see :func:`cleanup_workspace`), and resolving that client from the same profile
is what keeps the two clients one credential set rather than two.
"""

from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import attrs
import pyarrow.fs as pafs
from loguru import logger

from cosmos_curator.core.utils.storage.s3_client import S3Prefix, create_s3_client, is_s3path
from cosmos_curator.next.embeddings.schemas import EmbeddingColumnGroup
from cosmos_curator.next.recipes.embeddings.config import Modality
from cosmos_curator.next.recipes.embeddings.staged import STAGING_PROTOCOL_VERSION, staged_schema_fingerprint
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, filesystem_path, normalize_location_root
from cosmos_curator.next.utils.identity import canonical_digest

EMBEDDING_STAGING_ROOT_SUFFIX: Final = "__embedding_staging"

_RESULTS_SEGMENT: Final = "results"
_CHECKPOINTS_SEGMENT: Final = "checkpoints"


@attrs.frozen
class EmbeddingWorkspace:
    """Resolved locations for one (source snapshot, modality, computation) triple.

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
        """Return the root of the staged computed rows."""
        return f"{self.root_uri}/{_RESULTS_SEGMENT}"

    @property
    def checkpoints_uri(self) -> str:
        """Return the cluster-visible directory recording completed row ids."""
        return f"{self.root_uri}/{_CHECKPOINTS_SEGMENT}"


def default_staging_root(clips_lance_uri: str) -> str:
    """Derive the staging root beside the clips table, so a run needs no config to recover.

    A SIBLING of the table directory, mirroring ``action_pca_root_uri``'s
    ``__action_pca`` suffix so the recipe keeps one idiom for its derived
    locations. The ``rstrip`` is what keeps it a sibling: without it a URI ending
    in ``/`` would nest the staging directory INSIDE the Lance dataset, where a
    Lance reader would then have to ignore it.
    """
    return f"{clips_lance_uri.rstrip('/')}{EMBEDDING_STAGING_ROOT_SUFFIX}"


def resolve_workspace(  # noqa: PLR0913 - each argument independently decides the resolved path
    *,
    staging_root: str,
    source_uri: str,
    read_version: int,
    modality: Modality,
    recovery_contract: str,
    group: EmbeddingColumnGroup,
    storage_profile: str = "default",
) -> EmbeddingWorkspace:
    """Return the namespace this modality stages into, creating nothing.

    Pure path arithmetic plus a filesystem handle: nothing is written and nothing
    is claimed, so a modality that turns out to owe no work leaves no trace.

    Args:
        staging_root: Absolute local path, absolute local ``file://`` URI, or
            supported remote URI that contains the namespace.
        source_uri: Canonical (already normalized) URI of the clips table. Two
            spellings of one table must normalize to one value here, or the same
            work lands in two namespaces that cannot see each other.
        read_version: Lance version pinned AFTER the modality's columns exist, so
            a first run and its resume pin the same one.
        modality: Which modality stages here; becomes a path segment, and is what
            keeps two modalities' namespaces disjoint.
        recovery_contract: Digest of what this run will COMPUTE. Text and image
            pass their result identity; action passes its descriptor contract,
            because the final identity carries a basis fingerprint that is not
            known before the basis is bound.
        group: The modality's column group, whose staged schema is digested.
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
    base = normalize_location_root(staging_root)
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
    # The last two inputs are NOT part of what the run computes, and the
    # difference is deliberate. Neither changes a single vector, so neither may
    # invalidate canonical rows, but both change how a staged row is READ BACK,
    # so work across either boundary is incompatible with a resume. The namespace
    # is the only place that distinction belongs.
    #
    # The digest is a sha256, so the segment below is 64 hex characters: it can
    # carry neither a path separator nor a parent reference, and needs no
    # sanitizing.
    #
    # The modality is deliberately NOT among these. It is a path segment on every
    # resolved URI, so it separates the modalities whatever this digest says, and
    # both the contract and the staged fingerprint are already derived from the
    # modality's own column names. Adding it back would separate nothing.
    namespace = canonical_digest(
        {
            "source_uri": source_uri,
            "read_version": read_version,
            "recovery_contract": recovery_contract,
            "staging_protocol_version": STAGING_PROTOCOL_VERSION,
            "staged_schema_fingerprint": staged_schema_fingerprint(group),
        }
    )
    separator = "" if base.endswith("/") else "/"
    root_uri = f"{base}{separator}{modality}/{namespace}"
    logger.info(f"Embedding workspace for {modality} on {source_uri} at v{read_version} resolves to {root_uri}.")
    return EmbeddingWorkspace(
        root_uri=root_uri,
        filesystem=arrow_filesystem(root_uri, storage_profile=storage_profile),
        storage_profile=storage_profile,
    )


def cleanup_workspace(workspace: EmbeddingWorkspace) -> None:
    """Remove the namespace after its rows have been published.

    Safe without a manifest check because the namespace is derived, not claimed:
    anything under this path was staged by a run whose computation is identical to
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
    # botocore picks CRC32, which GCS names as accepted. PyArrow exposes no knob
    # to change its choice, so the client has to change instead. Both resolve the
    # same storage profile, so this stays one credential set.
    client = create_s3_client(workspace.root_uri, workspace.storage_profile, can_delete=True)
    if client is None:
        msg = f"no S3 client for S3 namespace {workspace.root_uri} under profile {workspace.storage_profile!r}"
        raise ValueError(msg)
    deleted = client.delete_prefix(S3Prefix(workspace.root_uri))
    logger.info(f"Removed {deleted} staged object(s) under {workspace.root_uri}.")
