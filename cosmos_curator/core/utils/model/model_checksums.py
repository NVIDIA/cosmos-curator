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

"""Source and verify sha256 checksums for downloaded model files.

Why this exists: a model download can fail in a way that leaves files of exactly the right
size on disk. The NGC SDK pre-allocates each output file to its final size
(``f.truncate(size)``) before worker threads fill it in, and those workers swallow their own
exceptions at DEBUG level, so a transfer that moved no bytes still "succeeds" and leaves
correctly-sized, zero-filled weights behind. A size check cannot catch that; a content hash
can.

Which file to checksum for a given model is chosen by a human editing ``all_models.json``
-- it is the artifact the loader actually reads, which no heuristic can reliably infer.
The tooling here only resolves and verifies the hashes for those named files.

Checksums come from Hugging Face rather than from whatever we happened to download, because
HF is the upstream the NGC copies are synced from. ``all_models.json``'s ``version`` is the
HF git revision and NGC's model version reuses that same string, so both sides pin to one
commit and a mismatch is a real finding rather than an artifact of comparing revisions. HF
serves an authoritative sha256 for LFS-tracked files without transferring them; the
remaining files are small (the largest across every model in ``all_models.json`` is ~10 MB)
and are fetched and hashed directly.

The same functions serve both callers: the pipeline's own download path verifies what it
just downloaded, and the standalone CLI verifies what is published in NGC.
"""

import hashlib
import pathlib
import typing

import requests
from loguru import logger

_HF_API_ROOT = "https://huggingface.co/api/models"
_HF_RESOLVE_ROOT = "https://huggingface.co"
_HTTP_TIMEOUT_S = 60

# Guard on fetching a non-LFS file just to hash it. Nothing in all_models.json comes close
# (largest is ~10 MB); this only stops a repo layout change from silently turning checksum
# generation into a multi-GB download.
_MAX_INLINE_FETCH_BYTES = 64 * 1024 * 1024


class ChecksumVerificationError(RuntimeError):
    """Raised when downloaded model files do not match their recorded sha256.

    Distinct from a transfer failure: the transfer reported success and produced files of
    the expected size, but their contents are wrong. Retrying the same download will not
    help, so callers should treat this as terminal rather than feeding it back into a retry
    loop. Subclasses RuntimeError so existing broad handlers keep working.
    """


class HfFile(typing.NamedTuple):
    """One file in a Hugging Face repo at a pinned revision."""

    filename: str
    size: int
    sha256: str | None  # None for non-LFS files, which HF addresses by git SHA-1


def sha256_file(path: pathlib.Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file, streamed so large weights stay off-heap.

    Args:
        path: file to hash.
        chunk_size: read size in bytes.

    Returns:
        Lowercase hex digest.

    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_hf_files(model_id: str, revision: str, token: str | None = None) -> list[HfFile]:
    """List a HF repo's files at a pinned revision, with sha256 for LFS-tracked ones.

    Two calls, because HF splits this: the revision endpoint lists filenames and sizes, and
    ``paths-info`` resolves LFS pointers to their sha256. Nothing is downloaded here.

    Args:
        model_id: HF repo id, e.g. ``"Sn4kehead/TransNetV2"``.
        revision: git revision to pin to (``all_models.json``'s ``version``).
        token: optional HF token, needed for gated repos.

    Returns:
        One HfFile per file in the repo, sorted largest first.

    Raises:
        requests.HTTPError: If either HF request fails (404 unknown repo/revision, 401/403
            gated repo needing a token).

    """
    headers = _hf_headers(token)

    listing = requests.get(
        f"{_HF_API_ROOT}/{model_id}/revision/{revision}",
        headers=headers,
        timeout=_HTTP_TIMEOUT_S,
    )
    listing.raise_for_status()
    names = [s["rfilename"] for s in listing.json().get("siblings", [])]
    if not names:
        return []

    info = requests.post(
        f"{_HF_API_ROOT}/{model_id}/paths-info/{revision}",
        headers=headers,
        json={"paths": names},
        timeout=_HTTP_TIMEOUT_S,
    )
    info.raise_for_status()

    files = [
        HfFile(
            filename=entry["path"],
            size=int(entry.get("size", 0)),
            sha256=(entry.get("lfs") or {}).get("oid"),
        )
        for entry in info.json()
        if entry.get("type") == "file"
    ]
    return sorted(files, key=lambda f: f.size, reverse=True)


def _hash_remote_file(model_id: str, revision: str, filename: str, token: str | None) -> str:
    """Download one small non-LFS file and return its sha256.

    HF only publishes sha256 for LFS-tracked files; everything else it addresses by git
    SHA-1, which is not what lands on disk. Hashing the bytes ourselves is the only way to
    cover those, and they are small enough for that to be cheap.
    """
    response = requests.get(
        f"{_HF_RESOLVE_ROOT}/{model_id}/resolve/{revision}/{filename}",
        headers=_hf_headers(token),
        timeout=_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    return hashlib.sha256(response.content).hexdigest()


def resolve_checksums_for(
    model_id: str,
    revision: str,
    filenames: list[str],
    token: str | None = None,
) -> dict[str, str]:
    """Resolve the sha256 of specific, named files at a pinned revision.

    Which file matters for a given model is a judgement call -- which artifact the loader
    actually reads -- so it is made by a human editing ``all_models.json`` and recorded as
    the keys of that model's ``sha256`` entry. This function only resolves the values; it
    never picks files itself.

    Args:
        model_id: HF repo id.
        revision: git revision to pin to (``all_models.json``'s ``version``).
        filenames: repo-relative paths to resolve, i.e. the existing ``sha256`` keys.
        token: optional HF token, needed for gated repos.

    Returns:
        Mapping of filename to sha256 for each name that could be resolved.

    """
    available = {f.filename: f for f in fetch_hf_files(model_id, revision, token=token)}

    checksums: dict[str, str] = {}
    for filename in filenames:
        entry = available.get(filename)
        if entry is None:
            logger.warning(f"{model_id}@{revision[:8]}: {filename} is not in the repo at this revision")
            continue
        if entry.sha256:
            # LFS-tracked: HF publishes the sha256, no transfer needed.
            checksums[filename] = entry.sha256
        elif entry.size <= _MAX_INLINE_FETCH_BYTES:
            # HF addresses non-LFS files by git SHA-1, so hash the bytes ourselves.
            checksums[filename] = _hash_remote_file(model_id, revision, filename, token)
        else:
            logger.warning(
                f"{model_id}: {filename} is {entry.size} bytes and not LFS-tracked, so its sha256 "
                "would require downloading it; skipping rather than doing that implicitly.",
            )
    return checksums


def verify_local_files(directory: pathlib.Path, expected: dict[str, str]) -> dict[str, tuple[str, str | None]]:
    """Check expected filename->sha256 pairs against files on disk.

    Args:
        directory: directory the expected files should live in. Files are matched by their
            relative path, then by name anywhere beneath it, since the NGC SDK nests
            downloads under a ``<name>_v<version>`` directory.
        expected: mapping of filename to expected sha256.

    Returns:
        Mapping of filename to ``(expected, actual)`` for each mismatch; empty means
        everything matched. ``actual`` is ``None`` when the file is missing entirely, which
        is how an aborted download can also present.

    """
    index = _index_files(directory)
    mismatches: dict[str, tuple[str, str | None]] = {}
    for filename, expected_hash in expected.items():
        found = _locate(directory, filename, index)
        if found is None:
            mismatches[filename] = (expected_hash, None)
            continue
        actual = sha256_file(found)
        if actual != expected_hash:
            mismatches[filename] = (expected_hash, actual)
    return mismatches


def _index_files(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    """Walk *directory* once, indexing files by relative path and by bare name.

    Built once per verification rather than globbing per file: a sharded model can list
    dozens of names, and re-walking the tree for each is needless IO.
    """
    index: dict[str, pathlib.Path] = {}
    for path in directory.rglob("*"):
        if path.is_file():
            index.setdefault(path.relative_to(directory).as_posix(), path)
            index.setdefault(path.name, path)
    return index


def _locate(directory: pathlib.Path, filename: str, index: dict[str, pathlib.Path]) -> pathlib.Path | None:
    """Resolve *filename* to a real file, preferring an exact relative-path match.

    Falls back to matching on the bare name because the NGC SDK nests downloads under a
    ``<name>_v<version>`` directory, so recorded paths are one level up from where they land.
    """
    direct = directory / filename
    if direct.is_file():
        return direct
    return index.get(filename) or index.get(pathlib.PurePosixPath(filename).name)


def format_mismatches(mismatches: dict[str, tuple[str, str | None]]) -> str:
    """Render mismatches from :func:`verify_local_files` as a single log-friendly line."""
    return ", ".join(
        f"{name}: expected {expected}, got {actual or 'MISSING'}" for name, (expected, actual) in mismatches.items()
    )


def is_all_zero(path: pathlib.Path) -> bool:
    """Report whether a file is non-empty and consists solely of zero bytes.

    This is the exact signature of the NGC SDK pre-allocating a file and never filling it,
    so it is worth naming explicitly rather than reporting a bare hash mismatch.

    Args:
        path: file to inspect.

    Returns:
        True if the file is non-empty and entirely zeros.

    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            if chunk.count(0) != len(chunk):
                return False
    return True


def describe_verification_failure(directory: pathlib.Path, mismatches: dict[str, tuple[str, str | None]]) -> str:
    """Build an error message for a failed verification, calling out zero-filled files.

    Args:
        directory: directory the files were verified in.
        mismatches: output of :func:`verify_local_files`.

    Returns:
        A message naming the mismatches, and the likely cause when files are all zeros.

    """
    message = format_mismatches(mismatches)
    index = _index_files(directory)
    zeroed = [name for name in mismatches if (found := _locate(directory, name, index)) and is_all_zero(found)]
    if zeroed:
        message += (
            f". {', '.join(zeroed)} contain only zero bytes, which means the transfer wrote nothing: "
            "the NGC SDK pre-allocates files to their full size, and its download workers log "
            "failures at DEBUG only. Enable logging.getLogger('ngcbase').setLevel(logging.DEBUG) "
            "to see the underlying error."
        )
    return message
