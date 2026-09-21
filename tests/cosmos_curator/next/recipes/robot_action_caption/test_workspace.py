# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace resolution: the PATH is the entire compatibility check.

``resolve_workspace`` writes nothing and claims nothing, so almost everything
worth pinning is a property of the URI it returns - two runs share completed
inference if and only if they resolve to one directory. The tests therefore
compare resolved paths, and touch the filesystem only where a real effect exists:
cleanup, and the assertion that resolution itself leaves no trace.

Five inputs decide that URI, and they split into two groups the assertions keep
apart:

    source_uri, contract_digest, shard   the run's own identity
    schema fingerprint, protocol         Phase A compatibility

The second group is why the golden namespace below is pinned as a literal. A
change to the staged fields or to the staging protocol has to isolate old
workspaces, and comparing two live resolutions cannot see it: both sides would
move together and agree.

The read version is deliberately not among the five, so an attempt can resume one
that pinned a different snapshot. Selection scope is not among them either, so a
widened attempt resumes the narrower one it widens. Resolution can express
neither - there is no version or scope argument left to vary - so the golden
literal is what turns red if either is digested again, and the reuse that buys is
pinned end to end in ``test_recovery_integration.py``.

Work-shard scope IS among the five, and for the opposite reason to those two: it
divides concurrent jobs rather than resuming one, and post-commit cleanup deletes
a whole namespace root, so two shards sharing one would have the first to publish
destroy the rest's staged rows. Ray's checkpoint recovery deletes a concurrent
job's pending entries and their data files for the same directory, so scoping
this side's cleanup could not rescue a shared one either. Its serialization
inside the digest is pinned rather than merely varied, because the namespace is a
durable directory name.

Location refusals stop a blank or relative local staging root before it can name
different physical directories in different processes. They arrive from two
layers: ``resolve_workspace`` decides whether a local path is absolute, while the
scheme and the ``file://`` authority are the shared Arrow seam's contract, which
resolution consults before hashing anything. The namespace segment is a sha256
this function derives itself, so a caller cannot express a separator or a parent
reference in it and there is nothing left to sanitize.
"""

import pathlib
from collections.abc import Callable
from unittest.mock import Mock

import pyarrow.fs as pafs
import pytest

from cosmos_curator.next.core.sharding import ShardConfig
from cosmos_curator.next.recipes.robot_action_caption import workspace

# Stand-ins for a caption contract digest. Their VALUES are never asserted in the
# comparison tests - only whether two resolutions agree - so any two stable,
# distinct strings serve; resolution does not parse the digest. Shaped as a
# sha256 anyway, because that is what a caller passes and a 64-hex namespace
# segment is what an operator will see. ``_DIGEST`` is also one of the golden
# namespace's inputs, so changing it moves that literal.
_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64

# The exact namespace one fixed identity resolves to. Derived once from the five
# inputs below and pasted here: see the module docstring for why a literal rather
# than a second live resolution. The staged-schema half of it is pinned
# independently in ``test_request.py``, so a fingerprint change turns both red and
# names itself.
_GOLDEN_STAGING_ROOT = "/staging"
_GOLDEN_SOURCE_URI = "/tables/clips.lance"
_GOLDEN_NAMESPACE = "3886681f1d9549627e9d0d397e23c03fd653e3923263ab45a6c67782043e4a21"

# The same identity with a work shard, pinned for the same reason: the shard's
# serialization inside the digest is a choice, and the only thing that can catch
# it drifting is a literal. A live comparison against an unsharded resolution
# would keep agreeing under any encoding.
_GOLDEN_SHARD = ShardConfig(count=8, index=3)
_GOLDEN_SHARDED_NAMESPACE = "de9c77fdec003334f30b69bc748e74761ed0c6d13988f3f57243ddeb74d3f06e"

# Resolves a workspace with one fixed identity, each input overridable; see
# ``resolve``.
WorkspaceFactory = Callable[..., workspace.CaptionWorkspace]


@pytest.fixture
def staging_root(tmp_path: pathlib.Path) -> str:
    """Return the durable root this test's namespaces resolve under."""
    return str(tmp_path / "staging")


@pytest.fixture
def resolve(staging_root: str) -> WorkspaceFactory:
    """Return a resolver over one identity, with every input overridable by keyword."""

    def build(**overrides: object) -> workspace.CaptionWorkspace:
        inputs: dict[str, object] = {
            "staging_root": staging_root,
            "source_uri": _GOLDEN_SOURCE_URI,
            "digest": _DIGEST,
            "shard": None,
        }
        inputs.update(overrides)
        return workspace.resolve_workspace(**inputs)  # type: ignore[arg-type]

    return build


def test_one_fixed_identity_resolves_to_the_golden_namespace(resolve: WorkspaceFactory) -> None:
    """The namespace is a durable directory name, so its serialization is pinned.

    Every input, the key names they are hashed under, and ``canonical_digest``'s
    own JSON form all feed this value. Re-deriving it inside the test would run
    the same code on both sides and pass through any of those changes.
    """
    resolved = resolve(staging_root=_GOLDEN_STAGING_ROOT)

    assert resolved.root_uri == f"{_GOLDEN_STAGING_ROOT}/inference/{_GOLDEN_NAMESPACE}"


def test_two_runs_with_the_same_inputs_resolve_to_one_namespace(resolve: WorkspaceFactory) -> None:
    """A resume must land in the directory the attempt it resumes staged into."""
    assert resolve().root_uri == resolve().root_uri


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_uri", "/tables/other.lance"),
        ("digest", _OTHER_DIGEST),
        ("shard", _GOLDEN_SHARD),
    ],
)
def test_a_changed_identity_input_resolves_to_a_different_namespace(
    resolve: WorkspaceFactory, field: str, value: object
) -> None:
    """Runs that must not share staged rows cannot see each other's directory.

    Another table, different caption semantics, and a different work shard are
    three different reasons, but they share one consequence: no per-row
    compatibility test exists downstream, so separation happens in the path or it
    does not happen.
    """
    assert resolve().root_uri != resolve(**{field: value}).root_uri


def test_one_sharded_identity_resolves_to_the_golden_sharded_namespace(resolve: WorkspaceFactory) -> None:
    """The shard's serialization inside the digest is pinned, not merely varied.

    A live comparison against the unsharded resolution stays green under any
    encoding of the pair, including one that collapses two distinct shards, so
    only a literal can catch that shape drifting.
    """
    resolved = resolve(staging_root=_GOLDEN_STAGING_ROOT, shard=_GOLDEN_SHARD)

    assert resolved.root_uri == f"{_GOLDEN_STAGING_ROOT}/inference/{_GOLDEN_SHARDED_NAMESPACE}"


def test_every_shard_of_one_campaign_resolves_to_its_own_namespace(resolve: WorkspaceFactory) -> None:
    """Concurrent shards must not share a root, because cleanup deletes the whole root.

    They agree on every other identity input by construction - one table, one
    contract - so the shard is the only thing that can separate them. If it
    cannot, the first shard to publish deletes every other's staged rows, and
    because generation is unseeded those captions cannot be reproduced.
    """
    count = 8
    roots = {resolve(shard=ShardConfig(count=count, index=index)).root_uri for index in range(count)}

    assert len(roots) == count


def test_an_unsharded_run_does_not_share_the_namespace_of_a_single_shard(resolve: WorkspaceFactory) -> None:
    """``None`` and a one-shard scope are different scopes and must serialize apart.

    They own the same fragments, so a digest that folded them together would look
    harmless. It is not: the two are launched by different campaigns, and a
    ``count=1`` job publishing would delete an unsharded attempt's staged rows.
    """
    assert resolve(shard=None).root_uri != resolve(shard=ShardConfig(count=1, index=0)).root_uri


def test_a_changed_staged_schema_separates_namespaces_while_the_contract_stays_equal(
    resolve: WorkspaceFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows written under one staged schema must not be read back under another.

    Both resolutions pass the same ``digest``, so the caption semantics are
    provably identical and no published row is stale. Staged Parquet written
    under the old field set still cannot be folded under the new one, so the
    namespace has to move anyway - the case that needs an identity input the
    contract digest does not carry.
    """
    under_current_schema = resolve()
    monkeypatch.setattr(workspace, "staged_schema_fingerprint", lambda: "c" * 64)

    assert resolve().root_uri != under_current_schema.root_uri


def test_a_bumped_staging_protocol_separates_namespaces_while_the_contract_stays_equal(
    resolve: WorkspaceFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A protocol bump isolates old workspaces without recaptioning anything.

    Separate from the staged-schema case because the two cover disjoint changes:
    the fingerprint moves when the persisted FIELDS change, and this moves when
    they do not - a different checkpoint-id source or partitioning leaves the
    schema byte-identical while making the rows unreadable to a resume.
    """
    under_current_protocol = resolve()
    monkeypatch.setattr(workspace, "STAGING_PROTOCOL_VERSION", workspace.STAGING_PROTOCOL_VERSION + 1)

    assert resolve().root_uri != under_current_protocol.root_uri


def test_resolving_a_workspace_creates_nothing_on_disk(resolve: WorkspaceFactory, staging_root: str) -> None:
    """A run that turns out to owe no work must leave no directory behind."""
    resolved = resolve()

    assert not pathlib.Path(resolved.root_uri).exists()
    assert not pathlib.Path(staging_root).exists()


def test_the_namespace_nests_under_the_staging_root(resolve: WorkspaceFactory, staging_root: str) -> None:
    """The staging root is the caller's, so nothing may resolve outside it."""
    assert resolve().root_uri.startswith(f"{staging_root}/")


def test_a_trailing_separator_on_the_staging_root_resolves_to_one_namespace(
    resolve: WorkspaceFactory, staging_root: str
) -> None:
    """Two spellings of one root must not split one run's work across two namespaces."""
    assert resolve(staging_root=f"{staging_root}/").root_uri == resolve().root_uri


def test_the_posix_root_builds_absolute_workspace_uris(resolve: WorkspaceFactory) -> None:
    """The POSIX root remains absolute when workspace segments are appended."""
    resolved = resolve(staging_root="/")

    assert (
        resolved.root_uri,
        resolved.results_uri,
        resolved.checkpoints_uri,
    ) == (
        f"/inference/{_GOLDEN_NAMESPACE}",
        f"/inference/{_GOLDEN_NAMESPACE}/results",
        f"/inference/{_GOLDEN_NAMESPACE}/checkpoints",
    )


def test_the_file_uri_root_builds_absolute_workspace_uris(resolve: WorkspaceFactory) -> None:
    """The slash that makes ``file:///`` absolute survives workspace joining."""
    resolved = resolve(staging_root="file:///")

    assert (
        resolved.root_uri,
        resolved.results_uri,
        resolved.checkpoints_uri,
    ) == (
        f"file:///inference/{_GOLDEN_NAMESPACE}",
        f"file:///inference/{_GOLDEN_NAMESPACE}/results",
        f"file:///inference/{_GOLDEN_NAMESPACE}/checkpoints",
    )


@pytest.mark.parametrize(
    ("staging_root", "expected_root_uri"),
    [
        ("/var/lib/caption///", f"/var/lib/caption/inference/{_GOLDEN_NAMESPACE}"),
        ("file:///tmp///", f"file:///tmp/inference/{_GOLDEN_NAMESPACE}"),
        ("s3://bucket-under-test///", f"s3://bucket-under-test/inference/{_GOLDEN_NAMESPACE}"),
    ],
    ids=["absolute_path", "absolute_file_uri", "s3_uri"],
)
def test_repeated_trailing_separators_are_removed_from_non_root_locations(
    resolve: WorkspaceFactory,
    monkeypatch: pytest.MonkeyPatch,
    staging_root: str,
    expected_root_uri: str,
) -> None:
    """Redundant trailing separators do not split a non-root workspace."""
    monkeypatch.setattr(workspace, "arrow_filesystem", lambda *_args, **_kwargs: pafs.LocalFileSystem())

    assert resolve(staging_root=staging_root).root_uri == expected_root_uri


def test_cleanup_removes_the_namespace_whole(resolve: WorkspaceFactory) -> None:
    """The staged rows and the checkpoint indexing them are siblings, so both go.

    Removing one and keeping the other would let a later run resume against half
    a namespace: a checkpoint claiming rows are done, with no staged row left to
    publish for them.
    """
    resolved = resolve()
    staged = pathlib.Path(resolved.results_uri) / "fragment_id=0" / "part-0.parquet"
    checkpoint = pathlib.Path(resolved.checkpoints_uri) / "0.json"
    for path in (staged, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"staged by an earlier attempt")

    workspace.cleanup_workspace(resolved)

    assert not pathlib.Path(resolved.root_uri).exists()


def test_cleanup_uses_the_workspace_filesystem_and_relative_path() -> None:
    """Cleanup must not re-resolve credentials independently of the workspace."""
    filesystem = Mock(spec=pafs.FileSystem)
    resolved = workspace.CaptionWorkspace(
        root_uri="file:///var/lib/caption/namespace",
        filesystem=filesystem,
    )

    workspace.cleanup_workspace(resolved)

    filesystem.delete_dir.assert_called_once_with("/var/lib/caption/namespace")


def test_cleanup_of_an_s3_namespace_goes_through_boto3_not_the_arrow_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which client deletes is the fix, not an implementation detail.

    Arrow's bulk delete sends a checksum GCS refuses, so a regression to
    ``delete_dir`` here is the leak returning. The profile is asserted alongside
    the prefix because a client resolved from a different one could reach a
    location the run never wrote to.
    """
    filesystem = Mock(spec=pafs.FileSystem)
    client = Mock()
    client.delete_prefix.return_value = 3
    resolved_with: list[tuple[str, str, bool]] = []

    def fake_client(target_path: str, profile_name: str, *, can_delete: bool) -> Mock:
        resolved_with.append((target_path, profile_name, can_delete))
        return client

    monkeypatch.setattr(workspace, "create_s3_client", fake_client)
    resolved = workspace.CaptionWorkspace(
        root_uri="s3://bucket/inference/namespace",
        filesystem=filesystem,
        storage_profile="captions",
    )

    workspace.cleanup_workspace(resolved)

    assert resolved_with == [("s3://bucket/inference/namespace", "captions", True)]
    assert [call.args[0].path for call in client.delete_prefix.call_args_list] == ["s3://bucket/inference/namespace"]
    filesystem.delete_dir.assert_not_called()


def test_cleanup_refuses_an_s3_namespace_whose_profile_resolves_no_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently skipping would report a cleanup that never ran."""
    monkeypatch.setattr(workspace, "create_s3_client", lambda *_args, **_kwargs: None)
    resolved = workspace.CaptionWorkspace(
        root_uri="s3://bucket/inference/namespace",
        filesystem=Mock(spec=pafs.FileSystem),
        storage_profile="captions",
    )

    with pytest.raises(ValueError, match="no S3 client for S3 namespace"):
        workspace.cleanup_workspace(resolved)


def test_resolution_carries_the_storage_profile_its_filesystem_was_built_from(
    resolve: WorkspaceFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup resolves its own client, so a dropped profile silently changes credentials.

    Asserted against a profile that is nobody's default, since the field's own
    default would satisfy the assertion without anything being propagated.
    """
    monkeypatch.setattr(workspace, "arrow_filesystem", lambda *_args, **_kwargs: pafs.LocalFileSystem())

    assert resolve(storage_profile="captions").storage_profile == "captions"


def test_a_relative_local_staging_root_is_refused(resolve: WorkspaceFactory) -> None:
    """A local namespace must mean the same physical directory in every process."""
    with pytest.raises(ValueError, match="local staging_root must be absolute"):
        resolve(staging_root="relative/staging")


def test_a_file_scheme_without_its_double_slash_is_refused_by_the_shared_uri_contract(
    resolve: WorkspaceFactory,
) -> None:
    """``file:relative/staging`` is a malformed URI, so it never reaches the local guard.

    Refused one layer down for the same reason as the ambiguous authority below:
    text claiming a scheme without ``://`` is not a path this function may read as
    relative, and ``filesystem_path`` owns that contract for every Arrow caller.
    What matters at this boundary is that the root is refused before the namespace
    is hashed, not which layer refuses it.
    """
    with pytest.raises(ValueError, match="scheme 'file' requires a file:// prefix"):
        resolve(staging_root="file:relative/staging")


def test_an_ambiguous_file_authority_is_refused_by_the_shared_uri_contract(
    resolve: WorkspaceFactory,
) -> None:
    """A ``file://host`` root is refused too, one layer down, and still before hashing.

    Which layer answers it is not this function's decision: ``filesystem_path``
    owns the URI contract for every Arrow caller, and resolution consults it
    before the namespace exists. Pinned here because what matters at this boundary
    is that the root is refused rather than silently read as a local path.
    """
    with pytest.raises(ValueError, match="file URI authority must be empty or localhost"):
        resolve(staging_root="file://relative/staging")


@pytest.mark.parametrize(
    "staging_root",
    [
        "/var/lib/caption/staging",
        "file:///var/lib/caption/staging",
        "s3://bucket-under-test/caption/staging",
    ],
    ids=["absolute_path", "absolute_file_uri", "s3_uri"],
)
def test_a_supported_absolute_staging_root_is_preserved(
    resolve: WorkspaceFactory,
    monkeypatch: pytest.MonkeyPatch,
    staging_root: str,
) -> None:
    """Supported absolute paths and remote URIs remain stable namespace roots."""
    monkeypatch.setattr(workspace, "arrow_filesystem", lambda *_args, **_kwargs: pafs.LocalFileSystem())

    assert resolve(staging_root=staging_root).root_uri.startswith(f"{staging_root}/inference/")


def test_a_blank_staging_root_is_refused(resolve: WorkspaceFactory) -> None:
    """A blank root would resolve the namespace against the worker's own cwd."""
    with pytest.raises(ValueError, match="non-empty absolute URI or path"):
        resolve(staging_root="   ")
