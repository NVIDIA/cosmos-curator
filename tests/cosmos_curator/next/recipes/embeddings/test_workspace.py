# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace resolution: the PATH is the entire compatibility check.

``resolve_workspace`` writes nothing and claims nothing, so almost everything
worth pinning is a property of the URI it returns - two runs share computed
vectors if and only if they resolve to one directory. The tests therefore compare
resolved paths, and touch the filesystem only where a real effect exists: cleanup,
and the assertion that resolution itself leaves no trace.

The URI is a digest over five inputs, in two groups the assertions keep apart,
plus a modality segment the digest deliberately does not carry:

    source_uri, read_version, recovery_contract   what the run computes
    staged schema fingerprint, staging protocol   how a staged row reads back

The second group is why the golden namespace below is pinned as a literal. A
change to the staged fields or to the staging protocol has to isolate old
workspaces, and comparing two live resolutions cannot see it: both sides would
move together and agree.

What must NOT reach the digest has no argument to vary, so it cannot be covered
by comparing two resolutions either - hence the exact-input-set test, which is
the only form that can turn red when a tuning knob is digested.

Trailing separators, filesystem roots, and unsupported schemes are deliberately
absent: the normalization is ``arrow_utils.normalize_location_root`` and the URI
contract is ``filesystem_path``, both pinned at their own homes. Only the two
refusals ``resolve_workspace`` decides itself are asserted here.
"""

import pathlib
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pyarrow.fs as pafs
import pytest

from cosmos_curator.next.embeddings.schemas import IMAGE_COLUMN_GROUP, TEXT_COLUMN_GROUP
from cosmos_curator.next.recipes.embeddings import workspace
from cosmos_curator.next.recipes.embeddings.config import Modality

# Stand-ins for a recovery contract digest. Their VALUES are never asserted in
# the comparison tests - only whether two resolutions agree - so any two stable,
# distinct strings serve; resolution does not parse the contract. Shaped as a
# sha256 anyway, because that is what a caller passes and a 64-hex namespace
# segment is what an operator will see. ``_CONTRACT`` is also one of the golden
# namespace's inputs, so changing it moves that literal.
_CONTRACT = "a" * 64
_OTHER_CONTRACT = "b" * 64

# The exact namespace one fixed identity resolves to. Derived once from the
# inputs below and pasted here: see the module docstring for why a literal rather
# than a second live resolution. The staged-schema half of it is pinned
# independently in ``test_staged.py``, so a fingerprint change turns both red and
# names itself.
_GOLDEN_STAGING_ROOT = "/staging"
_GOLDEN_SOURCE_URI = "/tables/clips.lance"
_GOLDEN_READ_VERSION = 7
_GOLDEN_MODALITY = Modality.IMAGE
_GOLDEN_NAMESPACE = "e7d5d8c1c6b9d3fb843431f2d0dd00bb9bea2b5ae67e5505a82e3e5fde8c70a8"

# Resolves a workspace with one fixed identity, each input overridable; see
# ``resolve``.
WorkspaceFactory = Callable[..., workspace.EmbeddingWorkspace]


@pytest.fixture
def staging_root(tmp_path: pathlib.Path) -> str:
    """Return the durable root this test's namespaces resolve under."""
    return str(tmp_path / "staging")


@pytest.fixture
def resolve(staging_root: str) -> WorkspaceFactory:
    """Return a resolver over one identity, with every input overridable by keyword."""

    def build(**overrides: object) -> workspace.EmbeddingWorkspace:
        inputs: dict[str, object] = {
            "staging_root": staging_root,
            "source_uri": _GOLDEN_SOURCE_URI,
            "read_version": _GOLDEN_READ_VERSION,
            "modality": _GOLDEN_MODALITY,
            "recovery_contract": _CONTRACT,
            "group": IMAGE_COLUMN_GROUP,
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

    assert resolved.root_uri == f"{_GOLDEN_STAGING_ROOT}/image/{_GOLDEN_NAMESPACE}"


def test_the_staged_rows_and_their_checkpoint_are_siblings_inside_the_namespace(
    resolve: WorkspaceFactory,
) -> None:
    """Both must move together when the namespace moves, so both derive from one root.

    Pinned as whole URIs because a resume reads the checkpoint to decide which
    staged rows exist; a checkpoint resolved beside a DIFFERENT namespace's
    results would report rows as done that nothing staged.
    """
    resolved = resolve(staging_root=_GOLDEN_STAGING_ROOT)
    namespace = f"{_GOLDEN_STAGING_ROOT}/image/{_GOLDEN_NAMESPACE}"

    assert (resolved.results_uri, resolved.checkpoints_uri) == (
        f"{namespace}/results",
        f"{namespace}/checkpoints",
    )


def test_two_runs_with_the_same_inputs_resolve_to_one_namespace(resolve: WorkspaceFactory) -> None:
    """A resume must land in the directory the attempt it resumes staged into."""
    assert resolve().root_uri == resolve().root_uri


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_uri", "/tables/other.lance"),
        ("read_version", _GOLDEN_READ_VERSION + 1),
        ("recovery_contract", _OTHER_CONTRACT),
    ],
)
def test_a_changed_computation_input_resolves_to_a_different_namespace(
    resolve: WorkspaceFactory, field: str, value: object
) -> None:
    """Runs that cannot reuse each other's staged vectors cannot see each other's directory.

    Another table, a moved snapshot, and a different model or basis invalidate the
    staged rows for three different reasons, but they share one consequence: no
    per-row compatibility test exists downstream, so separation happens in the
    path or it does not happen.
    """
    assert resolve().root_uri != resolve(**{field: value}).root_uri


def test_the_modality_segment_separates_two_modalities_by_itself(resolve: WorkspaceFactory) -> None:
    """Text and image stage disjoint columns, so one's rows can never publish the other's.

    The segment is what guarantees it, and the reason the modality is not digested:
    it is on every resolved URI, so it holds whatever the digest contains. Pinned
    with a shared ``recovery_contract`` so the digests are as close as a caller can
    make them, leaving the segment as the only separator under test.
    """
    under_text = resolve(modality=Modality.TEXT, group=TEXT_COLUMN_GROUP)
    under_image = resolve()

    assert under_text.root_uri != under_image.root_uri
    assert pathlib.PurePosixPath(under_text.root_uri).parent.name == "text"
    assert pathlib.PurePosixPath(under_image.root_uri).parent.name == "image"


def test_two_modalities_also_differ_beneath_the_segment(resolve: WorkspaceFactory) -> None:
    """Each modality's staged schema differs, which is the second reason the digest omits it.

    Kept separate from the segment test because this is what makes the omission
    safe rather than merely sufficient: even a caller that somehow resolved one
    modality under another's segment would not read the other's rows back, since
    the staged fields the digest covers are not the same fields.
    """
    under_text = resolve(modality=Modality.TEXT, group=TEXT_COLUMN_GROUP)
    under_image = resolve()

    # Compared live against each other rather than against the golden literal:
    # the claim is that the two digests DIVERGE, and against a fixed literal both
    # sides could collapse onto one value and still differ from it.
    assert pathlib.PurePosixPath(under_text.root_uri).name != pathlib.PurePosixPath(under_image.root_uri).name


def test_a_changed_staged_schema_separates_namespaces_while_the_contract_stays_equal(
    resolve: WorkspaceFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows written under one staged schema must not be read back under another.

    Both resolutions pass the same ``recovery_contract``, so the computation is
    provably identical and no published row is stale. Staged Parquet written under
    the old field set still cannot be folded under the new one, so the namespace
    has to move anyway - the case that needs an identity input the contract digest
    does not carry.
    """
    under_current_schema = resolve()
    monkeypatch.setattr(workspace, "staged_schema_fingerprint", lambda _group: "c" * 64)

    assert resolve().root_uri != under_current_schema.root_uri


def test_a_bumped_staging_protocol_separates_namespaces_while_the_contract_stays_equal(
    resolve: WorkspaceFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A protocol bump isolates old workspaces without recomputing any vector.

    Separate from the staged-schema case because the two cover disjoint changes:
    the fingerprint moves when the persisted FIELDS change, and this moves when
    they do not - a different checkpoint-id source or partitioning leaves the
    schema byte-identical while making the rows unreadable to a resume.
    """
    under_current_protocol = resolve()
    monkeypatch.setattr(workspace, "STAGING_PROTOCOL_VERSION", workspace.STAGING_PROTOCOL_VERSION + 1)

    assert resolve().root_uri != under_current_protocol.root_uri


def test_the_namespace_is_digested_from_exactly_the_five_compatibility_inputs(
    resolve: WorkspaceFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What must not reach the digest has no argument to vary, so the input set is the assertion.

    Two classes of key are wrong here and neither can be caught by comparing
    resolutions. A tuning knob - ``parquet_rows_per_file``, the batch sizes,
    ``max_fragments`` - changes how much a crash re-computes, not what a staged
    row MEANS, so digesting one would orphan every staged row whenever an operator
    retuned it. The ``modality`` is the other: harmless but inert, because the path
    segment separates the modalities on its own, and a redundant input reads to the
    next person as a guarantee the segment does not provide.
    """
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(workspace, "canonical_digest", lambda value: captured.append(value) or "d" * 64)

    resolve()

    assert set(captured[0]) == {
        "source_uri",
        "read_version",
        "recovery_contract",
        "staging_protocol_version",
        "staged_schema_fingerprint",
    }


def test_resolving_a_workspace_creates_nothing_on_disk(resolve: WorkspaceFactory, staging_root: str) -> None:
    """A modality that turns out to owe no work must leave no directory behind."""
    resolved = resolve()

    assert not pathlib.Path(resolved.root_uri).exists()
    assert not pathlib.Path(staging_root).exists()


def test_the_namespace_nests_two_levels_below_the_staging_root(resolve: WorkspaceFactory, staging_root: str) -> None:
    """Cleanup deletes the resolved root whole, so that root must never BE a shared parent.

    The staging root holds every modality's namespaces and the table's siblings
    hold the table itself; deleting either would take another run's staged rows,
    or the clips table, with it. Nothing guards against that at delete time - the
    two path segments appended here are the guard, so their presence is the
    assertion.
    """
    resolved = pathlib.PurePosixPath(resolve().root_uri)

    assert resolved.parent.parent == pathlib.PurePosixPath(staging_root)


def test_the_default_staging_root_is_a_sibling_of_the_table_never_inside_it() -> None:
    """A staging directory inside the Lance dataset is one a Lance reader must ignore."""
    root = workspace.default_staging_root("s3://bucket/robot/clips.lance")

    assert root == "s3://bucket/robot/clips.lance__embedding_staging"


def test_a_trailing_separator_on_the_table_uri_still_yields_a_sibling_root() -> None:
    """The separator is where nesting would happen, so the case is pinned on its own.

    Without the strip, ``clips.lance/`` would place the staging root INSIDE the
    dataset directory rather than beside it.
    """
    assert workspace.default_staging_root("s3://bucket/robot/clips.lance/") == (
        "s3://bucket/robot/clips.lance__embedding_staging"
    )


def test_cleanup_removes_the_namespace_whole(resolve: WorkspaceFactory) -> None:
    """The staged rows and the checkpoint indexing them are siblings, so both go.

    Removing one and keeping the other would let a later run resume against half a
    namespace: a checkpoint claiming rows are done, with no staged row left to
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


def test_cleanup_leaves_a_sibling_modalitys_namespace_untouched(resolve: WorkspaceFactory) -> None:
    """Modalities are published independently, so one's cleanup must not orphan another's.

    The closest namespace cleanup could reach by mistake, and the one a wider
    delete would actually hit: both live under the same staging root.
    """
    image = resolve()
    text = resolve(modality=Modality.TEXT, group=TEXT_COLUMN_GROUP)
    for resolved in (image, text):
        pathlib.Path(resolved.results_uri).mkdir(parents=True)

    workspace.cleanup_workspace(image)

    assert pathlib.Path(text.results_uri).exists()


def test_cleanup_uses_the_workspace_filesystem_and_its_own_path() -> None:
    """Cleanup must not re-resolve credentials independently of the workspace."""
    filesystem = Mock(spec=pafs.FileSystem)
    resolved = workspace.EmbeddingWorkspace(
        root_uri="file:///var/lib/embeddings/namespace",
        filesystem=filesystem,
    )

    workspace.cleanup_workspace(resolved)

    filesystem.delete_dir.assert_called_once_with("/var/lib/embeddings/namespace")


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
    resolved = workspace.EmbeddingWorkspace(
        root_uri="s3://bucket/image/namespace",
        filesystem=filesystem,
        storage_profile="embeddings",
    )

    workspace.cleanup_workspace(resolved)

    assert resolved_with == [("s3://bucket/image/namespace", "embeddings", True)]
    assert [call.args[0].path for call in client.delete_prefix.call_args_list] == ["s3://bucket/image/namespace"]
    filesystem.delete_dir.assert_not_called()


def test_cleanup_refuses_an_s3_namespace_whose_profile_resolves_no_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently skipping would report a cleanup that never ran."""
    monkeypatch.setattr(workspace, "create_s3_client", lambda *_args, **_kwargs: None)
    resolved = workspace.EmbeddingWorkspace(
        root_uri="s3://bucket/image/namespace",
        filesystem=Mock(spec=pafs.FileSystem),
        storage_profile="embeddings",
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

    assert resolve(storage_profile="embeddings").storage_profile == "embeddings"


def test_a_relative_local_staging_root_is_refused(resolve: WorkspaceFactory) -> None:
    """A local namespace must mean the same physical directory in every process."""
    with pytest.raises(ValueError, match="local staging_root must be absolute"):
        resolve(staging_root="relative/staging")


def test_a_blank_staging_root_is_refused(resolve: WorkspaceFactory) -> None:
    """A blank root would resolve the namespace against the worker's own cwd."""
    with pytest.raises(ValueError, match="non-empty absolute URI or path"):
        resolve(staging_root="   ")
