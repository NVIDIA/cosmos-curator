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

"""Action-modality CPU tests: the key-free descriptor extract / project contract.

Both phases are pure compute over positions: they read no identity column and
are cardinality- and order-preserving, so the fill worker can attach ``clip_id``
positionally. A clip that is missing / unreadable / undecodable / geometrically
rejected keeps its own row with a NULL descriptor (which projects to an all-NULL,
pending action group) rather than being dropped.

The action leg is Mecka-only: applicability is purely a non-empty
``action_data_uri`` (no ``source_dataset``, no dexterous registry), and every
artifact is read as a self-describing ACT2 ``.bin`` and aligned as mecka. A
non-mecka payload lacks the hand arrays and rejects geometrically to a NULL
descriptor rather than crashing the leg. The tests assert that no-drop contract,
the per-batch single-read memo, the fail-closed read/decode path, the
round-trip alignment of ``decode_descriptors``, and that the per-batch rejection
summary names a bounded, redacted sample of the rejected artifacts - never PCA
numerics, which are the fit's concern.
"""

import pathlib
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import ray

from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.next.embeddings.action import embedder as embedder_mod
from cosmos_curator.next.embeddings.action.embedder import (
    DualWristMotionDescriptorExtractor,
    DualWristMotionProjector,
    DualWristMotionReadConfig,
    _ActionPayloadReader,
    _RejectionLedger,
    decode_descriptors,
)
from cosmos_curator.next.embeddings.action.pca import PcaArtifact
from cosmos_curator.next.embeddings.action.wrist_motion import (
    DESCRIPTOR_DIM,
    DESCRIPTOR_VERSION,
    DescriptorRejection,
)
from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    ACTION_DIM,
    DESCRIPTOR_ROW,
    descriptor_batch,
)
from cosmos_curator.next.media.action_binary import encode_action_bin

# Ceiling on how long a stalled read waits for its successor. Only ever reached
# when the reads did NOT overlap, so it bounds a deadlock into a named failure
# rather than hanging the suite; a healthy run never waits measurably.
_REVERSAL_TIMEOUT_S = 30.0


def _source_batch(uris: Sequence[str | None]) -> pa.Table:
    """Build the source projection the action fill worker hands to ``extract``.

    Carries exactly the action leg's ``SOURCE_COLUMNS`` (``clip_id`` /
    ``action_data_uri``). ``clip_id`` is present because the worker's scan
    projects it to key its own write; ``extract`` itself must ignore it. The leg
    is Mecka-only, so there is no ``source_dataset`` column. ``action_data_uri``
    is ``large_string`` to match the clips table's ``OUTCOME_SCHEMA`` type.
    """
    rows = len(uris)
    return pa.table(
        {
            "clip_id": pa.array([f"clip{i}" for i in range(rows)], pa.string()),
            "action_data_uri": pa.array(list(uris), pa.large_string()),
        }
    )


def _only_rejection_summary(records: list[dict[str, Any]]) -> str:
    """Return the single per-batch rejection summary warning, failing if there is not exactly one.

    One summary per batch is itself part of the contract, so collapsing to one
    string here keeps every caller from re-asserting it.
    """
    summaries = [
        record["message"]
        for record in records
        if record["level"].name == "WARNING" and "descriptor_rejections" in record["message"]
    ]
    assert len(summaries) == 1
    return summaries[0]


def test_extractor_emits_one_valid_descriptor_row_per_clip(
    extractor: DualWristMotionDescriptorExtractor,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """Two readable clips yield the ``DESCRIPTOR_ROW`` schema, both rows valid, no key column."""
    uris = [make_mecka_bin(tmp_path / f"a{i}.bin", seed=i) for i in range(2)]

    out = extractor(_source_batch(uris))

    assert out.schema.names == DESCRIPTOR_ROW.names
    assert out.column("descriptor").is_valid().to_pylist() == [True, True]
    assert len(out.column("descriptor")[0].as_py()) == DESCRIPTOR_DIM


def test_extractor_keeps_missing_uri_row_as_null_descriptor(
    extractor: DualWristMotionDescriptorExtractor,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """An empty ``action_data_uri`` yields a NULL descriptor row, not a dropped row (cardinality preserved)."""
    good = make_mecka_bin(tmp_path / "a.bin", seed=1)

    out = extractor(_source_batch([good, ""]))

    assert out.num_rows == 2
    assert out.column("descriptor").is_valid().to_pylist() == [True, False]


def test_extractor_keeps_a_null_uri_row_distinct_from_an_empty_one(
    extractor: DualWristMotionDescriptorExtractor,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """A NULL ``action_data_uri`` survives to the emitted batch instead of collapsing to ``""``.

    Both mean "no artifact" for the extract itself, but the emitted URI is the
    identity the PCA sampling and de-dup key on, so the two must stay
    distinguishable downstream rather than being folded into one value.
    """
    good = make_mecka_bin(tmp_path / "a.bin", seed=1)

    out = extractor(_source_batch([good, None, ""]))

    assert out.column("descriptor").is_valid().to_pylist() == [True, False, False]
    assert out.column("action_data_uri").to_pylist() == [good, None, ""]


def test_extractor_emits_an_empty_batch_without_dispatching_any_reader() -> None:
    """A batch with no rows yields no rows, without the read pool being sized to zero workers.

    The worker count is ``min(read_concurrency, len(uris))``, which is zero for an
    empty batch, and zero is a ``max_workers`` value ``ThreadPoolExecutor`` rejects.
    The empty case therefore has to return before the pool is built.
    """
    extractor = DualWristMotionDescriptorExtractor(DualWristMotionReadConfig(read_concurrency=4))

    out = extractor(_source_batch([]))

    assert out.num_rows == 0
    assert out.schema.names == DESCRIPTOR_ROW.names


def test_extractor_keeps_unreadable_file_row_as_null_descriptor(
    extractor: DualWristMotionDescriptorExtractor,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """A missing artifact file yields a NULL descriptor row rather than raising or dropping."""
    good = make_mecka_bin(tmp_path / "a.bin", seed=1)

    out = extractor(_source_batch([good, str(tmp_path / "nope.bin")]))

    assert out.num_rows == 2
    assert out.column("descriptor").is_valid().to_pylist() == [True, False]


def test_extractor_keeps_geometrically_rejected_payload_as_null_descriptor(
    extractor: DualWristMotionDescriptorExtractor,
    non_dexterous_dataset: str,
    tmp_path: pathlib.Path,
) -> None:
    """An artifact with no hand arrays is geometrically rejected to a NULL descriptor (row kept).

    A non-mecka export decodes fine but lacks the hand arrays mecka alignment
    needs, so it rejects per-row rather than crashing the Mecka-only leg.
    """
    frames = 16
    payload = {
        "action": np.zeros((frames, 7), dtype=np.float32),
        "state": np.zeros((frames, 8), dtype=np.float32),
    }
    path = tmp_path / "plain.bin"
    path.write_bytes(encode_action_bin(payload, non_dexterous_dataset))

    out = extractor(_source_batch([str(path)]))

    assert out.num_rows == 1
    assert out.column("descriptor").is_valid().to_pylist() == [False]


def test_rejection_summary_names_the_rejected_artifact(
    extractor: DualWristMotionDescriptorExtractor,
    make_non_dexterous_bin: Callable[..., str],
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The per-batch rejection summary names the offending artifact, not only a count.

    A count on its own says a row is pending but gives no way to reach the
    artifact behind it, so it cannot be inspected, re-derived, or excluded.
    """
    uri = make_non_dexterous_bin(tmp_path / "no-hand-arrays.bin")

    extractor(_source_batch([uri]))

    assert uri in _only_rejection_summary(loguru_records)


def test_rejection_summary_counts_a_shared_artifact_once_across_views(
    extractor: DualWristMotionDescriptorExtractor,
    make_non_dexterous_bin: Callable[..., str],
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """Two view rows sharing one rejected artifact are summarized as a single artifact.

    The per-batch memo derives a shared artifact once, so the count is a count of
    artifacts; both rows are still left pending. That is why the summary reports
    artifacts rather than rows.
    """
    uri = make_non_dexterous_bin(tmp_path / "shared-span.bin")

    out = extractor(_source_batch([uri, uri]))

    assert out.column("descriptor").is_valid().to_pylist() == [False, False]
    assert "'count': 1" in _only_rejection_summary(loguru_records)


def test_rejection_ledger_escapes_a_control_character_in_an_artifact_uri(
    loguru_records: list[dict[str, Any]],
) -> None:
    """A newline inside a rejected artifact's URI is escaped in the ledger, so the summary stays one record.

    Asserted on the RETAINED example rather than on the emitted line: the summary
    renders through ``str(dict)``, whose ``repr`` of each value would escape the
    newline even if the ledger had stored it raw, so a log-line assertion would
    pass with the redaction gone. Artifact URIs come from the clips table, so no
    single row may forge a second log line in whatever aggregator collects it.
    """
    ledger = _RejectionLedger()
    ledger.record(DescriptorRejection.MISSING_ARM, "/data/action/a\nWARNING forged line.bin")

    DualWristMotionDescriptorExtractor._log_rejections(ledger)

    (example,) = ledger.examples[DescriptorRejection.MISSING_ARM]
    assert example == r"/data/action/a\u000aWARNING forged line.bin"
    assert "\n" not in _only_rejection_summary(loguru_records)


def test_rejection_ledger_redacts_the_recorded_uri() -> None:
    """The ledger routes an example URI through redaction before retaining it.

    Redacting on the way in is what keeps an embedded credential out of every
    consumer of the ledger, not just today's log line. Which secrets redaction
    strips is the redaction's own contract (``test_uri_redaction``); what this pins
    is that the ledger goes through it at all.
    """
    ledger = _RejectionLedger()

    ledger.record(DescriptorRejection.MISSING_ARM, "https://key:s3cret@bucket.example/action/a.bin")

    (example,) = ledger.examples[DescriptorRejection.MISSING_ARM]
    assert example == "https://bucket.example/action/a.bin"


def test_rejection_ledger_caps_examples_while_the_count_stays_exact() -> None:
    """Past the per-reason example cap the count keeps rising but no more URIs are retained.

    That split is what bounds the summary on a batch where every artifact rejects,
    without understating how many were rejected.
    """
    ledger = _RejectionLedger()
    recorded = _RejectionLedger.MAX_EXAMPLES_PER_REASON + 2

    for index in range(recorded):
        ledger.record(DescriptorRejection.MISSING_ARM, f"/data/action/{index}.bin")

    reported = ledger.summary()[DescriptorRejection.MISSING_ARM.value]
    assert reported["count"] == recorded
    assert len(reported["examples"]) == _RejectionLedger.MAX_EXAMPLES_PER_REASON


def test_rejection_ledger_keeps_each_reason_separate() -> None:
    """Every reason carries its own count and its own example URIs.

    Keying the summary by reason is only useful if an operator can attribute each
    artifact to the check that rejected it.
    """
    ledger = _RejectionLedger()

    ledger.record(DescriptorRejection.MISSING_ARM, "/data/action/arm.bin")
    ledger.record(DescriptorRejection.LENGTH_MISMATCH, "/data/action/length.bin")

    summary = ledger.summary()
    assert summary[DescriptorRejection.MISSING_ARM.value] == {"count": 1, "examples": ["/data/action/arm.bin"]}
    assert summary[DescriptorRejection.LENGTH_MISMATCH.value] == {"count": 1, "examples": ["/data/action/length.bin"]}


def test_extractor_keeps_undecodable_bytes_as_null_descriptor(
    extractor: DualWristMotionDescriptorExtractor,
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """Non-ACT2 bytes (e.g. a legacy pickle) drop to a NULL descriptor via the read/decode failure path.

    There is no format field or registry to gate on: the fail-closed behavior is
    that ``decode_action_artifact`` rejects the foreign bytes, the reader logs a
    read failure, and the row is kept as NULL.
    """
    path = tmp_path / "payload.pickle"
    path.write_bytes(b"not-an-act2-payload")

    out = extractor(_source_batch([str(path)]))

    assert out.num_rows == 1
    assert out.column("descriptor").is_valid().to_pylist() == [False]
    warnings = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    assert any("read failed" in message for message in warnings)


def test_reader_returns_none_on_undecodable_bytes(
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The reader fail-closes on non-ACT2 bytes: it returns ``None`` and logs a read failure.

    The reader is the last-resort fail-closed layer; handed foreign bytes directly
    it returns ``None`` (one dropped row) rather than propagating the decode error
    and failing the whole leg.
    """
    path = tmp_path / "payload.pickle"
    path.write_bytes(b"not-an-act2-payload")
    reader = _ActionPayloadReader(DualWristMotionReadConfig())

    assert reader.read(str(path)) is None
    warnings = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    assert any("read failed" in message for message in warnings)


@pytest.mark.parametrize(("separator", "escaped"), [("\n", r"\u000a"), ("\r", r"\u000d")])
def test_read_failure_warning_escapes_a_record_separator_in_the_uri(
    separator: str,
    escaped: str,
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A record separator inside an unreadable artifact's URI is escaped in the warning, not dropped.

    This warning interpolates the URI directly, and artifact URIs come from the
    clips table, so no single row may forge a second log line in whatever
    aggregator collects the warning. The escape is asserted PRESENT because a
    redaction that deleted the separator instead would also leave it absent - and
    would name an artifact that exists in no store.
    """
    reader = _ActionPayloadReader(DualWristMotionReadConfig())
    forged = str(tmp_path / f"a{separator}WARNING forged line.bin")

    assert reader.read(forged) is None

    (failure,) = [record["message"] for record in loguru_records if "read failed" in record["message"]]
    assert f"a{escaped}WARNING forged line.bin" in failure


def test_read_failure_warning_omits_a_presigned_signature_quoted_by_the_exception(
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A signature in the artifact URI stays out of the warning the failed read emits.

    Redacting the warning's own URI field is not sufficient: ``OSError`` renders
    the filename it failed on, so the raw URI arrives a second time inside the
    exception text and publishes the signature to whatever collects the record.
    """
    reader = _ActionPayloadReader(DualWristMotionReadConfig())
    signed = f"{tmp_path / 'missing.bin'}?X-Amz-Signature=deadbeef"

    assert reader.read(signed) is None

    (failure,) = [record["message"] for record in loguru_records if "read failed" in record["message"]]
    assert "deadbeef" not in failure
    assert "missing.bin" in failure


def test_extractor_multiview_fanout_shares_one_descriptor(
    extractor: DualWristMotionDescriptorExtractor,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """Two views sharing one artifact get identical descriptors on both of their rows."""
    shared = make_mecka_bin(tmp_path / "span.bin", seed=1)

    out = extractor(_source_batch([shared, shared]))

    assert out.column("descriptor").is_valid().to_pylist() == [True, True]
    left = np.asarray(out.column("descriptor")[0].as_py())
    right = np.asarray(out.column("descriptor")[1].as_py())
    np.testing.assert_array_equal(left, right)


def test_extractor_reads_each_shared_artifact_once(
    extractor: DualWristMotionDescriptorExtractor,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two views sharing one artifact URI trigger exactly one storage read (per-batch memo)."""
    read_urls: list[str] = []
    real_read_bytes = embedder_mod.read_bytes

    def counting_read_bytes(url: str, client: StorageClient | None) -> bytes:
        read_urls.append(url)
        return real_read_bytes(url, client)

    monkeypatch.setattr(embedder_mod, "read_bytes", counting_read_bytes)
    shared = make_mecka_bin(tmp_path / "span.bin", seed=1)

    out = extractor(_source_batch([shared, shared]))

    assert out.num_rows == 2
    assert read_urls == [shared]


def test_extractor_systemic_client_misconfig_fails_loud(
    extractor: DualWristMotionDescriptorExtractor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage profile that resolves for no row raises, rather than nulling every descriptor.

    Client resolution is hoisted out of the per-row drop handler: a systemic
    misconfiguration must fail the modality loudly, not be logged-and-dropped for
    every artifact until the whole action group is silently NULL.
    """

    def boom(_url: str, **_kwargs: object) -> None:
        msg = "unresolvable storage profile"
        raise ValueError(msg)

    monkeypatch.setattr(embedder_mod, "get_storage_client", boom)

    with pytest.raises(ValueError, match="unresolvable storage profile"):
        extractor(_source_batch(["s3://bucket/a.bin"]))


def test_reader_builds_one_client_per_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader memoizes one storage client per (scheme, bucket) backend.

    A clips table can mix backends across runs, so the reader must reuse a client
    per backend rather than rebuild it per row or latch onto the first URL's
    backend for every later read.
    """
    calls: list[str] = []

    def fake_get_storage_client(url: str, **_kwargs: object) -> object:
        calls.append(url)
        return object()

    monkeypatch.setattr(embedder_mod, "get_storage_client", fake_get_storage_client)
    reader = _ActionPayloadReader(DualWristMotionReadConfig())

    first = reader._ensure_client("s3://bucket-a/x.bin")
    again = reader._ensure_client("s3://bucket-a/y.bin")
    other = reader._ensure_client("s3://bucket-b/z.bin")

    assert first is again
    assert other is not first
    assert calls == ["s3://bucket-a/x.bin", "s3://bucket-b/z.bin"]


def test_projector_maps_valid_descriptors_to_the_action_group(
    extractor: DualWristMotionDescriptorExtractor,
    synthetic_pca: PcaArtifact,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """The projector maps each valid descriptor to one action group row with provenance."""
    uris = [make_mecka_bin(tmp_path / f"a{i}.bin", seed=i) for i in range(2)]
    descriptors = extractor(_source_batch(uris))

    out = DualWristMotionProjector(synthetic_pca).project(descriptors)

    assert out.schema.names == list(ACTION_COLUMN_GROUP.field_names)
    assert out.num_rows == 2
    assert len(out.column("embedding_action")[0].as_py()) == ACTION_DIM
    assert out.column("embedding_action_descriptor_version").to_pylist() == [DESCRIPTOR_VERSION] * 2
    assert out.column("embedding_action_pca_fingerprint").to_pylist() == [synthetic_pca.fingerprint] * 2


def test_projector_preserves_null_descriptor_as_a_null_action_group(
    extractor: DualWristMotionDescriptorExtractor,
    synthetic_pca: PcaArtifact,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """A NULL descriptor projects to an all-NULL (vector + provenance) action group, one row per input."""
    good = make_mecka_bin(tmp_path / "a.bin", seed=1)
    descriptors = extractor(_source_batch([good, ""]))

    out = DualWristMotionProjector(synthetic_pca).project(descriptors)

    assert out.num_rows == 2
    assert out.column("embedding_action").is_valid().to_pylist() == [True, False]
    assert out.column("embedding_action_descriptor_version").to_pylist() == [DESCRIPTOR_VERSION, None]
    assert out.column("embedding_action_pca_fingerprint").to_pylist() == [synthetic_pca.fingerprint, None]


def test_projector_handles_an_empty_descriptor_batch(
    extractor: DualWristMotionDescriptorExtractor,
    synthetic_pca: PcaArtifact,
) -> None:
    """An empty selection projects to zero rows with the action group schema, not an error."""
    empty = extractor(_source_batch([]))

    out = DualWristMotionProjector(synthetic_pca).project(empty)

    assert out.num_rows == 0
    assert out.schema.names == list(ACTION_COLUMN_GROUP.field_names)


def test_decode_descriptors_aligns_present_rows_to_the_valid_mask() -> None:
    """``decode_descriptors`` returns the per-row mask and a compact matrix of only the valid rows in order.

    A NULL row in the middle must not shift the surviving descriptors: the present
    matrix holds rows 0 and 2, never a misaligned row 1 (the round-trip-safe path
    that a Ray materialize's NULL-list compaction would otherwise break).
    """
    matrix = np.stack([np.full(DESCRIPTOR_DIM, float(i), dtype=np.float32) for i in range(3)])
    valid = np.array([True, False, True])
    table = descriptor_batch(["a", "b", "c"], matrix, DESCRIPTOR_DIM, valid)

    mask, present = decode_descriptors(table)

    assert mask.tolist() == [True, False, True]
    assert present.shape == (2, DESCRIPTOR_DIM)
    np.testing.assert_array_equal(present[0], matrix[0])
    np.testing.assert_array_equal(present[1], matrix[2])


def _concurrent_extractor(read_concurrency: int) -> DualWristMotionDescriptorExtractor:
    """Build an extractor that overlaps its artifact reads at the given width."""
    return DualWristMotionDescriptorExtractor(DualWristMotionReadConfig(read_concurrency=read_concurrency))


def _stall_reads_in_reverse(
    monkeypatch: pytest.MonkeyPatch, order: Sequence[str]
) -> Callable[[str, StorageClient | None], bytes]:
    """Make reads finish in reverse of ``order``, and record the order they finished in.

    Each URI waits for its successor to finish first, so the last submitted read
    completes first and the first completes last. That is a real inversion rather
    than a hint: with the reads serialized the barrier would deadlock, so a test
    using this cannot silently pass by never overlapping at all.
    """
    finished: dict[str, threading.Event] = {uri: threading.Event() for uri in order}
    real_read_bytes = embedder_mod.read_bytes

    def stalled_read_bytes(url: str, client: StorageClient | None) -> bytes:
        data = real_read_bytes(url, client)
        successor = order.index(url) + 1
        if successor < len(order):
            assert finished[order[successor]].wait(timeout=_REVERSAL_TIMEOUT_S), (
                f"read of {order[successor]} never completed; the reads did not overlap"
            )
        finished[url].set()
        return data

    monkeypatch.setattr(embedder_mod, "read_bytes", stalled_read_bytes)
    return stalled_read_bytes


def test_concurrent_extract_preserves_row_order_when_reads_complete_in_reverse(
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Descriptors land on their own input rows even when the reads finish in reverse order.

    The leg carries no identity column, so the caller attaches ``clip_id``
    positionally: a descriptor scattered by completion order rather than by input
    index would mislabel every row with no exception, no schema violation and no
    change in row count. Compared against the serial result so the assertion is the
    ordering itself, not a re-derivation of the geometry.
    """
    uris = [make_mecka_bin(tmp_path / f"a{i}.bin", seed=i) for i in range(4)]
    expected = _concurrent_extractor(1)(_source_batch(uris)).column("descriptor").to_pylist()
    _stall_reads_in_reverse(monkeypatch, uris)

    out = _concurrent_extractor(len(uris))(_source_batch(uris))

    assert out.num_rows == len(uris)
    assert out.column("descriptor").to_pylist() == expected


def test_concurrent_extract_keeps_a_mid_batch_unreadable_uri_as_null_on_its_own_row(
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """One unreadable artifact nulls its own row and no other, without failing the batch.

    A per-row read failure is a data problem, not a leg failure, and widening the
    reads must not let it displace a neighbour's descriptor.
    """
    uris = [
        make_mecka_bin(tmp_path / "a.bin", seed=0),
        str(tmp_path / "absent.bin"),
        make_mecka_bin(tmp_path / "c.bin", seed=2),
    ]

    out = _concurrent_extractor(len(uris))(_source_batch(uris))

    assert out.num_rows == len(uris)
    assert out.column("descriptor").is_valid().to_pylist() == [True, False, True]


def test_concurrent_extract_reads_each_shared_artifact_once(
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Views sharing one artifact still trigger exactly one read at a width above one.

    The distinct-URI set is resolved before anything is submitted, so widening the
    reads cannot multiply the reads this change exists to reduce.
    """
    read_urls: list[str] = []
    real_read_bytes = embedder_mod.read_bytes

    def counting_read_bytes(url: str, client: StorageClient | None) -> bytes:
        read_urls.append(url)
        return real_read_bytes(url, client)

    monkeypatch.setattr(embedder_mod, "read_bytes", counting_read_bytes)
    shared = make_mecka_bin(tmp_path / "span.bin", seed=1)

    out = _concurrent_extractor(4)(_source_batch([shared] * 4))

    assert out.num_rows == 4
    assert read_urls == [shared]


def test_concurrent_extract_records_rejections_in_first_seen_order(
    make_non_dexterous_bin: Callable[..., str],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """Rejected artifacts are summarized in input order even when derived out of order.

    The ledger documents its examples as first-seen, and it is written on the
    calling thread walking the distinct URIs precisely so arrival order cannot
    reorder them. Reads are forced to complete in reverse, so a ledger written from
    the workers would report the reversed order here.
    """
    uris = [make_non_dexterous_bin(tmp_path / f"r{index}.bin") for index in range(3)]
    _stall_reads_in_reverse(monkeypatch, uris)

    out = _concurrent_extractor(len(uris))(_source_batch(uris))

    assert out.column("descriptor").is_valid().to_pylist() == [False] * len(uris)
    summary = _only_rejection_summary(loguru_records)
    assert f"'count': {len(uris)}" in summary
    assert [summary.index(uri) for uri in uris] == sorted(summary.index(uri) for uri in uris)


def test_concurrent_extract_fails_loud_on_a_systemic_client_misconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage profile that resolves for no row raises even at a width above one.

    Clients are warmed on the calling thread before any worker is submitted, so a
    misconfiguration that is wrong for every row stays one loud failure instead of
    degrading into N per-artifact drops that silently empty the group.
    """

    def boom(_url: str, **_kwargs: object) -> None:
        msg = "unresolvable storage profile"
        raise ValueError(msg)

    monkeypatch.setattr(embedder_mod, "get_storage_client", boom)

    with pytest.raises(ValueError, match="unresolvable storage profile"):
        _concurrent_extractor(4)(_source_batch([f"s3://bucket/{index}.bin" for index in range(4)]))


def test_concurrent_extract_requests_cancellation_of_the_reads_queued_behind_a_failing_worker(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker's exception tears the pool down asking for the queued reads to be dropped.

    Asserts that cancellation is requested, not that a particular read was skipped:
    whether any single queued read starts is a race between a worker picking it up
    and the calling thread reaching the shutdown, which cannot be ordered from
    outside ``concurrent.futures``. Requesting it is the whole mechanism -- the
    context manager passes ``cancel_futures=False``, under which every queued read
    runs and can spend its own full retry budget before the failure surfaces.
    """
    shutdowns: list[bool] = []

    class _RecordingPool(ThreadPoolExecutor):
        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:  # noqa: FBT001, FBT002
            shutdowns.append(cancel_futures)
            super().shutdown(wait, cancel_futures=cancel_futures)

    def exploding_read_bytes(_url: str, _client: StorageClient | None) -> bytes:
        msg = "endpoint unreachable"
        raise RuntimeError(msg)

    monkeypatch.setattr(embedder_mod, "ThreadPoolExecutor", _RecordingPool)
    monkeypatch.setattr(embedder_mod, "read_bytes", exploding_read_bytes)
    # Bare paths, never opened: the read is replaced wholesale, so real artifacts
    # would only add setup that the test cannot exercise.
    uris = [str(tmp_path / f"a{index}.bin") for index in range(4)]

    with pytest.raises(RuntimeError, match="endpoint unreachable"):
        _concurrent_extractor(2)(_source_batch(uris))

    assert shutdowns == [True]


def test_read_config_rejects_a_read_width_below_one() -> None:
    """A non-positive read width is refused at construction, not at pool creation.

    Left to the executor it would surface as a bare ``max_workers must be greater
    than 0`` from deep inside a worker, naming nothing the caller configured.
    """
    with pytest.raises(ValueError, match="read_concurrency"):
        DualWristMotionReadConfig(read_concurrency=0)


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_action_extractor_is_independent_across_ray_blocks(
    extractor: DualWristMotionDescriptorExtractor,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
) -> None:
    """The extractor runs correctly across multiple Ray Data blocks (per-batch state is independent).

    The bins are written in the driver, and each worker decodes the
    self-describing ACT2 ``.bin`` with no registry lookup, so a monkeypatched
    dataset name resolves identically driver-side and worker-side. Cardinality is
    preserved AND every row embeds validly, so the extractor's per-row result -
    not just the row count - is independent of how Ray partitions the blocks. Only
    per-row validity is asserted (not partition count), since Ray Data block
    granularity is not a function of ``repartition``'s argument.
    """
    uris = [make_mecka_bin(tmp_path / f"c{i}.bin", seed=i) for i in range(4)]
    table = _source_batch(uris)

    mapped = ray.data.from_arrow(table).repartition(2).map_batches(extractor, batch_format="pyarrow")

    rows = mapped.take_all()
    assert len(rows) == 4
    assert all(row["descriptor"] is not None for row in rows)
