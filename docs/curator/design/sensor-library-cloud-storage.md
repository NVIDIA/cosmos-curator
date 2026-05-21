# Sensor Library Cloud Storage — Validation Results

Validation of `make_index_and_metadata(..., FROM_HEADER)` and
`CameraSensor.sample()` against `s3://` sources, with the caller wrapping the
cloud object as a `BinaryIO` (the sensor library does **not** accept URIs
directly).

For end-user usage (the caller-side `BinaryIO` wrapping pattern, credential
plumbing for S3 / Azure, code examples) see
[`docs/curator/guides/sensor-library-cloud-storage.md`](../guides/sensor-library-cloud-storage.md).

## Summary

**`FROM_HEADER` stays bounded over `s3://` for well-formed MP4s.** S3
reads exactly the same 67 KB the local path does (~4% of the 1.76 MB
fixture); `FULL_DEMUX` reads the whole file as expected; decoded frames
and `VideoIndex` are byte-identical local vs S3.

**Wall time on S3 is latency-bound, not bandwidth-bound.** Every PyAV
`seek()` produces a fresh `GetObject` (roughly one per seek empirically).
For our fixture, FROM_HEADER takes ~1.7 s on S3 vs ~50 ms locally;
`sample()` takes ~6.8 s vs ~0.4 s. Bytes consumed are identical to
local; the gap is 13–34 HTTP round-trips on a residential network.

**File layout (`faststart` vs `nonfaststart`) does not matter on S3.**
Ranged GETs make `moov`-at-end as cheap as `moov`-at-front, contrary to
the conventional "always lay out faststart for streaming" advice.

**One stream per `CameraSensor` lifecycle (post-`DataSource` refactor).**
The sensor library no longer accepts URIs; callers wrap their own
seekable `BinaryIO`. `CameraSensor.__init__` and `.sample()` now share a
single underlying stream — the index build and the decode loop both seek
within the same `smart_open.open(...)` connection family. Compared to
the pre-refactor two-stream lifecycle this leaves bytes-read, reads, and
S3 GetObject count unchanged (34 total for `sample` at 1 fps × 10 s) but
shaves wall time materially (~28 s → ~6.8 s on this network) because the
second smart_open setup is no longer needed.

Azure was not measured this pass. The code path is structurally identical;
the validation gap is empirical, not architectural.

## Validation Harness

`cosmos_curator/core/sensors/scripts/cloud_io_benchmark.py` exposes two
subcommands:

```bash
python -m cosmos_curator.core.sensors.scripts.cloud_io_benchmark index \
  --source <local | s3:// | az://> \
  [--reference-source <local-path>] [--skip-full-demux]

python -m cosmos_curator.core.sensors.scripts.cloud_io_benchmark sample \
  --source <local | s3:// | az://> \
  --target-fps <f> --duration-s <s> \
  [--reference-source <local-path>]
```

Per measurement it reports:

- wall time
- `bytes_read` at the file-like layer
- number of `read()` and `seek()` calls
- (S3 only) HTTP-level GetObject request count from a
  `before-send.s3.GetObject` boto3 hook
- with `--reference-source`, byte-equality of the resulting `VideoIndex`
  (and pixel-equality of decoded frames for `sample`)

`bytes_read` is the source of truth for bytes returned from cloud to
libav. Bytes-on-the-wire is not separately reported because `smart_open`
issues open-ended ranges (`Range: bytes=START-`) and closes mid-stream,
so the request `Range` header doesn't carry the response size.

## Results

Numbers below were measured on darwin 25.4.0 against
`s3://dtzeng-cosmos-test-bucket/cloud-validation/...` with the
`cosmos-curator-dev` AWS profile, fixture `test_clip_10s.mp4` (see
appendix). The S3 wall times are network-snapshot-sensitive (residential
broadband; per-`GetObject` RTT dominates) — treat them as
order-of-magnitude markers, not benchmarks. Bytes-read, reads, seek
counts, and S3 GetObject counts are stable across runs.

### Faststart MP4 (`moov` first; ~1.76 MB total)

| Source  | Method      | Wall time | bytes_read | reads | seeks | S3 requests | parity vs local |
| ------- | ----------- | --------: | ---------: | ----: | ----: | ----------: | --------------- |
| local   | FROM_HEADER |   0.051 s |     67,516 |     3 |    13 |         n/a | reference       |
| `s3://` | FROM_HEADER |   1.667 s |     67,516 |     3 |    13 |          13 | OK              |
| local   | FULL_DEMUX  |   0.004 s |  1,796,909 |    56 |    13 |         n/a | reference       |
| `s3://` | FULL_DEMUX  |   1.871 s |  1,796,909 |    56 |    13 |          13 | OK              |

### Non-faststart MP4 (`moov` after `mdat`; ~1.76 MB total)

| Source  | Method      | Wall time | bytes_read | reads | seeks | S3 requests | parity vs local |
| ------- | ----------- | --------: | ---------: | ----: | ----: | ----------: | --------------- |
| local   | FROM_HEADER |   0.006 s |     67,517 |     3 |    13 |         n/a | reference       |
| `s3://` | FROM_HEADER |   1.696 s |     67,517 |     3 |    13 |          13 | OK              |
| local   | FULL_DEMUX  |   0.003 s |  1,796,911 |    56 |    13 |         n/a | reference       |
| `s3://` | FULL_DEMUX  |   1.855 s |  1,796,911 |    56 |    13 |          13 | OK              |

### `CameraSensor.sample()` lifecycle (10 s @ 1 fps, faststart fixture)

| Source  | Wall time (idx + decode) | bytes_read | reads | seeks | S3 requests | parity vs local  |
| ------- | -----------------------: | ---------: | ----: | ----: | ----------: | ---------------- |
| local   | 0.419 s (0.020 + 0.399)  | 10,613,481 |   326 |    35 |         n/a | reference        |
| `s3://` | 6.759 s (1.605 + 5.153)  | 10,613,481 |   326 |    35 |          34 | OK (frames + ts) |

The seek count is +1 across the index measurements (and +2 across the
sample lifecycle, one per phase) compared to pre-refactor measurements
because `open_video_container` now `seek(0)`s the underlying stream
before handing it to PyAV. This is intentional — it makes
`io.BufferedIOBase` `DataSource`s usable as absolute-offset random-access
buffers regardless of the position the caller (or a previous phase)
left them at, see `cosmos_curator/core/sensors/utils/video.py`. The
extra seek is byte-free for `FROM_HEADER` over S3 (smart_open elides the
no-op seek when already at position 0; GetObject count stays at 13) but
costs one extra `GetObject` for `FULL_DEMUX` (12 → 13). For the sample
lifecycle the GetObject count is unchanged at 34 because the new
`seek(0)`s land on a stream that's already at 0 (start-of-phase) and
smart_open elides them.

### Header-fallback fixture

Not measured. The obvious "truncate to N bytes" approach corrupts the
container before the header-index branch is reached. A reliable fixture
(fragmented MP4 with `frag_keyframe+empty_moov`, or MKV without cues) is
captured as a follow-up. Behavior in code is well-defined: `FROM_HEADER`
raises `_HeaderIndexUnavailableError` and `make_index_and_metadata`
silently falls back to `FULL_DEMUX` unless `allow_header_fallback=False`.

## Limitations

1. **Azure not measured this pass.** Code path is identical to S3 (same
   `smart_open` plumbing, same seekable backend in `smart_open >= 7`);
   just no numbers yet. HTTP-level request counts for Azure would
   additionally require splicing a custom `azure.core` policy or
   monkey-patching `download_blob` (both private API); file-like-layer
   `bytes_read` would still answer the bounded-read question without that.
2. **Per-seek HTTP RTT dominates on S3.** Empirically one `GetObject`
   per `seek()` that actually moves position. For our fixture and
   network the per-request RTT × 13–34 requests dominates the wall-time
   gap vs. local (≈ 130–700 ms per request depending on link, vs ≤ ms
   for local seeks). Bounded-read property holds; latency property does
   not.
3. **`CpuVideoDecoder.open` re-parses the container on every
   `CameraSensor.sample()` call.** Each call calls
   `open_data_source(self._source)` → `av.open(stream)` and the libav
   demuxer re-reads `moov` to set up its packet iterator. For
   caller-owned `BinaryIO` sources (the post-refactor cloud path) this
   stays on the same `smart_open` stream rather than spinning up a fresh
   one, so the per-sample HTTP cost is bounded to whatever GetObjects
   PyAV's seeks happen to trigger — but the parse itself is still
   visible as the gap between `index_build_s` and `sample_s` in the
   lifecycle table.
4. **`smart_open >= 7` is required.** Older versions don't provide
   seekable cloud streams. Pinned at `7.6.1` in `pixi.lock`.
5. **PyAV `>= 17` is required.** The header-index path uses
   `stream.index_entries`, exposed only in v17+. Pinned at `==17.0.0`
   in `pixi.toml` (see comment about a 17.0.1 regression).

## Caller-Owned `BinaryIO` Lifecycle Note

After the `DataSource` refactor, the sensor library is backend-agnostic and
callers wrap cloud objects as `BinaryIO` themselves (see the
[usage guide](../guides/sensor-library-cloud-storage.md)). `CameraSensor`
reuses the **same** `BinaryIO` across the index-build and `.sample()` phases;
PyAV seeks within it for both.

Empirically, this changes the `sample` lifecycle as follows (10 s @ 1 fps,
faststart fixture):

| Aspect            | Pre-refactor (two smart_open opens) | Post-refactor (shared `BinaryIO`) |
| ----------------- | ----------------------------------- | --------------------------------- |
| `bytes_read`      | 10,613,481                          | 10,613,481                        |
| `reads`           | 326                                 | 326                               |
| `seeks`           | 33                                  | 35 (+2 from in-wrapper `seek(0)`) |
| S3 GetObjects     | 34                                  | 34                                |
| Wall time         | 28.022 s                            | 6.759 s                           |
| Frame / TS parity | OK                                  | OK                                |

Bytes-on-the-wire and the S3 request count are unchanged. The wall-time
improvement comes from no longer doing a second `smart_open.open(...)` →
TCP/TLS setup → initial `GetObject` between the index-build and decode
phases; the second phase just reuses the stream that's already open. The
two extra `seeks` are the `open_video_container` / `image_sensor._load_frame`
`seek(0)`s that position PyAV / PIL at the container origin; they're
HTTP-free because the stream is already at offset 0 when each phase starts.

## Follow-Up Tasks

- **(M) Cache the `VideoIndex` / `moov` box on `CameraSensor`** so
  repeated `sample()` calls don't re-fetch the header. Two designs:
  (a) hold an open caller-provided stream on the sensor instance;
  (b) cache `moov` bytes in memory and reconstruct a `BytesIO`-backed
  PyAV container.
- **(M) Connection pooling / stream reuse across `seek()`s.** Empirical
  pattern is one new `GetObject` per seek (~600 ms each); pooling the
  underlying TCP connection or reusing the open stream and only doing
  `seek()`+`read()` on it would amortise network setup. Likely the
  highest-leverage perf work for cloud sources.
- **(S) Verify Azure end-to-end.** Harness already supports
  `--azure-profile-name`; this is a measurement gap, not a code gap.
  Repeat the matrix against an Azure container.
- **(S) Build a reliable header-fallback fixture** (fragmented MP4 or
  MKV without cues) and measure it. Behavior is well-defined; this is
  empirical confirmation and a regression net.
- **(M) Support `gs://` (GCS).** Mirrors the s3/azure code path; would
  need a parallel `GcsClient` under `cosmos_curator.core.utils.storage`.

## How to Reproduce

Prerequisites:

- `smart_open >= 7` and PyAV `>= 17` in the Python env (the
  `cosmos-curate` micromamba env or any pixi env). Older PyAV silently
  makes every `FROM_HEADER` measurement raise.
- Working AWS credentials for the target bucket (verified with
  `aws s3 ls s3://<bucket>/`).

Build fixtures (PyAV-only, no system `ffmpeg` needed):

```bash
mkdir -p /tmp/cloud-fixtures
python - <<'PY'
import av

SRC = "tests/cosmos_curator/pipelines/video/data/test_clip_10s.mp4"

def remux(out_path: str, movflags: str | None) -> None:
    options = {"movflags": movflags} if movflags else {}
    with av.open(SRC) as src, av.open(out_path, "w", options=options) as dst:
        in_streams = list(src.streams)
        out_streams = [dst.add_stream(template=s) for s in in_streams]
        mapping = {id(i): o for i, o in zip(in_streams, out_streams, strict=True)}
        for packet in src.demux():
            if packet.dts is None:
                continue
            packet.stream = mapping[id(packet.stream)]
            dst.mux(packet)

remux("/tmp/cloud-fixtures/faststart.mp4",    "faststart")
remux("/tmp/cloud-fixtures/nonfaststart.mp4", None)
PY
aws --profile <prof> s3 cp /tmp/cloud-fixtures/ \
    s3://<bucket>/cloud-validation/ --recursive
```

Run the matrix:

```bash
S3_BASE=s3://<bucket>/cloud-validation
S3_PROF=<prof>
LOCAL=/tmp/cloud-fixtures

for fixture in faststart nonfaststart; do
  python -m cosmos_curator.core.sensors.scripts.cloud_io_benchmark index \
      --source $LOCAL/$fixture.mp4
  python -m cosmos_curator.core.sensors.scripts.cloud_io_benchmark index \
      --source $S3_BASE/$fixture.mp4 --s3-profile-name $S3_PROF \
      --reference-source $LOCAL/$fixture.mp4
done

python -m cosmos_curator.core.sensors.scripts.cloud_io_benchmark sample \
    --source $LOCAL/faststart.mp4 --target-fps 1 --duration-s 10
python -m cosmos_curator.core.sensors.scripts.cloud_io_benchmark sample \
    --source $S3_BASE/faststart.mp4 --s3-profile-name $S3_PROF \
    --target-fps 1 --duration-s 10 \
    --reference-source $LOCAL/faststart.mp4
```

For Azure, swap to `az://<container>/...` and use `--azure-profile-name`.

## Appendix: Fixtures

Both fixtures are 854x480 H.264 at ~24 fps for 9.96 s, ~1.76 MB total.

| ID             | Codec | Resolution | Duration | Layout              | Built from                                                                                      |
| -------------- | ----- | ---------- | -------- | ------------------- | ----------------------------------------------------------------------------------------------- |
| `faststart`    | h264  | 854x480    | 9.96 s   | `moov` first        | `tests/cosmos_curator/pipelines/video/data/test_clip_10s.mp4` remuxed with `movflags=faststart` |
| `nonfaststart` | h264  | 854x480    | 9.96 s   | `moov` after `mdat` | same source, no `movflags`                                                                      |
| `fallback`     | n/a   | n/a        | n/a      | header-bad          | not built; see Follow-Up                                                                        |
