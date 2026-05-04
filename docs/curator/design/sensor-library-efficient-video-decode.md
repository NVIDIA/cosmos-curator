# Efficient Sparse Video Decode

## Overview

This document describes the design for decoding video at a sample rate lower than the
source frame rate — for example, extracting frames at 2 Hz from a 30 Hz video — with
minimal I/O, minimal decode work, and bounded memory usage.

The key insight is that `VideoIndex` already contains every frame's presentation
timestamp and keyframe flag, read from the container index (moov/cues atom) without
demuxing.  This lets us pre-compute a complete seek plan as a pure numpy operation
before opening the decoder at all.

**Primary location**: `cosmos_curator/core/sensors/utils/video.py`

## Problem Statement

A naive decode-and-sample approach decodes every frame in the video and discards the
ones that don't fall on target timestamps.  For a 60-second 30 Hz video decoded at
2 Hz:

- Naive: ~1800 frames decoded, ~1800 `to_ndarray()` calls, full sequential read
- Sparse: ~120 seeks, ~120 decoded frames (plus inter-GOP frames needed to reach each
  target), 120 `to_ndarray()` calls, non-sequential but minimal reads

The speedup grows as the ratio of source fps to target fps increases.

### Video Codec Background

Compressed video is organised into **Groups of Pictures (GOPs)**.  Each GOP begins
with an I-frame (keyframe) — a fully self-contained image — followed by P- and B-frames
that are encoded as deltas relative to other frames.  You cannot decode a P- or B-frame
without first decoding its reference frames back to the nearest preceding keyframe.

This means the minimum unit of random access is one GOP: seek to a keyframe, then
decode forward until you reach the target frame.

## MP4 Container Structure and Header Extraction

The MP4 format (ISO Base Media File Format, ISO 14496-12) stores metadata and media
data in a nested hierarchy of typed **boxes** (also called **atoms**).  Each box begins
with a 4-byte size followed by a 4-byte type code, so the entire tree can be navigated
without reading any payload.  The key insight for fast index extraction is that **all
structural metadata is concentrated in a single `moov` box** — the decoder never needs
to touch the encoded bitstream (`mdat`) to build `VideoIndex`.

### Box Hierarchy

```text
ftyp   ← file type and compatibility brands (always first)
moov   ← movie container — ALL metadata lives here
  mvhd ← movie header: total duration, time scale, creation time
  trak ← one per stream (video, audio, subtitles, …)
    tkhd ← track header: duration, pixel dimensions
    mdia
      mdhd ← track time scale, language
      hdlr ← handler type ('vide', 'soun', …)
      minf
        stbl ← Sample Table Box  ←  the index
          stsd ← sample description: codec, width, height, …
          stts ← Time-to-Sample: run-length encoded DTS deltas
          ctts ← Composition Time Offset: DTS→PTS per sample (B-frames only)
          stss ← Sync Sample: sorted list of keyframe sample indices
          stsz ← Sample Size: byte size of every packet
          stsc ← Sample-to-Chunk: maps packets to chunk groups
          stco ← Chunk Offset: byte offset of each chunk in the file
          co64 ← 64-bit Chunk Offset (used when offsets exceed 4 GB)
mdat   ← media data: the raw encoded bitstream (bulk of the file)
```

### The Sample Table Box (`stbl`)

`stbl` is the video index.  Its sub-boxes together let libavformat reconstruct full
per-packet metadata without reading any encoded data:

**`stts` (Time-to-Sample, mandatory)**
Run-length encodes the duration of each sample in track time_base units.  The DTS of
every sample is derived by prefix-summing these deltas.  For constant-frame-rate video
this is usually a single entry: `(count=N, delta=d)`, so the entire DTS timeline for
an hour-long video fits in 8 bytes.

**`ctts` (Composition Time Offset, optional)**
Present only when B-frames exist.  For each sample, `pts = dts + offset`.  Absent for
I/P-only streams (H.264 with `max_b_frames=0`, most HEVC-encoded dashcam footage).
When present, libav adds this offset to the DTS to produce the PTS stored in
`VideoIndex.pts_stream`.

**`stss` (Sync Sample, optional)**
A sorted list of 1-based sample indices that are I-frames (keyframes / sync points).
Absent when every sample is a keyframe (e.g. MJPEG).  `VideoIndex.is_keyframe` and
`VideoIndex.kf_pts_stream` are derived directly from this list.

**`stsz` (Sample Size)**
One entry per sample giving its size in bytes — used to populate `VideoIndex.size`.

**`stsc` (Sample-to-Chunk) + `stco`/`co64` (Chunk Offset)**
Samples are grouped into *chunks* (contiguous runs on disk).  `stco` gives the byte
offset of each chunk; `stsc` maps each sample to its chunk.  Together they let libav
compute the exact byte offset of any individual packet — `VideoIndex.offset`.

### Where `moov` Lives and I/O Cost

The position of `moov` relative to `mdat` determines how much of the file must be read
to build the index:

```text
Fast-start layout (web-optimized, qt-faststart):
┌──────┬────────────┬──────────────────────────────┐
│ ftyp │    moov    │             mdat              │
└──────┴────────────┴──────────────────────────────┘
         ↑ sequential read — all metadata, no media data

Standard layout (typical camera recorder output):
┌──────┬──────────────────────────────┬────────────┐
│ ftyp │             mdat             │    moov    │
└──────┴──────────────────────────────┴────────────┘
  reads first 8 bytes, seeks to end, reads moov
```

In the fast-start layout libavformat never seeks backward: it reads `ftyp`, then `moov`,
and the full sample table is available.  In the standard layout two seeks are required
(forward past `mdat` to find `moov`), but still no encoded video data is read.

In either case, the total bytes read to build `VideoIndex` is proportional to the number
of entries in `stts`, `ctts`, `stss`, and `stsz` — roughly **5–15 bytes per frame**.
For a 30-minute 30 Hz video (~54k frames) this is 1–3 MB, and the operation completes
in a few milliseconds.

### `FROM_HEADER` vs `FULL_DEMUX`

`make_index_and_metadata` exposes this via `VideoIndexCreationMethod`:

- **`FROM_HEADER`** (default): libavformat reads the `stbl` sub-boxes from `moov` as
  described above.  No encoded bitstream is read.  Milliseconds, O(MB).

- **`FULL_DEMUX`**: walks every packet in the file, building the index from actual
  demuxed packet headers.  Produces identical results for well-formed MP4s but requires
  reading and partially parsing the entire file.  Use only for validation or for
  containers without a complete `stbl` (e.g. truncated recordings or MPEG-TS streams,
  which have no random-access index at all).

`FROM_HEADER` is what makes sparse decode practical: the per-frame PTS and keyframe
information needed to build a complete seek plan is available in the container header
alone, before any video data is read.

## Architecture

```text
VideoIndex (pts_ns, pts_stream, kf_pts_stream, time_base)
        │
        ▼
sample_closest_indices(pts_ns, grid) → indices, counts
        │  pure numpy, no I/O
        ▼
pts_stream[indices] → sampled_pts_stream
        │
        ▼
make_decode_plan(kf_pts_stream, sampled_pts_stream, counts)
        │  pure numpy, no I/O
        ▼
list[(kf_pts_stream, group_targets)]  ← one entry per unique governing GOP
        │
        ▼
decode_closest_frames(source, decode_plan, time_base)
        │  one seek + forward decode per entry
        ▼
list[(target_ns, actual_ns, rgb24_array)]
```

### VideoIndex

`VideoIndex` is a structure-of-arrays container built from the container's index
entries (no demux required).  The fields used by the decode path are:

| Field | Type | Description |
| --- | --- | --- |
| `pts_ns` | `ndarray[int64]` | Presentation timestamp of every packet, in nanoseconds, in ascending order |
| `pts_stream` | `ndarray[int64]` | Presentation timestamp of every packet in stream-native time_base units, in ascending order |
| `kf_pts_stream` | `ndarray[int64]` | Keyframe PTS in stream-native time_base units — passed directly to `make_decode_plan` and `container.seek` |
| `time_base` | `Fraction` | Stream time_base (e.g. `Fraction(1, 15360)`); enforces `pts_to_ns(pts_stream, time_base) == pts_ns` at construction |
| `is_keyframe` | `ndarray[bool]` | True for I-frame packets |
| `offset` | `ndarray[int64]` | Byte position in the file of each packet (per-packet lookup, not scan order) |

All arrays are sorted by PTS before being stored.  For non-B-frame video this is a
no-op.  For B-frame video, the container stores packets in decode order (DTS order),
which produces a non-monotonic PTS sequence such as `[0, 3, 1, 2, 6, 4, 5, ...]`.
`get_video_index` argsorts by PTS so that `pts_ns` is always monotonically
increasing and `pts_ns[-1]` is always the true last presentation timestamp.

A side-effect of this sort is that `offset` is **not** monotonically increasing for
B-frame video — adjacent entries may point to non-sequential file positions.

`VideoIndex` is built once per video.  For a 1-hour 30 Hz video (~108k frames) this
takes a few milliseconds and uses ~a few MB of memory.

## Algorithms

### `make_decode_plan`

**Signature**: `make_decode_plan(kf_pts_stream, pts_stream, counts) → DecodePlan`

Computes which keyframe to seek to for each target timestamp, and groups multiple
targets that fall in the same GOP so they are resolved in a single seek+decode pass.
All arguments are in stream-native time_base units (not nanoseconds).

```python
# For each target, find the last keyframe at or before it.
# searchsorted(..., side='right') - 1  gives the index of that keyframe.
insert = searchsorted(kf_pts_stream, pts_stream, side='right') - 1
clip(insert, 0, len(kf_pts_stream) - 1)    # clamp out-of-range targets

governing_kf = kf_pts_stream[insert]       # keyframe PTS for each target

# Group and deduplicate.  np.unique preserves ascending order,
# so all seeks are strictly forward — no backward seeks.
for each unique kf in np.unique(governing_kf):
    emit one plan entry: kf and the (pts_stream, count) pairs for that GOP
```

**Output contract**:

- One entry per unique governing keyframe — GOPs with no targets are absent.
- `len(plan) ≤ number of keyframes` (sparse sampling skips most GOPs).
- Entries are in strictly ascending keyframe order (all seeks are forward).
- Every input target appears in exactly one group.
- Targets in the last GOP (after the last keyframe, before the last frame) are valid
  and governed by the last keyframe.

**Raises** `ValueError` for any of the following:

- `kf_pts_stream` is empty (no keyframes).
- `kf_pts_stream` is not sorted in strictly ascending order.
- `pts_stream` and `counts` differ in length.
- `pts_stream` is not sorted in strictly ascending order.
- Any value in `pts_stream` is before the first keyframe in `kf_pts_stream`.

Callers are responsible for ensuring inputs satisfy these preconditions.
Silent clamping for out-of-range targets was rejected because it produces subtly wrong
output at scale — a target at t=0.05s silently returning the frame at t=0.10s is harder
to debug than a loud error.

### `decode_closest_frames`

For each `(kf_pts_stream, group_targets)` in the plan:

1. `container.seek(kf_pts_stream, stream=stream)` — seeks directly using the
   stream-native PTS stored in `VideoIndex`.  No ns→stream conversion is needed;
   see [Time-Base Conversion](#time-base-conversion) for why the inverse is never used.
2. `codec_context.flush_buffers()` — required after every seek to discard stale
   reference frames from the previous decode position.
3. Decode forward frame by frame, tracking `prev_frame`.  For each target in the
   group (in order), when a decoded frame first passes the target PTS, pick the
   closer of `prev_frame` and `curr_frame`.
4. Break out of demux as soon as all targets in the group are resolved — no further
   packets are read from disk.

**Closest-frame selection** (bracket-and-pick):

```python
while frame_pts_ns >= current_target:
    if prev_frame is None:
        best = curr_frame               # first frame, no bracket available
    elif |curr_pts - target| < |prev_pts - target|:
        best = curr_frame
    else:
        best = prev_frame
    emit (target, best_pts, best_frame.to_ndarray())
    advance to next target
```

Decoded frames arrive in presentation (PTS) order from the codec, so the bracket is
always valid.

### Time-Base Conversion

`VideoIndex` stores timestamps in two forms: nanoseconds (`pts_ns`, `kf_pts_ns`) for
MCAP-compatibility and external APIs, and stream-native time_base units (`pts_stream`,
`kf_pts_stream`) for seeks and decode comparisons.

The stream→ns direction (used once at `VideoIndex` build time) uses exact `Fraction`
arithmetic to avoid floating-point drift at epoch-scale timestamps:

```python
# VideoIndex build  (stream → ns, done once)
pts_ns = int(entry.pts * time_base * 1_000_000_000)   # Fraction arithmetic, exact
```

The inverse (ns→stream) is **never performed in the decode path**.  Floor division is
lossy for fps-rate time_bases: with `Fraction(1, 30)`, frame 5 has `pts_stream=5`,
which converts to `pts_ns=166_666_666`, which converts back via
`166_666_666 * 1 // (1 * 1_000_000_000) = 0` — pointing at frame 0 instead of frame
5.  At 30 fps this causes seeks to land in the wrong GOP and decode extra frames.

Instead, `pts_stream` and `kf_pts_stream` are preserved verbatim from the container
index and flow through `sample_closest_indices` → `make_decode_plan` →
`decode_closest_frames` without ever being converted back.

## Memory and I/O Characteristics

| Property | Value |
| --- | --- |
| Memory per decoded frame set | `O(N_targets)` — only selected frames held |
| Disk reads per seek | One keyframe packet + inter-GOP packets up to target |
| Seeks per call | `len(make_decode_plan(...))` ≤ number of keyframes |
| All seeks forward | Yes — `np.unique` on governing keyframes preserves order |
| Container formats | MP4/MOV, MKV (index-based seek). MPEG-TS not supported. |

## Testing

Tests live in `tests/cosmos_curator/core/sensors/utils/test_video.py`.

### `make_decode_plan` test coverage

| Category | Cases |
| --- | --- |
| **Errors** | Empty / unsorted `kf_pts_ns`, length mismatch `pts_ns` vs `counts`, unsorted `pts_ns`, targets before first keyframe → `ValueError` |
| **Edge** | Empty `targets` → `[]` |
| **Normal** | Single GOP; two GOPs one target each; two GOPs multiple targets |
| **GOP skipping** | 5 keyframes, 2 targeted — 3 absent from output |
| **All keyframes** | Every frame is a keyframe; targets are keyframe timestamps |
| **Exact match** | Target PTS equals a keyframe PTS |
| **Out of range** | Target before first keyframe → `ValueError`; target after last frame → `ValueError`; mixed in/out-of-range → `ValueError` |
| **Boundary** | Target in last GOP (valid); single keyframe covers all in-range targets |
| **Ordering** | Output entries are strictly ascending; skipped GOPs absent |
| **Epoch timestamps** | Large `int64` values (~`1.7e18 ns`) — no overflow or precision loss |

Tests validate the output contract, not implementation details.  Any correct
implementation of the same contract should pass.
