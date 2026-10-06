# Sensor Library MP4 Parsing Design

This note documents the generic ISO-BMFF/MP4 box-parsing implementation in the Cosmos
Curator Sensor Library: `cosmos_curator.core.sensors.mp4` (box walking, track parsing) and
`cosmos_curator.core.sensors.data.mp4` (`Mp4FrameIndex`/`Mp4Metadata`, the decode-relevant
typed views built from a parsed track). It does not define the `mp4-index` recipe's own
output contract (manifest schema, sidecar naming, publication, reconciliation) — see
[curator-next-mp4-index.md](curator-next-mp4-index.md) for that.

## Why a custom reader, not PyAV

This codebase's existing demuxer, PyAV, doesn't expose two things `mp4-index` needs:

- **User-defined boxes.** A unit can write its own proprietary boxes alongside the standard
  ISO-BMFF ones; PyAV only surfaces boxes it recognizes.
- **`ctts`'s own composition-offset table.** PyAV resolves presentation timestamps internally but
  doesn't expose the undecoded box a B-frame track's composition offsets live in -- this package
  needs the raw ticks, not an already-computed result (see "Design principle" below).

Both require reading the box tree directly instead of through a demuxer.

## Why this is a separate package from any recipe

`core.sensors` is self-contained: nothing under it may import from `cosmos_curator.next`,
`cosmos_curator.core.utils`, or anywhere else in the tree (enforced by
`tests/cosmos_curator/core/sensors/test_import_boundary.py`). MP4 box parsing lives here,
not inside the `mp4-index` recipe, because it has no dependency outside this package: it
takes a byte range or a stream, returns a typed, undecoded view of what the container
states, and does no I/O, no storage-client work, no Lance. Recipe-specific concerns —
discovering candidate files under an S3 prefix, publishing a manifest, naming sidecars —
depend on `cosmos_curator.core.utils.storage` and live in
`cosmos_curator.next.recipes.mp4_index` instead.

## Scope

Box-level only. No decode, no muxing, no coverage of every ISO-BMFF box — `udta`/`meta`,
audio tracks, and anything outside one video track's own `moov` entry are out of scope.
Both non-fragmented MP4 (one pooled `moov` sample table) and fragmented MP4 (one
`moof`/`traf`/`trun` at a time) are covered, producing the same `VideoTrack` type either
way.

## Design principle: faithful decode, judgment calls deferred

Every parser in this package decodes exactly what a box states and leaves any judgment
call the box itself doesn't resolve as an explicit `None`, rather than guessing or
defaulting on a caller's behalf:

- A sample entry's fourcc (`avc1`, `hev1`, ...) is never mapped to a friendly codec name
  here — `VIDEO_FOURCC_CODEC_NAMES` (`box_primitives.py`) and `Mp4Metadata.from_video_track`
  (`data/mp4.py`) own that.
- A track whose `ctts` reorders samples (B-frames) is parsed, not rejected or
  special-cased.
- An absent `stss` is never defaulted to "every sample is a keyframe" — that's a
  spec-sanctioned but still-a-choice inference, left to the caller.
- Composition offsets, edit lists (`elst`), and AVC profile/level fields are all persisted
  as the undecoded ticks/bytes the box states, not pre-converted to nanoseconds or
  human-readable form. "Persist undecoded, interpret in code": a fix to the
  interpretation (e.g. edit-list application) then applies to every already-extracted
  sidecar on its next read, with no re-extraction needed — only a true parsing bug (what
  gets extracted from the source bytes) requires that.

## Module map

| Module | Owns |
|---|---|
| `box_primitives.py` | Generic header parsing, child walking, path descent, `find_video_trak`. `VIDEO_FOURCC_CODEC_NAMES` is the single table for both "is this fourcc a video track" and "which codec is it." |
| `byte_range.py` | `FetchRange`, the transport contract `box_locate.py`'s guess-then-fallback pattern shares: arbitrary, independent offsets. Contracts only, no concrete implementation. |
| `box_locate.py` | Locates a non-fragmented file's `moov` box: `get_moov_full_walk` (always correct, forward header walk), `get_moov_tail_guess`/`get_moov_head_guess` (round-trip-optimized guesses for unit output that leads or trails with `moov`), `get_moov_auto` (head guess, then tail guess, which falls back to the full walk). |
| `moov_track.py` | Parses `mvhd`/`tkhd`/`mdhd`/`stsd` and defines `VideoTrack` — shared by both the non-fragmented and fragmented combiners, since codec description and track/media timing don't depend on fragmentation. |
| `moov.py` | The non-fragmented combiner: struct-of-arrays decode of one `moov`'s pooled `stbl` (`stts`/`ctts`/`stss`/`stsz`/`stsc`/`stco`/`co64`) into a `VideoTrack`. |
| `fragment_table.py` | The fragmented combiner: decodes `moov` + every `moof`/`traf`/`tfhd`/`tfdt`/`trun` into the same `VideoTrack` shape. Assumes a fixed layout verified against real production files (`tfhd` flags `0x020008`, `trun` flags `0x000601`, `sample_count=1`) and refuses rather than guessing at anything else. |
| `moof_walk.py` | Walks a fragmented MP4 once, yielding every `moof` box verbatim (header included), unparsed — reads in large sequential chunks through one open connection, carrying a trailing partial box across chunk boundaries. |
| `moof_schema.py` | `MOOF_TABLE_SCHEMA` — the sidecar shape `moof_walk.py`'s output is written to: one row per fragment, each `moof`'s own bytes plus its absolute file offset (the one fact not recoverable from the bytes alone). |
| `errors.py` | `UnindexableVideoError` — the one base class every error in this package raises, so a caller can catch a single type regardless of which reader failed. |
| `data/mp4.py` | `Mp4FrameIndex`/`Mp4Metadata` — typed, decode-relevant views built from a `VideoTrack`, either freshly parsed (`from_video_track`) or read back from an already-published sidecar (`from_frame_index_table`/`from_row`). One interpretation path either way. |

## Error contract

Every exception this package raises is a subclass of `UnindexableVideoError` (`errors.py`)
— "this video cannot be read by this package's readers, for any reason, not a bug in the
caller." Notable subclasses:

- `BoxNotFoundError` — a required box is absent within the searched range.
- `AmbiguousVideoTrackError` — more than one `trak` under `moov` has a video-codec `stsd`
  entry.
- `MalformedBoxHeaderError` — a box declares a total size smaller than its own header.
- `MalformedSampleTableError` (`moov.py`) — a `stbl`'s own sample-count cross-validation
  (`stsz`/`stsc`-expanded count vs. `stts`-expanded count) disagrees.
- `MalformedCompositionOffsetError` (`moov.py`) — a `ctts`'s own sample count doesn't match
  `stts`'s, leaving ISO/IEC 14496-12 8.6.1.3.3's `CT(n) = DT(n) + CTTS(n)` undefined for
  some sample.
- `EmptySampleTableError` (`moov_track.py`) — a parse produced zero samples, from either
  combiner.
- `NoFragmentsFoundError` (`moof_walk.py`) — a file declared fragmented has zero `moof`
  boxes.
- `FragmentLayoutUnsupportedError`/`CompositionOffsetUnsupportedError`
  (`fragment_table.py`) — a fragment's layout, or a nonzero composition offset, doesn't
  match the one shape verified against real production data.
- `UnsupportedEditListError` (`data/mp4.py`) — an `elst` box is present but not the one
  shape (`from_frame_index_table`) knows how to apply to `pts_ns`.

A recipe-specific error (e.g. an unavailable sof/moe join) subclasses
`UnindexableVideoError` in that recipe's own error module, not here.

## What's deliberately out of this package's scope

- **Storage/transport.** `FetchRange` is a contract a caller builds over `smart_open`, a
  presigned URL, or a local file; `moof_walk.py` takes an already-open file-like handle
  directly, owned and closed by its caller. This package picks no transport library.
- **Publication I/O and schemas.** `FRAME_INDEX_SCHEMA`/`VIDEO_METADATA_SCHEMA` and every
  read/write sidecar function live in `cosmos_curator.next.recipes.mp4_index.contracts`
  — the producer-owned contract. This package holds only the typed reader views
  (`Mp4FrameIndex`/`Mp4Metadata`) that consume Arrow arrays and metadata rows.
- **Friendly codec names, presentation-timestamp policy, keyframe-absence policy.** All
  resolved one layer up, by whatever reads a `VideoTrack` or an already-published sidecar
  next — see "Design principle" above.

## Known constraints

- **Fragmented layout is fixed, not general.** `fragment_table.py` only accepts one
  `tfhd`/`trun` flag combination (`tfhd` flags `0x020008` — `default-base-is-moof` +
  `default-sample-duration-present`; `trun` flags `0x000601` — `data-offset-present` +
  `sample-size-present` + `sample-flags-present`; `sample_count=1`), verified against
  production files. Any other combination raises `FragmentLayoutUnsupportedError`
  rather than being parsed — this is a real, current scope boundary, not a hypothetical
  one, and broadening it needs verification against real files in that other shape before
  the refusal can be relaxed.
- **A nonzero composition offset in a fragmented track is refused**
  (`CompositionOffsetUnsupportedError`), not computed — never observed in real production
  data, so there is nothing to verify the decode against yet.
- **`UnsupportedEditListError`** covers only the one `elst` shape (single entry, normal
  rate, non-negative `media_time`) verified against a real B-frame clip. A multi-entry
  edit list, an empty edit (`media_time=-1`), or a non-1x rate is refused rather than
  guessed at.
- **Even the supported `elst` shape only applies `media_time`, never `segment_duration`.**
  `segment_duration` is in the *movie* timescale (`mvhd`'s own), not the *media* timescale
  (`mdhd`'s) `media_time` and `dts_ticks` already share — a field this package has never
  parsed, so an edit list that also trims a track's tail, not just its lead-in, still
  publishes every trailing sample. B-frames are the one real source of `elst` in
  camera-originated footage (an encoder's own reordering lookahead needs trimming from the
  front) — strongly discouraged for training data regardless, since decoding them makes
  both memory and compute depend on reference-frame distance rather than staying constant
  per frame, so this gap hasn't been worth closing against an already-discouraged encoding
  choice.
- **`box_locate.py`'s guess sizing is calibrated, not arbitrary.** The initial head/tail
  guess (8 MiB) is sized against real production files whose entire `moov` (header, sample
  table, and any shared `uuid` extension box) measured well under that bound; the one retry
  multiplies by 4x before falling back to the full walk.
- **Guessing avoids `get_moov_full_walk`'s one-round-trip-per-box cost.** A full walk pays
  one fetch for every top-level box before `moov`; `get_moov_tail_guess`/`get_moov_head_guess`
  resolve in a single fetch when the guess hits, which is the common case for unit output
  that leads or trails with `moov`.
- **`box_locate.py` returns `moov`'s bytes directly, not an offset pair.** A caller never
  needs a second fetch for bytes already in hand — splitting "locate" from "fetch" would pay
  for the same bytes twice on every guess that hits.
- **`moof_walk.py` reads in large sequential chunks, not one header probe per box.** A real
  production file can carry tens of thousands of fragments, so round-trip cost is the whole
  reason this chunked design exists.

## Implementation status

Implemented and tested against hand-packed synthetic box layouts, three real fixture files
checked into CI (`tests/cosmos_curator/pipelines/video/data/test_clip_10s.mp4`,
`test_clip_10s_bframes.mp4`, `test_video_30s.mp4`, cross-checked against PyAV's own
`make_index_and_metadata` for offset/size/`is_keyframe`/`pts_ns` agreement on all three),
and a full real-corpus pass across every session and camera in a production indexing run.

- `cosmos_curator/core/sensors/mp4/` — tests under `tests/cosmos_curator/core/sensors/mp4/`
  (`test_box_primitives.py`, `test_box_locate.py`, `test_moov.py`,
  `test_fragment_table.py`, `test_moof_walk.py`).
- `cosmos_curator/core/sensors/data/mp4.py` — tests under
  `tests/cosmos_curator/core/sensors/data/test_mp4.py`.

## References

- ISO/IEC 14496-12 (ISO Base Media File Format) — the governing spec for every box this
  package parses (`moov`, `trak`, `mdia`, `minf`, `stbl`, `stts`, `ctts`, `stss`, `stsz`,
  `stsc`, `stco`/`co64`, `mvhd`, `tkhd`, `mdhd`, `stsd`, `edts`/`elst`, `moof`, `traf`,
  `tfhd`, `tfdt`, `trun`, `mvex`/`trex`).
- ISO/IEC 14496-15 (Carriage of NAL unit structured video in the ISO base media file
  format) — governs `avcC`/`hvcC` and the undecoded AVC/HEVC fields this package persists
  verbatim (`avc_profile_idc`, `chroma_format_idc`, `bit_depth_luma_minus8`, etc.).
