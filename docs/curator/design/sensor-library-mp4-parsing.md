# Sensor Library MP4 Parsing Design

The sensor library’s MP4 parser builds video metadata and frame indexes while exposing custom
UUID boxes to registered handlers. It supports nonfragmented sample tables and a subset of
fragmented layouts. Callers can use it independently of an indexing recipe or storage format.
PyAV and the FFmpeg CLI do not provide a general interface for retrieving arbitrary MP4 UUID
payloads.

`box_probe.index_from_boxes` is an independent indexer: it indexes a video with PyAV from the
extracted `moov` and `moof` boxes alone, laid out at their source offsets with zeros in place of
media, and returns libav's `VideoProbe` with `Mp4Metadata` built from it. It indexes the stream
libav ranks best, keeps libav's timestamps, reads fragment layouts this parser refuses, and leaves
null the metadata fields libav does not report; it does not run UUID handlers. The
underlying per-packet index is `video_probe` in `cosmos_curator.core.sensors.utils.video`, which
opens a source with FFmpeg's `ignore_editlist` and reports each packet's decode time, unshifted by
the edit list, with its composition offset, byte range, duration, and keyframe flag in decode order.

The implementation lives in `cosmos_curator.core.sensors.mp4`; its typed metadata and frame-index
views live in `cosmos_curator.core.sensors.data.mp4`. Callers supply the inputs and decide how to
store and use the results.

## Parsing inputs and results

The library provides separate parsing functions for nonfragmented and fragmented MP4s. Both return
a `VideoTrack` containing video metadata and frame-index arrays.

| Entry point | Input |
| --- | --- |
| `moov.parse_video_track` | One complete `moov` box, including its header |
| `fragment_table.parse_fragment_table` | One complete `moov` and a list of `(moof_start, moof_bytes)` pairs in file order; each offset is absolute within the source MP4 |

For nonfragmented files, the parser selects the video track and expands its `stbl` sample tables.
For fragmented files, it reads the video description from `moov` and combines the supplied
`moof` records. Both paths provide decode timestamps, sample offsets and sizes, keyframe information,
and codec metadata through the same result type.

`VideoTrack` represents frame fields as NumPy arrays in decode order, rather than one Python object
per frame. Composition offsets retain the run-length representation of `ctts` until a consumer
expands them. Byte offsets refer to the source MP4; timestamp ticks use the track's media timescale.

The box parsers preserve source fields and absence where possible:

- Missing `ctts` leaves composition offsets as `None`.
- Missing `stss` leaves nonfragmented keyframe information as `None`.
- The codec fourcc, decoder configuration bytes, and edit-list bytes remain available for interpretation.
- UUID payloads directly under `moov` and within the video sample entry are also retained as
  `(identifier, payload)` pairs, whether or not a handler is registered.

Direct box access preserves these fields and provides an extension point for UUID payloads alongside
standard MP4 parsing.

## Typed metadata and frame indexes

`Mp4Metadata.from_video_track` builds metadata for decoding, including codec names and time-base
information. `Mp4FrameIndex.from_video_track` expands composition-offset runs, treats all samples as
keyframes when `stss` is absent, and computes:

- Decode and composition-offset timestamps in integer nanoseconds.
- Media presentation timestamps (`pts_ns`) from decode time plus composition offset.
- Edited presentation timestamps (`edited_pts_ns`) after the supported edit-list shift.
- Keyframe indexes and a stable presentation-order permutation.

The frame arrays stay in decode order. `pts_sort_order` maps a presentation-order rank to a
sample index; it does not reorder the arrays.

The same interpretation is available for stored records through `Mp4Metadata.from_row` and
`Mp4FrameIndex.from_frame_index_table`. The latter accepts an Arrow table plus the video's time base
and optional edit list. The index retains its time base and edit-list bytes. Both `pts_ns` and
`edited_pts_ns` are computed during construction and remain in decode order. Unsupported edits
raise `UnsupportedEditListError` during construction. `pts_sort_order` is computed from media PTS,
because supported edits shift every presentation timestamp equally and cannot change sample ordering.

The three container timelines expose read-only NumPy views. Absent or all-zero composition offsets
let `pts_ns` share the `dts_ns` buffer. An absent edit list or zero edit shift lets `edited_pts_ns`
share the `pts_ns` buffer. Nonzero transformations allocate separate arrays.

`StreamingCameraSensor` takes a source and an `alignment_field` selected from
`VideoFrameAlignmentField`: `PTS_NS` (the default), `EDITED_PTS_NS`, `DTS_NS`, `EOF_NS`, `SOF_NS`, or
`MOE_NS`. Its `timestamps_ns` property reports the selected index timestamps. Absent or partially
missing timestamps raise an error; selecting SOF or MOE does not substitute EOF or infer values.
Decoded frames always stream in presentation order. Alignment retains the current frame and one
lookahead frame; windows do not collect decoded pixels. DTS remains readable in decode order for
integrity checks. Sampling with DTS raises when decode and presentation order differ, since
reordering decoded frames would require an additional pixel buffer.

The constructor accepts optional `metadata` and `frame_index`, supplied together. Otherwise, accessing
`metadata`, `frame_index`, or `timestamps_ns` builds and stores both objects from the source once.
`Mp4Indexer.PYAV` is the default backend; `Mp4Indexer.CUSTOM` uses the box parser. Neither backend requires
extracted artifacts: the source reader fetches `moov` and `moof` boxes and seeks past media payloads.
Custom indexing supports one video track at `stream_idx=0`; PyAV selects the requested video stream.
PyAV does not interpret recorder-specific UUID timestamps, so selecting EOF/SOF/MOE requires a supplied
index containing those values or suitable timestamp extraction.

Data-integrity checks select `DTS_NS` explicitly for stored indexes. Legacy external timestamp files
continue through `CameraSensor`; they are not relabeled as EOF/SOF/MOE. Required manifest reads propagate
unsupported-edit errors; optional manifest reads retain their existing fallback to opening the media.

EOF/SOF/MOE are independently optional; recipes decide which timestamps are
required. The frame-index reader represents entirely null optional columns as `None`. Partially missing
SOF/MOE columns become NumPy masked arrays with `int64` values and an independent boolean mask:
`True` means that frame's timestamp is unavailable. Valid timestamps retain their exact nanoseconds,
including values beyond float64 precision. Fully populated columns remain ordinary NumPy arrays.
Partially null composition-offset and EOF columns are rejected. Optional externally supplied timing
arrays stay associated with their frames and are not adjusted by MP4 time-base conversion or edit-list
shifts. The generic reader preserves supplied values; recorder-specific sentinel normalization belongs
to the recipe.

These readers consume Arrow tables and metadata rows supplied by the caller. They do not open
datasets or depend on a storage client.

`Mp4Metadata.track_id` is the selected track's native unsigned MP4 ID, preserved by both indexers.
`StreamingCameraSensor(source, metadata=metadata, frame_index=frame_index)` decodes the track identified
by `metadata.track_id`; a missing track is an error. Without a supplied pair, `stream_idx` selects the
video stream for lazy indexing, and the resulting metadata identifies the track to decode.

## Sampling windows and decoded-frame memory

A clip is a logical output span; sampling windows partition delivery of its reference timestamps.
One clip can be supplied through one window or several consecutive windows. For a fixed source
timeline, reference grid, and sampling policy, flattening those windows produces identical reference
timestamps, selected source timestamps, and decoded pixels. Window bounds do not restrict eligible
source observations: a nearest frame may lie before a window's start or after its end, subject to the
same `NearestTimestampPolicy.max_delta_ns` tolerance.

`StreamingCameraSensor.sample()` yields one `StreamingCameraData` payload per window. The payloads
share one forward decode pass, so windows must be non-overlapping and their frames must be consumed
in order. Creating payloads does not decode their pixels; decoding advances as `payload.frames()`
is iterated. Callers can feed successive payloads into one clip output. The streaming recipe's current
one-window-per-clip mapping does not limit this sensor-library behavior.

| Sensor | Decoded-frame retention |
| --- | --- |
| `CameraSensor` | Materializes a window's selected frames as a batch. Smaller windows reduce that batch. |
| `StreamingCameraSensor` | Yields frames lazily. Alignment retains the current frame and one lookahead frame, independently of window length. |

Increasing the number of windows is therefore not a decoded-pixel memory control for streaming
consumption. This bound does not describe total actor memory: codec and I/O buffers, indexes,
timestamp arrays, output handling, and frames retained by the caller all add to it. Collecting
`payload.frames()` into a list retains those decoded frames and gives up the streaming memory benefit.

## UUID handlers during traversal

Both parsing entry points accept `uuid_handlers`, a sequence of `UuidHandler` registrations.
Each registration supplies a 16-byte identifier, a set of exact parent paths, and a callback.
`BoxTraversal` invokes the matching callback while scanning a visited scope's children.

The callback receives the payload after the UUID identifier and a `BoxLocation` describing the box:

| Context | Meaning |
| --- | --- |
| `offset`, `content_start`, `end` | Bounds relative to the current `moov` or `moof` buffer; `content_start` follows the box header |
| `parent.path` | The enclosing box path, excluding the UUID box itself |
| `parent.track_id` | Selected track identity when the enclosing scope identifies a track; otherwise `None` |
| `parent.fragment_index` | Zero-based position in the supplied fragment list for `moof` scopes; otherwise `None` |

For example, `("moov",)` matches UUID children of `moov`; `("moof", "traf")` matches children
of a visited `traf`. A UUID directly under `moof` has a fragment index but no track identity.
For the selected video track, traversal visits `trak`, `mdia`, `minf`, and `stbl`,
so UUID children at those paths receive the selected track ID. Parent paths include the
sample-entry fourcc when matching boxes inside a codec description.

Traversal descends through container layouts that the standard parsers understand. Registering a
path does not make the parser search arbitrary containers, other tracks, or media payloads. Unknown
UUID identifiers do not invoke callbacks; their headers and identifier lengths are still validated
when their scope is scanned.

Registrations belong to the parsing invocation, with no global registry. Each buffer has its own
traversal; revisiting a scope uses its cached child locations and does not dispatch the same UUID
box again. Duplicate registrations for the same identifier and parent path raise `ValueError`.

Callbacks collect or interpret extension data separately from `VideoTrack`; their return values
are not used to add fields to the standard index. The caller owns payload validation and any
association with samples. Callbacks can run before a later parsing failure, so use fresh collectors
for each attempt and consume their results only after parsing succeeds.

### Example: collecting a custom label

This example defines a UTF-8 label payload directly under `moov`. A real extension supplies its
own identifier, supported parent paths, and payload decoder.

```python
from cosmos_curator.core.sensors.mp4.box_traversal import BoxLocation, UuidHandler
from cosmos_curator.core.sensors.mp4.moov import parse_video_track
from cosmos_curator.core.sensors.mp4.moov_track import VideoTrack


def parse_with_labels(moov_bytes: bytes) -> tuple[VideoTrack, list[tuple[BoxLocation, str]]]:
    labels: list[tuple[BoxLocation, str]] = []

    def collect_label(payload: bytes, location: BoxLocation) -> None:
        labels.append((location, payload.decode("utf-8")))

    handler = UuidHandler(
        identifier=bytes.fromhex("00112233445566778899aabbccddeeff"),
        parent_paths=frozenset({("moov",)}),
        callback=collect_label,
    )
    track = parse_video_track(moov_bytes, uuid_handlers=(handler,))
    return track, labels
```

## Finding and fetching boxes

Transport belongs to the caller. `FetchRange` is an `(offset, length) -> bytes` callable, which can
be backed by a local file or a storage client. `get_moov_auto` tries a head range, then a tail range,
then a top-level header walk. The guesses default to 8 MiB with one 4x retry. A successful guess
returns the bytes already fetched, avoiding a second request for the same box.

`walk_moofs` reads an already-open stream in chunks and yields complete `moof` boxes with their
absolute offsets. It carries partial boxes across chunk boundaries and skips other box contents
without accumulating a whole media-data box in memory. It reads through the stream; it does not
turn skipped media into independent range requests. The caller opens and closes the handle.

Callers may cache extracted box bytes to reuse them on subsequent parsing attempts.

## Supported layouts and validation

The parser handles one recognized video track (`avc1`, `avc3`, `hev1`, or `hvc1`). It rejects ambiguous
video-track selection and multiple sample-description entries. Audio decoding, muxing, and general
support for every ISO-BMFF container are outside its scope.

Nonfragmented sample tables must agree on frame count: `stsz` and the expanded `stsc`/`stco` or
`co64` mapping must cover the samples defined by `stts`; `ctts`, when present, must cover that same
sample count. B-frame composition offsets are retained for presentation-time calculation.

Fragmented support covers single-track, single-run profiles: one track in `moov`, one `traf`
and one nonempty `trun` per `moof`, with a `tfdt` decode time and an explicit `trun.data_offset`.
Initial sample tables in `moov` must be absent or empty. Files combining initial `moov` samples
with `moof` samples are rejected, because this path indexes only the fragment samples.
Each fragment's `tfhd.track_ID` must match the track selected from `moov`, and `tfdt` must use
version 0 or 1; unknown decode-time layouts are rejected.
The base is either `tfhd.base_data_offset` or `default-base-is-moof`; implicit addressing and
multiple tracks or runs are refused. Sample durations, sizes, and flags resolve from `trun`,
then `tfhd`, then `trex`. First-sample flags apply only to the first sample. Multi-sample runs
require durations to compute subsequent decode timestamps. Nonzero fragment composition offsets
are rejected.

Both single-sample camera fragments and FFmpeg `frag_keyframe+empty_moov` output are covered.
The checked-in two-fragment fixture is verified against PyAV packet timestamps, positions, sizes,
keyframe flags, and payload bytes.

Fragmented duration comes from the decode-time span plus the last sample's duration. If that duration
is missing, the parser estimates it from the mean decode interval; a single sample uses its available
duration or zero. This avoids relying on a potentially stale `mdhd.duration`.

Presentation-time interpretation supports a single edit-list entry at normal playback rate with
nonnegative `media_time`. It applies that shift but does not trim samples using `segment_duration`.
Multiple entries, empty edits, and other playback rates are rejected.

## Errors

Source-format errors use `UnindexableVideoError` subclasses, including missing boxes, malformed
headers or truncated parsing buffers, inconsistent sample tables, unsupported fragment layouts,
and unsupported edit lists. The parsing entry points translate buffer-read `struct.error` and
`IndexError` failures into `MalformedBoxHeaderError`.

This is not a catch-all for transport failures, invalid API arguments, or custom callback errors.
Invalid UUID registrations and Arrow-column inputs can raise `ValueError`; fetcher errors propagate
unchanged. Handler failures raise `UuidCallbackError`, retaining the original exception in `error`
and as its cause. Callers decide whether that exception represents invalid source data or a handler bug.

## Implementation map

| Module | Responsibility |
| --- | --- |
| `box_primitives.py` | Box headers, basic path lookup, and video-track selection |
| `box_traversal.py` | Scoped child scans, UUID dispatch, and location context |
| `byte_range.py`, `box_locate.py` | Range-fetch contract and `moov` location strategies |
| `moof_walk.py`, `moof_schema.py` | Sequential fragment extraction and the Arrow schema for caching fragment bytes and offsets |
| `moov_track.py` | `VideoTrack`, track headers, sample descriptions, and composition-offset runs |
| `moov.py` | Nonfragmented sample-table parsing |
| `fragment_table.py` | Fragmented sample-index assembly |
| `errors.py` | Source-format error base class |
| `data/mp4.py` | Typed metadata and frame-index interpretation |

Tests live under `tests/cosmos_curator/core/sensors/mp4/` and
`tests/cosmos_curator/core/sensors/data/test_mp4.py`. They cover box extraction, sample-table parsing,
UUID dispatch and context, fragment defaults and duration, and typed-reader behavior using synthetic
boxes and MP4 fixtures.

## References

- ISO/IEC 14496-12: ISO Base Media File Format, including sample tables and movie fragments.
- ISO/IEC 14496-15: Codec configuration records for AVC and HEVC in ISO-BMFF.
