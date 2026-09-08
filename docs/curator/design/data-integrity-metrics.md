# Data Integrity Metrics

A reference for the data-integrity metrics: what each one measures, how it works, its measurement fields, and corner cases.

Metrics are pure measurements — each computes a fact about the data and carries no pass/fail policy (that is the evaluators' job). Timestamp metrics take a 1-D array of `int64` nanosecond timestamps; `FrameReorderingPresentMetric` takes a boolean frame-reordering flag. Time-domain values are nanoseconds in the `int64` range, matching `core.sensors`: array fields (the input timestamps) are `np.int64`, while scalar discrete fields (`expected_period_ns`, `max_gap_ns`) are Python `int` constrained to that range. Aggregate statistics computed over them are float nanoseconds, since a mean or spread is not generally an integer (for example `actual_mean_period_ns`).

Every result also reports `is_defined` — whether the input met the metric's computational minimum (for example, ≥2 samples to classify a step, or ≥1 valid interval for a rate). Metrics never raise on thin data; they return a measurement with `is_defined == False`, and callers must check `is_defined` before evaluating. The per-metric corner cases below note each one's minimum.

For how these fit in the framework and which ship in the first release, see [data-integrity-design.md](data-integrity-design.md).

## Input contract

Every metric validates its input before changing any state, so a bad call raises rather than leaving the instrument half-updated. This is shared here rather than repeated in each metric's corner cases below.

All four timestamp metrics (`TimestampOrderingMetric`, `RateMetric`, `TimestampGapMetric`, `JitterMetric`) take a 1-D `int64` NumPy array:

- not a NumPy array → `TypeError`; not 1-D or not `int64` → `ValueError`
- empty and single-sample arrays are valid input (no error); they may still be undefined — see each metric's minimum

`RateMetric`, `TimestampGapMetric`, and `JitterMetric` *subtract* consecutive timestamps to measure interval magnitudes, so they additionally:

- reject an interval too large to represent in `int64` (for example a corrupt sentinel beside an epoch timestamp) → `ValueError`, rather than a silently wrapped value
- validate `expected_hz` at construction: non-positive or non-finite → `ValueError`; so high the discrete period rounds below 1 ns → `ValueError`; so low the period exceeds `int64` nanoseconds → `ValueError`

`TimestampOrderingMetric` instead compares consecutive timestamps pairwise for ordering, without fixed-width subtraction (array comparison within a window, arbitrary-precision Python subtraction at the seam), so it never overflows — it accepts the entire `int64` domain (a step from `INT64_MIN` to `INT64_MAX` is simply a large forward step) and takes no `expected_hz`.

`FrameReorderingPresentMetric` takes a single boolean flag: a non-boolean → `TypeError`, and a second `update()` → `RuntimeError`.

## Timestamp provenance

A metric measures the timestamps it is handed and cannot tell where they came from, so the value of every timing result depends on the clock behind its input. That makes the clock part of the result, and every stream reports which one it was measured on as `timestamp_source`: `capture` or `container_pts`. A session reports the roll-up of its streams as `timestamp_source` too: `capture` or `container_pts` when every contributing stream agrees, `mixed` when they do not, and null when nothing contributed a clock — an empty session, or one whose streams could not be read.

**Why it matters.** Presentation timestamps are the container's playback instructions, not a record of when anything was captured. Whether they reflect capture depends on how the file was muxed. A variable-frame-rate source carries the real sample times, and the metrics below measure exactly what they claim. A constant-frame-rate source does not: the encoder discards capture times and regenerates them on a fixed grid, and each file's grid restarts at `0`. Against such a source the interval-based metrics (`RateMetric`, `TimestampGapMetric`, `JitterMetric`) see a metronome and report a flawless stream by construction, while the session-grain metrics see every sensor starting at the same instant and report zero spread with total overlap — in both cases regardless of what the hardware actually did. There is no standard for how a vendor relates the two: some anchor `pts[0]` to the first frame's capture time and offset the rest, others space PTS evenly while the true capture times drift.

Measured on one internal 12-camera session (605 frames, 30 Hz), container PTS gives an interval of exactly 33.333 ms at every step and a cross-sensor spread of `0`. The same session's capture timestamps — recorded in per-camera sidecar files on a clock shared across the rig — give intervals ranging 33.306–33.374 ms and put the cameras 133 ms apart, for 1.3% non-overlap. So a perfect score on container PTS means *nothing was measurable*, not *the data was verified healthy*.

**Where capture times come from.** A recording that has them carries a `<video>.mp4.timestamps.npy` sidecar: one `int64` epoch-nanosecond entry per displayable frame, on a clock shared across the rig. `CameraSensor` accepts that timeline and reports it in place of PTS, everywhere — `start_ns`, `end_ns`, `timestamps_ns`, `stream_timestamps()`, and the payload clock on a sampled window. The container's own PTS stays intact for seeking and decode planning, which is what the muxer's numbers are actually for.

The sidecar is an interface, not the format: it is where this codebase's one available capture-timestamp source happens to live, deliberately converted out of any vendor's own representation before it reaches the sensor. A recording that carries capture times some other way (MCAP, for instance, where each message's own timestamp is the capture time) is served by teaching the recipe to produce the same array, with nothing below that changing.

The sidecar is also not proof, which is why finding one is the recipe's job rather than the sensor's: `CameraSensor` reports any timeline it is handed as `capture`, because a library that is given an array and no provenance has nothing else to say. `multimodal-split` writes an evenly spaced resampling grid under the same filename, and a grid believed is worse than no timeline at all — it produces exactly the flawless-by-construction numbers described above, with a provenance stamp saying they are trustworthy. See [data-integrity-pipeline.md](data-integrity-pipeline.md) for how the recipe tells the two apart.

**When there is no sidecar.** Most sources have none, so its absence is not a finding: the stream is measured on container PTS and says so. A sidecar that exists but is unreadable, or whose length disagrees with the video's frame count, is a different matter and fails the stream — silently falling back would hide the one thing an operator can act on. Because a timeline is matched to frames by position, a count that disagrees leaves *every* frame's capture time in doubt rather than only the missing ones'.

Reading provenance beside the numbers is therefore how a reader tells a rig that is genuinely steady from a mux that invented the timeline; both produce near-zero jitter, and only one of them is a measurement. It is reported in both CLIs' output and persisted as a `stream.lance` column, so a query for streams whose cadence is real can filter on it.

### The session grain skips without a shared capture clock

The per-stream metrics still run on container PTS: a regenerated cadence is misleading rather than meaningless, the reader can see the clock beside it, and a per-stream number does at least describe the file that was read. Both session-grain metrics are different in kind. Each compares *one camera's timeline against another's*, and a constant-frame-rate mux puts every camera on the same grid restarting at `0`, so what they would compare is the encoder's arithmetic rather than any two cameras. That does not produce a misleading number, it produces a number with no subject.

So a metric declaring `requires_capture_timestamps` — today both of them — runs only when the session's rolled-up clock is `capture`. Otherwise it reports `SKIPPED`, with a reason naming what was measured instead: every sensor on container PTS, the sensors disagreeing, or no sensor reporting a clock at all. `mixed` skips for the same reason `container_pts` does rather than being treated as partial evidence: a spread between one camera's real capture times and another's synthesized grid is the offset between two unrelated clocks.

The skip is recorded as *never ran* — `is_defined` null with every metric column null — not as a measurement marked skipped. A stored zero spread would be a fact in the store, and [`di-reevaluate`](data-integrity-store-schema.md) would rebuild it and hand it to the evaluator, turning the mux's artifact into a `PASS` under any threshold. There is nothing to rebuild from a null row, which is what makes the skip hold. Because a never-ran row cannot explain itself either, the session's `timestamp_source` column carries the cause, and a re-judge reads it back to say the same thing the original run did.

Two skips have to be told apart, and the provenance check runs first for that reason. One camera measured on capture times is a session that *was* measured and had nothing to compare, so it stores real numbers with `is_defined = false` and `num_sensors = 1`. One camera on container PTS was never measurable, so it stores nulls. The distinction matters downstream: only the first is a fact a later policy can re-judge.

This is honest about the tool's reach rather than a limitation to route around. Capture timestamps have no cross-vendor standard, and the sidecar is the one source this codebase can read, so skipping says "this dataset cannot answer the cross-sensor question" instead of answering it wrong.

## TimestampOrderingMetric

Classifies how a timestamp stream steps — increasing, equal, or decreasing — so one pass tells you whether it is non-decreasing, strictly increasing, or neither.

**How it works:** compares each consecutive timestamp pair — without fixed-width subtraction, so it stays exact across the whole `int64` range — and classifies the step: a larger value is increasing, an equal value is an adjacent duplicate, a smaller value is decreasing (backward). It reports the two failure counts separately, so evaluation policy chooses which matter:

- non-decreasing (duplicates allowed): require `decreasing_count == 0`
- strictly increasing (no duplicates): require `strict_violation_count == 0`

`duplicate_count` counts *adjacent* duplicates only, but that is exact for the property that matters: a stream with `decreasing_count == 0` and `duplicate_count == 0` is strictly increasing, and therefore globally unique — a far-apart duplicate would require a decreasing step somewhere, which `decreasing_count` catches. (Adjacent-only is also what lets this run in bounded memory while streaming — see below.)

**Measurement fields:**

- `decreasing_count` — number of backward steps (`t[i] < t[i-1]`)
- `duplicate_count` — number of adjacent duplicate steps (`t[i] == t[i-1]`)
- `first_decreasing_index`, `first_duplicate_index` — first offending sample of each kind, or `None`
- `num_samples` — number of timestamps analyzed
- `strict_violation_count` (derived) — `decreasing_count + duplicate_count`; for a defined measurement, zero iff strictly increasing (an undefined measurement — fewer than two samples — also reports zero)

**Corner cases:**

- fewer than two samples → an undefined measurement (`is_defined == False`): all counts zero, both indices `None`
- a step is decreasing or duplicate, never both; the two are counted independently
- indices point at the offending sample, not the interval
- `duplicate_count` is adjacent-only; a global duplicate census on *unordered* data is a different, non-streamable problem

**Streaming:** the metric instrument itself folds the stream — feed successive windows with `update(timestamps_ns=...)` and call `measurement()` to finalize. It carries only the previous window's last timestamp (to classify the step across the window boundary) plus a running index offset, so its state is O(1) regardless of stream length. A one-shot measurement over a whole array is a single `update()` + `measurement()`.

## RateMetric

Measures how far the mean sample *period* deviates from the expected period, as a percentage — average timing accuracy, not worst-case jitter. The result is period-space (a 2× rate error reads as 50%).

**How it works:** the expected period is `expected_period_ns = round(1e9 / expected_hz)`, rounded half up to the nearest integer nanosecond (so 2.5 ns → 3 ns, not 2) — the same discrete period the gap metric uses. Consecutive intervals are compared to it; the deviation is `|sum(interval − expected_period_ns)| / (expected_period_ns × n) × 100`. Because it sums *signed* errors before taking the absolute value, fast and slow intervals partially cancel — the result is the relative deviation of the mean interval from the expected period (net drift), not per-sample variation.

**Measurement fields:**

- `period_deviation_percent` — relative deviation of the mean interval from the expected period, as a percentage (period-space; a 2x rate error reads as 50%)
- `expected_hz`, `expected_period_ns` — the target rate and period
- `actual_mean_period_ns` — the mean inter-sample interval; `actual_mean_hz` — its reciprocal rate (`1e9 / actual_mean_period_ns`), not the arithmetic mean of per-interval rates
- `num_samples`, `num_intervals`, `num_filtered` — sample count, intervals analyzed, non-monotonic intervals dropped

**Corner cases:**

- non-monotonic (`<= 0`) steps are dropped as invalid cadence intervals; every forward interval is kept, so a dropout counts toward the deviation (flagging the dropout itself is `TimestampGapMetric`'s job)
- no valid intervals (empty or all non-monotonic) → an undefined measurement (`is_defined == False`), not an error
- alternating fast/slow intervals can cancel to a small deviation — that is correct (it is rate accuracy, not spread; use `JitterMetric` for spread)
- `period_deviation_percent` is a dimensionless ratio, but this API is nanoseconds only (`expected_period_ns` derives from the fixed `1e9` ns/s): timestamps must be nanoseconds, and passing another unit changes the result. The ratio is unit-independent only if every time quantity — the intervals *and* the seconds constant — is converted together, which the fixed `1e9` does not let you do

## TimestampGapMetric

Flags inter-sample intervals large enough to imply missing samples — *inferred* gaps, summarized by how many there are, the largest, and where the first one is.

**How it works:** the expected period is `round(1e9 / expected_hz)`, rounded half up to the nearest integer nanosecond (so 2.5 ns → 3 ns, not 2). Each consecutive interval is compared to it with exact integer arithmetic: an interval is a gap when `interval >= (3 × period + 1) // 2` — equivalently `interval / period >= 1.5`, with an exact 1.5-period tie counting as a gap. (Writing the cutoff as `round(interval / period) >= 2` would leave the half-period tie ambiguous and lose precision at large values; the integer form fixes both.) A gap thus implies one or more missing samples. The metric summarizes gaps rather than logging each one: it reports the count, the largest gap, and the index of the first gap — not a per-event list (see the corner cases). The metric *infers* loss from the expected cadence: it assumes a fixed rate, so a large interval could instead be jitter, an expected pause, or a clock adjustment; the gaps are missing-sample *candidates*, not certainties. The 1.5-period cutoff is a fixed v1 heuristic, not an externally established threshold.

**Measurement fields:**

- `max_gap_ns` — the largest gap interval (later minus earlier timestamp)
- `expected_period_ns`, `expected_hz` — the target period and rate
- `num_samples`, `num_gaps` — sample count and number of gap events
- `first_gap_index` — index of the first sample immediately following a gap (the later endpoint of the first gap interval), or `None` if there are no gaps

**Corner cases:**

- fewer than two samples, including empty → an undefined measurement (`is_defined == False`)
- an interval must reach 1.5 periods (the `interval >= (3 × period + 1) // 2` cutoff), so sub-period jitter (for example 1.3× the period) is not a gap; a backward or duplicate step is never a gap
- `max_gap_ns` is the raw gap interval (later − earlier timestamp); the excess over one period is derivable as `interval − expected_period_ns`
- the measurement is an O(1) summary, not a per-event log: `first_gap_index` locates only the *first* gap and `max_gap_ns` sizes only the *largest*, so the individual middle gaps are counted but not otherwise described. This mirrors `TimestampOrderingMetric` (which reports `first_decreasing_index`, not every backward step) and keeps space bounded on pathological input; a full per-event log can be added later if a consumer needs one.

**Streaming:** the metric instrument folds the stream — feed successive windows with `update(timestamps_ns=...)` and call `measurement()` to finalize. It carries only the previous window's last timestamp (to check the interval across a window boundary) plus the O(1) running summary, so its state is bounded regardless of how many gaps the stream contains.

## JitterMetric

Measures short-term variation (spread) in inter-sample timing, independent of the average rate.

**How it works:** the sample standard deviation of the inter-sample intervals, expressed as a percentage of the expected period. Non-monotonic steps are dropped first. This is deliberately different from `RateMetric`: a stream can have a correct mean rate (low `period_deviation_percent`) while still being unstable sample-to-sample (high `jitter_percent`).

**Measurement fields:**

- `jitter_percent` — interval standard deviation as a percentage of the expected period
- `expected_hz` — the target rate
- `num_samples`, `num_intervals` — sample count and intervals analyzed
- `num_filtered` — non-monotonic (`<= 0`) intervals dropped

**Corner cases:**

- fewer than two intervals → an undefined measurement (`is_defined == False`); `jitter_percent` is `nan`
- uses the sample standard deviation (`ddof=1`) and normalizes by the *expected* period, not the observed mean

## FrameReorderingPresentMetric

Records a caller-supplied frame-reordering flag — normally `VideoMetadata.has_bframes`, which reflects whether a video's header signals B-frames (for H.264). The metric never opens a video itself; it just carries the flag. B-frames complicate frame-accurate seeking and GPU decode scheduling.

**How it works:** records the frame-reordering flag the caller supplies as a plain boolean — normally `VideoMetadata.has_bframes`, which is libavcodec's `has_b_frames` (the reorder-buffer size the decoder parses from the header). The metric itself sees only that boolean: it does not open the video, read a header, or scan the stream. For H.264/AVC a non-empty reorder buffer reliably indicates B-frames. An authoritative maximum-consecutive count would require a whole-stream pass over every frame's type (far more than the single header field behind this flag), and the encoder's configured maximum (libavcodec `max_b_frames`) is unusable here because it is "decoding: unused" — `0` on a decoded stream.

**Measurement fields:**

- `has_reordering` — the supplied frame-reordering flag (B-frames for H.264; normally `VideoMetadata.has_bframes`)

**Corner cases:**

- no `update()` call → an undefined measurement (`has_reordering is None`, `is_defined == False`), like the other metrics on thin data (its minimum is one `update()`)
- reports presence, not a count of B-frames actually emitted or their longest run
- video-specific: it consumes a boolean frame-reordering flag (for example `VideoMetadata.has_bframes`), not a timestamp array
- input errors (non-boolean, or a second `update()`) are covered by the shared [Input contract](#input-contract)

## Ordering policy: non-decreasing vs strict increase

Non-decreasing and strictly increasing are not separate metrics — they are two evaluation policies over the single `TimestampOrderingMetric`, chosen by which field the evaluator reads:

- **non-decreasing** (duplicates allowed) → threshold `decreasing_count` at 0
- **strictly increasing** (no duplicates) → threshold `strict_violation_count` at 0

Keeping the measurement single and the policy in the evaluator keeps the catalog small and failures diagnostic — you always see the backward-step and duplicate counts separately, and you pick the strictness at check time rather than by choosing a different metric.

## Session-grain metrics

The two metrics below measure a **session**, not a stream: their subject is a set of sensors, and no one stream in it can carry the answer. They take recording bounds rather than timelines — one `(start_ns, end_ns)` pair per sensor, folded one sensor at a time — so a caller that has just measured a session's streams can feed them what it already has.

They live in a separate registry (`SESSION_INSTRUMENTS`) from the five above, because everything that iterates the per-stream registry does so once per stream. `di-session` and the Ray pipeline both run them, print or log them, and store them in the three session-grain Lance tables specified in [data-integrity-store-schema.md](data-integrity-store-schema.md) section 5. A failure exits `di-session` nonzero; the pipeline counts it, since findings never reach its exit status.

Bounds are validated at the boundary like every other input: a non-integer bound → `TypeError` (a `bool` included, since it is an `int` subclass), one outside `int64` nanoseconds → `ValueError`. Ordering is deliberately *not* required — a sensor whose last sample precedes its first is corrupt rather than unrepresentable, and each metric reports that through its own arithmetic instead of raising, leaving `TimestampOrderingMetric` to name the stream responsible.

### MultiSensorSpreadMetric

Measures how far apart a set of sensors begin and end — a session-level alignment signal across sensors.

Named for the spread rather than for a gap (the design doc's earlier `MultiSensorGapMetric`): what it measures is how far apart the sensors start and stop, and a second "gap" beside `TimestampGapMetric` — which is about missing samples *within* one stream — would name two unrelated things alike.

**How it works:** from each sensor's recording bounds (`start_ns` / `end_ns`), it takes the spread of the starts (`max(starts) − min(starts)`) and the spread of the stops (`max(stops) − min(stops)`). Large spreads indicate sensors that came up or shut down at very different times.

**Measurement fields:**

- `start_spread_ns` — spread of the sensors' start times (`max(starts) − min(starts)`)
- `stop_spread_ns` — spread of the sensors' stop times (`max(stops) − min(stops)`)
- `num_sensors` — number of sensors compared
- `max_spread_ns` (derived) — the worse of the two spreads; the field policy judges, since a session is as badly aligned as its furthest-out sensor

**Corner cases:**

- fewer than two sensors → an undefined measurement (`is_defined == False`); one sensor is never out of step with itself
- a sensor with no timestamps has no bounds to compare, and is not counted in `num_sensors`
- folding order does not matter; the metric is symmetric in its inputs

The bounds must share a clock across sensors for the spread to mean anything, and this metric declares `requires_capture_timestamps` for that reason: unless every contributing stream reported `capture`, it does not run at all and its verdict is `SKIPPED`. See [The session grain skips without a shared capture clock](#the-session-grain-skips-without-a-shared-capture-clock) for why a constant-frame-rate video source cannot satisfy it, and why the skip is recorded as *never ran* rather than as a measured zero.

### MultiSensorOverlapMetric

Measures how much of a recording session all sensors' recording bounds overlap — the overlap of bounding intervals. This is not a guarantee of usable data throughout: it compares each sensor's `start_ns` / `end_ns` only, so internal gaps can still leave little or no simultaneously-available data inside the overlapping window.

**How it works:** the effective duration is `min(stops) − max(starts)` (clamped to zero), the total session span is `max(stops) − min(starts)`, and the overlap fraction is their ratio. It looks only at the bounding intervals, never inside them.

**Measurement fields:**

- `overlap_fraction`, `non_overlap_fraction` — bounding-interval overlap fraction and its complement, or `nan` when undefined
- `effective_duration_ns`, `total_duration_ns` — overlapping-bounds span and full session span
- `num_sensors` — number of sensors compared
- `non_overlap_percent` (derived) — `non_overlap_fraction` as a percentage; the field policy judges, because a threshold here is always a ceiling and an overlap is the one quantity a session wants *more* of

**Corner cases:**

- fewer than two sensors → an undefined measurement (`is_defined == False`), with both durations reported as `0` rather than as the lone sensor's own span, which would read as a measured overlap
- when the latest start is after the earliest stop there is no common window, so the overlap is zero — a real finding, not an undefined one
- when the total session span is zero (every sensor shares the same zero-duration bounds) the ratio is undefined, so the measurement is undefined rather than a division by zero
- a non-*positive* total span is undefined for the same reason: it means the bounds contradict each other (some sensor's last sample precedes the first sensor's start). Both durations stay on the measurement, since they are what shows a reader which of the two situations it is
- a sensor with no timestamps has no bounds to contribute (as with `MultiSensorSpreadMetric`)
- folding order does not matter

Like `MultiSensorSpreadMetric`, this metric declares `requires_capture_timestamps`: a mux that restarts every camera's timeline at `0` reports total overlap for any rig at all, so without a shared capture clock the verdict is `SKIPPED` and the row is recorded as *never ran* rather than as a measured total, leaving a later policy nothing to rebuild. See [The session grain skips without a shared capture clock](#the-session-grain-skips-without-a-shared-capture-clock).
