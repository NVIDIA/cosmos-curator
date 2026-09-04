# Data Integrity Design

## Goal

Define the design for `cosmos_curator.core.sensors.data_integrity` — a generic data-integrity framework for sensor-library use, not a pipeline-specific validation layer.

This document scopes the first, high-confidence deliverable: the **metric and evaluation kernel** — the streaming measurement instruments and the pure PASS/FAIL evaluators that judge them. The orchestration that runs the kernel at recording scale — and the full-fidelity sensor input path it consumes — is deliberately out of scope for this first release — a set of open design problems for later; see [Future integration](#future-integration). Leaving it out keeps the first release small and lets that layer be designed against real sensor-input experience rather than imagined usage.

The design aim is not to get the public API right the first time — it is to keep the public surface small and the churn-prone parts (evaluation policy, and later whatever configuration surface the orchestration grows) isolated, so that inevitable changes stay cheap and local rather than forcing a wide migration.

The metric catalog lives in a companion doc, [data-integrity-metrics.md](data-integrity-metrics.md).

## Scope (v1)

v1 is the measurement + evaluation kernel: the metric instruments and the evaluators that judge their measurements. It is generic across sensors, with one deliberately video-specific metric, and is clearly open-sourceable:

- single-sensor timing metrics (timestamp ordering, rate deviation, gaps, jitter)
- one video-specific single-sensor metric (frame-reordering / B-frame presence), included because it is easily implemented and broadly useful
- pure PASS/FAIL evaluators (`below_threshold`, `above_threshold`, `within_range`) over metric measurements
- unit tests with synthetic fixtures

The kernel deliberately stops at the metric/evaluator boundary. The orchestration that runs it, and the full-fidelity sensor input path it consumes, are left as open design problems for a later release (see [Future integration](#future-integration)). Multi-sensor metrics were out of scope for v1 and have since been added, along with the session-grain tables that store their measurements and verdicts (see [Metric Categories](#metric-categories)). What is out of scope entirely is captured in [Non-Goals](#non-goals).

## Design Principles

### Generic Metrics

Metrics should be generic and reusable.

That means:

- a metric computes a fact about data
- a metric should not encode a pass/fail policy
- a metric should avoid modality-specific assumptions unless it is explicitly a modality-specific metric
- metric names and fields should describe what was measured, not how it will be judged

For example:

- `RateMetric` measures deviation from expected cadence
- `TimestampOrderingMetric` classifies increasing / equal / decreasing steps
- `MultiSensorOverlapMetric` measures session-level overlap

This is the level of abstraction for `core.sensors.data_integrity`.

### Separate Measurement From Evaluation

The framework should preserve a hard split between:

- metric computation
- evaluation policy

This gives reuse properties:

- the same metric can be evaluated with multiple thresholds
- metric values can be inspected even when no pass/fail judgment is needed
- templates can define standard evaluations without making the metric itself policy-bound

### Open-Source-Safe Constants and Sources

- metrics are standard, publicly known operations — no proprietary or novel algorithm
- code, comments, tests, and docs cite only public sources — never internal systems, branches, or proprietary datasets or tuning
- default constants (for example, the gap threshold at which a missing interval is treated as a dropout) must be justifiable from first principles or a public reference, or exposed as neutral, user-configurable defaults — never a value tuned on a proprietary dataset

### Package Layout

The reusable framework lives at:

`cosmos_curator/core/sensors/data_integrity/`

The v1 kernel is two modules:

- `metrics.py`
  - metric instruments and their immutable measurement types
- `evaluation.py`
  - evaluators (`below_threshold`, `above_threshold`, `within_range`) and evaluation results

Three modules have since joined them, all still backend-agnostic: `results.py` (the vocabulary a result is expressed in), `instruments.py` (which threshold applies to which measurement) and `engine.py` (running every metric over one already-open sensor and judging the lot). A CI test (`test_kernel_boundary.py`) keeps the package free of `argparse`, `lance` / `pyarrow` and cloud clients, so the sensor library can depend on it without paying for a workflow.

The orchestration that was out of scope for v1 now exists, and lives outside the sensor library at:

`cosmos_curator/next/recipes/data_integrity/`

That package owns the `di-check` and `di-session` CLIs, stream discovery, the concurrent session runner, report rendering and the Lance result store ([store schema](data-integrity-store-schema.md)) — the half that takes URIs, credentials and command-line flags. What remains undecided is the orchestration *above* a single session: declaring checks as data, and driving many metrics from one read — see [Future integration](#future-integration).

Repo convention note:

- `__init__.py` files in this codebase do not re-export package symbols
- callers should import from concrete module paths

## Core Data Model

Metric logic depends on plain arrays, not on any sensor interface. This keeps
measurement decoupled from the sensor model and avoids inventing abstractions
that concrete sensors do not implement.

### Metrics Consume Arrays, Not Sensors

A metric is a function of one or more array arguments plus scalar params (its
constructor). Its signature is the contract, and it knows nothing about sensors.
The array arguments are not constrained to one dtype or one dimension: often 1-D
timestamps (`int64`), but a value-domain metric may take a float array, and lidar
checks may take multi-dimensional arrays (point clouds, range images).

```python
RateMetric(expected_hz=...).update(timestamps_ns=ts)        # array arg: timestamps_ns; param: expected_hz
RangeBoundsMetric(low=..., high=...).update(values=values)  # array arg: values;        params: low, high
```

Resolving each array argument to a concrete array — from whichever sensor
attribute a caller names — is out of the metric's scope: the metric measures
whatever `update(...)` hands it. That resolution is exactly what the future
orchestration and sensor adapters will own (see [Future integration](#future-integration));
keeping it out of the metric is what lets the same metric run over a recording, a
sampled payload, or an aligned frame without change.

## Metrics

The full metric catalog — including planned and modality-specific metrics — lives in the companion [metrics doc](data-integrity-metrics.md). This section defines the metric *shape*, not the catalog.

### Metric Shape

A metric is a mutable measuring *instrument*; the value it produces is an
immutable **measurement** object with typed fields. The instrument carries the
running state (that is what `update` is for); the measurement carries facts only.

A measurement object should:

- contain the measured values
- be independent of thresholds
- report `is_defined` — whether the input met the metric's computational minimum so the values are well-defined (the discrete analogue of `NaN`); callers must not evaluate an undefined measurement
- be serializable or easily convertible to plain data later

Measurements share one small protocol; metrics do not (each is a concrete
instrument following the same `update` / `measurement` shape, with no shared base
protocol):

```python
class Measurement(Protocol):
    @property
    def is_defined(self) -> bool: ...


# Each metric is a concrete class following this shape:
metric = RateMetric(expected_hz=30.0)
metric.update(timestamps_ns=window)   # fold one window (repeatable)
result = metric.measurement()         # finalize the immutable facts
```

Each built-in metric defines its own instrument plus the concrete measurement it produces.

Examples:

- `TimestampOrderingMeasurement`
  - `decreasing_count`, `duplicate_count`
  - `first_decreasing_index | None`, `first_duplicate_index | None`
  - `strict_violation_count` (derived)
- `RateMeasurement`
  - `period_deviation_percent`
  - `expected_hz`
  - `actual_mean_hz`
- `MultiSensorOverlapMeasurement`
  - `overlap_fraction`
  - `non_overlap_fraction`
  - `effective_duration_ns`
  - `total_duration_ns`

### Metric API

A metric is constructed with its scalar params, folds one or more windows with
`update(**arrays)`, and finalizes with `measurement()`. The instrument is
mutable; the measurement it returns is frozen. A one-shot measurement over a whole
array is a single `update()` followed by `measurement()`.

Recommended pattern — a frozen measurement plus the instrument that produces it:

```python
@attrs.define(frozen=True)
class RateMeasurement:
    period_deviation_percent: float
    expected_hz: float
    actual_mean_hz: float

    @property
    def is_defined(self) -> bool: ...


class RateMetric:
    def __init__(self, *, expected_hz: float) -> None: ...

    def update(self, *, timestamps_ns: npt.NDArray[np.int64]) -> None:
        ...

    def measurement(self) -> RateMeasurement:
        ...
```

### Units

Raw and discrete time-domain values are nanoseconds in the `int64` range, matching `core.sensors` (which uses `int64` ns throughout): array fields are `np.int64`, while scalar discrete fields (for example the configured period and the largest gap) are Python `int` constrained to that range. Aggregate statistics computed over them are float nanoseconds, since a mean or spread is not generally an integer (for example `actual_mean_period_ns`). Non-time fields keep their natural unit: rates in Hz (`expected_hz`, `actual_mean_hz`), ratios as a percent or fraction (`period_deviation_percent`). The unit is always explicit in the field name; the unit follows the quantity.

### Metric Categories

Metrics come in two arities:

- single-sensor metrics
- multi-sensor metrics

The distinction is a property of the metric (how many sensors it consumes), not a
type hierarchy and not separate metric classes — metrics stay flat, and both
arities are the same `InstrumentSpec`.

v1 shipped single-sensor metrics only. The two multi-sensor metrics
(`MultiSensorSpreadMetric`, `MultiSensorOverlapMetric`) have since been added: they
fold one sensor's recording bounds per `update()` rather than a window of one
sensor's timeline, so the instrument shape carried them unchanged. What arity did
cost is a second registry — `SESSION_INSTRUMENTS` beside `INSTRUMENTS` — because
every caller that iterates the per-stream registry does so once per stream, and a
session-arity spec in there would have each of them measure a session per stream.

Persistence cost the same split for the same reason: a session-grain verdict is keyed
by session rather than by stream, so it lands in three tables parallel to the
per-stream ones instead of widening them. See
[data-integrity-store-schema.md](data-integrity-store-schema.md) section 5.

## Streaming and Windows

A sensor recording can be far larger than memory (a lidar file may be tens of GB), so metrics must measure over a stream of windows rather than one in-memory array — and a single read should be able to feed many metrics at once.

### The instrument is the accumulator

A metric is a stateful instrument, so streaming needs no second abstraction: the same object folds windows and produces the measurement.

- `update(**arrays)` — fold one window (the metric's named array arguments)
- `measurement()` — finalize the immutable measurement

The instrument carries whatever **seam state** stitches consecutive windows — most often the previous window's last timestamp, so the step across a window boundary is classified correctly, plus a running sample offset so reported indices stay global. State is O(1) in the stream length for every metric: each keeps only running summaries (counts, extremes, the first offending index), never a per-event log. A one-shot measurement over a whole array is a single `update()` + `measurement()`, so the math has one source of truth.

### One read, many metrics

A single pass over the windows can feed many instruments at once — a present property of the instrument design, not something a future layer adds. The illustrative loop runs against the kernel as shipped:

```text
metrics = [TimestampOrderingMetric(), RateMetric(expected_hz=...), JitterMetric(expected_hz=...)]
for window in windows:                 # consecutive, in-order windows over the full data
    for metric in metrics:
        metric.update(timestamps_ns=window.timestamps_ns)
measurements = [metric.measurement() for metric in metrics]
```

What a future orchestration layer would add is not this capability but the automation of it: resolving each metric's array inputs from a sensor (via some source mapping) and packaging the results, instead of hand-writing this loop. The metrics themselves just fold whatever `update(...)` hands them; there is no `metric.inputs` introspection in the code today.

Where the windows come from is deliberately out of scope here: the data-integrity kernel consumes a stream of consecutive, in-order, full-fidelity windows and does not care how they were produced. Wiring a concrete sensor to *produce* that stream — a full-fidelity read, distinct from lossy sampling — is a separate step, tracked as a sensor-library follow-up (see [Future integration](#future-integration)).

### Streamability notes

Most metrics are cleanly online (running sums, min/max, Chan's parallel-variance algorithm for jitter). One needs a note:

- **Adjacent duplicates** — `TimestampOrderingMetric` counts *adjacent* duplicates, which is O(1) state; exact global duplicate detection would need every value seen (unbounded). That is not a loss for the property that matters: `decreasing_count == 0 ∧ duplicate_count == 0 ⟺ strictly increasing ⟺ globally unique`. A global duplicate census on unordered data is a separate, non-streamable concern.

## Evaluation

### Evaluation Result

Evaluation results should be small immutable objects that capture:

- status — `PASS` or `FAIL`
- margin — signed distance from the threshold: positive means headroom (passing with room to spare), negative means the amount by which the value is outside the acceptable region

Recommended shape:

```python
class EvaluationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@attrs.define(frozen=True)
class EvaluationResult[T: (int, float)]:
    status: EvaluationStatus
    margin: T
```

### Evaluator Functions

Evaluators should be functions, not classes:

Recommended built-ins:

- `below_threshold`
- `above_threshold`
- `within_range`

Each evaluator should:

- accept a metric measurement object
- accept an accessor or field-selection mechanism
- return an `EvaluationResult` with status `PASS` or `FAIL`

Evaluators never return `UNDEFINED` and never reason about data sufficiency: they assume a well-defined measurement. The caller checks `measurement.is_defined` and skips evaluation when it is False. Keeping evaluators to `PASS` / `FAIL` is what keeps them generic.

## Future integration

The kernel above measures and judges. Everything needed to *run* it at recording scale is deliberately not designed here — it should be designed once the full-fidelity sensor-input path exists and real usage informs the contracts, rather than around imagined usage.

This section names the **problems** that layer must solve, and the **constraints** the kernel imposes on any solution — not the solutions themselves. The shapes suggested below are examples to make the problem concrete, not commitments; the design is open for whoever picks it up.

Open problems, in rough dependency order:

- **Full-fidelity sensor input** — producing a stream of consecutive, in-order, full-fidelity windows (distinct from lossy sampling) from a concrete sensor, for the instruments to fold. This is separate sensor-library work, and the prerequisite for any recording-level check.
- **Capture timestamps in preference to container PTS** — `CameraSensor.timestamps_ns` returns presentation timestamps, which a constant-frame-rate mux synthesizes on a fixed grid starting at `0` per file. Against such a source the timing metrics and both multi-sensor metrics report a perfect score whatever the hardware did, so the result says nothing was measurable rather than that the data is healthy (see [Timestamp provenance](data-integrity-metrics.md#timestamp-provenance) for the measured comparison). Also sensor-library work, and what makes the timing suite meaningful on video.
- **Composing metrics into runnable units** — how a metric, an evaluator, and a sensor selection combine into something nameable and reusable (a "check").
- **Declaring and configuring checks** — how checks are described as data and loaded. Format and schema are undecided.
- **Driving many metrics from one read** — large recordings (a lidar file can be tens of GB) mean a single pass over the windows must feed many instruments at once; how that pass resolves each metric's array inputs from a sensor and packages the results is open.
- **An end-to-end entry point** — running a configured set of checks against a sensor (a CLI, or otherwise).

Constraints any solution must respect:

- an undefined measurement (`is_defined == False`) must not be evaluated; how the orchestration surfaces that (for example, a check-level status distinct from `PASS` / `FAIL`) is its own call
- the input is a stream of full-fidelity windows, larger than memory; one read should be able to drive many metrics, so buffering the whole recording is not an option
- measurement stays separate from evaluation policy (the split the kernel is built around)

## Testing Strategy

### Unit Tests

Use synthetic timestamp arrays.

Cover:

- perfect cadence
- jitter
- large gaps
- duplicate / non-monotonic edge cases where applicable
- evaluator margin sign and boundary/equality cases
- input-contract rejection (non-1-D, non-int64) and `int64` overflow rejection on interval math

## Acceptance Criteria

The v1 kernel is complete when:

- `cosmos_curator.core.sensors.data_integrity` exists as a new package with `metrics.py` and `evaluation.py`
- the v1 metric set and the `below_threshold` / `above_threshold` / `within_range` evaluators are implemented
- every metric reports `is_defined` and never raises on thin data
- unit tests cover the metrics and evaluators with synthetic fixtures
- the metric catalog is documented ([metrics](data-integrity-metrics.md))

## Non-Goals

- a general plugin/discovery architecture
- pipeline-specific orchestration or workflow integration
- a broad reporting platform (beyond structured Python results)
- a general modality-specific validation framework or plugin system (individual modality-specific metrics, such as the video frame-reordering check, are in scope)
- a replacement for all internal data-quality tooling

The framework should stay small, explicit, and generic enough that the first implementation is straightforward. Coverage — and the orchestration in [Future integration](#future-integration) — can expand after that foundation is in place.
