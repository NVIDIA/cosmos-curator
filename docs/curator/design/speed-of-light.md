# Speed-of-Light Design

## Motivation

Pipelines without captioning are fairly lightweight — video decode, filtering, and preprocessing are embarrassingly
parallel and GPU-accelerated. Even transcoding, while not free, scales well. These stages are not the bottleneck.

Captioning is. With the current default model (Qwen 2.5 VL), autoregressive inference already dominates pipeline
wall-clock time. We are planning to move to larger models (e.g., Qwen 3.5-27B), which will only widen this gap.
Therefore, the pipeline's theoretical throughput ceiling — its "speed of light" — is determined entirely by how fast the
captioning model can produce output tokens. Optimizing anything else yields marginal returns.

The `vllm_async` captioning stage replaces the synchronous offline `LLM` with an in-process `AsyncLLM` engine
that supports continuous batching natively — sequences backfill as others complete, eliminating drain tails *within* the
engine's scheduler. This is the necessary foundation.

However, the engine still lives inside `process_data()`, which acts as a hard barrier. When a `process_data()` call
finishes and the stage waits for Xenna to dispatch the next batch of tasks, the engine has nothing to work on. The
barrier has moved up a level, but it is the same barrier. Closing this gap — keeping the engine continuously fed — is
the primary optimization target.

## The One Metric: Output Tokens/s

All optimization work should be measured against a single number:

> **Total caption output tokens generated / pipeline wall-clock time**

This metric captures everything — batching efficiency, pipeline bubbles, prefill overhead, queue stalls — in one ratio.
To evaluate how close the pipeline is to the hardware ceiling, compare this number against a standalone vLLM benchmark
on the same model and GPU configuration. The gap is the pipeline overhead.

### How to Measure

1. **Capture token counts.** vLLM's `RequestOutput` contains both `prompt_token_ids` (input tokens from the vision
   encoder + text prompt) and `outputs[0].token_ids` (generated caption tokens). Extract `len()` of each and store
   alongside the caption text as output metadata — token counts are a property of the generated data (useful for cost
   accounting, quality analysis, downstream filtering), not ephemeral perf instrumentation.

2. **Report in summary.json.** Aggregate the per-caption token counts into the existing summary output.

3. **Console output.** When captioning is enabled, print a throughput summary at the end of every pipeline run so the
   metric is immediately visible without digging into `summary.json`:
   ```
   Captioning throughput
   ─────────────────────────────────────
   total prompt tokens:       2,220,560
   total output tokens:         220,134
   total windows:                   444
   avg prompt tokens/window:      5,001
   avg output tokens/window:        495
   output tokens/s:               506.0
   ─────────────────────────────────────
   ```

4. **Time-series monitoring.** Xenna already exports per-stage metrics (progress, queue depths, actor utilization,
   process time) as Ray Gauges scraped by Prometheus and visualized in a pre-built Grafana dashboard. Emit token counts
   through the same path — e.g., cumulative output/input token counters per captioning stage — so that tokens/s can be
   plotted over time. This gives two things the aggregate summary.json number cannot: intra-run visibility (throughput
   dips between `process_data()` calls reveal the barrier gap directly) and cross-run trending (catch regressions,
   measure optimization impact). Input tokens/s is worth tracking alongside output tokens/s — a drop in input
   throughput points to vision-encoder / prefill bottlenecks, while a drop in output throughput points to decode-side
   stalls.

5. **Baseline benchmark.** Run `vllm bench throughput` on a single GPU to establish the per-GPU throughput ceiling.
   Use the pipeline's actual average prompt and output token sizes (reported in the captioning throughput summary) as
   the benchmark parameters so the workload profile matches. Scale linearly by GPU count to get the multi-GPU
   baseline:
   ```bash
   vllm bench throughput \
     --model <model_path> \
     --dataset-name random \
     --random-input-len <avg_prompt_tokens_per_window> \
     --random-output-len <avg_output_tokens_per_window> \
     --num-prompts 1000
   ```
   The ratio `pipeline_output_tokens_per_s / baseline_output_tokens_per_s` is the pipeline efficiency — the gap is
   the overhead from pipeline orchestration (batch barriers, queue stalls, data transfer).

## Approaches

### 1. Saturate the Engine from Pipeline Stages

Lowest-effort changes within the existing `CuratorStage` contract. The goal is to keep the `AsyncLLM` engine's batch
full for the duration of each `process_data()` call.

- Ensure the upstream prep/render stages produce inputs faster than the engine consumes them, so the engine never
  starves within a batch.
- Tune `max_inflight_requests`, `stage_batch_size`, and worker counts to maximize the number of concurrent requests
  in the engine's scheduler.
- Profile to find whether prefill (vision encoder) or decode (autoregressive) is the dominant component of each batch
  cycle, and tune accordingly.

### 2. Eliminate the `process_data()` Barrier in Xenna

Changes in the Xenna runtime to eliminate the inter-batch gap — the period between one `process_data()` return and the
next dispatch where the engine has no work.

- Reduce or eliminate the synchronization barrier between `process_data()` calls so new tasks can be dispatched before
  the previous batch fully drains.
- Allow the scheduler to over-provision input to GPU stages, maintaining a ready queue so the next batch starts
  immediately.
- This is the harder problem and likely requires Xenna-level changes, but it is where the remaining throughput is
  hiding.

### 3. Disaggregated ViT Encoding (vLLM EPD)

Separate the vision encoder (ViT) from the autoregressive decoder into distinct pipeline stages.

- ViT encoding is compute-bound; AR decode is memory-bandwidth-bound. They have different scaling profiles and compete
  for GPU resources when co-located. Disaggregating lets each be independently scaled and scheduled.
- The AR stage receives pre-computed visual embeddings instead of raw frames, eliminating prefill bubbles — it can start
  decoding immediately.
- **vLLM already supports this natively.** The EPD (Encoder-Prefill-Decode) feature, merged in v0.11.1, runs the vision
  encoder and LLM as separate vLLM server instances. An Encoder Cache (EC) connector transfers embeddings between them
  via shared storage (filesystem or shared memory). Each instance scales independently on separate GPUs.
- Published benchmarks show **2-2.25x throughput** on multi-image workloads and up to **57% end-to-end improvement**
  offline (4x A100, Qwen3-VL-4B). Gains grow with image count and model size.
- References: [vLLM EPD docs](https://docs.vllm.ai/en/latest/features/disagg_encoder/),
  [arXiv:2501.05460](https://arxiv.org/abs/2501.05460).
- Open question: whether to use vLLM's built-in EC connector for embedding transfer or integrate via Ray Direct
  Transport for tighter coupling with the pipeline's data flow.

### 4. Ray Serve + Ray Data

Full architectural shift: vLLM behind Ray Serve, pipeline on Ray Data.

- Ray Serve can host vLLM with autoscaling and continuous batching.
- Ray Data streaming datasets naturally produce backpressure-driven pipelining without explicit batch barriers.
- This is the most invasive change but aligns the pipeline with the direction of the Ray ecosystem and removes the
  stage-barrier problem structurally.

## Task List

- [x] **Instrument token counting** — extract output token counts from `RequestOutput` in `process_vllm_output()`,
  accumulate, and write to `summary.json`.
- [ ] **Export token metrics to Grafana** — emit cumulative input/output token counters as Ray Gauges from the
  captioning stage; add a tokens/s panel to the existing Grafana dashboard.
- [ ] **Establish baseline** — run `vllm bench throughput` on a single GPU for each supported captioning model, using
  `--random-input-len` / `--random-output-len` from the pipeline's avg prompt/output tokens per window; scale
  linearly by GPU count.
- [ ] **Measure current gap** — run a representative pipeline (e.g., 4k videos, 4 nodes) and compare
  `output_tokens_per_s` against the baseline.
- [ ] **Tune async stage saturation** — experiment with `max_inflight_requests`, `stage_batch_size`, and upstream
  worker counts to maximize engine utilization within each `process_data()` call.
- [ ] **Eliminate `process_data()` barrier** — identify what Xenna changes would allow continuous task dispatch to the
  engine without waiting for the previous batch to complete.
- [ ] **Prototype vLLM EPD** — deploy encoder and PD as separate vLLM instances using the built-in EC connector; measure
  throughput improvement on representative workloads.
- [ ] **Evaluate Ray Serve + Ray Data** — prototype the full architectural shift and measure throughput vs. complexity
  trade-off.
