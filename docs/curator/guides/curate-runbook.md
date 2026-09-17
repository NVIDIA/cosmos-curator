# Curate Runbook — run the selection leg

Operator guide for running the **Curate** leg (Curator Next): shrink a large,
redundant clip corpus into a compact, de-duplicated, task-balanced selection.
This is the leg that runs **after** embedding — it reads `clips.lance`, fuses
three embedding blocks into one distance, clusters, optionally drops
near-duplicates, balances the survivors across fairness groups, and writes the
verdict back onto the rows it read.

**Wide table in, wide table out.** A successful run adds two columns to the table
it read and commits one new Lance version. There is no selection side table, no
staging tree, and no report file in that committed output — so nothing extra to
name, garbage-collect from what the version references, or correlate afterwards
except the table's own version number.

- What one run does and its configuration: [`recipes/curation/README.md`](../../../cosmos_curator/next/recipes/curation/README.md)
- Architecture, schemas, distributed model: [Curator Next Curation](../design/curator-next-curation.md)
- The cluster substrate this runbook launches on: [Managed Ray Clusters on Slurm](../design/curator-next-slurm-ray.md)

## Entry point

Curate is the `curate` pipeline kind, so it is launched by the generic pipeline
CLI from a config file. There is no recipe-specific runner and no flag set:

```bash
cosmos-curator pipeline template curate > curate.yaml   # edit clips_lance_uri and target
cosmos-curator pipeline validate curate.yaml           # config faults, no table read
cosmos-curator pipeline render   curate.yaml           # every resolved value, defaults included
pixi run --as-is -e cuml run-pipeline curate.yaml      # execute
```

Only `cosmos-curator` is an installed script; `run-pipeline` is a Pixi task, so
it needs the `pixi run` prefix unless you are already inside a `pixi shell`. The
*Path A* section below explains the environment choice.

**The link from a table version back to its rules is stored, not a convention you
maintain.** The commit stamps a digest of the config, and the centroids artifact
beside the table archives that config as text under the same
`centroids_fingerprint`, so
"what was version 47 selecting for?" is answerable from the table alone — see
[Reading a version's identity back](#reading-a-versions-identity-back).

Keeping the config file under version control is still worth doing, for review and
for re-running, but it is no longer the only record of a selection.

`validate` and `render` are config-only: they resolve the file and stop, without
opening the table. Data-contract checks are the run's own driver preflight, which
still fails in seconds before any GPU work.

Individual settings are adjustable per invocation without editing the file:

```bash
pixi run --as-is -e cuml run-pipeline curate.yaml \
  --set target.target_count=25000 --set dedup_eps=null
```

`--set` assigns into the loaded mapping *before* validation, so an overridden
value is checked by the same rules as a written one. Dotted paths reach nested
fields. Values are read as YAML, which is how `null` above disables
de-duplication and why a quoted `"25000"` is rejected rather than coerced.

A dotted path sets one leaf and *merges* with what the file already wrote, so
switching between the two mutually exclusive `target` forms takes two overrides.
The line above works on a file that writes `target_count`; against one that
writes `target_fraction` it is refused, because both forms would then be set:

```bash
pixi run --as-is -e cuml run-pipeline curate.yaml \
  --set target.target_count=25000 --set target.target_fraction=null
```

The refusal names the invariant rather than the fix, so reach for this shape
whenever an override switches a form instead of adjusting one in place.

`--set` reaches **every** field, not only the operational ones. That includes the
result-defining ones — `weights.*`, `kmeans_random_state`,
`fairness_residual_seed`, `merge_theta_task`, `subtask_clusters`,
`within_group_order` — and nothing stops you. Overriding one
is the case this guide argues against throughout: the committed version records
the run identity, but a shell-only override does not appear in the checked-in
config diff. A reviewer can recover it from the committed version later, but not
from the file review alone. Treat `--set` as being for the fields that do
not change what a run means, and put the rest in the file. `weights` deserves
particular care, because a zero weight also drops its block from the eligibility
predicate: raising one from zero can take a large part of the corpus out of the
eligible set, and the newly ineligible rows come back NULL rather than keeping the
previous run's verdicts. That is safe but not reversible by accident — see
[Re-runs and recovery](#re-runs-and-recovery).

The fusion `weights` are the one part of the config the template deliberately
omits: they define the distance the whole corpus is clustered on, so changing one
makes two runs incomparable. Set them as a reviewed change, never as a per-run
adjustment.

Before reaching for the weights, note what the fused distance is for. It answers
"is this recording redundant", and its two consumers are locality clustering and
near-duplicate detection. It is **not** a task signal, and the action block in
particular is measurably task-blind (task-separation ratio 0.990, same-task-closer
AUC 0.543, against 0.853 for text and 0.879 for image on the same 131,602 clips
with repaired labels). So lowering `action` will weaken duplicate detection
without making anything more task-accurate, and raising it will not help a task
question either. Anything you want grouped or balanced *by task* is served by the
canonical `task_name` label, which the fairness path reads directly and which
no weight affects. Level-2 fairness uses `embedding_text_subtask` cells. See
[the curation design doc](../design/curator-next-curation.md).

### Config fields

`cosmos-curator pipeline schema curate` is authoritative — it publishes the model
itself. The bounds and defaults below are a convenience copy of it, so check the
schema before relying on one. `pipeline render` prints the resolved values of one
file, defaults included.

Every field below except `clips_lance_uri`, `storage_profile` and
`dedup_concurrency` is **result-defining**: two runs that differ in one of them are
not comparable. That includes the two envelope keys. The table's own address is
excluded so a table copied to a second URI still reports the identity of the rules
that produced it.

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | Always `1`. Required rather than defaulted, so a config written for a later generation is rejected instead of read under this one's field meanings. |
| `kind` | yes | Always `curate`. The discriminator the CLI routes on. |
| `clips_lance_uri` | yes | The `clips.lance` to read **and widen**. Input and output are the same table. |
| `target.target_count` | one of | Keep this many survivors (clamped to the population). Mutually exclusive with `target_fraction`. |
| `target.target_fraction` | one of | Keep this fraction of **survivors** in `(0, 1]`, rounded half-up and never below 1. Neither target ⇒ keep every survivor. |
| `weights` | no | Per-block fusion weights (`subtask` `0.6`, `image` `0.2`, `action` `0.2`), which must sum to 1. The one field the template omits on purpose — see above. |
| `within_group_order` | no | Tie-break inside a funded fairness group: `farthest` (default, atypical-first), `nearest`, or `neutral` (`clip_id` only). "Atypical" means unusual instruction, appearance, and motion — not an unusual task; task spread comes from the quotas that run *before* this ordering, not from this field. |
| `target_mean_cluster_rows` | no | Target mean rows per cluster (default `200000`); `k = ceil(eligible_rows / this)`. The only cluster-count control. |
| `fit_sample_rows` | no | Budget for the single-GPU k-means fit sample (default `4000000`), counted in **eligible** (predicate-passing) rows rather than physical ones. It bounds the fit's host and device matrices, and also the *read*: the sample is a manifest prefix that crosses the budget, so a sparsely-eligible corpus extends the prefix further down the manifest to reach that many usable rows. Raising it therefore costs read volume as well as memory — raise it only if the fit task has both to spare. See [Why the fit runs on a sample](../design/curator-next-curation.md#why-the-fit-runs-on-a-sample). |
| `kmeans_random_state` | no | Seed for the k-means fit (default `42`). |
| `fairness_residual_seed` | no | Seed for the residual order at **both** fairness levels (default `0`). Decides which tasks win the leftover clips, and inside each funded task which of its subtask cells do; at a target below the task count that is the whole allocation, so it decides which tasks appear at all. Change it to draw a different, equally valid subset; two runs differing only here are both correct and not comparable. Deliberately separate from `kmeans_random_state`, which keys a persisted centroids artifact: sharing one seed would make "redraw the subset" require refitting the basis. |
| `dedup_eps` | no | Duplicate above `1 - eps` similarity (default `0.01`); larger drops more near-duplicates. Either `null` (skip) or a float in `(0, 1]`: the lower end is open because the retention test is strict, so `0` would drop nothing while still paying for every GPU comparison, and the upper end is `1` by policy — `1 - cos` spans `[0, 2]`, but larger `eps` would mark almost every pair a duplicate, so values above `1` fail at parse time. Omitting the key keeps the default, so de-duplication **runs**; skipping is opt-in. |
| `dedup_concurrency` | no | Cap on concurrent whole-GPU per-cluster dedup tasks; `null` (default) fans out to every free GPU. Scheduling only — the retention result does not depend on it. Ignored when `dedup_eps` is `null`. |
| `merge_theta_task` | no | Cosine above which two **task** labels become one fairness group (default `0.95`). |
| `subtask_clusters` | no | Level-2 fairness cells: how many k-means clusters the subtask text embedding is partitioned into (default `16`). The level-2 group ceiling is `merged tasks x (this + 1)`. Two ceilings bound it and compute is neither, though it is not free — the subtask fit sample is `min(fit_sample_rows, k x 100,000)`, so it grows with `k` and then saturates. The two are `k <= target_count / merged tasks` — a pre-run number only when `target_count` is set, and otherwise a post-run check against the resolved keep-count; above it a task's cells are funded by `fairness_residual_seed` rather than by the corpus — and `k` well below the subtask spellings inside one task, estimated at ~40 (above it the partition enumerates wordings instead of grouping them). At the 250M envelope the second binds, and since it is an estimate rather than a measurement **the only supported direction is down** — move it only when a scarce target pushes the first ceiling below 16. See [Choosing `subtask_clusters` before the run](#choosing-subtask_clusters-before-the-run). |
| `storage_profile` | no | Storage profile for the table (default `default`). |

`dedup_eps: null` is how a run asks for **no de-duplication at all**, and it is
not the same as a tiny `eps`: the stage does not run. It **changes verdicts** —
see [When to skip de-duplication](#when-to-skip-de-duplication).

`CurateConfig` is `extra="forbid"`, so a config naming a removed key —
`staging_root`, `selection_label`, `n_clusters`, `pool_size`, `merge_theta_subtask`,
`subtask_cluster_k` — fails at parse time
with that key named rather than running with a default you did not choose.
`pipeline validate` is the cheapest place to find that out.

`subtask_cluster_k` is the renamed one, and it has **no alias**: a config carrying
the old key fails with that key named rather than silently taking the new field's
default. Rename it to `subtask_clusters`.

### What the target does to the shape of the selection

The target is not only "how many clips" — it decides how strongly the run rebalances at
all. The allocator finds one integer fill line `L` and sorts every group into two
populations: groups at or under `L` are kept **in full**, groups above it are cut to `L`
rows regardless of size, plus at most one more if the residual reaches them. A lower target
lowers `L`, which moves more groups into the second population. The full derivation is in
the [design doc](../design/curator-next-curation.md#allocation-integer-max-min-quotas-applied-twice);
what follows is what it means for a target you are about to set.

| `target / survivors` | What the output looks like |
|---|---|
| near 0 | `L` falls to zero. Most groups contribute **nothing at all** and exactly `target` groups are funded one clip each, chosen by `fairness_residual_seed`. Maximum flattening, but also maximum exclusion — this is the regime the degeneracy WARNING covers. |
| middle | Small groups pass through whole while the largest are trimmed to `L`. The head is flattened, the tail keeps its shape. |
| 1 (or omitted) | Every group is saturated. **The fairness stage is a no-op** and the run is de-duplication only. |

Two consequences worth expecting rather than discovering. First, a target at or above the
survivor count is a no-op, because the target clamps to the population. The run says so on
one line before any verdict is written — `curate fairness: N survivor(s) across G merged
group(s), target=N` — and the verdict counts confirm it: no `below_quota` **and** no
`unfunded`. Both are needed. Absent `below_quota` alone means only that every group was
funded to its capacity *or* to zero, which is also what a corpus of one-survivor groups
produces at a scarce target, and that run excluded most of its groups rather than keeping
everything. Second, equal *counts* are not equal *rates*: at a low target a 50-clip group
and a 1,000,000-clip group both contribute `L`, so their retention rates differ by orders of
magnitude. This design equalizes counts on purpose; if you need rates, no target setting
will give them to you.

## Prerequisites

- **Input exists and passes preflight.** `clips.lance` must carry the vector columns
  for every weighted block, **plus `embedding_text_task` whatever the weights are** —
  the task label merge runs on every run — and `clip_id` and `task_name` must both be
  `string`. `embedding_text_subtask` is required exactly when the `subtask` weight is
  above zero, which is also when the level-2 fairness cells are fitted.
  `subtask_name` is not read. At least one row
  must satisfy the eligibility predicate. Preflight validates **data contracts
  only** and fails on the driver in seconds, before any Ray or GPU work:
  `ValueError` for a contract failure, `FileNotFoundError` for an absent table.
  Fragment geometry and per-modality NULL counts are logged, not asserted.
- **One producer per consumed embedding group.** Every group the run reads must
  carry its provenance columns and name no more than one identity in each — one
  text model, one image model, one action descriptor version and PCA fingerprint.
  A table filled by two producers is refused, because the run would compare both
  in one cosine space where their vectors share no geometry while every distance
  it yields still looks plausible. Two identities in one group normally means an
  embeddings rebuild is still in flight: re-run the embeddings leg to completion —
  an ordinary run recomputes every row whose recorded identity names another
  computation — then confirm the group reports one identity before curating. The
  resolved identities appear in the preflight log and in the centroids artifact.
  The identity is a digest of the whole computation, so a re-pin or a pooling
  change is visible to this gate rather than passing it silently. The requirement
  is per **column** and is read against the
  vectors this run actually consumes, so the action group must record both its
  descriptor version and its PCA fingerprint, and the text group is judged on
  `embedding_text_task` always plus `embedding_text_subtask` only when that block
  carries weight. Two states look alike from outside and are treated
  differently: a group that is present but entirely unfilled names no producer
  and is **not** refused here — the eligible-row count is what reports it — while
  a group that holds vectors this run reads but leaves a provenance column NULL
  **is** refused, naming that column and the number of rows affected. The remedy
  for the second is the same as for a double fill: re-run the embeddings leg, which
  selects any row whose provenance is NULL and rewrites the vector and its
  attribution together.
- **Environment: `cuml`.** The fit and the per-cluster dedup are the only GPU
  steps and run cuML/cuPy; everything else is CPU Ray Data. Both GPU stages name
  the `cuml` pixi environment through their runtime environment, and the driver
  should be launched there too — `pixi run -e cuml` (or `--as-is`).
- **GPUs.** Reference hardware is 8x H100-80GB with 1 TB host RAM per node.
  Single-node and multi-node allocations are both supported; a smaller corpus runs
  on fewer GPUs. GPU memory no longer scales with the corpus — the fit is sampled
  and dedup is bounded by the largest single cluster. On an 80 GiB card that bound
  is **11,152,540 rows** in one cluster, roughly 56x the default mean; a cluster
  above it makes the run refuse rather than OOM (see
  [When one cluster is too large to score](#when-one-cluster-is-too-large-to-score)).
  A GPU is required even at `dedup_eps: null`: the fit still runs, because
  `curate_cluster_id` and the default within-group ordering both come from it.
- **Host RAM per concurrent dedup task** (not applicable at `dedup_eps: null`). A
  dedup task also holds the cluster's
  Arrow-decoded matrix and its retention-order gather at once — about twice the
  cluster's vectors, ~77 GB for a cluster at the device ceiling — and *nothing
  refuses on that number*. Size a node for one cluster per GPU, or cap the fan-out
  with `dedup_concurrency`.
- **Driver memory scales with fragment *count*, not with corpus size.** Every run
  writes both columns on **every** fragment, so the driver collects one metadata
  payload per fragment before it commits — measured at ~740 bytes on a nine-column
  table, most of it the fragment's own file metadata. A fragment is one producer
  publish batch — `clips_per_publish_batch`, default 8,000 rows — so 250M rows is
  ~31,250 fragments and ~22 MiB of payloads, and 500M rows ~62,500 and ~44 MiB.
  That is a *lower* bound, since a production fragment references more data files
  than the measured fixture, and a table compacted into ~1,000-row fragments would
  put 250M rows into 250,000 fragments and roughly 176 MiB on the driver.
  *Nothing refuses on this number.* Larger fragments cut it, and cut per-fragment
  task overhead in both read passes — but they are not free, because the same
  division sets the fit sample's cluster count. Read
  [Before raising the producer's publish batch](#before-raising-the-producers-publish-batch)
  before treating fragment count as a number to minimize.
- **Node-local scratch, not a shared filesystem.** There is no `staging_root`
  and no shared-FS prerequisite. What scales is the `__dedup_key` shuffle: roughly
  865 GB at the 250M-row envelope, routed through the Ray object store and
  spilling to **node-local** scratch, plus two ~20 GB `materialize()` barriers
  cluster-wide. Size node-local scratch — it is the limit a run hits first, and
  curation deliberately does not check it. That shuffle is the de-duplication
  stage's own grouping: at `dedup_eps: null` a narrow projection takes the stage's
  place, so the fused vectors are shed without being regrouped first.

## Path A — one interactive allocation (simplest)

Curate is a single driver-orchestrated pass, so one GPU allocation is the
simplest way to run it. `cosmos-curator slurm shell` brings up the container on a
compute node and runs one command inside it. Run it **on the Slurm login node**,
not from a laptop: every path it takes is resolved on the machine that invokes
`srun`. (For the raw `enroot start` equivalent, see the
[Interactive Slurm guide](slurm-interactive.md).)

Two mounts decide where the config file has to live:

| Flag | Container path | Holds |
|---|---|---|
| `--workspace-path` | `/config` | the `curate.yaml` you are about to run |
| `--curator-path` | `/src/cosmos-curator`, symlinked onto `/opt/cosmos-curator` | a source checkout, when you are running one instead of the image's own code |

So write the config into the workspace directory and name it by its *container*
path:

```bash
cosmos-curator pipeline template curate > "${HOME}/cosmos_curator_local_workspace/curate.yaml"
# edit clips_lance_uri, storage_profile and target

cosmos-curator slurm shell \
  --account <account> --partition <partition> \
  --no-exclusive --gres gpu:1 --time 01:00:00 \
  --workspace-path "${HOME}/cosmos_curator_local_workspace" \
  --cache-path "${HOME}/.cache" \
  --mount-s3-creds \
  --curator-path "${HOME}/src/cosmos-curator" \
  --container-mounts "${HOME}/.config/cosmos_curator/config.yaml:/cosmos_curator/config/cosmos_curator.yaml:ro" \
  --container-image "${HOME}/container_images/<image>.sqsh" \
  --pixi-envs cuml \
  -- pixi run --as-is -e cuml run-pipeline /config/curate.yaml
```

`run-pipeline` is a core task and `cuml` includes `core`, so it is already on the
path in that environment. Use `--as-is` for runtime launches: it skips Pixi
environment validation and fails fast if `cuml` is not installed rather than
trying to install packages while Ray workers start.

Three of those flags are load-bearing and none substitutes for another:

- `--pixi-envs cuml` warms the environment during slim-image startup; `-e cuml`
  on the inner `pixi run` selects it for the driver. The GPU fit and dedup tasks
  name `cuml` through their own Ray runtime environment either way, so an unwarmed
  image resolves it while Ray workers are starting.
- The explicit `--container-mounts` entry is what supplies `cosmos_curator.yaml`,
  where the `storage_profile` your config names is defined. `slurm shell` mounts
  that file automatically only for `model_cli` commands.
- `--no-exclusive` alongside a GPU subset is deliberate: an exclusive allocation
  that does not claim every GPU on the node is cancelled by the idle-GPU monitor.

Add `--json` for a machine-readable result on stdout, which is also what makes a
failure's class explicit: a config fault is reported as `invalid`, anything raised
by the run itself as `runtime`. Without `--json` a config fault still prints its
message and exits 2, while a run failure surfaces as an ordinary traceback.

`runtime` covers more than a mid-run failure: the driver preflight runs inside the
run, so a rejected input is labelled `runtime` even though it precedes any Ray
work and publishes no verdict. The label tells you where the failure arose,
not whether anything was written — for that, read the message and consult
*Re-runs and recovery*, which is organised by what the table is left holding.

## Path B — managed Ray cluster (multi-node)

For capacity that must be assembled from independent Slurm jobs, submit the same
driver command to a managed run-scoped Ray cluster
([design](../design/curator-next-slurm-ray.md)). The pipeline command is the
argument vector after `--`:

```bash
cosmos-curator slurm ray template > cluster.yaml   # edit account, partitions, worker_lanes
cosmos-curator slurm ray submit cluster.yaml -- \
  pixi run --as-is -e cuml run-pipeline curate.yaml
```

The `pixi run` prefix is required here and is not decoration: the cluster head
execs that argument vector directly, with no shell and no Pixi wrapper around it,
so `run-pipeline` alone would fail as an unknown executable — it is a Pixi task,
not an installed script. The only console script this project installs is
`cosmos-curator`.

The environment is chosen by the `-e cuml` in the submit command above, not per
lane: `cluster.yaml` has no per-worker environment field. On a slim image, list it
under `runtime.pixi_envs` so the environment is warmed before Ray starts rather
than resolved during startup. Monitor and tear down with
`cosmos-curator slurm ray status <run-id>` and `... stop <run-id>`.

> **Compatibility caveat.** Curate is a single-driver pass and is **not yet
> qualified for elastic worker loss** (see the Slurm-Ray design doc's *Pipeline
> Compatibility*). Run it against a **stable** allocation: provision enough GPUs
> up front and set the worker walltime equal to the head walltime so lanes do not
> churn mid-run. Managed renewal/preemption is for pipelines qualified to
> tolerate changing membership.

## Reading the outcome

A run reports its outcome twice: as the `CurateResult` the run returns, and as
two columns on the table. The result is in-process only — nothing persists it, so
capture what `run-pipeline` prints. Without `--json` that is a one-line summary;
with `--json` it is the whole result as a JSON object. Its full contract is in
[the design doc](../design/curator-next-curation.md#what-a-run-reports-instead-curateresult);
the fields an operator reads first:

| Field | Meaning |
|---|---|
| `read_version` | the pinned version every read used |
| `committed_version` | the version this run created — the handle to this selection. Its basis is found through the commit's `centroids_fingerprint`, not by building a name from this number |
| `eligible_rows` | rows where every weighted vector is present |
| `written_rows` | verdict rows the write claimed; always equal to `eligible_rows`, because the run refuses any other total rather than committing it |
| `reason_counts` | one count per reason; sums to `eligible_rows` |
| `requested_k` / `effective_k` | the `k` asked for, and the cluster count actually fitted after any downward clamp to the sample |
| `fit_rows` | usable rows the k-means sample held |
| `target` | the resolved keep-count the quota was allocated at |
| `subtask_k` | level-2 cells the fit returned; `0` when no level-2 basis was fitted at all. That has two causes and the number cannot tell them apart — read the `curate fit:` line, which reports the benign one at INFO and the granularity loss as a WARNING |
| `fairness_groups` | merged `(task, subtask cell)` groups holding at least one survivor. Bounded above by `merged tasks x (subtask_k + 1)`, and normally below it: a group whose every row was a duplicate does not appear at all |
| `unfunded_groups` | how many of those groups got no budget at this target. Read it as a share of `fairness_groups`, not as an absolute — the same count means very different things against 50 groups and against 50,000 |
| `merge_stats` | the task merge: labels in, representatives out, **clips moved** between fairness groups, wall-clock. The clip count is the one that matters to a selection: folding 20 labels is unremarkable unless one of them held a third of the corpus |
| `centroids_uri` | where the basis for this version was written |

The runner logs a counts-and-versions line, a reasons line, the task merge line, the
cluster-radius percentiles, the duplicate-score percentiles and counterfactual ladder,
the unfunded count and share, the fit sample's fraction of eligible rows, and the
centroids URI — so a collapsed task vocabulary, an over-fine level-2 partition and a
mis-set duplicate threshold are all visible in the log without querying anything.

Three of those lines are **unconditional**, meaning they appear on every run rather than
only when something looks wrong: clips moved by the task merge, the unfunded count and
share, and the fit sample's fraction of eligible rows. They are the inputs to
[Calibrating a new corpus](#calibrating-a-new-corpus). No distribution reaches
`CurateResult` — the two histograms and the sample fraction are log-only, so capture the
log if you intend to calibrate from them.

### The two columns

| Column | Values |
|---|---|
| `curate_selection_reason` | `selected`, `duplicate`, `below_quota`, `unfunded`, `invalid_embedding` — or NULL when the run did not claim the row |
| `curate_cluster_id` | the locality partition the row was de-duplicated within; NULL when the run did not claim the row, and NULL for a claimed row that bypassed clustering as `invalid_embedding` |

There is no `false` and no `selected` boolean: "not selected" is
`curate_selection_reason <> 'selected'`, "not curated" is `IS NULL`. Read the
outcome with predicates against the table itself:

The [`robot-action-caption` recipe](../design/curator-next-robot-action-caption.md)
uses `curate_selection_reason = 'selected'` as its default input scope. Run
Curate before captioning, or set `input.selected_only: false` explicitly when a
standalone caption run should process every otherwise-pending row.

```python
import lance

ds = lance.dataset("s3://bucket/run/clips.lance")
print(ds.version)
print(ds.count_rows(filter="curate_selection_reason = 'selected'"))
print(ds.count_rows(filter="curate_selection_reason = 'unfunded'"))
```

Three checks worth running after every commit:

| Check | Expected |
|---|---|
| `curate_selection_reason IS NULL AND <every weighted vector IS NOT NULL>` | **0** on a quiescent table — no eligible row was lost. The run enforces the equivalent *before* it commits, against the version it pinned, so a non-zero count here means rows were appended after the run pinned that version |
| `curate_selection_reason IS NOT NULL AND <any weighted vector IS NULL>` | **0**, unconditionally. The write is total, so an ineligible row is blanked rather than left holding an earlier run's verdict. A non-zero count is a defect in the write-back, not an operator condition |
| `curate_selection_reason = 'invalid_embedding'` | small; a non-trivial count means an upstream embed leg wrote non-finite or zero-norm vectors |

`curate_cluster_id` is a **computational locality partition, not a semantic
category**. Do not build reporting, filtering, or dataset documentation that
presents a cluster id as a class of clip. If the run warns that `k == 1`, the
corpus was small enough that every row landed in one cluster: de-duplication is
exhaustive and the *result* is strictly better, but the column carries no
information at all.

### Artifacts

One artifact per run, named by the hash of its own bytes:

```text
{clips_lance_uri}__curate_centroids/{sha256}.npz
```

It is written **before** the commit, and the commit records that hash as
`centroids_fingerprint` — so it is still locatable from the table alone, by reading
the version's commit rather than by constructing a filename. It holds the raw,
unnormalized `(k, 865)` centroid array plus the block column names, block dimensions
and fused width, the weights, the effective `k`, the **read** version the sample was
drawn from, the fit's row count, the k-means seed, the ids of the sampled fragments,
and the run's config as text (`resolved_config`) beside its `config_digest`. The
centroids live in the **fused** space, so a reader needs the block order and weights
to interpret any coordinate; the seed and fragment ids are what make the fit
reproducible from the artifact alone. Nothing cleans old artifacts up, at roughly
9 MB each.

Do **not** locate a basis by picking a file out of that directory. The names carry no
version, so only the commit says which object a given version's cluster ids were
assigned against.

**The write is not best-effort, and it cannot leave you with a repair to do.** It
precedes the commit, so a failure to publish the basis ends the run with the table
unchanged, exactly like any other pre-commit failure: no verdict is published, and
re-running the same config is safe. Expect a retry to leave the failed attempt's object
behind: the GPU fit is not bitwise reproducible, so a retry normally writes a new object
rather than landing on the old name. A run that dies between the write and the commit
leaves one behind the same way. Nothing references those objects, no reader can reach
them, and you can ignore them.

The **committed Lance version is the sole completion signal**. There is no
`_SUCCESS` marker, no `complete` flag, and no report file to wait for. Every object
a commit references is durable before that commit lands, so a version that is
visible is complete.

### Reading a version's identity back

The commit records which rules produced it, so a table version can be identified
without your own notes:

```python
import lance

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options

dataset = lance.dataset(uri, storage_options=get_lance_storage_options(uri, profile_name=profile))

transaction = dataset.read_transaction(version)
if transaction is None:
    raise SystemExit(f"v{version} has no transaction record")

properties = transaction.transaction_properties or {}
if properties.get("kind") != "curator-next-curation":
    raise SystemExit(f"v{version} was not written by Curate (kind={properties.get('kind')!r})")

properties["config_digest"]          # the rules this version applied
properties["centroids_fingerprint"]  # the basis its cluster ids were assigned against
transaction.read_version             # the table state this run was computed against
```

Both guards are load-bearing. `read_transaction` returns `None` for a version whose
transaction record is absent, and any sibling leg that writes this table commits its
own versions, so a version picked by number alone may not be Curate's.

`config_digest` is a `sha256:` hash of every result-defining config field — the
target, weights, thresholds, both seeds, and `within_group_order`, but *not* the
table's address, the storage profile, or `dedup_concurrency`. Use it as an equality
check only:

- **Equal digests** on two versions mean both runs applied the same *rules*.
- **Unequal digests** mean they did not, and the digest cannot tell you *which*
  field differs — read the archived config for that.

The hash covers a second half no config file names, under a `__contract__` key: the
order, source vector column and width of every fused block, the
`curate_selection_reason` vocabulary, both columns level-1 fairness reads (`task_label`,
the task label the groups are keyed on, and `task_vector`, the vector the merge compares
those labels by), and the label-canonicalization rule run over a fixed probe set. A
release that reorders the blocks, repoints one at another embedding column, or folds
task labels differently therefore moves the digest even though every config file is
byte-identical.

Equal rules are still not the same claim as equal behavior. What the digest covers is
the config plus that enumerated contract — not the build that read them, so a change
that moves selection from outside both is invisible here. Three are worth naming, because
all three are orderings and **none of them is `within_group_order`**: the manifest-ordered
fragment prefix the k-means sample is drawn from, of which only the row budget
(`fit_sample_rows`) is a config field and the prefix rule consuming it is not; the blake2b
keying rule behind the quota residual tie-break, of which only the *seed* is a config
field; and the `np.lexsort` retention order inside de-duplication, which no config field
feeds at all.

The first is the one this release moved, and it reaches furthest: it selects the rows the
locality basis is fitted on, so it moves every `curate_cluster_id` and every duplicate and
quota verdict decided against one. Two runs differing only there are separated by the
commit's `centroids_fingerprint` rather than by `config_digest` — a different prefix fits a
different basis, and the artifact is named by the hash of its own bytes — so read the two
properties together.

`schema_version` moves when the config surface changes, which catches the largest of
those. When you are comparing across a period in which the leg itself changed, pin the
code version the way you would for any other computation.

Adding a result-defining field — or a new entry to the `__contract__` half — also moves
the digest of a config that did not change: the hash covers the canonical text of the
whole payload, so a version curated before `fairness_residual_seed` existed hashes
differently from a re-run of the same file today, and so does one curated before the
contract half gained an entry. The second case is the one to expect after a leg release
and the easier one to misread, because there is no new field to point at — the operator's
config is byte-identical, no field was added, and the digest still differs. Read both
archived configs before concluding two versions applied different rules across such a
boundary.

To see the field values, read `resolved_config` from the centroids artifact this
version's commit names. The fingerprint is both the object's name and a checksum, so
re-hashing the bytes confirms you loaded the right basis before you trust anything in
it:

```python
import hashlib
import io

import numpy as np

from cosmos_curator.core.utils.storage.storage_utils import get_storage_client, read_bytes

fingerprint = properties["centroids_fingerprint"]
artifact = f"{uri}__curate_centroids/{fingerprint}.npz"
client = get_storage_client(artifact, profile_name=profile)

payload = read_bytes(artifact, client)
assert hashlib.sha256(payload).hexdigest() == fingerprint

with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
    text = str(archive["resolved_config"])
    assert int(archive["read_version"]) == transaction.read_version

assert f"sha256:{hashlib.sha256(text.encode()).hexdigest()}" == properties["config_digest"]
print(text)  # the field values themselves
```

The three assertions check different things and all three are worth keeping. The
first says the object at that name is the one the commit meant; the second says the
basis was fitted on the same table state the commit was computed against; the third
says the archived config text is the one the digest was taken over.

`np.load` cannot open a remote URI, which is why the bytes are fetched through the
storage client first — the same path `scripts/inspect_curate_columns.py` uses.

A digest that changes when you expected it not to is not a fault: the covered set
is defined by exclusion, so any config field not on the short exclusion list joins
the identity. That direction is deliberate — it can cost you a needless comparison,
but it cannot report two genuinely different runs as one.

## Calibrating a new corpus

Three fields cannot be chosen from documentation, because each one means something
different on every corpus: `dedup_eps`, `subtask_clusters`, and
`target_mean_cluster_rows`. The shipped defaults are starting points, not
recommendations. **The first run on a new corpus is a calibration run** — treat its
log as the measurement and its selection as provisional.

The procedure is one run and five readings:

1. **Run once**, at the target you actually want, with the defaults. Nothing here
   needs a special mode; every figure below is logged unconditionally.
2. **Read the cluster-radius histogram** — `distance_to_centroid` p50 and p95. This
   is the extent of a locality cluster in the same fused metric `dedup_eps` is
   expressed in, so it is what makes the threshold interpretable at all. If p50 sits
   far below your `eps`, clusters are tighter than the duplicate gate and the gate is
   dropping clips that are merely typical; if p95 sits far above it, the gate is
   barely firing.
3. **Read the duplicate-score histogram and its counterfactual ladder** — p99, p99.9
   and max of each row's best similarity to a strictly earlier row, then the count
   that *would* have been flagged at each candidate `eps`. The ladder carries **six
   static rungs** — `0.001, 0.005, 0.01, 0.02, 0.05, 0.10` — **plus the run's own
   `dedup_eps`**, marked with a trailing `*` so the operating point is readable
   beside its counterfactuals. The log reports each rung as an **absolute count over
   the whole corpus**, not a rate. Both percentiles and counts cover the **scored**
   rows only: a row that bypassed the similarity pass is excluded from the histogram
   and from its denominator, which the line states as "over N scored row(s)". This is
   the reading that decides `dedup_eps`, and it decides it without a second run.

   The marked rung will read slightly *under* the `duplicate` count in the verdict
   line, and that gap is expected rather than a counting bug: each rung counts from
   the first bin whose lower edge reaches `1 - eps`, so the bin straddling the
   threshold is dropped whole and every figure is a lower bound accurate to one bin
   width. See [the design doc](../design/curator-next-curation.md#what-is-deliberately-not-persisted).

   Read the marked rung against the neighbours below it. A threshold sitting several
   rungs above the point where the counts stop being small is not de-duplicating: it
   is thinning the corpus by locality, because at that distance "duplicate" has come
   to mean "closer than typical to some earlier clip" rather than "near-identical to
   one". The `max` is the check that settles it — if the corpus's highest similarity
   barely clears `1 - eps`, there is no duplicate population for a looser threshold
   to find, only ordinary neighbours.
4. **Read the three unconditional metrics** — unfunded group count and share, clips
   moved by the task merge, and the fit sample's fraction of eligible rows. The
   unfunded share confirms `subtask_clusters`, but read the WARNING to know which
   level starved — at most one of the two fires, and a task-level shortfall wins,
   because there `subtask_clusters` is the wrong knob. It only *confirms* the value
   because, unlike the other two fields, `subtask_clusters` has a ceiling you can
   compute before running anything — see below. The sample fraction tells you how much of
   the corpus the basis was actually fitted on, which is what bounds how far you can
   trust the radius reading in step 2.
5. **Then set the three fields** and re-run. Only now is the selection meaningful,
   because only now do the thresholds refer to this corpus's geometry.

### Choosing `subtask_clusters` before the run

`dedup_eps` and `target_mean_cluster_rows` genuinely need a measurement first. This one
does not: its upper bound is arithmetic, and getting it wrong is the most common way a
run produces a plausible selection that means nothing.

```text
k <= target_count / merged tasks    arithmetic ceiling (absolute targets only)
k well below ~40                    semantic ceiling
```

The quotient is only a pre-run number when `target_count` is set. Under
`target_fraction`, or with no target at all, the keep-count is
`SelectionTarget.resolve(survivor_count)` and is not fixed until de-duplication
has run, so the same ceiling becomes a post-run check against the logged
survivor count.

Take the merged task count from a previous run's log (`curate merge: task L=... -> R=...`,
where `R` is the number you want) or, absent one, from the annotation schema. Then:

Write the quotient as `q`, reading `target` as `target_count` where it is set and otherwise
as the resolved keep-count the previous run logged. The bands are total, so every value
lands in exactly one:

| `q = target / merged tasks` | What to set |
|---|---|
| `q >= 16` | leave the default at 16 |
| `1 <= q < 16` | set `subtask_clusters` to `floor(q)` |
| `q < 1` | `subtask_clusters` is irrelevant — the target is below the task count and **level 1** is starving; raise the target or lower `merge_theta_task` |

The value is clamp-only: the fit returns as many centroids as the sample supports, which on a
small corpus can be fewer than you set. Check the `curate fit: subtask k clamped from ... to
...` line and redo the arithmetic against the clamped number, because that is the `k` the
groups were actually built from.

Above the arithmetic ceiling a task's budget is smaller than the number of cells it
occupies, so its level-2 fill line is zero and every one of its clips is allocated by the
residual pass. The selection stays exact, uniform and spread across the partition, but
which cells reach the output is decided by `fairness_residual_seed` and not by the corpus
— a cell holding thousands of clips and one holding a dozen have identical odds, because
capacity never enters an order. Reseeding moves *which* cells lose. It moves *how many* only
within the level-1 leftover, and that leftover is bounded: a task that wins an extra clip at
level 1 starves one fewer cell, and a task's quota can differ by at most that one clip between
seeds, so **at most one group per merged task can flip**. In practice the reported count moves
by a group or two at a few dozen tasks, and stays well inside a percent of the group count even
at a hundred-plus tasks. Expect that drift rather than reading it as a defect — it will not
move you out of the regime.

Do not rely on the degeneracy WARNING to catch this for you. It fires above 20% of groups,
which leaves a band where the quota is already a lottery and the run stays quiet. Take a
target of 10,000 over 2,287 merged tasks — a ceiling of 4 — and set `k = 5`: level 1 hands
852 tasks five clips and the remaining 1,435 only four, so those 1,435 cannot reach their
fifth cell. Their level-2 fill line is zero and the seed places every clip they contribute,
but with a survivor in all five cells of every task the corpus-wide unfunded share is
`1,435 / 11,435 ≈ 12.5%` and nothing warns. The arithmetic costs nothing and covers the
band.

Raising `subtask_clusters` above the default is not supported at any target. A single task
is estimated to hold roughly 40 distinct subtask spellings, so a partition finer than that
stops grouping wordings and starts enumerating them — restoring the unbounded free-form key
that the cell index exists to replace. Nothing in the run reports this, because the run no
longer reads the spellings. Treat the 40 as a direction rather than a cut-off: it is an
estimate, and confirming it means counting distinct `subtask_name` values per `task_name`
on the clips table yourself. Do that before arguing for a larger `k`.

### Worked example: one 131,602-row corpus

The numbers below come from a single real corpus of 131,602 clips. They are **one
corpus's geometry, not production guidance** — do not copy them as defaults. They
are here because the *shape* of the reading is the transferable part.

| Reading | Value |
|---|---|
| nearest-neighbour fused cosine, p50 | 0.7581 |
| p90 | 0.8371 |
| p99 | 0.9163 |
| p99.9 | 0.9609 |
| max | 0.9824 |
| counterfactual yield per 20,000 rows at `eps = 0.001` | 0 |
| at `eps = 0.005` | 0 |
| at `eps = 0.01` | 0 |
| at `eps = 0.02` | 2 |
| at `eps = 0.05` | 44 |
| at `eps = 0.10` | 352 |
| actual duplicate pairs corpus-wide at `eps = 0.01` | **1**, at fused cosine 0.9917 |

The ladder rows above are **normalized per 20,000 rows** so they transfer to a corpus
of another size; your log emits the same six rungs as **absolute counts over the whole
corpus**, so scale one to the other before comparing them.

Read the first block against the gate. At `eps = 0.01` a duplicate needs fused
cosine above `1 - 0.01 = 0.99`, and the sample's **maximum** is 0.9824 — below the
threshold. So on this corpus the default `eps` *provably cannot flag anything in the
sample*, and the single real pair found corpus-wide sits at 0.9917: a genuine
outlier well clear of the bulk, not the tip of a distribution pressing against the
gate. That is the healthy shape. The unhealthy shapes are a p99 sitting just under
`1 - eps` (the threshold is arbitrating a dense region, so small changes move many
verdicts) and a ladder that jumps by orders of magnitude between adjacent rungs (the
corpus has a duplicate population the current `eps` is missing entirely).

The ladder is the actionable half. Going from `eps = 0.01` to `0.10` on this corpus
buys 352 flags per 20,000 rows instead of 0 — which is a real choice an operator can
make, and one they can only make from these counts.

### Which block is actually binding

The fused distance is a **conjunction**, and one block usually decides the outcome.
Under the fusion identity, a duplicate needs

```text
sum over blocks of  w_m * (1 - cos_m)  <  eps
```

so block `m` on its own must satisfy

```text
cos_m  >  1 - eps / w_m
```

At the default weights and `eps = 0.01` that resolves to:

| Block | Weight | Its own gate |
|---|---:|---:|
| subtask text | 0.6 | `cos > 0.9833` |
| image | 0.2 | `cos > 0.95` |
| action | 0.2 | `cos > 0.95` |

Note that every gate is strictly **below 1**, and that is true for any positive
weight and any positive `eps`. No weight/eps combination can make duplicates
impossible by construction, which is why nothing warns about the weights — the
formula is an explanatory aid, not a guard.

On the same corpus the binding block is **action**, not text: only **22** rows in
20,000 have a neighbour whose motion clears `cos > 0.95`, against **3,286** clearing
the text gate. So text agreement is common and motion agreement is rare, and the
duplicate rate is set by the rare conjunct. An operator who lowers `eps` expecting
the text block to bind has mis-modelled their own corpus; the ladder plus these two
counts is how you find out which block you are actually tuning.

### Revisiting the shipped default

`dedup_eps` defaults to `0.01`, and that default has only ever been calibrated
against corpora orders of magnitude smaller than the supported envelope. **A run at
250M rows or above, and its duplicate-score histogram, is the intended trigger for
changing it.** Until such a run exists the default is a placeholder that happens to
be conservative, and the calibration procedure above is the supported way to choose
a value — not a fallback for when the default disappoints.

### When to skip de-duplication

`dedup_eps: null` is right when the corpus has **no structural source of duplicates**.
The clearest case: a corpus whose `span_group_id` is unique on every row has no
multi-view or multi-take grouping behind it, so there is no mechanism that would
have produced two recordings of the same physical event. Paying for a corpus-scale
GEMM to confirm that is waste; the calibration run's ladder is the cheaper way to
establish it, and once established, skipping is the honest configuration.

Two properties to hold onto:

- **Skipping changes verdicts.** Rows that would have carried `duplicate` become
  selection candidates: they compete for budget, they change every fairness group's
  survivor count, and they change which rows clear a quota. A skip-path run and a
  dedup-path run over the same corpus are **not comparable**, in the same way two
  runs at different targets are not.
- **Everything else is unchanged.** The fit still runs, `curate_cluster_id` is still
  written, and the default `farthest` ordering still works, because both come from
  the scan rather than from de-duplication. The output schema is identical on both
  paths.

### Before proposing a different fit sampler

The fit sample is a deterministic **manifest prefix**, and the natural objection is
that a prefix is not a uniform sample. Half of that objection is already settled, so
do not spend a query on it: fragments **are** segregated by dataset. The ingest
config takes one required `source_dataset` per run and stamps it on every row, and a
run's fragments are appended sequentially, so the table is a concatenation of
dataset-pure slabs. A prefix is therefore fitted on whichever datasets were written
first, and no measurement is needed to establish that.

What is unsettled is whether it **matters** — whether distinct datasets occupy
different regions of the fused space, or merely different labels. That is the
prerequisite for any work on the sampler, and it is a query about geometry rather
than about provenance. Per fragment, report the distribution of `task_name` and
`subtask_name`, then compare the prefix's fragments against the whole manifest:

- If the two distributions are close, each dataset already spans the corpus's
  content, so a prefix is near enough to a random sample and the concern is moot.
- If the prefix's distribution is narrow against the manifest's, the basis is being
  fitted on a genuinely narrower slice of content, and the concern is real.

Without that comparison, a change to the sampler is a fix for a problem nobody has
shown exists. Note that the figures once cited as evidence against changing it are
**synthetic** — see the design doc's
[alternatives](../design/curator-next-curation.md#alternatives-considered) for what
still argues for the prefix, which is determinism and needing no scan rather than
any measurement.

### Before raising the producer's publish batch

Fragment count and the fit sample's cluster count are the same division, so the
obvious lever on one moves the other. The producer pins `max_rows_per_file` to
`clips_per_publish_batch`, which fixes both:

```text
fragments = rows             / clips_per_publish_batch   ->  62,500 at 500M rows
C         = fit_sample_rows  / clips_per_publish_batch   ->  500
```

Whole-fragment sampling is cluster sampling, so `C` is what the basis is actually
fitted over. Doubling the publish batch to cut a 500M-row table to 31,250 fragments
therefore drops `C` to 250, giving back half of the fit quality the producer's
fragment count currently supplies for free.

Two things follow. First, this is **not Curate's field** — `clips_per_publish_batch`
belongs to the producer, so changing it is a cross-package request, and Curate is the
consumer that silently pays for it. Second, do not open that request on the strength
of a fragment count alone. The number sounds large next to the range where Lance
manifest cost is usually discussed, but nobody here has measured the manifest cost of
this table: report manifest open and scan-planning latency at the current fragment
count, and whether it grows enough to register against the scan it precedes.

If it does register, **compaction is the remedy that leaves the producer's field
alone** — it rewrites fragments after the fact, so `clips_per_publish_batch` and the
recovery unit it sizes stay where they are, and no cross-package request is needed.
It does not leave `C` alone, though, and that is the trap: the sampler takes a
fragment prefix of the table as **read**, so a compaction target divides
`fit_sample_rows` exactly as the publish batch does. Lance's `compact_files` targets
1,048,576 rows per fragment unless told otherwise, which on a 500M-row table yields
about 477 fragments. Fragment count is what bounds read parallelism for every recipe
whose unit of work is a fragment, and 477 still leaves that ample; what it does not
leave ample is `C`, which falls from 500 to about 4 — a far heavier loss than the
halving that doubling the publish batch would cost. Pick a target explicitly, from
the width of the widest reader of this table and the `C` you are willing to keep.
Reach for the publish batch only if compaction is unavailable.

## Comparing two runs

This is the one operator-facing procedure that **moved** rather than disappeared.
There used to be one selection table per target, so comparing two targets meant
reading two tables side by side. Now there is one selection per table and no run
naming, so comparison is **Lance version time-travel** over the single table. The
mechanics and their rationale are in the design doc's
[rerun section](../design/curator-next-curation.md#rerun-rebuild-and-comparing-targets);
the procedure is:

1. **List the Curate versions and read their digests.** Each Curate run commits
   exactly one version, and the repository runs no version cleanup, so prior
   selections persist by default. Not every version is a Curate commit — appends
   and embedding fills create their own — and the `kind` property is what tells
   them apart.

   ```python
   import lance

   ds = lance.dataset("s3://bucket/run/clips.lance")
   for v in ds.versions():
       tx = ds.read_transaction(v["version"])
       if tx is None:
           continue
       properties = tx.transaction_properties or {}
       if properties.get("kind") == "curator-next-curation":
           print(v["version"], v["timestamp"], properties["config_digest"])
   ```

   Two versions with the **same** digest were selecting for the same thing, so a
   difference between them is a difference in the corpus. Two with **different**
   digests are not comparable as selections, and
   [Reading a version's identity back](#reading-a-versions-identity-back) is how you
   recover which rules each one applied.

2. **Open both versions and count.** The previous version's `curate_*` column
   files stay referenced by the previous manifest, so an older selection is read,
   not reconstructed.

   ```python
   prev = lance.dataset("s3://bucket/run/clips.lance", version=6)
   cur = lance.dataset("s3://bucket/run/clips.lance", version=7)

   for reason in ("selected", "duplicate", "below_quota", "unfunded", "invalid_embedding"):
       f = f"curate_selection_reason = '{reason}'"
       print(reason, prev.count_rows(filter=f), cur.count_rows(filter=f))
   ```

3. **Diff membership when the counts are not enough.** Scan `clip_id` from each
   version under `curate_selection_reason = 'selected'` and difference the two
   sets to get what entered and what left the selection. This is a two-pass scan
   of a narrow projection, so run it as a Ray Data job on a corpus-scale table
   rather than in the driver.
4. **Load the matching centroids** for either curation version from
   `__curate_centroids/{fingerprint}.npz`, taking the fingerprint from that
   version's `centroids_fingerprint` commit property. The archive also carries
   that run's `resolved_config`, which is where the digest's field values live.

A prior selection is also exactly reproducible by **re-running its config**:
curation is a full recomputation with a fixed seed, so the same source version,
weights, thresholds and seed give the same verdicts.

## Re-runs and recovery

- **Re-running is safe and overwrites.** A run recomputes everything and
  overwrites both columns; there is no incremental invalidation, no compatibility
  fingerprint, and no resumption. Re-running the same config against the same
  source version reproduces the same verdicts.
- **The commit is all-or-nothing.** Any fragment failure aborts the run *before*
  the commit, so no verdict is published. This differs from the embeddings leg,
  which skips a failed fragment and commits the rest — a Curate verdict is decided
  against a corpus-global quota, so a partial commit would publish verdicts
  derived from a population that was never written.
- **A first run against a table leaves one version behind even if it fails.**
  Preflight widens the schema with the two `curate_*` columns in a metadata-only
  commit before any verdict exists, so a failure after that point leaves a version
  whose columns are entirely NULL. It is harmless: it stamps no `kind`, so
  `inspect_curate_columns.py` reports the table as never curated rather than as a
  run that claimed nothing, and a retry re-uses it instead of adding another. Later
  runs find the columns present and commit nothing extra.
- **Files left behind by a failed run are inert.** Workers that finished before
  the abort left column files at the target, but no manifest references them, so
  no version can read them. That is garbage to collect, never a partial commit.
- **Appended clips read NULL.** New fragments are simply "not curated" until the
  next run; the run does not special-case them. An append *during* a run rebases
  cleanly against the commit: the run's own completeness check is evaluated
  against the version it pinned, so the appended rows are outside it and the run
  succeeds. They read NULL on the committed version until the next run.
- **Turning de-duplication off or on is a full re-decision, not a filter change.**
  Every eligible row is re-claimed and overwritten, so no verdict is left stale —
  but the two runs are not comparable, because a row the earlier run called
  `duplicate` is a selection candidate in the later one. Record which path a
  version came from; nothing on the table does.
- **Every run is a full run.** The write is total: every row of every fragment is
  written on every run, NULL where the run did not claim it. So a re-run under
  narrower weights blanks the rows it no longer curates rather than leaving them
  carrying the previous run's verdicts. No pre-run cleanup, no detector query, and
  no procedure to remember — a non-NULL value is always the latest run's.
- **A rebuild is a re-run**, optionally dropping the two columns first. Dropping is
  no longer needed for correctness; it only removes the columns from the schema.

### Narrowing the eligible set

Raising a weight from zero, or curating a corpus where one modality is missing for
a whole dataset, takes rows out of the eligible set. Those rows come back **NULL**
in both columns, so the table never shows a mix of two runs' criteria.

What that costs is the previous verdicts: they are gone from the *latest* version.
Recovering them means reading the previous version, which is exactly what the
version history is for:

```python
previous = lance.dataset(uri, version=committed_version - 1)
```

So the care a narrowing run needs is not a repair afterwards, it is knowing the
version number before you start. Note the run's committed version from the
`curate: committed v<N>` log line — the table records which *rules* produced each
version, but finding the one you want still means walking its history.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Driver exits immediately with a named error before any Ray work | Preflight rejected the input: a missing vector column, a `clip_id` / `task_name` type mismatch, a consumed embedding group filled by more than one producer, a consumed group holding vectors whose provenance column records nothing, or no row satisfying the eligibility predicate raises `ValueError`; an entirely absent table raises `FileNotFoundError`. Preflight runs inside the run, so both are reported as `runtime` under `--json`, and without it as a traceback whose last line is the message. | Read the last log line; fix the table or the config. For either producer-identity failure the message names the group and the column, and the remedy is to re-run the embeddings leg to completion, which recomputes every row whose recorded identity is missing or names another computation, so one producer fills the vectors and their attribution together. Preflight is deliberately eager so bad inputs never reach a Ray actor, and it precedes the verdict commit — so no verdict is published. On a first run the schema widening may already have committed; see *Re-runs and recovery*. |
| Parse-time error naming a key you did not expect to be wrong | A pre-rewrite config or launch script | The key is gone rather than renamed-and-tolerated; see the *Config fields* table for the current surface. |
| Run dies after the whole dedup pass with a message about rows carrying no reason, or not accounting for every eligible row | A stage dropped or duplicated rows | Nothing was committed — the check runs before the write. Report it: this is an internal invariant, not an input problem. |
| Run dies with a message that no row survived to be selected, or that every eligible row already carries a reason | Every eligible row is a duplicate, has an invalid embedding, or already carries a verdict from a prior run | Raise `dedup_eps` scrutiny first (a very large `eps` collapses everything), then check `invalid_embedding` volume in the previous run. At `dedup_eps: null` no row can be a duplicate, so the whole eligible set was `invalid_embedding` and the vector columns are the only place to look. Nothing was committed. |
| Run dies naming a `__dedup_key`, its row count, and a maximum number of rows the device holds | One cluster holds more rows than a GPU can score at **any** tile size. Not a corpus-size problem: the k-means basis concentrated the corpus into too few clusters. Nothing was committed. | Lower `target_mean_cluster_rows` (raises `k`), or raise `fit_sample_rows` so the basis stops concentrating rows. **Do not split the cluster** — see [When one cluster is too large to score](#when-one-cluster-is-too-large-to-score). |
| Run dies with node-local disk exhaustion mid-shuffle | Node-local scratch too small for the `__dedup_key` shuffle | Size scratch for roughly 865 GB per 250M eligible rows, or reduce the eligible set. Curation does not pre-check the environment. |
| Import/CUDA errors at startup | Not launched in the `cuml` environment | Run under `pixi run -e cuml …` (or `--as-is`); the fit and dedup tasks need cuML/cuPy. |
| The run warns that `k == 1` | Corpus smaller than `target_mean_cluster_rows` | Benign: dedup is exhaustive and the result is strictly better. Only `curate_cluster_id` is degenerate — it is `0` for every row. Lower `target_mean_cluster_rows` if you want a real partition. |
| The `k == 1` warning also says the eligible rows exceed what one device can score | At `k == 1` there is only one cluster, so the per-group device ceiling (11,152,540 rows at 865 wide on 80 GiB) applies to the whole eligible set | Advisory, emitted on the driver before the scan; the binding refusal still happens inside the dedup task. Lower `target_mean_cluster_rows` so `k > 1`, which is the only way to split the work. |
| `curate fit sample:` warns that an estimated **device peak** exceeds 80 GiB | The sampled matrix times the measured ~4x k-means peak does not fit one card. Advisory only — the run continues and the fit may still succeed. | Lower `fit_sample_rows` to the number the warning names (~6.2M at the fused width). The `GiB host matrix` figure on the preceding INFO line is a different cost with no threshold — see [The fit sample's two memory lines](#the-fit-samples-two-memory-lines). |
| `curate fit:` warns that the subtask block carries weight but **no level-2 basis was fitted** | No sampled row carried a usable `embedding_text_subtask` direction, so every row takes the reserved cell and fairness granularity falls back to the canonical task label alone. The artifact cannot tell you this — it records an empty subtask basis for the benign case too. | Check `embedding_text_subtask` on the fit sample's fragment prefix for zero-norm or non-finite vectors. If you meant to select without level-2 cells, set the subtask weight to zero instead, which reports at INFO. |
| `curate fit:` says at INFO that no subtask basis was fitted because the block carries no weight | The documented escape, and a configured state rather than a fault. `subtask_k` is `0` and each task is one level-2 group. | Nothing. Weight the block if you want level-2 cells. |
| `curate fit: subtask k clamped from … to …` | `subtask_clusters` exceeds the usable subtask rows in the fit sample, so k-means clamped it | Advisory: the level-2 group ceiling is the clamped value, not the one you asked for. Lower `subtask_clusters` or raise `fit_sample_rows`. |
| Extra column files at the table URI after a failure | A worker finished before the run aborted | Nothing references them; no version can read them. Collect them as garbage. |
| An object in `__curate_centroids/` that no version's commit names | A run published its basis and then failed before committing | Nothing references it and no reader can reach it. Collect it as garbage; the failed run committed nothing. |
| The inspector reports that a version was committed by Curate but carries no `centroids_fingerprint` | That version was curated before the basis became content-addressed, so its commit records no hash to resolve. Reported as a note, not a failure — the data is sound, only the reference is in the older form | Its basis is the version-keyed `v<N>.npz` object in the same root, which nothing records the hash of, so it cannot be verified. Re-curate the table if you need a checkable basis; nothing needs repairing otherwise. |
| The inspector fails saying a version carries cluster ids but no Curate commit names a basis | Either the search window (64 versions) does not reach back to the Curate commit — the message says so when that is possible — or something other than Curate wrote the `curate_*` columns | If the window is the cause, inspect the Curate version directly (`--version`). If it is not, treat the columns as untrusted: no basis means no way to interpret a cluster id. |
| Rows that carried a reason now read NULL after a re-run | The eligible set narrowed, so the total write blanked them. Working as designed, not a fault | Nothing to repair. If you need the previous verdicts, read the previous table version — see [Narrowing the eligible set](#narrowing-the-eligible-set). |
| `target_count` and `target_fraction` both given | A cross-field validator on `SelectionTarget` refuses a target expressed twice, so this fails at parse time | Set at most one; omit both to keep every survivor. |
| A warning that the level-2 quota is **degenerate**: too large a share of groups received nothing | The target cannot reach most of the cells the run created, so which cells are funded is decided by `fairness_residual_seed` rather than by the corpus. The allocation is still exact, uniform, and spread across the partition rather than concentrated on low cell ids, but it has stopped meaning what level-2 fairness is for. Fires above **20%** of the group count — the same number reported in step 4 — and is structurally silent at the 250M envelope, where the group ceiling is far below any realistic target. It is a backstop, not a detector: a `subtask_clusters` modestly above the ceiling leaves the quota a lottery while staying under 20%, so the absence of this warning does not certify the value. | Either raise the target or lower `subtask_clusters`, using the arithmetic in [Choosing `subtask_clusters` before the run](#choosing-subtask_clusters-before-the-run) rather than by trial. Reseeding does **not** help: it moves which cells are funded, and moves the count only within the level-1 leftover — at most one group per merged task, never out of the regime. Do **not** read it as a bug in the allocator: see [Calibrating a new corpus](#calibrating-a-new-corpus). |
| A warning that the target is below the task count, so some tasks draw no selected clips | A distinct and more serious condition than the line above, and it is unthresholded because it is binary: a task starves only when the level-1 fill line is zero, which happens exactly when `target` is below the number of merged tasks. Every survivor in those tasks lands as unfunded — rows already carrying `duplicate` or `invalid_embedding` keep the verdict they had. **While this fires the level-2 line is suppressed**, because below the task count every funded task holds a quota of exactly one, which makes the level-2 share a reading of `subtask_clusters` rather than of the corpus. | Raise the target above the task count, or **lower** `merge_theta_task` to fold the vocabulary harder — a label joins a representative only *above* the cosine, so a lower theta merges more and `1.0` merges nothing. Lowering `subtask_clusters` **cannot help** — the shortfall is at the task level, and even one cell per task cannot be funded. Which tasks survive is decided by `fairness_residual_seed`, so it is arbitrary but not correlated with the label's spelling — and, as with the row above, reseeding changes which tasks survive, never how many. |
| A warning that the task merge moved too large a share of clips between fairness groups | More than **10%** of eligible clips changed fairness group because their task label was absorbed by a representative. The label counts alone would not have shown this — a few absorbed labels can hold most of the corpus. | Check `merge_theta_task`. This is the loudest available correlate of the degenerate merge the design doc's Limitations describe, and nothing on the table records it, so investigate at the time it fires. |
| `curate cluster radius: distance_to_centroid p50=… p95=…` | Informational: this is what makes `dedup_eps` interpretable, since `k` rises with the corpus while `eps` is fixed. | If p50 sits far below `dedup_eps` the clusters are tighter than the duplicate threshold and the gate is dropping legitimately distinct clips; if p95 sits far above it, the gate is barely firing. |
| `curate cluster radius: no scored row carried a distance` | Every eligible row was `invalid_embedding`, so nothing reached the similarity pass | The run committed a selection of nothing useful. Check the vector columns for the weighted blocks before re-running. |
| Duplicate-score percentiles and a counterfactual ladder over candidate `eps` values | Informational, and the primary input to choosing `dedup_eps`. Log-only: no percentile or bin count reaches `CurateResult`, so capture the log. | Read it per [Calibrating a new corpus](#calibrating-a-new-corpus). A maximum below `1 - eps` means the threshold cannot fire on this corpus at all. |
| `curate dedup score: no row carried a retention score; de-duplication was skipped or every row bypassed the similarity pass` | INFO rather than a warning, because a skip is configured and not degenerate. The message cannot tell its two causes apart, but you can: if the config set `dedup_eps: null` the stage never ran, which is the expected case. If it did **not**, every eligible row bypassed the similarity pass, so the whole eligible set is `invalid_embedding`. | At `dedup_eps: null`: expected, and there is nothing to calibrate against on a skip-path run — use a run with an `eps` set to get the ladder, then decide whether to skip. Otherwise treat it as the "no scored row carried a distance" case above and check the weighted vector columns. |
| A line reporting the fit sample as a fraction of eligible rows | Informational: how much of the corpus the centroid basis was actually fitted on | A small fraction does not invalidate the run — a locality cluster is a computational partition — but it does bound how far you should trust the cluster-radius reading, and it is the figure to quote if a skewed basis is suspected. |
| `dedup_eps: 0.0` | Zero is refused by the field's open lower bound, because the retention test is strict: `0.0` would pay for every GPU comparison and drop nothing, including exact copies | `null` is the spelling for "skip de-duplication"; the error names the field. Omitting the key keeps the default, which **runs** de-duplication — skipping is opt-in. |

### When one cluster is too large to score

Dedup scores one cluster per GPU with a tiled cosine GEMM, and it sizes that tile
from the cluster's row count and the card's total memory. When even a tile of one
row does not fit, the cluster's vectors alone exhaust the card and no tile choice
helps, so the stage **refuses** instead of letting CUDA fail. The message is
self-contained by design — the raise happens inside a Ray task, where the
exception type is erased and only the text reaches you — and it names the cluster
id, its row count, the device size, and the maximum rows that device holds.

Read it as a statement about **cluster skew, not corpus size**. The ceiling is
11,152,540 rows on an 80 GiB card at the fused width of 865, roughly 56x the
default `target_mean_cluster_rows`, so hitting it means the fitted basis pushed
a large share of the corpus into one cluster — not that the corpus outgrew the
hardware. The two remedies the message names attack exactly that:

- **Lower `target_mean_cluster_rows`**, which raises `k` and partitions the
  corpus more finely.
- **Raise `fit_sample_rows`**, so the basis is fitted on more of the corpus and
  concentrates it less. The sample is a *manifest prefix*, so a table whose write
  order correlates with content is the case most likely to produce a skewed basis.

**Splitting the offending cluster is not a remedy, and no setting does it.** A row is
a duplicate iff its similarity to some strictly earlier row *within its own
cluster* exceeds `1 - eps`, so which rows share a cluster decides which
comparisons happen at all. Cutting a cluster up removes comparisons: a partition
that ignores position — anything hash-like or round-robin — keeps only about `1/g`
of the duplicate drops for `g` parts, and every stage still reports success. Only
a split made in the *same geometry* the duplicate relation is defined in comes
close to preserving verdicts, which is exactly what the two remedies above do by
changing the partition through the basis. The run refuses rather than splitting for
that reason. The reasoning is in
[the design doc](../design/curator-next-curation.md#device-memory-the-derived-tile-and-the-cluster-ceiling).

**Each remedy has a ceiling, and the two guards do not model each other.** The
refusal is computed from one cluster's row count inside the dedup task; the fit's
memory check is computed from the sample's row count on the driver and contains no
`k` term at all. Neither can warn that the change the other asked for has run out
of room, so a remedy pushed past its ceiling trades a refusal that names its cause
for a failure that names nothing.

| Remedy | Ceiling, in operator units | What binds it |
|---|---|---|
| Lower `target_mean_cluster_rows` | none enforced anywhere; the real limit is `fit_sample_rows / k` **rows per centre** | fit quality — no guard, no log line, no reported figure |
| Raise `fit_sample_rows` | **6,206,600** rows at the fused width of 865, about **1.55x** the `4000000` default | the fit's device-peak warning, then a fit-task OOM |

Lowering `target_mean_cluster_rows` raises `k` but does not grow the sample, so
the same rows are spread across more centres. On a 250M-row corpus the default
`200000` gives `k = 1250` and 3,200 sample rows per centre; `20000` gives
`k = 12500` and 320; `2000` gives `k = 125000` and 32. Nothing reports that
quotient, and a basis fitted from tens of rows per centre is not a meaningful
partition of anything. The two ceilings also compound: even at the maximum sample,
`k = 12500` is still under 500 rows per centre.

Above 6,206,600 sample rows the fit logs the advisory line described in [The fit
sample's two memory lines](#the-fit-samples-two-memory-lines) — computed against an
**assumed** 80 GiB card, so on a smaller device the real ceiling is lower than the
warning believes and nothing refuses.

Neither ceiling is GPU time. De-duplication is shuffle-bound rather than
compute-bound — the ~865 GB moved at 250M eligible rows (1.73 TB at 500M) against
14 and 29 GPU-minutes of matrix multiply — so raising `k` is nearly free in the
stage it fixes, and it is the fit, not the dedup, that limits how far you can raise
it. Reasoning:
[Why the fit runs on a sample](../design/curator-next-curation.md#why-the-fit-runs-on-a-sample).

### The fit sample's two memory lines

The fit logs two numbers about the same sample, and they answer different
questions:

| Line | What it is | Threshold |
|---|---|---|
| INFO `… a ~N GiB host matrix` | the sample matrix in node RAM, at `min(fit_sample_rows, eligible)` rows | none — reported as fact |
| WARNING `~N GiB estimated device peak …` | that matrix times the measured ~4x cuML k-means peak | 80 GiB, one card |

Only the warning is a judgement, and it is **advisory**: it is computed on the
driver, before the fit task is scheduled and in an environment with no GPU to
ask, so it assumes an 80 GiB card. On a smaller device it can stay silent when it
should not; on a larger one it can fire when the fit would have fitted.

If you ran an earlier build, both numbers may have **dropped** — sharply on a corpus
the eligibility predicate thins out, and not at all where the prefix reaches
`fit_sample_rows`, which is the reference case. They used to be modelled on the
*physical* rows of the sampled fragment prefix — so on a table of few large
fragments the warning over-reported by the whole overshoot (measured 11.8x on a
two-fragment table) and then advised lowering `fit_sample_rows` at the very moment
that field was already the binding constraint. Both now scale by the rows the fit
allocates, which is each matrix's own budget bounded by the prefix's **eligible**
rows: the sample scan reads only predicate-passing rows, so on a corpus whose
weighted modalities are thinly embedded the buffers are sized for the rows that
arrive rather than for the rows the prefix stores. When the peak does exceed one
card the warning names the `fit_sample_rows` value that would fit. The host matrix
figure carries no threshold because nothing here knows the fit task's share of node
RAM; that one is yours to check against the allocation.
