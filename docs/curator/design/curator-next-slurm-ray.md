# Managed Ray Clusters for Curator Pipelines on Slurm

## Summary

Cosmos Curator can build a run-scoped Ray cluster from independent Slurm jobs:

- one non-requeueable job hosts the Ray head, pipeline driver, and application work
- each worker lane is one requeueable job providing additional compute capacity
- head and workers default to exclusive CPU or GPU nodes and can independently request shared-node slices
- a preempted lane can rejoin the same Ray cluster when Slurm requeues it
- a lane whose walltime is shorter than the run renews through successive allocations
- users can add or remove lanes without restarting the driver

`cosmos-curator slurm submit` places a whole Ray cluster inside one multi-node allocation, and remains the simpler
option when the resources are known and can be scheduled together. Independent allocations exist for when they cannot:
a single-node head can start working while additional capacity is still queued, capacity can arrive and leave without ending
the run, and worker walltime and preemption no longer bound the cluster. The tradeoff is explicit and pays for all of
it — worker loss becomes ordinary Ray node loss, which the pipeline has to tolerate. Managed clusters use the separate
`cosmos-curator slurm ray` command group.

## Initial Scope

This is not a general-purpose Ray-on-Slurm service that runs arbitrary Ray workloads. It is built for running Cosmos
Curator Ray Data pipelines, initially on NVIDIA Slurm clusters. Two consequences follow:

- **Scope is narrow by choice.** Only the deployment surface those pipelines need on those clusters is modeled.
- **The mechanism is deliberately not Curator-specific.** Workers join a standard Ray cluster and run whatever Ray or
  Ray Data schedules; there is no Slurm-side work assignment protocol, so pipeline code runs unchanged.

One command creates the cluster and runs the pipeline on it, so the cluster belongs to that submitted pipeline command
and the head job defines its lifetime. That coupling buys a single run identity, automatic teardown, and a manifest
that records a driver outcome rather than only a cluster state.

The first implementation owns the launcher lifecycle:

- validate deployment configuration
- submit one head and the requested worker lanes
- start the driver when Ray advertises compute resources, including those on the head
- report status and support manual scale and stop operations
- allow a requeued lane to rejoin
- tear down Ray and every recorded Slurm job when the run ends

It deliberately does not:

- cover every Slurm selector a site might use; the config models the account, partition, QOS, walltime, and GPU count
  the target clusters need, and anything further is added when a target requires it
- publish the generated batch scripts as a standalone contract to be submitted by hand; they depend on run state the
  submitting client creates, which is what lets lifecycle commands act on exactly the jobs that were recorded
- validate site policy, partitions, filesystems, or network topology before submission
- guarantee recovery if the submitting process disappears between a successful `sbatch` and recording its job ID
- make existing Curator pipelines elastic or preemption-safe
- add Ray authentication or transport security
- maintain a background desired-state controller or autoscaler

Each of these is deferred rather than rejected, and should be justified by a target that needs it.

Pipeline migration follows the launcher implementation. Until a pipeline is explicitly qualified, this command provides
the cluster substrate rather than a claim that the pipeline is safe under worker loss.

## Topology

Each run has a unique ID, one head job, and zero or more worker lanes. All worker lanes use the same configured node
shape.

The head runs the Ray control plane, a small supervisor, the pipeline driver, and application tasks and actors. Like
workers, it advertises its allocated CPU and GPU resources and the configured `curator_io` slots to Ray. It is
non-requeueable because losing the Ray head ends the cluster.

Head and worker allocations use the same resource fields and defaults, but their configurations are independent:
changing a worker setting does not change the head. Each defaults to an exclusive whole node with all its memory;
either can set `exclusive: false` and request a slice with `cpus` and `memory`. The head can use a different node
shape, partition, QOS, and walltime from the worker lanes, such as a long-lived GPU allocation alongside shorter,
preemptible GPU allocations.

Both roles reserve free ports at startup because a sliced allocation can share a node with another run. The head
publishes its ports in the run's bootstrap record. The reservation covers every port Ray would otherwise pre-select,
and Ray worker processes use random ports to avoid collisions with the node's ephemeral port range.

Each worker lane is an independent requeueable Slurm job on one node. Its Slurm job ID remains
stable across requeues even though a new incarnation may run on a different physical node. A restarted incarnation
joins as a fresh Ray node and carries no worker-local state from its predecessor.

A lane whose walltime is shorter than the head's is submitted as a throttled job array instead of a single job, so it
renews through as many allocations as it takes to cover the run. This exists because the two ways an allocation ends
are not the same: Slurm requeues a preempted job, but a job that reached its time limit is simply finished. Preemption
therefore re-runs one array task without consuming the lane's budget, while a time limit lets another task become
eligible. Array task IDs are unordered identifiers rather than progress counters. The array job ID is the lane's
identity for the whole run, so recorded IDs, cancellation, and scale are unchanged; only the reported state of a lane
becomes an aggregate over its tasks.

This matters because sites commonly price walltime against priority. Long allocations may only be available on a
low-priority or preemptible partition, while the higher-priority partition caps walltime well below a multi-day run.
Renewal lets a run take the higher-priority partition without ending when its first allocation expires. It has no
setting of its own: it is what a worker walltime shorter than the head's means, so the default pair of equal
walltimes is the plain one-allocation-per-lane shape. Renewal also makes worker churn routine, since every lane goes
dark for a scheduling gap each time its walltime expires. When no worker lanes are available, the head can continue
work that fits its resources. Stages requiring resources available only on the workers stall until a lane returns.
The compatibility requirements below therefore remain load-bearing.

Lanes are not offset from one another to spread their renewals out: a timed dependency would leave later lanes
ineligible rather than merely lower priority, unable to take capacity that freed up during the offset.

The head starts the driver as soon as Ray advertises compute resources on the head or a worker. It does not wait for
worker lanes to join: pending workers are a normal scheduler condition, and later workers contribute after they
arrive. A run can start with `worker_lanes: 0` and use only the head, then add lanes through `scale` if needed.

Different shapes among the additional worker lanes would be a separate design.

## Ownership and Invariants

Each system remains authoritative for the state it already owns:

| State | Authority |
|---|---|
| Run configuration, command, and submitted job IDs | Run manifest |
| Head and worker allocation lifecycle | Slurm |
| Driver outcome | Head supervisor |
| Live nodes, tasks, actors, and objects | Ray |
| Inputs, outputs, and checkpoints | Pipeline-selected durable storage |

The launcher follows these invariants:

- lifecycle commands act only on exact job IDs recorded for the run
- the manifest records Slurm's `ClusterName`, and a lifecycle command rejects a login node connected to another
  cluster before it interprets those cluster-local job IDs
- all manifest writers use the state and runtime implementation shipped with that run
- manifest updates are atomic and every one of them is judged against the state it finds
- `SUBMITTING` advances to `STARTING` only after every initial job ID has been recorded
- `STOPPING` means teardown is still in progress
- a run becomes terminal only after Ray is down, worker allocations are terminal, and the head wrapper is exiting
- status observations never replace Slurm or Ray as an authority

One advisory lock on the run directory protects every write. A lifecycle operation spans several separate processes
and cannot hold that lock end to end, so it does not try to: each mutation takes the lock, re-reads state, and is
accepted or refused against the state machine. Ordering between operations therefore comes from the state machine
rather than from a lock held across round trips. The case that has to be ordered is teardown against a scale already
in flight, and it is: head cleanup moves the run to `STOPPING` before it snapshots the lanes it is about to cancel,
and a run that has begun stopping refuses to record a new lane, so the scale cancels the job it just submitted
instead of leaving one behind the snapshot. A lane that still slips through has a second line of defense, because a
worker refuses to join a run whose head the scheduler says is gone. A mutation decided from an earlier observation —
the one reconciliation `status` performs — is additionally conditional on the manifest revision it read.

Whole operations are not serialized against each other. Two concurrent `scale` commands can interleave, and the
losing one is rejected when it tries to record a lane number that already exists.

Run state follows this high-level progression:

```text
SUBMITTING -> STARTING -> ACTIVE -> STOPPING -> SUCCEEDED | FAILED | STOPPED
```

The first two states divide work the launcher owns from work the cluster owns. `SUBMITTING` covers only the
launcher's own recording and ends when every initial job ID is recorded, so it lasts seconds and names the one window
in which a vanished client can strand a run. `STARTING` means every job ID is recorded and the run is waiting on the
cluster: the head may still be queued or Ray may still be coming up.
That is a normal condition and can last as long as the queue does, so it must not be reported as though submission were
still in progress. `ACTIVE` means the driver is running, even if every worker lane is still pending.

`SUCCEEDED` and `FAILED` reflect the driver or launcher outcome. `STOPPED` means an explicit user stop. If cleanup
finishes but the final manifest write does not, a later status request can reconcile the terminal state from the
persisted outcome and current Slurm state.

## Run State and Discovery

Each run has a private directory on storage visible to the login, head, and worker nodes. It contains the manifest,
bootstrap and status observations, generated job wrappers, and the exact launcher runtime used by the run.

The manifest records the resolved deployment config, command argument vector, concrete runtime paths, Slurm cluster
name, head job ID, worker lane IDs, and driver outcome. It is a run ledger, not a replacement for live scheduler state.

A run directory is always `<state_dir>/<run_id>`, so a run ID plus its state directory locate the manifest by
arithmetic and no index of runs is kept. `list` enumerates the state directory, so deleting a run directory removes
the run from every view with nothing left to reconcile, and is how an operator recovers a forgotten run ID.

The state directory is launcher-wide rather than part of a run config: `submit` and every command that takes a run ID
accept `--state-dir`, while `COSMOS_CURATOR_SLURM_RAY_STATE_DIR` changes their shared default. The common case therefore
stays a bare run ID without embedding the locator inside the object it locates. A run directory whose manifest cannot
be read is reported as `UNREADABLE` rather than skipped, because that is damaged state rather than an absent run.
If the directory is visible from more than one cluster, `list` names the cluster recorded by each run, while `status`,
`scale`, and `stop` require the connected scheduler to report the same `ClusterName`.

The bootstrap record carries the head's address and the ports it reserved, and binds workers to the run and head job
they were submitted for. Every worker incarnation validates that identity and refuses to join after the head is no
longer active. Only an answer the scheduler actually gave may condemn a run: an unreachable controller and a departed
head are the same nonzero exit from a direct job query, so a starting lane retries rather than spending one of its
allocations on a scheduler hiccup.

## Submission and Lifecycle

Submission creates a `SUBMITTING` manifest, submits the head, submits each initial lane, and records every returned job
ID. It then advances the run to `STARTING`; the head does not start Ray before it sees that state.

If submission fails while the CLI is still running, the launcher enters `STOPPING` and cancels every job ID it has
recorded. It publishes `FAILED` itself only when it had recorded nothing; otherwise the run stays `STOPPING` until
head cleanup or a later `status` confirms those allocations are terminal, because `FAILED` must not be published
while a recorded allocation might still be running. Deterministic per-run job names allow recovery when an `sbatch`
response is lost but the client remains alive. There remains a narrow failure window when the client itself
disappears after Slurm accepts a job but before its ID is recorded.

The head supervisor owns Ray and driver processes. Once the driver exits, it records the outcome and tears down Ray.
The outer Slurm wrapper then runs the same shipped runtime once more, outside the container, to cancel recorded lanes
and publish the terminal state after allocation cleanup; running outside the container means this still happens when
the supervisor itself was killed. A stop request similarly enters `STOPPING`, requests worker cancellation before head
cancellation, and is finalized only after Slurm reports every recorded allocation as terminal.

Scale is manual and reconciles the current number of nonterminal lanes to a requested target. Scale-up creates fresh
lane numbers; scale-down cancels the highest-numbered lanes first. The target applies to that command only. A later
terminal lane is not automatically replaced.

Partial lifecycle operations report the exact job IDs they submitted or canceled. They do not attempt to roll back
earlier successful scale changes; rerunning the same target performs a fresh reconciliation.

## Configuration and CLI

Managed topology and runtime settings live in a versioned JSON or YAML config. The pipeline command remains a separate
argument vector after `--`, keeping deployment and pipeline configuration composable.

The config covers:

- initial worker-lane count and optional display name
- independent head and worker Slurm selections, exclusivity, CPU, memory, and GPU requests
- common container image, source, mounts, credentials, environment, and Pixi settings
- optional node-local Ray storage

The container image is normally a host path to a squashfs. A registry reference such as
`registry.example.com#org/repository:tag` or `registry.example.com#org/repository:sha256:<digest>` is also accepted:
the node runtime imports it at job start.

Both walltimes are always stated and must be finite. The head's walltime is the run's lifetime, and the ratio of the
two decides how many allocations a lane renews through, so neither is left to a partition default the launcher cannot
see; a config that omits one gets the launcher's own default rather than the site's.

Both roles default to `exclusive: true`, which requests `--exclusive --mem=0`. Setting `exclusive: false` uses the
role's `cpus` and `memory` instead, defaulting to 16 CPUs and 64G. CPU and memory settings are ignored for exclusive
allocations. Each role can request GPUs with either `gpus` or `gpus_per_node`; the two fields are mutually exclusive.
Omitted head fields use their own defaults, regardless of the worker configuration.

```yaml
schema_version: 1
worker_lanes: 2
slurm:
  head:
    partition: gpu_long
    time: 7-00:00:00      # bounds the run; the head cannot be renewed
    gpus: 8              # application work also runs on the head
  worker:
    partition: batch      # higher priority, 4-hour cap
    time: "04:00:00"      # 42 allocations per lane cover the head's 7 days
    gpus: 8
```

The config model is strict, generates JSON Schema, supports small `--set` overrides, and can render the canonical
resolved form stored in the manifest. Validation checks the configuration contract; site-dependent failures remain the
responsibility of Slurm and the allocated jobs.

Mounts whose source exists only on an allocated node are created before the container starts. Sources the launcher
itself introduces, such as a configured Ray temporary directory, are always created; user-declared node-local sources
are opt-in, so a site can choose to have a missing path fail rather than be created with default permissions.

The command group exposes config tooling plus five lifecycle operations:

```text
cosmos-curator slurm ray template [--json]
cosmos-curator slurm ray schema
cosmos-curator slurm ray validate CONFIG [--set PATH=VALUE]
cosmos-curator slurm ray render CONFIG [--set PATH=VALUE]

cosmos-curator slurm ray submit CONFIG [--state-dir PATH] [OPTIONS] -- COMMAND ...
cosmos-curator slurm ray list [OPTIONS]
cosmos-curator slurm ray scale RUN_ID --workers N [OPTIONS]
cosmos-curator slurm ray status RUN_ID [OPTIONS]
cosmos-curator slurm ray stop RUN_ID [OPTIONS]
```

`submit` and commands that take a run ID accept `--state-dir`; `list` uses the same option as its discovery root. Every
lifecycle command accepts `--json` for machine-readable output, including on failure. Lifecycle commands use the
existing local or SSH Slurm transport; no command requires SSH access to compute nodes.

A typical session:

```console
$ cosmos-curator slurm ray template > cluster.yaml   # then edit account, partitions, and worker_lanes
$ cosmos-curator slurm ray submit cluster.yaml -- python -m my_pipeline --input s3://bucket/shards
Run ID: cc-ray-8f3c1d2a4b60
Slurm cluster: example-cluster
Head job: 4812733
Worker jobs: 4812734, 4812735
Manifest: /home/user/slurm-ray/cc-ray-8f3c1d2a4b60/manifest.json
Logs: /home/user/slurm-ray/cc-ray-8f3c1d2a4b60/logs

$ cosmos-curator slurm ray status cc-ray-8f3c1d2a4b60
Run cc-ray-8f3c1d2a4b60 on example-cluster: ACTIVE
Head 4812733: RUNNING
Driver: RUNNING
Lane 0 (4812734): RUNNING, array_task=2, restarts=0
Lane 1 (4812735): PENDING, restarts=0, reason=Priority
Ray: live (cn1234:6379)
Logs: /home/user/slurm-ray/cc-ray-8f3c1d2a4b60/logs

$ cosmos-curator slurm ray scale cc-ray-8f3c1d2a4b60 --workers 4
$ cosmos-curator slurm ray stop cc-ray-8f3c1d2a4b60
```

## Status and Failure Semantics

`status` combines three views without conflating them:

- manifest phase and recorded driver outcome
- current Slurm state for the exact head and lane job IDs
- a timestamped observation written by the head about Ray nodes and resources

An old Ray observation is reported as stale. Unknown Slurm state prevents mutations that could otherwise act on an
incorrect view. A terminal run reports Ray as stopped only after allocation cleanup has been confirmed.
The one reconciliation write performed by `status` is conditional on the manifest revision from which its Slurm query
was built. If another writer advances that revision, the observation is discarded and a later status call retries from
a coherent snapshot.

Worker loss is exposed to Ray as node loss. Task and actor recovery follow the policies chosen by the pipeline. A lane
that becomes terminal without being requeued remains absent until a user scales the run again. A renewing lane is
terminal only once it has no array task left. Between tasks it is pending; when one is running, `status` may report its
unordered array task ID but never presents that ID as progress through the lane's budget.

Unexpected head loss fails the run. Remaining worker jobs are canceled by wrapper cleanup when possible and can always
be targeted later from the recorded manifest IDs.

## Pipeline Compatibility

The launcher accepts any driver command that connects to the managed Ray cluster, but a pipeline is suitable only if it
can tolerate changing membership. In particular, it must:

- allow late workers to contribute and tolerate periods with reduced capacity
- configure appropriate Ray task and actor recovery
- treat worker-local files and worker-owned Ray objects as ephemeral
- keep durable outputs and checkpoints outside worker-local storage
- make retried external side effects idempotent or atomically committed
- initialize every new worker incarnation rather than only workers present at driver startup

Existing pipelines are migrated and qualified after the launcher exists. Coordinated multi-node inference replicas and
pipelines that permanently size themselves from startup resources are outside the initial compatibility target.

## Deployment and Security Assumptions

A supported site provides:

- a head partition/QOS whose policy and walltime support the whole run, plus CPU or GPU capacity for worker lanes
- preemption policy that requeues eligible worker jobs
- job arrays, when worker allocations are shorter than the run and lanes have to renew
- shared storage with reliable advisory locking and atomic rename
- common runtime, data, credentials, outputs, and checkpoints visible where required
- hostname resolution and network connectivity between independent allocations
- node-local storage when configured
- `python3` of at least 3.8 on login and compute nodes

The interpreter is the one requirement the container does not cover. Lifecycle commands run the run's own state
module on the login node, and both head cleanup and a starting lane's head check run its runtime module on a compute
node outside the container — the first so that teardown survives a killed supervisor, the second so that the
scheduler logic a lane needs before it starts anything exists once, in Python, rather than a second time in
batch-script shell. All of them use whatever `python3` resolves to there, so those two modules are held to 3.8
rather than the 3.13 the rest of Curator assumes: surveyed clusters run login nodes from 3.8 through 3.12, and below
3.8 the state model would need a third-party typing backport, defeating the point of running with no installation at
all. Submission checks the login node it is given against that floor before it creates anything, so a site that
cannot meet it fails immediately and by name. Later lifecycle commands assume the login nodes of one cluster are
interchangeable and do not repeat the check. This is the one site property the launcher validates ahead of time,
because it is the one it can check without guessing at site policy.

The launcher targets a trusted private Slurm network. It does not configure Ray authentication or TLS, and it does not
expose the Ray dashboard. Sites with mutually untrusted users sharing the compute network are outside the initial
security boundary.
