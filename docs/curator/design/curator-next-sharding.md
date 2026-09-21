# Curator Next: Deterministic Work Sharding

## Summary

A **work shard** is one independently launched job's slice of a single logical
run. Sharding lets an operator split a run over hundreds of millions of rows into
several jobs that can execute on different clusters, at different times, and
retry independently, while the union of the jobs covers the whole work scope with
no gap and no overlap.

The primitive is one small shared type,
[`ShardConfig`](../../../cosmos_curator/next/core/sharding.py), which answers
exactly one question: **which work units belong to this job?** It is a pair of
integers and a modulo, with no coordinator, no manifest, no lease, and no shared
state between jobs.

What it is deliberately **not**:

- not a scheduler — nothing launches, sizes, retries, or sequences the jobs;
- not an orchestration service — there is no run registry and no campaign state;
- not a DAG framework — a shard executes the recipe's existing plan unchanged;
- not a correctness mechanism on its own — adopting `ShardConfig` does not make
  a recipe safe to shard (see [Adoption checklist](#adoption-checklist)).

Configuration is two fields, and the whole object is optional. Absent means one
job owns every work unit, which is the behavior every recipe had before this
existed.

## The gap this closes

Ray Data's job-level row checkpointing already covers a large class of failures,
and sharding does not replace or duplicate any of it. The two mechanisms answer
different questions, and reading them as alternatives is the main way to get this
wrong.

| | Row checkpointing | Work sharding |
| --- | --- | --- |
| Question | which rows of **this job** are already done? | which work units are **this job's** at all? |
| Granularity | one row | one work unit (a Lance fragment, today) |
| Scope | inside one job's lifetime | across independently launched jobs |
| Covers | a task, actor, or node dying; preemption; a crash mid-plan | a whole job failing; one cluster being unavailable; a campaign spanning days or sites |
| State | a durable checkpoint directory | none — a pure function of the work-unit id |
| Recovery | the next attempt in the **same** namespace | the next launch of the **same shard index** |

Checkpointing makes an attempt cheap to resume. It does nothing about the ceiling
on a single attempt: one job is one Ray cluster, one allocation, one queue
position, and one blast radius. At corpus scale that ceiling is the operational
problem — a run wide enough to need a thousand GPUs is a run that cannot get them
all at once, in one place, for as long as it takes.

The two compose, and the composition is the point:

```text
                        one logical run
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │     ← SHARDING splits
   ┌───────────┐         ┌───────────┐         ┌───────────┐  the work universe
   │ shard 0/3 │         │ shard 1/3 │         │ shard 2/3 │  into independent
   │ cluster A │         │ cluster A │         │ cluster B │  jobs
   └───────────┘         └───────────┘         └───────────┘
         │                     │                     │
   attempt 1 ✗           attempt 1 ✓           attempt 1 ✗
     crashed                                     preempted
         │                     │                     │
   attempt 2 ✓                 │               attempt 2 ✓     ← CHECKPOINTING
     resumes at the            │                 resumes at the   resumes an
     row it reached            │                 row it reached   attempt WITHIN
         │                     │                     │            one shard
         └─────────────────────┴─────────────────────┘
                               │
                    the whole work scope, covered
```

Each layer makes the other more useful. Sharding bounds what one failure can
cost — a lost job is one shard's worth of in-flight work, not the corpus.
Checkpointing bounds what a *retry* costs inside that shard, so a failure in the
last hour of a shard's work does not restart the shard. Without sharding, a
checkpointed run is still a single point of scheduling failure. Without
checkpointing, a shard failing late is as expensive as never having sharded.

## Terminology

Three different things partition data in this system, at three different layers,
and they are routinely confused. This document fixes the names.

```text
Lance fragment        physical storage unit, owned by the PRODUCER leg.
                      Durable, stable id in the table manifest.
      │
      │  ← sharding divides these, by manifest id
      ▼
work shard            one JOB's execution scope: the fragments whose id
                      satisfies this job's ownership function.
      │
      │  ← the job's own plan then reads its owned fragments
      ▼
Ray Data block        in-memory unit of parallelism INSIDE one job.
                      Transient, sized for execution, never durable.
```

- A **Lance fragment** is the producer's durable storage concern. A consuming
  recipe does not create, merge, split, or reorder them.
- A **work shard** is an *execution scope* over those fragments. It exists only
  in one job's configuration and log lines.
- A **Ray Data block** is the unit the executor moves and parallelizes over
  within a job. Blocks are re-bundled, split, and fused by the plan; nothing
  durable may be keyed on them.

Two rules follow from the layering:

- **Never shard on a Ray Data block.** Blocks are not stable across plans or
  versions, so a block-keyed scope is not reproducible — and the executor is
  free to re-bundle, split, and fuse them, so the same table can yield different
  blocks on two runs of the same config.
- **Never treat a shard as storage geometry.** A shard does not rewrite,
  compact, or repartition the table. Fragment geometry belongs to the producer
  leg, whose publish batch size is what decides it — see
  [`robot-action-split`](curator-next-robot-action-split.md) for the producing
  side. A consumer that reshaped fragments to suit its own parallelism would be
  rewriting another leg's durable storage to buy execution convenience it can
  get from Ray blocks for free.

## The mechanism

`ShardConfig` carries `count` and `index`, and assigns a work unit by modulo over
its stable id:

```text
owns(work_unit_id)  ≡  work_unit_id % count == index
```

That single expression is the whole primitive, and everything the design needs
follows from it being a pure function of the id alone:

- **Disjoint.** Each id has exactly one residue, so exactly one owner. No two
  jobs can claim the same work unit, which is what makes concurrent execution
  safe without a lock.
- **Covering.** Every residue in `[0, count)` is some shard's, so launching
  indices `0` through `count - 1` covers the universe. No id is unowned.
- **Stateless.** Ownership needs no manifest, no coordinator, and no agreement
  between jobs. Two jobs planning a week apart, on different clusters, derive the
  same partition from the same table.
- **Order-free.** Shards may run in any order, concurrently or serially, and a
  shard may be re-launched any number of times.

Both fields are **required** when the object is present, and that is deliberate:
a half-configured scope — a count with no index, or an index with no count — would
silently run one shard while the remaining work was claimed by no job at all. The
optionality lives on the enclosing config field instead, where `None` means
unsharded and says so.

`index >= count` is refused at parse time by a cross-field validator, because
each field's own bound cannot express the relation between them. An out-of-range
index owns nothing, so without the check a typo produces a job that starts, finds
no work, exits zero, and reports success.

### Why the id is an integer, and why there is no configurable shard key

The ownership function takes one `int`. It does not take a column name, a key
expression, or a hash function, and it should not.

- **A configurable key is a correctness surface, not a feature.** Every
  additional key shape is another way for two jobs in one campaign to disagree
  about the partition, which is the precise defect the shared type exists to
  prevent. A campaign whose jobs shard on different keys has gaps and overlaps,
  and nothing detects it.
- **The work unit is already decided by the storage layer.** A consuming recipe
  reads a Lance table, and the durable, stable, manifest-recorded unit of that
  table is the fragment. Its id is a small integer. Offering to shard on an
  arbitrary column would invite sharding on something with none of the properties
  the partition needs.
- **Hashing a string key buys nothing here and costs recovery.** A clip id could
  be hashed to a residue, and the partition would be just as disjoint, but the
  scope would no longer align with fragments — so every job would read every
  fragment and discard most of its rows, and the read cost of the campaign would
  multiply by `count`.

If a future recipe's durable work unit is genuinely not a fragment, the correct
change is to give that recipe a work-unit id of its own and pass it to the same
`owns()`, not to make the ownership function configurable.

### Why composition, not a mixin

A recipe adopts sharding by holding a `ShardConfig | None` field, not by
inheriting from a shard-aware base config.

- **A mixin cannot express the optionality.** Inheriting puts `count` and `index`
  on the recipe's config directly, so "unsharded" would have to be spelled as
  both fields being absent — which is exactly the half-configured scope the
  required pair rules out. Holding an optional *object* keeps "no shard" and "a
  complete shard" as the only two representable states.
- **A mixin's `model_config` reaches every adopter.** Pydantic v2 merges config
  across bases (measured at 2.13.5: a `frozen=True` mixin combined with an
  `extra="forbid"` base yields both), so a shard mixin carrying any config of
  its own would silently change the strictness of every recipe that adopted it.
  That is a wide blast radius for two integers.
- **There is no base class to extend.** Recipe configs share no common ancestor
  today, so a mixin would introduce a cross-recipe inheritance edge whose only
  cargo is those two integers — coupling every adopter to every other one's
  future changes.

## A shard is execution scope, never result semantics

This is the invariant that keeps sharding from contaminating the data it
produces: **which job produced a row must not be able to change what that row
says.**

So a shard index and count belong in a run's logs, its summary, and its recovery
namespace — and in **no result digest, fingerprint, or durable per-row column**.
Concretely:

- Changing `count` or `index` must not make an already-produced row stale. A row
  produced by shard 3 of 8 satisfies the result contract exactly as the same row
  from an unsharded run would, so a re-plan under a different campaign shape must
  find it complete and not re-own it. (The row's *value* may be unreproducible —
  a generative recipe is unseeded — which is precisely why re-owning it is
  destructive rather than merely wasteful.)
- A shard may therefore not appear in a result contract digest. If it did, the
  campaign's own shape would invalidate its own output, and re-sharding a corpus
  would recaption, re-embed, or re-score all of it.
- Rows a shard does not own are simply **left owed**, to be produced identically
  by whichever job does own them. "Not mine" is never a verdict written to a row.

The shard **does** belong in a recovery namespace, and that is not a
contradiction. A namespace answers *may these two attempts share staged, uncommitted
work?* — and two different shards must not, because they are concurrent writers
producing different rows. Separating them by namespace is what makes concurrent
shards safe. A namespace is not result identity; it is deleted after a commit.

Two mechanisms make a shared namespace destructive rather than merely untidy,
and the second is not the adopting recipe's to fix. A recipe that deletes its
namespace after a commit deletes every other shard's staged rows with it. And
Ray Data's own checkpoint recovery is job-blind: measured at Ray 2.58.0, each
job's `load_checkpoint` deletes *every* pending checkpoint entry it finds in the
directory along with the data files those entries name, including entries a
concurrently running job wrote seconds earlier. Scoping the recipe's cleanup to
the fragments one job owns therefore would not buy shared staging; it would only
leave the checkpoint index unprunable. An adopter wanting shard-independent reuse
gets it from committed rows and the owed-work predicate, not from staging.

Nor can the checkpoint index be split per work unit to escape that. A Ray Data
job carries exactly one `CheckpointConfig`, whose `checkpoint_path` is a single
string set on the `DataContext`; there is no per-operator or per-input-partition
scoping at 2.58.0. A checkpoint directory per work unit therefore means a Ray
Data **job** per work unit, paying engine-pool startup for each one — which is
why the shard, not the work unit, is the namespace input.

### Publish early so that committed work is the reuse

Since committed rows are the shard-independent unit, the practical lever is to
stop deferring them. A recipe whose publication holds every result for one
terminal commit converts any late failure into work that only its own namespace
can recover, and a re-launch under a different `count` cannot reach it. Splitting
the publication into a bounded number of commits makes each landed one reusable
under every later campaign shape.

Two properties make that split safe and affordable, both measured at pylance
11.0.0:

- An `Update` naming fragments no earlier commit touched **lands against a stale
  read version**, so a sequence of commits needs no re-open and no re-planning —
  the same fragment-disjointness that lets two shards commit concurrently.
- A commit costs one manifest, and a manifest enumerates **every** fragment of
  the table rather than the ones the commit touched (82 bytes per fragment,
  linear from 100 to 4,000). N commits therefore cost O(N × F) manifest bytes, so
  the commit **count** must be capped rather than the fragments per commit: one
  commit per fragment is O(F²), roughly 73 GiB of manifests for a 250M-row
  table's ~31k fragments.

This shortens the exposure but does not remove it. Publication runs after
inference, so a failure during the inference phase still strands that job's
staged rows against a `count` change. Choosing `count` per campaign remains the
rule; what changes is that finishing inference, not finishing the whole run, is
what makes progress survive a re-shape.

## Owed work under a shard

Recipes here select work with a predicate — "rows whose output column is still
null, or stale under the current contract". Sharding interacts with that
predicate in one specific way that must be got right, and the order is what
matters.

**Narrow the work-unit universe by ownership BEFORE counting owed rows.**

```text
   fragment universe        every fragment the pinned manifest holds
         │
         ▼
   shard ownership          decided on the manifest id ALONE, so it is
         │                  independent of how much work remains
         ▼
   owed-row count           one filtered count per OWNED fragment
         │
         ▼
   read tasks               one per owed fragment
```

Two reasons, and the second is the subtle one:

1. **Cost.** A filtered count per fragment is a real scan-planner call. Counting
   the whole table in every job multiplies that driver cost by `count` for
   nothing.
2. **Honest diagnostics.** Every owed figure a job reports must be *its own*. A
   job that prints a table-wide pending count is describing work it will not do,
   and an operator reading it concludes the job is broken or stuck. The rule is
   that a shard reports what it owns: owned work units, owed work units, owed
   rows within them, and nothing table-wide.

The same rule applies to the "nothing owed" diagnostic, which is where it is
easiest to get wrong. When a job finds no work, an explanation naming a
table-wide population is actively misleading — the rows it names may belong
entirely to other shards. The count must be scoped to the fragments this job
owns, and the message must say so, so that the number cannot be read as a
corpus-wide claim.

Ownership being decided on the id alone has a second benefit worth naming: a
shard's scope does not drift as work completes. Shard 3 of 8 owns the same
fragments whether none, half, or all of their rows are already done, so a
re-launch of that shard is reproducible.

## Adoption checklist

Adopting `ShardConfig` is a small change; being *safe* to shard is not. A recipe
works through all five of these before adding the field.

1. **Is the checkpoint id stable across every mutation a multi-job campaign
   permits?** Answer it by enumerating those mutations, not by classifying the id
   as physical or logical. A Lance row address is stable under far more than its
   "meaningful only within a snapshot" caveat suggests: measured at pylance
   10.0.0, a metadata-only `add_columns` and a per-fragment `update_columns`
   commit — a sibling shard's publication included — leave every address
   untouched. What breaks it is a **fragment rewrite**, and compaction rewrites
   into fresh fragment ids, so a `(fragment_id, business_key)` composite is not
   the hedge it looks like — its fragment half moves with the rewrite too, at
   several times the width. Prefer the narrow `uint64` when the campaign's
   commits do not rewrite fragments, since the id width sets the driver's
   checkpoint-index size. Outliving a rewrite needs a purely **logical** key —
   the business key alone, or a durable work-unit id the storage layer does not
   assign — which costs both the width and a uniqueness contract the recipe must
   then enforce itself. Either way, state the excluded mutation as an
   operational rule. This is the prerequisite that most often requires real
   work.
2. **Are the work units genuinely independent?** A recipe whose output depends on
   a corpus-wide reduction — a fitted basis, a global k-means, a
   population-derived threshold — produces *different output* when its rows are
   divided, and does so silently. Such a recipe must not shard on the reduced
   axis at all.
3. **Does every owed-work figure scope to the shard?** See
   [Owed work under a shard](#owed-work-under-a-shard). A table-wide count in a
   sharded job is a diagnostic defect, not a cosmetic one.
4. **Is the shard out of every result digest, and in the recovery namespace?**
   Both halves. Out of the digest, or a re-shard invalidates prior output; out of
   the namespace, and two concurrent shards share a staging directory.
5. **Is the write path safe for concurrent committers?** Sharding makes
   concurrent commits to one table the *expected* case rather than an accident,
   so every commit the recipe issues must tolerate losing a race and retrying.
   A commit scoped to the rows a job owns can retry safely; a whole-table
   rewrite cannot, because retrying it after losing a race would discard the
   concurrent shard's commit rather than merge with it. A metadata-only schema
   widening is the one commit several jobs will genuinely race on, since each
   job independently needs the columns to exist before it can plan.

## Limitations

- **Balance is by work-unit count, not by cost.** Modulo distributes fragment
  *ids* evenly; it does not know how many rows a fragment still owes. A corpus
  whose remaining work is concentrated in a few fragments will be unevenly
  divided. This is accepted: a cost-balanced assignment would need a coordinator
  or a manifest, which is the state this design exists without. Where skew
  matters, raise `count` — more, smaller shards reduce the worst-case share
  without adding any coordination.
- **Nothing verifies that a campaign completed.** No registry records which
  indices ran, so launching indices 0, 1, and 3 of 4 leaves shard 2's work owed
  with nothing reporting it. Completion is observable — re-plan and read the owed
  count — but it is not *asserted*. Campaign bookkeeping belongs where jobs are
  launched.
- **Changing `count` mid-campaign re-partitions everything.** Ownership is
  modulo `count`, so a different count is a different partition. Work already
  committed is unaffected (a shard is not result identity), but in-flight
  uncommitted progress under the old count is stranded in namespaces the new
  shape will not resolve to. Choose `count` for the campaign and keep it.
- **Sharding does not make a mutating source safe.** It bounds what one job
  loses; it does not license compaction or any other rewrite of the source while
  jobs hold progress. A campaign spanning days widens the window in which
  somebody might try, which makes the operational rule *more* important, not
  less.

## Operator workflow

A campaign is `count` launches of the same config, differing only in the shard
index. There is no campaign-level command, by design — the adopting recipe's
config is the whole interface, so a campaign is driven by whatever already
submits that recipe's jobs.

```bash
# One config file per index, identical but for the shard index. Indices are
# ZERO-BASED, so a count of 8 means 0 through 7 inclusive.
for index in $(seq 0 7); do
  yq ".input.shard = {\"count\": 8, \"index\": ${index}}" run.yaml \
    > "run-shard-${index}.yaml"
done
# Then submit each file the same way a single unsharded run is submitted.
```

The shard is echoed in the job's own log line and in its run summary, so an
operator reading one job's output can tell which slice it was. Checking a
campaign is therefore per-shard: a shard that failed is re-launched with the same
index, and it resumes — from its checkpoint if its staged progress survived, from
its owed set if it did not.

Confirming a campaign finished means **re-launching every index**, not sampling
one: a job reports only its own slice, so one shard reporting nothing owed says
nothing about the others. That is cheap in a recipe that plans before it acquires
anything — a job owing no work returns from its driver plan without a GPU, which
is how the [reference
adoption](curator-next-robot-action-caption.md#driving-a-campaign) behaves — and
it makes the same action both the check and the fix: shards already complete cost
a plan, and shards that are not do the remaining work. A recipe that acquires
before planning does not get this property, and should say so.
