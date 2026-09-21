# Curator Next: Robot Action Caption

## Summary

`robot-action-caption` captions the egocentric manipulation clips in the
`clips.lance` table produced by Curator Next
[`robot-action-split`](curator-next-robot-action-split.md). By default it reads
only clips whose final Curate verdict is `curate_selection_reason = 'selected'`;
`input.selected_only: false` explicitly broadens the attempt to every otherwise
pending clip. It generates one natural-language description with a globally
registered model through the recipe-private vLLM executor and fills five nullable
caption columns on the rows it read. `model.variant` names a key in
`cosmos_curator/configs/all_models.json`; the registry pins artifact identity,
while the real vLLM construction decides runtime compatibility.

The executor stays inside this recipe while Curator Next recipe designs incubate
independently. `engine.py` owns processor construction, batching, horizontal
replication, and restart bounds; `datasource.py` and `inference.py` own staged-row
identity and checkpoint composition. The recipe also owns the prompt, media
binding, five output columns, publication policy, and recovery identity.

Execution is two-phase, for one reason: generation is **stochastic and unseeded**,
so re-running a clip does not reproduce its caption — it produces a different one.

1. **Phase A — checkpointed inference.** One streaming Ray Data plan renders
   prompts, fetches media, runs inference, and writes terminal rows to durable
   Parquet under a snapshot-scoped workspace. Ray Data's job-level row
   checkpointing records completed rows, so a restart pays only for clips that had
   not finished. Phase A never touches Lance.
2. **Phase B — validate everything, then commit in bounded waves.** The staged
   rows are validated as one set, every touched fragment is prepared, and the
   prepared set is published by at most 16 `LanceOperation.Update` transactions —
   exactly one when the publication is narrow.

A row is pending when no run has claimed it, when the run that did used different
semantics or model identity, or when its five-column caption group is not a valid
terminal row. Latest contract wins, and malformed current rows are re-owed: both
are recomputed and overwritten by the same publication.

At corpus scale one job is not enough, so `input.shard` divides the table's
fragments across independently launched jobs — different clusters, different
times, independent retries — whose union covers the work with no gap and no
overlap. That is a second, complementary reliability mechanism: row checkpointing
makes one attempt cheap to resume, and sharding bounds what one job's total
failure can cost. Neither reaches the caption contract, so which job captioned a
clip cannot change what that caption says. See
[Sharding a caption campaign](#sharding-a-caption-campaign) and, for the shared
primitive, [Deterministic Work Sharding](curator-next-sharding.md).

For one pinned attempt, **owed rows are the intersection of caption-pending rows
and the configured input scope**. Every phase uses the same Lance predicate for
that intersection. Curate verdicts other than `selected`, including NULL, are
untouched in default mode; NULL means Curate did not claim the row, not "selected
by default." Selection changes which rows run, never what caption a row would
produce, so it is absent from the caption contract digest.

## Context and Responsibilities

`robot-action-split` owns row creation, fragment creation, span identity, and clip
media. It writes `clip_id`, `clip_uri`, `task_name`, `subtask_name`,
`camera_motion_annotation`, and the rest of the span-owned schema. This recipe
adds columns and never adds, removes, reorders, or rewrites a row.

| Leg | Reads | Writes | Relationship |
| --- | --- | --- | --- |
| [`robot-action-split`](curator-next-robot-action-split.md) | LeRobot/Mecka shards | the whole clip row | upstream producer; append-only |
| [`embeddings`](curator-next-embeddings.md) | `clip_uri`, labels | `embedding_*` column groups | sibling additive writer |
| **`robot-action-caption`** | `clip_id`, `clip_uri`, `curate_selection_reason` in default mode, and whichever context columns the snapshot supplies | the five `caption_*` columns | this document |
| [`curate`](curator-next-curation.md) | embedding columns | `curate_*` columns | sibling additive writer and producer of captioning's default input scope; run first unless caption selection is disabled |

All three enrichment legs follow the additive nullable feature-group contract:
one owner per column group, metadata-only widening, and a column-local atomic
commit. Two of them are relevant here as **siblings on the same table** — a
Curate run and a caption run can each commit their own version of `clips.lance`
without seeing each other's columns, because neither rewrites base columns. What
is outside the consistency model is two concurrent writers of the *same* group.

That sibling independence holds for writes that are **time-separated or
fragment-disjoint**, and only for those — which is precisely why a work shard
owns whole fragments rather than rows. Concurrent shards of one campaign land in
the fragment-disjoint case by construction, so their `Update` commits do not
conflict with each other however they interleave. The one commit they *do* race
on is the shared `add_columns` widening, which is handled where it is issued
([Concurrent shards and the schema
widening](#concurrent-shards-and-the-schema-widening)). Lance decides an `Update` commit's
conflicts by fragment id, so two sibling legs whose commits interleave across
each other's pinned versions on a **shared fragment** do not silently merge —
the later commit is **rejected** with a retryable conflict error. A sibling
`add_columns` is stronger still: it widens the schema, so every fragment's file
list becomes stale at once and the pinned commit is rejected without fragment
ids being consulted at all. `fields_modified` plays no part in that decision; it
is index invalidation, not conflict scope.

Rejection is the safe outcome rather than a defect. `LanceOperation.Update`
replaces a fragment's *entire* file list, so committing a list assembled before
the sibling's file existed would unlink that file — which is exactly the
cross-leg data loss the check prevents. For this recipe a conflict rejection
surfaces as an `OSError` — Lance routes commit conflicts there, not past it — so
it lands in
the [ambiguous-commit path](#an-ambiguous-commit-is-verified-not-trusted), which
re-reads the table, finds the wave's rows still pending, and raises. That is one
of three outcomes there, and it is the one a sibling conflict necessarily reaches:
a rejected `Update` or `add_columns` leaves the wave's fragments in place, so the
unverifiable branch cannot fire. Nothing is published and the staged rows are
intact.

The consequence is therefore an **availability** property, not a correctness one,
and it has a leg-scheduling implication: sibling legs sharing this table should
be scheduled so their commits do not interleave. Overlapping them costs the loser
its publication attempt — cheap for this recipe, because a re-run republishes the
staged rows without re-inferring, but not free, and a leg whose commit is not
similarly verified may pay more. Note that this recipe's own widening is an
`add_columns`, so a first run against a table without the caption columns will
reject any sibling's pinned `Update` unconditionally.

The recipe relies on these upstream properties:

- `clip_id` is unique and immutable as an **upstream table-domain contract**, and
  `clip_uri` is immutable for an existing row. That is what makes a pinned table
  version a sufficient stand-in for media identity — there is no per-row media
  fingerprint. This recipe does not re-scan the table to prove global `clip_id`
  uniqueness, and does not need to: neither half keys on `clip_id` alone.
  Phase A's checkpoint id is the row's own Lance address, and Phase B scopes
  staging, coverage, and `update_columns` by fragment, so the identity is always
  a row or a fragment rather than a bare clip. Equal ids in different fragments
  therefore cannot fold together or receive each other's captions; consumers that
  need the producer's global uniqueness guarantee validate it at the producer
  boundary.
- `clip_id` and `clip_uri` **exist and hold usable text**. They are the only two
  structurally required columns, because without them no row is captionable at
  all: one is the caption's identity, the other its media location. In default
  selected-only mode, `curate_selection_reason` must also exist with exact Arrow
  `string` type. Curate writes it nullable; captioning does not require the
  nullable schema flag because equality has the same semantics either way. Its
  only selected value is `selected`; every
  other enum value and NULL is outside the attempt. Setting
  `input.selected_only: false` removes that requirement and uses the caption
  pending predicate alone. All checks run on the driver, before the widening, so an invalid
  source is reported with a usable error and without having committed a schema
  change to it. See `require_source_columns` in
  [`table.py`](../../../cosmos_curator/next/recipes/robot_action_caption/table.py).

  The two get **different type gates**, because the recipe uses them
  differently. `clip_uri` is only ever read, so any encoding a scan hands back as
  `str` serves — `string`, `large_string`, `string_view`, or a dictionary of
  those, the wide `is_text_type` predicate. `clip_id` is also the **join key**
  publication updates on, and is held to exactly `string`. Publication types its
  update table from the *stored* schema so `update_columns` never coerces, which
  means a `large_string` or dictionary-encoded id would be joined at that type on
  both sides — not a documented capability of that call, and a join that matches
  nothing is not an error: it would commit zero updated rows inside a Ray task,
  where the exception type is erased, and the run would report success having
  published no caption. Widening this gate means proving the join handles the
  type, not merely that the value reads back as `str`.
- Every **context** column is optional in the sense that its *absence* is
  allowed. A column the snapshot does not supply is not part of that run's
  contract: it is resolved away rather than required, the prompt renders one line
  fewer for every row in the run, and the resolved subset is recorded in the
  digest.

  A column that **exists but does not hold text** is a different state and is
  **refused** — `resolve_context_fields` raises `ValueError`. Treating it as
  absent would silently drop one prompt line from every caption in the run while
  stamping the corpus `success`, and the pending predicate would then skip those
  rows forever. Absent is a property of the snapshot; present-but-mistyped is a
  producer-side contract fault, and only the second has an owner who can fix it.
  The check runs on the driver *before* the widening, so the refusal leaves no
  caption columns on a table this run never captioned. See
  `resolve_context_fields` in
  [`table.py`](../../../cosmos_curator/next/recipes/robot_action_caption/table.py).
- Splitting is append-only. Rows appended after a run pins its snapshot are not
  part of that attempt and stay pending for the next one.

Caption text is **semantic only**. Nothing downstream may derive geometry,
localization, or motion from it; those come from the deterministic action
descriptor described in the [embeddings design](curator-next-embeddings.md).

## Design Invariants

1. **Lance is canonical.** Readers never need the staging workspace.
2. **Publication accepts all or nothing, and publishes in whole fragments.** Every
   staged row is validated before the first transaction, so a run either refuses
   with the table untouched or commits only rows it has already accepted. The
   writing is then split into at most 16 waves, so an *interrupted* publication
   can leave earlier waves canonical — but never a partially published fragment,
   which is what keeps the pending predicate honest at every instant.
3. **Completed inference is durable and reusable.** A crash before publication
   loses no generation whose staged row and checkpoint were committed.
4. **Two runs share staged work if and only if every namespace input matches.**
   The workspace path encodes the source URI, the contract digest, the shard, and
   the staging layout, so runs any of those separate cannot see each other's rows
   and no per-row compatibility check exists. Input scope is deliberately absent,
   so a widened attempt resumes the narrower one it widens; the pinned read
   version is absent too, so a resume reuses work across a commit that moved the
   version without moving rows. What the path does *not* declare — the identity of
   the table at the source URI, and which other run is writing one namespace — is
   operator discipline rather than an invariant ([Limitations](#limitations)).
5. **A shard divides execution, never results.** Which job captioned a clip
   cannot change what that caption says, so the shard reaches the recovery
   namespace and the run's logs and summary, and reaches no digest or durable
   column ([Sharding a caption campaign](#sharding-a-caption-campaign)).
6. **Result identity is independent of throughput tuning.** Replica counts, batch
   sizes, retry budgets, and CPU reservations do not appear in the digest.
7. **One corrupt clip does not fail a run; one bad credential does not become
   millions of error rows.** The invariant governs **classification, not
   extent**: it decides which side of the line a fault lands on, and nothing
   bounds how many rows end up carrying a row-local verdict. See
   [Error taxonomy](#error-taxonomy).
8. **An attempt is finite.** Work is selected from one pinned snapshot.

## End-to-end Data Flow

The step order is load-bearing rather than incidental. This section is the
authoritative explanation; the compact
[`pipeline.py`](../../../cosmos_curator/next/recipes/robot_action_caption/pipeline.py)
module contract links here. Five orderings each prevent a specific failure,
called out under the diagram.

```text
Driver
  require_source_columns(ds)   refuse a bad table BEFORE mutating it
        │
        ▼
  require_selection_column(ds, selected_only=...)
        │                      default scope requires Curate's string verdict
        ▼
  ds = ensure_caption_fields(uri)  add the five nullable columns if absent
        │                      (metadata-only; opens its own handle per attempt
        │                       and retries a LOST schema race — see below)
        ▼
  context_fields = resolve_context_fields(ds)
        │                      same snapshot as the pin below
        ▼
  read_version = ds.version    pin from the handle that widened the table
        │
        ▼
  digest = caption_contract_digest(model, context_fields,
        │                          generation=, video=, media=)
        ▼
  row_filter = caption_work_filter(digest, model, selected_only=...)
        │
        ▼
  plan_caption_scan(ds, row_filter, shard=...)
        │                      keep the fragments this job OWNS, then one
        │                      owed-row count per owned fragment
        ├── nothing owed ──►  return; no model validation/workspace/Ray
        │
        ▼
  validate_model_directory(...)       config plus recognized weight file
  resolve_workspace(...)              pure path arithmetic; writes nothing
  ensure_ray_initialized(...)
        │
        ▼
Phase A — Ray Data, one streaming plan
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Read      pinned Lance fragments, one block per owed fragment        │
  │           emits clip_id, fragment_id, clip_uri, prompt_text,         │
  │           __checkpoint_id (the row's Lance address)                  │
  │             │                                                        │
  │             ▼  Ray CheckpointFilter (already-completed rows dropped) │
  │ repartition lightweight rows into bounded, fragment-local fetch      │
  │             │  blocks (strict=False; identity columns unchanged)     │
  │             ▼                                                        │
  │ map_batches ClipMediaFetch → video_data_url | media_error            │
  │             │               systemic fault or spent budget → raise   │
  │             ▼                                                        │
  │ pipe        build_request  pre-seed outputs; route media_error into  │
  │             │      __inference_error__ → vLLM → read_generation      │
  │             ▼                                                        │
  │ map_batches canonicalize: classify the terminal status, stamp the    │
  │             │             digest and model id, select STAGED_SCHEMA  │
  │             ▼                                                        │
  │ Write     write_parquet(partition_cols=["fragment_id"], mode=append) │
  └──────────────────────────────────────────────────────────────────────┘
        │
        ▼
Phase B — publication
  read staged Parquet → fold duplicate clip_ids → validate → prepare each
  touched fragment with update_columns (parallel Ray tasks, no commit)
        │
        ▼
  refuse set mismatch, duplicate envelopes, or metadata/envelope id mismatch
        │
        ▼
  at most 16 LanceOperation.Update waves over the prepared fragments
        │
        ▼
  clips.lance @ v+K        ← a clip is complete once its wave lands
        │
        ▼
  cleanup of the staging namespace (after commit, never fatal)
```

- **Validating before widening** means an invalid source is refused without
  having committed a schema change to somebody else's table.
- **Validating context before widening** happens inside
  `ensure_caption_fields` on every reopen (`_validated_context_fields`): a
  present-but-wrong-typed context column is refused before any metadata commit,
  so the table is not left with five new caption fields from a run that never
  captioned. **Resolving** which optional columns supply the prompt happens
  **after** widening, from the same handle whose version is pinned: that call
  may reopen past an earlier driver open, and the digest must describe the
  pinned snapshot, not a stale one.
- **Pinning after the widening** means the run reads a snapshot whose caption
  columns exist, so its own predicate can reference them. It is pinned from the
  handle `ensure_caption_fields` **returned** rather than from the handle the
  caller held, because that call may have lost a schema race to a concurrent
  shard and reopened: a stale handle would pin a version predating the columns.
  Unlike the earlier design, the pinned version no longer has to *match* between
  a run and its resume — it is not a namespace input.
- **Owning before counting** means each job's owed figures describe its own
  slice. Counting first and filtering after would have every shard scan the
  whole table rather than the fragments it owns, and make every job report work
  it will not do.
- **Planning before acquiring anything** means a run that owes nothing needs no
  model-directory validation, workspace, Ray cluster, or model initialization.
  That is the case a monitoring loop re-runs constantly, and it is the one that
  must be cheap. It is also what makes re-launching a finished shard safe and
  nearly free, which is how a campaign is closed out.

The prompt is rendered **once**, on the scan, and travels as a `prompt_text`
column. It is not re-derived closer to the engine: two code paths normalizing the
same fields is how one table ends up holding two subtly different prompts under a
single contract digest. See
[`prompt.py`](../../../cosmos_curator/next/recipes/robot_action_caption/prompt.py).

## Caption Result Contract

Five flat nullable columns, matching the convention Curate already uses on this
table rather than the caption-plus-metadata struct that
[`video-caption`](curator-next-video-captioning.md) uses on the `video_split`
table. Names, Arrow types, and the terminal vocabulary are module constants in
[`contracts.py`](../../../cosmos_curator/next/recipes/robot_action_caption/contracts.py)
(`CAPTION_COLUMNS`, `CAPTION_FIELDS`, `CaptionStatus`).

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `caption` | `large_string` | Generated text. Nonblank iff the status is `success` or `truncated`. |
| `caption_status` | `string` | One of `success`, `truncated`, `error`. |
| `caption_error` | `large_string` | Nonblank iff the status is `error`. Human-readable, no stack trace. |
| `caption_contract_digest` | `string` | Which semantics produced the row. Also what makes the row current or stale. |
| `caption_model` | `string` | Resolved model id, for triage without parsing the digest. |

Every field is nullable so the whole set can be added to an existing table with a
metadata-only `add_columns` — no data-file rewrite and no backfill.

### The field set is registered atomically, and a partial group is refused

`ensure_caption_fields` recognizes exactly **three** states of the five-column
group, and there is no fourth:

| State | What happens |
| --- | --- |
| all five absent | registered in one `add_columns` commit |
| all five present | validated against `CAPTION_FIELDS` exactly — Arrow type and nullability — with **no commit** |
| a non-empty strict subset | **refused**, naming which are present and which are missing |

The subset is refused rather than topped up because **a partial group has
ambiguous ownership**. The group is registered atomically, so this recipe cannot
produce a strict subset by crashing; something else did. Silently completing it
would normalize an unexplained schema into a valid published one and take
ownership of columns this recipe never registered — and if that other writer owns
the missing fields, the repair is the data loss. Refusing costs an operator one
diagnosis; repairing costs the wrong answer to a question nobody asked.

Three details are load-bearing:

- **Validation precedes any commit.** The context-column type check runs first,
  fail-closed, so a metadata commit can never precede reporting a malformed
  source field; and a mistyped caption field is refused without having committed
  the columns around it. Type and nullability are published contracts and cannot
  be narrowed in place.
- **The complete-group path commits nothing, deliberately.** It is a contract,
  not an optimization, and for two reasons that survive the read version having
  been removed from the namespace identity (see the comment in
  [`workspace.py`](../../../cosmos_curator/next/recipes/robot_action_caption/workspace.py)
  — two attempts pinning different versions are *meant* to share completed work,
  so a second commit no longer splits namespaces). First, this function runs on
  every launch, resumes included, and widening happens at most once per table per
  contract generation: committing here would advance the table's version once per
  launch forever, for a schema that never changes, on the one table every
  concurrent job reads. Second, it is what makes a contended first launch
  terminate — every job that loses the race reopens onto this path instead of
  committing a widening of its own.
- **The state is re-read one more time than it is committed.** Registration gets
  a small fixed number of commit passes — not a configurable one, because the
  bound exists so a deterministic storage failure still surfaces, and the number
  of jobs racing to widen is not something an operator tunes against. Each pass
  re-opens the table, because a lost commit cannot be retried from the handle
  that lost it: the group the loser needs exists only at a version that handle
  cannot see. After the last commit the loop looks once more without committing,
  because a conflict on that commit is itself evidence that *some* writer
  committed — and if that writer registered this same group, there is nothing
  left to do. That final look is an ordinary classification pass, so a group that
  turns out to be a strict subset is refused as partial rather than reported as
  absent.

`add_columns` advances the handle in place, which is why the caller can pin the
read version on the very next line without re-opening the table. See
`ensure_caption_fields` in
[`table.py`](../../../cosmos_curator/next/recipes/robot_action_caption/table.py).

### Terminal states

There are exactly three. `terminal_row()` validates the pairing when a row is
created, and publication validates it again after the Parquet round trip before
preparing any fragment update. A row that violates the invariant has no correct
publication: there is no way to tell whether a status-less caption succeeded.

| Status | `caption` | `caption_error` | Meaning |
| --- | --- | --- | --- |
| `success` | nonblank | null | Generation stopped short of **both** bounds that could have cut it. |
| `truncated` | nonblank | null | Generation reached **either** bound — the output-token cap, or the context length net of the prompt. The text is usable but cut off. |
| `error` | null | nonblank | A row-local media fault, a per-request engine failure, or an empty generation. |

`truncated` exists because a cut-off caption is usable but must be
distinguishable. Detection is a token comparison rather than a `finish_reason`
read, because the Ray Data LLM engine stage does not surface that field — see
`caption_status()`.

The comparison is against **two bounds, not one**. A generation can stop because
it exhausted its sampling cap (`generation.max_output_tokens`), or because prompt
*plus* generation reached the engine's context length
(`generation.max_model_len`) — and a generation the second bound cut stops
*below* the first, so a one-bound rule would publish it as `success`. Both ask
the same question about the generated count, so both collapse into one ceiling:

```text
ceiling = min(generation.max_output_tokens,
              generation.max_model_len - prompt_token_count)
truncated when generated_token_count >= ceiling
```

Config enforces `max_output_tokens < max_model_len`, so the ceiling is positive
for any prompt that leaves room at all.

**Both token tallies are required, not optional.** `read_generation` indexes the
engine's `num_generated_tokens` and `num_input_tokens` keys directly and
`_token_count` refuses a non-integer or negative value, so a missing or unusable
tally raises rather than degrading to a one-bound comparison. That is the
deliberate choice: a silently degraded ceiling publishes a truncated caption as
`success`, and unlike a raise nothing downstream can tell that it happened. See
`read_generation` and `_token_count` in
[`request.py`](../../../cosmos_curator/next/recipes/robot_action_caption/request.py).

The prompt tally is the engine's own. If it counts multimodal placeholders
*before* expansion, subtraction leaves the ceiling above the real headroom and
the classifier can miss a context-bound truncation. An unverified estimate is
not safer: one below the real headroom would instead invent truncations.

At the separate bound comparison, equality is conservative. A generation that
ends naturally exactly on either declared bound is indistinguishable from one
the bound cut off, so that exact tie is reported as `truncated`. This local
tie-break does not establish a global preference between false-success and
false-truncation outcomes.

`canonicalize` in
[`request.py`](../../../cosmos_curator/next/recipes/robot_action_caption/request.py)
is the **sole author** of every published caption value. Neither engine hook
receives the digest or the model id, and the media stage writes no caption column
at all, so there is exactly one place a stamped digest can come from. A second
author is how two rows in one table come to disagree about what produced them.

### One work predicate combines caption validity with input scope

`caption_work_filter(digest, model_id, selected_only=...)` in `table.py` is the
single definition of what an attempt owes. Its private pending half considers a
row complete only when all of these hold:

- `caption_contract_digest` equals the current digest;
- `caption_model` equals the resolved model id;
- `caption_status` is exactly `success`, `truncated`, or `error`;
- `success` and `truncated` carry a nonblank caption and a null error;
- `error` carries a null caption and a nonblank error.

Every other shape is pending, including a current digest beside an unsupported
status, contradictory caption/error values, or a missing or foreign model id.
The next publication overwrites that entire sibling group. This fail-closed
choice keeps one pushdown predicate as the source of truth and prevents
`run_config` from returning ordinary no-work success for malformed current rows.
Foreign or missing digests remain **latest contract wins**: they are re-owed
exactly as before, so a contract change recaptions every row in the configured
scope instead of silently mixing semantics within that scope.

The public helper then intersects that pending definition with
`curate_selection_reason = 'selected'` when `input.selected_only` is true.
Lance's three-valued logic naturally excludes NULL as well as every non-selected
Curate verdict. When the switch is false, the helper returns the pending half
unchanged. The column and selected value are local read-contract constants in
the caption recipe: captioning consumes persisted table data and does not import
the Curate recipe at runtime.

Four details are non-obvious and each one is load-bearing:

- **Every nullable member has an explicit null arm.** In SQL three-valued logic a
  comparison against NULL evaluates to NULL rather than true, so merely negating
  the valid shape can lose a malformed row from the scan.
- **Identifiers stay bare, never quoted.** On the Lance scanner filter surface a
  double-quoted name parses as a *string literal*, which compares two constants
  and matches every row instead of raising.
- **The digest is validated before it is interpolated.** `caption_work_filter` refuses
  anything but a 64-character lowercase hex sha256, and that accepted alphabet
  cannot express a quote or an operator — which is what makes the interpolation
  injection-free rather than merely conventional.
- **The model id is escaped as a string literal.** Registry validation guarantees
  it is nonblank, but model names may contain punctuation that must remain data
  rather than SQL. Nonblank caption/error checks use Lance's supported regular
  expression predicate; its filter planner does not accept `trim`.

The same predicate drives every question the recipe asks about the table:
`plan_caption_scan` counts it per fragment, the datasource filters on it,
`owed_clip_ids` enumerates it within a fragment, publication validates exact
staged coverage against it, and ambiguous-commit verification re-uses it at the
latest observed version. An unselected caption-pending row therefore neither
launches inference nor blocks publication, and remains untouched for a later
broader or newly selected run. A column group its owner has never run is **absent
from the schema**, which is a different state from present-and-null; test presence
before nullity, as with every additive group on this table.

### Staging layout is versioned in the namespace, never in the digest

A change to how staged rows are written alters no caption's text, but staged
Parquet written under the old layout cannot be read back under the new one. So
such a change must **isolate old workspaces without recaptioning the corpus** —
which the namespace can express and the contract digest cannot. Two inputs carry
it, and they divide by *what* versus *how*:

| Input | Covers | Derived from |
| --- | --- | --- |
| `staged_schema_fingerprint()` | **what** a staged row carries — the ordered field names and Arrow types | hashed from `STAGED_SCHEMA` itself, so it cannot drift from the schema it describes |
| `STAGING_PROTOCOL_VERSION` | **how** those rows are written, checkpointed, and read back | a hand-maintained constant |

The protocol version exists only for changes the fingerprint cannot see: ones
that leave `STAGED_SCHEMA` byte-identical while changing a staged row's meaning
or recovery. Three kinds qualify:

- **how a completed row is identified** — the checkpoint id column, the value it
  derives from (the Lance row address today), and whether the checkpoint
  outlives a successful Phase A;
- **where and in what format the rows land** — the `results` and `checkpoints`
  segment names, the partition columns and their Hive `key=value` scheme, the
  Parquet format, the append-versus-overwrite mode;
- **what the sink's transaction is assumed to guarantee** — that a staged file
  which became durable without its checkpoint entry is discarded and replayed, so
  a Ray upgrade that moves the commit protocol or the checkpoint index format
  qualifies as well.

The enumerated trigger list is maintained beside the constant in
[`workspace.py`](../../../cosmos_curator/next/recipes/robot_action_caption/workspace.py),
which is where somebody about to bump it reads; keeping one copy is what stops
the two from drifting apart. Adding, removing, retyping, or reordering a staged
field needs no bump; the fingerprint already moved. Neither do throughput
choices: batch sizes and the rows-per-file hints change how many staged files a
phase leaves behind, never whether one can be read.

What the fingerprint hashes is each field's name, type, and nullability, in
order — and deliberately **not** Arrow schema metadata, at either the schema or
the field level. `STAGED_SCHEMA` carries none today, and metadata cannot change
how a staged row is read back: publication selects columns by name through
`pads.dataset(...).to_table(columns=...)`, and a metadata key round-trips into
`schema.metadata` without moving a column's type or its values under the
project's dependency contract. The one metadata mechanism that *does* change a
logical type on read — an Arrow extension type, whose name travels in serialized field
metadata — is already covered, because the extension identity appears in the
hashed type string (`extension<arrow.uuid>`) while `field.metadata` on such a
field is `None`. Hashing metadata would therefore be strictly weaker than what is
hashed now, and it is not free: any new fingerprint input moves every namespace
digest, abandoning every workspace that holds progress and re-inferring its
clips. Revisit only if a staged field gains metadata that a reader consults.

Both live beside what they version — the fingerprint with `STAGED_SCHEMA` in
[`request.py`](../../../cosmos_curator/next/recipes/robot_action_caption/request.py),
the constant in
[`workspace.py`](../../../cosmos_curator/next/recipes/robot_action_caption/workspace.py)
— and neither is a digest input.

## The Prompt

Frozen in
[`prompt.py`](../../../cosmos_curator/next/recipes/robot_action_caption/prompt.py):
the template text, `PROMPT_VERSION`, the supported context field tuple, and the
normalization rule.

`SUPPORTED_CONTEXT_FIELDS` is every column this prompt knows how to render, in
render order, each paired with its label: `task_name` (`Task`),
`task_description` (`Task description`), `subtask_name` (`Current action`), and
`camera_motion_annotation` (`Camera motion`). One definition drives the
projection, the rendering, the diagnostics, and the digest, so a column cannot be
projected without a label or labelled without being read.

Which of those a given run uses is **resolved per snapshot** against the source
schema, not per row. Resolution is what keeps one attempt's prompt contract
single-valued: every row in the run renders the same field set, the digest records
exactly that set, and `SCAN_SCHEMA` never varies with which optional columns
happen to exist — the resolved subset collapses into one `prompt_text` value
before any block leaves a read task.

Listing a column in the tuple therefore changes no digest on its own. **Landing it
upstream does**, because the resolved subset grows. That is intended — those
captions really were produced without it — and it is a scheduled event rather than
an incidental one. Reordering the tuple is likewise a deliberate digest change.
`task_description` sits directly after `task_name` because it elaborates the same
label.

Each resolved field is normalized identically — collapse internal whitespace runs
to a single space and strip. Collapsing is what preserves the `Label: value` line
structure, and so also what stops a column's free text from posing as a further
labelled line.

There is **no per-field length cap**. The reason is measured, not structural, and it
is worth stating precisely because the obvious assumption is wrong: these are not all
short label columns. Over the 131,602-row `mecka` corpus:

| Field | Max chars | Distinct | Shape |
|---|---:|---:|---|
| `task_name` | 48 | 2,738 | genuinely a short label (`folding_clothes`) |
| `subtask_name` | 188 | 107,341 | free-form prose, near-unique per row |
| `camera_motion_annotation` | 538 | 59,821 | prose; enumerates motion segments |

Caps are omitted because the **whole rendered prompt** still peaks at 1,241 characters
(min 554 / p50 666 / p95 910 / p99 1,038), roughly 310 tokens — negligible beside the
video frames in the same request. Capping individual fields would have truncated
nothing on this corpus.

That bound is a property of the **corpus**, not of the code:
`camera_motion_annotation` length scales with the number of motion segments detected
in a clip, so a corpus of longer or busier clips lengthens it with no code or schema
change. The measured corpus also did not supply `task_description`, so the numbers
above describe a three-line prompt and say nothing about what a fourth line adds.
Re-measure per corpus. The consequence of being wrong is severe rather than
graceful: the engine stage pins `should_continue_on_error=False`, so an over-length
request raises at the engine and fails the whole run instead of producing one terminal
error row.

A null or blank field is **omitted from the rendered prompt entirely**. On a
snapshot supplying `task_name`, `subtask_name`, and `camera_motion_annotation`:

```text
You are annotating egocentric video recorded during robot manipulation data collection. Describe only what is visible in the video.

Task: Put the bowl in the sink
Current action: grasp the bowl
Camera motion: slow pan left while the wrist approaches

Describe this clip in two to four sentences. Cover the manipulated objects, the hand or gripper motion, and how the scene changes from start to end. Write plain declarative prose with no preamble and no bullet points. Treat the context above as background only: do not restate it, and do not mention the camera, the recording, or the annotation task.
```

The same clip with `camera_motion_annotation` null loses the line rather than
rendering it empty:

```text
You are annotating egocentric video recorded during robot manipulation data collection. Describe only what is visible in the video.

Task: Put the bowl in the sink
Current action: grasp the bowl

Describe this clip in two to four sentences. Cover the manipulated objects, the hand or gripper motion, and how the scene changes from start to end. Write plain declarative prose with no preamble and no bullet points. Treat the context above as background only: do not restate it, and do not mention the camera, the recording, or the annotation task.
```

A bare `Camera motion:` line teaches the model that absence is itself a value, and
it reliably provokes commentary about the missing annotation instead of about the
video. A column absent from the whole snapshot produces the identical rendering —
the difference is that the digest records it, so the two populations are never
confused.

**To change the prompt, edit `prompt.py` and bump `PROMPT_VERSION`** — the digest
change is what makes prior staged work unreachable rather than silently mixed in.

## Result Identity: the Contract Digest

`caption_contract_digest` answers exactly one question: *is an existing output
valid under this run's semantics?* It therefore covers every value that changes
the text a model produces, and nothing that only changes how fast we get there.
The payload is `normalized_caption_contract()`, hashed by the repository's
`canonical_digest` over sorted compact JSON — do not hand-roll a hash.

| Block | Included | Why it is result-defining |
| --- | --- | --- |
| `model` | resolved model id, revision, precision | different weights, different text |
| `prompt` | `PROMPT_VERSION`, the system and output instruction text, the normalization rule, and the resolved context fields with their labels | the instruction *is* the contract, and the context decides which row values reach the model |
| `request` | role, content item order, media type, video items per prompt, whether media preparation runs | decides what the model is *shown*, not merely how the bytes get there |
| `sampling` | `generation.temperature`, `.top_p`, `.max_output_tokens`, `.max_model_len` | the output cap bounds the generation directly; the context length is the quieter second bound, cutting a caption short without anything raising — `caption_status` measures against it, so such a row publishes as `truncated`, and a row produced under a different context length was produced under different semantics |
| `video` | exact `video.media_io_kwargs()` and `video.mm_processor_kwargs()` outputs, plus `DO_SAMPLE_FRAMES` | records only the temporal and pixel override groups the runtime config requested, under the same key names vLLM receives; an empty mapping means the selected backend or artifact owns that behavior |
| `media` | `media.max_clip_bytes` | see below |
| `terminal` | `TERMINAL_POLICY_VERSION`, the status vocabulary, **both** truncation bounds — the output cap and the context length — and the verdict for an empty generation and a deterministic media fault | a row's shape and vocabulary are part of its meaning |

The terminal states are sorted by `.value` rather than definition order, so
renaming a member is a deliberate digest change while a cosmetic reorder is not.

The context length appears under both `sampling` and `terminal`, and that is not
redundancy: the two entries record different facts — that the engine was *built*
with it, and that the terminal verdict is *computed against* it — so a run under
a different output cap and a run under a different context length stamp
differently.

### `TERMINAL_POLICY_VERSION` covers what the thresholds cannot

The `terminal` block stamps the two threshold **values**. No value it carries can
describe the **shape of the comparison drawn against them**, and that gap is live
rather than theoretical. Every one of these re-verdicts every caption in the
configured scope while leaving the thresholds untouched:

- flipping `>=` to `>`, which moves the boundary by one token;
- swapping or dropping the `min` that folds the two bounds into one ceiling;
- changing how the classifier treats an unusable token tally.

Left unstamped, the pending predicate would keep skipping the rows the new rule
would label differently, and two populations would end up in one column with
nothing on a row to tell them apart — the same failure mode as [weights replaced
under a stable revision label](#limitations).

`TERMINAL_POLICY_VERSION` is what closes it. It is a digest input covering
comparison boundaries, fallback rules, and the deterministic-fault-versus-
systemic-failure boundary — everything about the classification *policy* that no
threshold value expresses. **A change of that kind must increment it**, or leave
the configured scope needing a recaption to stay comparable. It is deliberately separate
from the numeric bounds beside it, which are recorded explicitly and move on
their own. `contracts.py` carries the constraint as a code comment at the
`terminal` entries, so the next reader of `caption_status` meets it there too.

Five of those inputs are not module constants. `contracts.py`'s module docstring
names them at their owning boundaries, and they are the ones worth knowing before
they surprise someone:

- **The resolved model** comes from the shipped registry, so editing the selected
  alias's pinned revision in
  [`configs/all_models.json`](../../../cosmos_curator/configs/all_models.json)
  recaptions the configured scope. `resolve_registered_model` in
  [`models/registry.py`](../../../cosmos_curator/models/registry.py) is what
  turns that entry into the typed identity the digest stamps.
- **The resolved context subset** comes from the source snapshot. Landing a new
  supported context column upstream grows the subset, changes the digest, and
  therefore recaptions the configured scope. A column *disappearing* has the same effect in
  the other direction. Whether a row is recaptioned is consequently a
  property of the table's schema at read time, not only of this recipe's code.
- **`generation`, `video`, and `media`** are the three operator-configurable
  sections, and every field in them is a **result-contract value, not a tuning
  knob**. Raising `media.max_clip_bytes` converts terminal `error` rows into
  captions; changing `generation.temperature` changes the text outright;
  the video FPS, frame bounds, or pixel override changes what the caption
  describes. Two runs differing in any of them produce different populations
  from the same table, so treat all three sections as deployment-wide settings
  rather than per-run dials.

Excluded, deliberately: engine replica count, the concurrent-batch count, media
batch size, CPU reservations, retry budgets, and the staged field set and staging
protocol — which the [workspace namespace](#the-path-is-the-whole-compatibility-mechanism)
hashes instead. None of them move the text. Including them would invalidate a
checkpoint whenever somebody retuned throughput — which is exactly when a resume
matters most. The config surface in
[`config.py`](../../../cosmos_curator/next/recipes/robot_action_caption/config.py)
is built around that split, and it is the reason `generation`, `video`, and
`media` are sections of their own rather than corners of `execution`.

## Recovery

### The gap this closes

Without row checkpointing, the unit of lost work on a crash is the fragment. A
fragment holds thousands of clips, each costing seconds of GPU time, so a crash
one clip short of the end throws all of it away.

Inference is the expensive step. Prompt rendering and media fetch are cheap and
worth redoing, so the checkpoint records completed inference and nothing else.

The stronger reason is correctness. Sampling is stochastic and unseeded (see
`generation.temperature` / `.top_p`, which default to 0.7 and 0.8 and have no
seed field), so re-running a completed clip does not reproduce its
caption — it publishes a **different** one. Recovery therefore preserves the run's
own output; saving GPU hours is the secondary benefit. The same property is why
conflicting duplicate staged rows are a hard error rather than last-write-wins:
two differing rows for one `clip_id` are two different answers, not a retry
artifact.

### The path is the whole compatibility mechanism

A checkpoint is reusable only by a run pinning the same source snapshot and the
same contract. The namespace *is* the check:

```text
<staging_root>/inference/<namespace digest>/     digest over:
        │                                          canonical source URI
        ├── results/                               contract digest (model, prompt,
        └── checkpoints/                             sampling, frames, caps)
                                                   shard (count and index, or
                                                     "unsharded")
                                                   staging protocol version
                                                   staged schema fingerprint
```

The last two are in the namespace but **not** in the contract digest, because
they describe staged layout. Neither changes one row's caption, so neither may
invalidate canonical captions; both can make staged work unreadable by a resume
and therefore separate workspaces. The schema/protocol
asymmetry is the [staging-layout
split](#staging-layout-is-versioned-in-the-namespace-never-in-the-digest):
staged Parquet written under the old field set or protocol cannot be read back
under the new one.

The **shard** is in the namespace for a different reason, and the distinction
matters: it is not that two shards' staged rows would be *incompatible*, but that
two shards are **concurrent writers**. Separating them by namespace is what makes
a campaign safe to run at once, and it is why the shard reaches the namespace
while staying out of the contract digest entirely — see
[Sharding a caption campaign](#sharding-a-caption-campaign).

The **pinned read version is deliberately absent**, and that is a change from an
earlier design that digested it. A version that moved without moving rows
invalidates nothing, so digesting the version forfeited every finished caption to
any commit that advanced it — including this recipe's own metadata-only widening,
which is a commit the recipe *itself* issues between a crash and its retry. Under
unseeded generation that is lost output, not lost time. The removal landed
alongside a since-removed composite id and was attributed to it, but the two
are independent: the [row address](#the-checkpoint-id-is-the-row-address) that replaced it is equally
unmoved by those commits, as measured. What the version can still
invalidate is a *compaction*, and that is now handled by naming it an
operational rule rather than by discarding progress on every version bump; see
[No recovery across compaction](#no-recovery-across-compaction).

### Why input scope is not a namespace input

`input.selected_only` decides which rows an attempt owes, so it is a plausible
namespace input and was one until it was measured against the case that actually
arises. Two attempts differ in scope only on the **crash path** — a successful
attempt commits and deletes its namespace, leaving nothing for the flag to
separate — and on that path the two directions are not alike:

| Resume | Staged rows against what it owes | Outcome |
| --- | --- | --- |
| **Widened** (`true` → `false`) | a subset | the checkpoint drops the finished clips at the read, per-fragment coverage balances, and the crash's captions are published |
| **Narrowed** (`false` → `true`) | a superset | publication refuses the extras **before** the commit, with every staged row intact |

Digesting the flag buys the second row's refusal by imposing the first row's
cost on every widening: the crashed attempt's completed captions are stranded in
a namespace nothing will read again — cleanup follows only a commit — and those
clips are recaptioned with *different* text, because generation is unseeded. The
refusal is recoverable and the stranding is not, so scope stays out.

Relaxing publication's extra-row check is **not** the alternative fix. A narrowed
attempt that published a superset would then reach `cleanup_workspace`, which
deletes the whole namespace, destroying the wider attempt's staged rows for every
fragment outside the narrow plan. That check stays strict; see
[When publication refuses](#when-publication-refuses-and-what-to-delete).

Two runs share staged rows if and only if they land in the same directory, so
runs that any digested input separates cannot see each other's data and there is
nothing to validate per row. `resolve_workspace` is **pure path arithmetic plus a filesystem handle**:
it writes nothing, claims nothing, and creates nothing, so a run that turns out to
owe no captions leaves no trace at all. A local staging root is absolute at this
boundary: config resolution canonicalizes configured local paths first, and
`resolve_workspace` rejects any unresolved relative local path before hashing the
namespace or selecting a filesystem. Ray workers need not share the driver's
working directory, so resolving the path lazily in each process would split one
logical namespace across different physical directories. Absolute bare paths,
absolute `file://` URIs with no host or `localhost`, and supported remote URIs
such as `s3://<bucket>/...` remain valid. A one-slash `file:/...` form and an
S3 URI without a bucket authority are malformed rather than local-path
fallbacks. The workspace removes redundant trailing
separators from non-root locations while preserving the slash that makes `/` and
`file:///` filesystem roots.

There is no manifest and no lock, and their absence is a design choice rather than
an omission. A manifest could only ever restate what the path already pins, and
being a second copy of that state it could disagree with it — at which point the
run has to arbitrate between two sources of truth for a question the directory
name already answered.

### One writer at a time per namespace

Two runs that resolve to one directory are **compatible as far as the namespace
declares**: the inputs it digests are the ones that could make one run's staged
captions invalid for another, or make the two concurrent. One further input
could — the identity of the table that occupies the source URI — and it is
deliberately [not among them](#limitations). Compatible is not the same as
concurrent either, and that distinction is the precondition this design carries:

- **Compatibility** asks whether two runs *may* share rows. The path answers it.
- **Concurrency** asks whether two runs may *write* the same rows at once. The
  path cannot express that, and under unseeded generation the answer is no even
  when the first answer is yes, because the two writes disagree on content.

So the precondition is **one writer at a time per namespace**. A resume satisfies
it: the earlier attempt has stopped, and the checkpoint leaves the two clip sets
disjoint. Two genuinely concurrent runs do not — they generate different captions
for the same clip, both `APPEND` into the same `results/fragment_id=<id>/`
partition, and publication then refuses the differing pair because there is no
basis on which to pick one. Worse, the refusal does not clear itself: cleanup runs
only after a commit, and every input to the namespace digest is stable across
re-runs, so every later attempt reads the same poisoned partition and refuses
again. The [remedy is to delete the whole
namespace](#when-publication-refuses-and-what-to-delete), discarding completed
captions.

It is **documented and not enforced**. There is no lease, no lock file, no claim
record, no PID check, and no manifest, and none is wanted: every in-recipe
enforcement adds durable state *beside* the path, which is exactly the second copy
of state the no-manifest argument exists to reject, and it arbitrates at the wrong
layer. Mutual exclusion belongs where runs are launched — a Slurm job dependency,
an NVCF concurrency limit, or a monitoring loop that confirms the previous
invocation finished before starting the next. The recipe's contribution is naming
the precondition and the recovery, not policing admission.

This is a **different** exposure from the [cross-leg Lance
conflict](#context-and-responsibilities), and the difference is worth keeping
straight. Two concurrent *legs* on one table are rejected at the commit, loudly,
with nothing published and a re-run that works. Two concurrent *caption runs* in
one namespace are not rejected anywhere until publication, and the state they
leave behind is terminal until an operator deletes the directory.

Note the precondition is *per namespace*, not per table. Two caption runs on one
table are safe when they resolve to different namespaces and own disjoint rows,
which is exactly what [sharding](#sharding-a-caption-campaign) arranges. The
prohibition is on two writers in **one** directory, and a shard is what keeps
concurrent captioning out of that case rather than an exception to it.

There is likewise no completion marker. The Lance commits are the completion
signal: a clip is complete once the wave carrying it has landed.

The workspace carries **its own filesystem**, and the staged rows and the
checkpoint that indexes them both use that bound capability. It also carries the
**storage profile** that filesystem was resolved from, because post-commit
cleanup deletes through a different client (see
[Why cleanup deletes through boto3](#why-cleanup-deletes-through-boto3)) and
resolving that client from the same profile is what keeps the two one credential
set rather than two. A local namespace still goes through the filesystem's own
recursive delete, on the filesystem-relative path.
`installed_checkpoint` reads the active Ray configuration back before yielding
and requires its id column, path, and filesystem object to match what it installed.
One process lock makes the checkpoint configuration's guard, installation, and
teardown atomic, so a concurrent plan is refused instead of replacing or clearing
the active plan's process-global configuration. Two independently resolved
filesystems would let the sink write where the checkpoint cannot read, or let
cleanup run under different credentials. The first
failure surfaces as a resume that silently redoes everything rather than as an
access error; the second leaves an uncollected namespace after the
already-canonical commit and is logged without failing the run.

An earlier design carried a per-row `request_fingerprint` so a checkpoint could
survive a source-version change. It was dropped: it bought a narrow case (a
metadata-only schema add between attempts) at the cost of a per-row column, a
per-row comparison at load time, and a subtle argument about which input changes
count as semantic. That rejection has only hardened, because the narrow case
turned out not to need buying at all: a metadata-only add does not move a
[row address](#the-checkpoint-id-is-the-row-address) either, as measured.

### Snapshot pinning order

The read version must be pinned **after** the caption columns exist, or the first
run and its resume pin different versions and share no checkpoint. Because
`ensure_caption_fields()` is a no-op on the second call — and asserted not to bump
the version — the common case (first run adds the columns, the resume finds them)
pins the same version both times. The order is the first three steps of
[the data flow diagram](#end-to-end-data-flow).

### The checkpoint id is the row address

`__checkpoint_id` is the row's `uint64` **Lance address**, carried straight off
the scan in
[`datasource.py`](../../../cosmos_curator/next/recipes/robot_action_caption/datasource.py).
It is unique per row within the pinned snapshot by construction, so recovery
leans on no uniqueness contract of its own. The same value supplies
`fragment_id`, its high 32 bits, which is what makes a row's checkpoint entry and
its staged partition unable to disagree — a join back to the table could resolve
differently.

Keying on `clip_id` alone would not do. Publication scopes staging, coverage, and
`update_columns` by fragment, so two fragments holding one clip id would see the
resume filter mark the second complete and no attempt would ever caption it — a
gap with no error. That is a real hazard rather than a hypothetical one, because
this recipe deliberately declines to rely on table-wide `clip_id` uniqueness;
`require_exact_coverage` asserts it per fragment instead of re-deriving it.

**Stability is scoped to a fragment rewrite, not to a table version.** This is
the point the design previously got wrong, in both directions, so it is now
stated as a measurement. Probed at the locked pylance 10.0.0 and at 11.0.0, with
identical results, against the mutations that can occur between a crash and its
resume:

| Mutation | `_rowaddr` |
| --- | --- |
| `add_columns` with the all-nullable caption schema (`ensure_caption_fields`) | preserved |
| `fragment.update_columns` + `Transaction(Update)` on one fragment (`prepare_fragment`, `commit_prepared`) | preserved |
| a later wave on a **sibling** fragment, which is the concurrent-shard case | preserved |
| the pending predicate pushed into the scan | true addresses, not renumbered |
| `delete` | survivors preserved; a tombstone moves no offset |
| `compact_files` | **changed**, into a fresh fragment id |

An earlier design replaced the address with the string pair
`"<fragment_id>:<clip_id>"` on the belief that every commit advancing the version
discarded the attempt's completed inference — the recipe's own metadata-only
widening included, which under unseeded generation is lost output rather than
lost time. The table above is the refutation. The compensating claim, that the
pair survived "a compaction moving a row within its fragment", was unreachable
too: compaction **never reuses** a fragment id, measured both merging three
fragments into the single id `3` and rewriting one fragment carrying two
tombstoned rows from id `0` to id `1`. The pair's fragment half therefore moved
exactly where the address moves. Same stability set, roughly 71 bytes per
completed clip against 8, so the pair was removed.

Never reusing a fragment id also fixes the **direction** of the compaction
failure: a stale id matches no row rather than aliasing onto a different one, so
the cost is re-inference and never a silently skipped clip. See
[no recovery across compaction](#no-recovery-across-compaction).

That safety rests on **two** dependency behaviours, and both are recorded beside
`STAGING_PROTOCOL_VERSION` in
[`workspace.py`](../../../cosmos_curator/next/recipes/robot_action_caption/workspace.py)
as re-measure items on a pylance bump — the caveat that went wrong here went
wrong by being left implicit, so neither premise is left to follow from the
other. Uniqueness *within* a snapshot is Lance's own construction, but a stale id
never naming a different **live** row needs both that fragment ids are never
recycled and that an offset inside a surviving fragment is never reassigned. The
second is the one other writers on `clips.lance` could break, so it was measured
rather than inferred: an append opens a new fragment instead of extending an
existing one, a delete tombstones without renumbering survivors, and the offsets
a delete frees are not handed to later appended rows.

Two properties the key must keep, both easy to break in the same two places:

- **Index memory.** Ray's checkpoint filter loads the completed ids into an array
  and matches each block against it, so the id width sets the index size, and the
  **driver** process pays it. At 8 bytes the index is about 4 GB at 500M
  completed clips; the string pair was roughly 71 characters — 64 hex, one
  separator, a decimal fragment id — which scaled naively to tens of GB. Both
  figures are derivations from the id width, not measurements. The narrow id is
  also what keeps checkpointability independent of campaign shape: the pair was
  affordable only because [sharding](#sharding-a-caption-campaign) divides the
  index by the shard count, whereas ~4 GB is affordable unsharded.
- **Id survival.** Ray commits a row as complete by reading the configured id
  column off the **written output**, so the column must survive into the staged
  Parquet. A staged schema that drops it checkpoints nothing while appearing to
  work.

The id must be present on the **first** operator of the dataset — a column added
later in the plan is invisible to the filter — and it must reach the terminal
write.

The column name is spelled once, in
[`datasource.py`](../../../cosmos_curator/next/recipes/robot_action_caption/datasource.py),
beside the scan that fills it. The producing scan and Ray's checkpoint filter are
two halves of one mechanism that fails *silently* when they disagree — a mismatch
records no progress while the run appears to work — so the string has exactly one
definition, in the module that produces its values. A generic home would make it
look like an independent contract a leg merely opts into, which is the opposite of
how it behaves.

Retyping the id did not bump `STAGING_PROTOCOL_VERSION`, and did not need to:
the type is part of `STAGED_SCHEMA`, so `staged_schema_fingerprint` moved and
older namespaces are isolated automatically. Publication is untouched — it joins
on `clip_id` and addresses a partition by `fragment_id`, and only Ray's filter
reads this id.

### Staged Parquet, and why it is needed alongside the checkpoint

They are not redundant; they record different things. Ray's checkpoint is an
**index over safely written output** — it says which rows are done, not what they
said. The staged Parquet holds the caption values themselves, and it is what makes
a failed publication cheap to retry: the captions survive the failure, so the next
attempt republishes them instead of regenerating them.

What the split separates is the **data plane, not the control plane**. There is one
entry point — `run_config` in
[`pipeline.py`](../../../cosmos_curator/next/recipes/robot_action_caption/pipeline.py)
— and publication cannot be invoked on its own. A re-run that owes nothing but a
republish still resolves the model, verifies the staged weight files, initializes
Ray, and enters Phase A, whose engine pool is sized from the count this job
*owes* — nothing has been published yet, so that is the whole wave it is
republishing, and on a sharded run it is owed within the fragments this job owns
rather than across the table — and starts its
initial replicas whether or not the checkpoint filter leaves them a row to
caption. A republish therefore costs an **engine cold start and no inference**: the
split removes the expensive half of the bill, not the whole bill. An operator
planning one should budget a fresh run's startup and none of its GPU time.

The staged schema is `clip_id`, `fragment_id`, `__checkpoint_id`, plus the five
output columns, written with `partition_cols=["fragment_id"]` so Phase B reads
only the fragments it needs, and `mode="append"` so a second attempt's staged
files land beside the first's. `APPEND` is passed explicitly even though it is Ray's current
default: were that default ever to become `OVERWRITE`, a resume would silently
delete the captions it exists to preserve, and unseeded generation makes them
unrecoverable.

The terminal write is also a correctness boundary for job-level checkpointing:
Ray reads the checkpoint id back from the sink output before marking a row
complete. Replacing it with `materialize()` would leave no write operator for
that protocol and can disable recovery with only a warning, so the plan must
remain one logical read-to-write DAG.

Sink sizing is a real concern: `write_parquet` otherwise emits one file per output
block, which is O(batch count) staged files and a listing cost every resume pays. Both
`min_rows_per_file` and `max_rows_per_file` are set from
`execution.parquet_rows_per_file`.

The two halves are not equally advisory, and the difference is what sets recovery
granularity. Ray's sink adopts the **minimum** as the Write operator's
`min_rows_per_bundle` and the bundler accumulates blocks until it reaches that
floor — measured at Ray 2.58.0, sixteen 32-row blocks coalesced into a single
512-row bundle under a 4096-row floor. One bundle is one checkpoint, so what a
preemption re-generates is:

```text
rows at risk per in-flight bundle = max(parquet_rows_per_file,
                                        rows in the block reaching the sink)
```

A bundle can never hold fewer rows than the block it receives, which is why
lowering this knob beneath that block size splits files without checkpointing any
more often. The **maximum**, by contrast, only ever splits a bundle into files and
never changes what is durable. Exact file sizes remain unguaranteed either way, so
correctness must not depend on them, and no sink-side repartition is added to
enforce them: paying another data movement step to hit an exact file size would
cost more than the uneven files it avoids.

The default of 256 sets the floor equal to `lance_read_batch_size`, so the two
coincide and neither silently dominates the other. Raising one alone re-creates a
gap in which the larger is the only bound that matters; raise both together to
trade recovery granularity for fewer staged objects.

The phase also depends on Ray's two-phase commit around that write — pending
checkpoint first, then the data file, then commit by rename, with a later attempt
deleting any staged file left uncommitted. A staged file that became durable
*without* its checkpoint entry would be regenerated as a different caption under
unseeded sampling, and publication would then refuse the conflicting pair
permanently rather than merely repeat the work.

### No recovery across compaction

Staged rows are partitioned by fragment id and keyed by the
[row address](#the-checkpoint-id-is-the-row-address), so every commit
this recipe issues costs nothing: measured at the locked pylance 10.0.0, neither
the metadata-only widening nor a per-fragment caption update moves an address. A
compaction is the case that breaks: it rewrites fragments and **never reuses**
their ids, so the staged partitions point at fragments that no longer exist and
no owed row's id matches a staged one.

There is no transparent recovery across that and none will be built. The rule is
operational: **do not compact or otherwise mutate the source's fragment geometry
while a workspace holds progress.**

Nothing refuses it, and the reason is worth being explicit about because it
changed. The namespace no longer digests the read version, so a compacted source
resolves to the **same** directory rather than a fresh one — a compaction is no
longer self-isolating. What that costs is bounded and non-corrupting: the staged
rows are keyed by fragment ids the new manifest does not contain, so no owed row
can match one, and the attempt re-infers its whole owed set instead of reusing
anything. The stale partitions are never read and are not published from; they
are dead weight in the namespace until it is deleted, not a source of wrong
captions.

That is a deliberate trade against the alternative it replaced. Digesting the
version made a compaction isolate itself, at the cost of discarding completed
captions on **every** version bump — including the recipe's own widening — which
is the far more frequent event. Paying re-inference on the rare compaction is
cheaper than paying it on every routine commit, and both cost only inference:
neither can publish a caption the current contract did not produce.

Because fragment ids are never reused, the stale key cannot collide with a live
row either, which is what bounds the cost to re-inference — the failure is lost
work, never a clip silently marked complete.

### There is no switch that disables reuse

Reuse is not a feature with an off position. The namespace is *derived* from the
snapshot and the contract, so two runs that would share staged rows are by
construction runs whose outputs are **interchangeable in meaning** — a flag
guarding against semantic mismatch would have nothing to guard. What such a flag
could never have addressed is two runs writing the namespace *at the same time*,
which is a launch-side precondition rather than a compatibility question; see
[One writer at a time per namespace](#one-writer-at-a-time-per-namespace).

The consequence is worth being explicit about, because the flag that used to exist
worked by removing and recreating the directory: nothing in the recipe ever
deletes staged work except the cleanup that follows a successful commit. Deleting
staged rows before a commit does not cost a re-run, it costs *different captions*,
and no configuration should be able to ask for that by accident.

To make a run ignore what is already staged, change something the digest covers
(the model registry pin, the prompt, `max_clip_bytes`) or remove the namespace
directory by hand. The first is a semantic decision and recaptions the configured scope by
design; the second is an operator action taken deliberately, with the run's own log
line naming the directory it resolved.

## Sharding a Caption Campaign

`input.shard` divides the table's fragments across independently launched jobs,
this job taking the shard at the given zero-based index. Absent — the default —
means one job owns every fragment, which is how this recipe behaved before the
field existed.

```yaml
input:
  clips_lance_uri: s3://bucket/clips.lance
  shard:
    count: 8      # the campaign is eight jobs
    index: 3      # this job is the fourth of them
```

The ownership mechanism, the terminology it fixes, and the checklist a recipe
works through before adopting it are generic and live in
[Deterministic Work Sharding](curator-next-sharding.md). This section is this
recipe's adoption: what made it safe, what it changes about the run, and how an
operator drives a campaign.

One note on the word, because this document had two other uses for it before this
one. An upstream **LeRobot/Mecka shard** is a source directory
`robot-action-split` reads. A staged file written by Ray's Parquet sink was also
called a shard, and those passages now say **staged file** instead. Throughout
this section, and in `input.shard`, the word means one job's **execution scope**
and nothing else.

### Why a caption run needs two reliability mechanisms

Row checkpointing already makes an attempt cheap to resume. What it cannot do is
raise the ceiling on a single attempt, and at corpus scale that ceiling is the
binding constraint: 500M clips is more GPU-hours than one allocation holds, in
one place, for as long as it takes. The two mechanisms are layered, and each one
makes the other worth having.

```text
        one logical caption run over clips.lance
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
 shard 0/3            shard 1/3           shard 2/3     ← SHARDING divides the
 fragments 0,3,6…     fragments 1,4,7…    fragments 2,5,8…  fragment universe.
 cluster A            cluster A           cluster B      Separate namespaces,
      │                   │                   │          separate commits.
      │                   │                   │
 attempt 1 ✗ OOM     attempt 1 ✓        attempt 1 ✗ preempted
      │                                       │
 attempt 2 ✓              ← CHECKPOINTING     attempt 2 ✓
 resumes from the           resumes an        resumes from the
 last committed             attempt inside    last committed
 row                        one shard         row
      │                   │                   │
      └───────────────────┼───────────────────┘
                          ▼
            three Update commits, one per shard,
            each touching only the rows it owns
```

- **Sharding bounds what one failure costs in scheduling terms.** A job lost to
  a dead cluster is one shard's in-flight work, not the corpus, and the other
  shards neither wait for it nor notice.
- **Checkpointing bounds what a retry costs inside a shard.** Without it, a
  shard failing in its last hour would restart from nothing, which would make
  shards have to be small — and small shards mean many jobs, which is its own
  operational cost.
- **Sharding also bounds the checkpoint index.** Ray's filter loads the completed
  ids of *its own namespace*, so the driver-side index divides by the shard
  count. It mattered more under the wide composite id that
  [preceded the row address](#the-checkpoint-id-is-the-row-address), where the index
  was the expensive part of recovery; at 8 bytes per completed clip the index is
  affordable at 500M rows whether or not a campaign is sharded.

### What a shard owns, and what it never touches

A shard owns **whole fragments**, assigned by the manifest's own fragment id. It
does not own rows, subsets of fragments, or anything the plan derives at runtime.

Ownership is decided on the id **before** any owed-row count, and both halves of
that order matter. A fragment's owner does not change as its rows complete, so
re-launching shard 3 of 8 is reproducible; and a job never pays for a filtered
count over fragments it will not read, so the driver cost of a campaign does not
grow with its width.

What a shard does **not** do:

- It does not rewrite, compact, or repartition the source. Fragment geometry
  stays [`robot-action-split`](curator-next-robot-action-split.md)'s.
- It does not appear in the contract digest, so re-sharding a corpus recaptions
  nothing. A caption produced by shard 3 of 8 is identical in meaning to the same
  caption from an unsharded run, and the pending predicate cannot tell them
  apart — which is the property that makes a campaign's shape freely changeable
  between campaigns.
- It does not write a "not mine" verdict. Rows this job does not own are left
  **owed**, for whichever job does own them.

### Why shards cannot share one staging namespace

`input.shard` is a namespace input, so `count=4 index=1` and `count=4 index=3`
stage under different directories. That is deliberate, and two independent
mechanisms — one ours, one Ray's — destroy in-flight work if they share one.

- **Post-commit cleanup is whole-namespace.** `cleanup_workspace` deletes the
  namespace root, not this job's fragment partitions. Measured end to end with
  the digest forced to ignore shard scope: shard 1 of 2 staged four captions and
  one committed checkpoint, shard 0 then published successfully, and the commit's
  cleanup left zero staged rows and zero checkpoint files — so shard 1 recaptioned
  all four clips. Generation is unseeded, so that is destroyed GPU work rather
  than a repeated one.
- **Ray's checkpoint recovery is job-blind.** Every job's `load_checkpoint` first
  runs `_clean_pending_checkpoints_task`, which lists *every* `*.pending.parquet`
  in the checkpoint directory, deletes any data file whose name they prefix, and
  then deletes the pending entries. Measured at Ray 2.58.0 on a shared directory
  holding one job's committed pair and two jobs' in-flight pairs: both in-flight
  pairs were deleted, including the one belonging to the other job, which was
  still running. Committed entries survived. Nothing on this side can scope that
  — it runs inside Ray before the first row is read — so shard-scoping our own
  cleanup would not make sharing safe.

Sharing is also unnecessary for most of the reuse it looks like it would buy.
Completed work becomes reusable by being **committed**, not by being staged: the
pending predicate reads the caption columns and the contract digest, so a job
under any `count` finds committed clips complete and never re-owns them. Only
uncommitted staged rows are namespace-scoped, and those live for one attempt by
design.

Publication commits in [waves](#why-publication-commits-in-waves-and-why-not-one-commit-per-fragment)
rather than once precisely to make that reuse start earlier: an interrupted
publication leaves the waves that landed reusable under any later `count`. It
cannot reach further back than that, though. Publication runs only after
inference finishes, so a failure **during Phase A** still strands every staged
row of that job against a `count` change, and no commit granularity alters it.
That is the residual cost of choosing `count` per campaign.

What a shared directory *would* add, beyond the two mechanisms above, is a
checkpoint index no commit ever prunes: entries are named by Ray's write uuid and
task index and hold whatever ids their task carried, so they cannot be filtered
down to one shard without reading and rewriting every entry.

### Concurrent shards and the schema widening

Sharding makes concurrent commits on one table the expected case, and the recipe
issues exactly one commit every job needs before it can do anything: the
metadata-only widening that registers the five caption columns. Eight jobs
launched together will race on it, and seven of them will lose.

`ensure_caption_fields` therefore takes the **URI** rather than an open dataset,
and owns its own handle:

```text
attempt 1   open uri  →  columns absent  →  add_columns  →  ✗ lost the race
                                                │
attempt 2   open uri  →  columns PRESENT  →  validate  →  ✓ return this handle
                (reopening is the point: the handle that lost the race
                 cannot see the commit that beat it)
```

A retry that held the original handle would re-read a snapshot predating the
winner's commit, conclude the columns are still absent, and fail the same way
until its budget ran out. Reopening per attempt is what makes the race
self-resolving. The function returns the advanced handle so the read version is
pinned from the snapshot that actually has the columns — a structural guarantee
rather than a call-site convention.

The retry budget is bounded and small. A lost race resolves on the next attempt
because the winner has already committed; a failure that persists across the
budget is not a race, and surfacing it is correct.

### What made this recipe safe to shard

Two prerequisites had to be met first, and neither was cosmetic.

- **A checkpoint id that survives a sibling's commit.** Concurrent shards commit
  constantly, so a key invalidated by a version bump would let each bump
  invalidate every other shard's staged work — fatal to a campaign rather than
  merely wasteful. The [row address](#the-checkpoint-id-is-the-row-address)
  survives one: measured, a caption wave on one fragment leaves every other
  fragment's addresses intact. A composite id was adopted for this reason and
  later removed, because what the campaign actually needed was for the
  **namespace** to stop digesting the read version, and that is independent of
  the key.
- **Shard-scoped owed accounting.** Every figure a job reports is now its own —
  owned fragments, owed fragments, owed rows within them. A table-wide count in
  a sharded job describes work the job will never do, and an operator reading it
  concludes the job is stuck.

Independence was already true and did not need work: a caption is a pure function
of one clip's media and the contract digest. There is no corpus-wide reduction,
so dividing the rows cannot change what any caption says. A recipe with a fitted
basis or a population-derived threshold would not have this property, which is
why the [generic checklist](curator-next-sharding.md#adoption-checklist) asks.

### What a sharded run logs and reports

Every owed figure is shard-relative, and the scope is named on the line so the
number cannot be misread as a corpus-wide claim:

```text
Caption work shard 3/8 on <uri>: this job owns the fragments whose manifest id
  mod 8 equals 3. Indices are zero-based, so a complete campaign launches
  indices 0 through 7.
Caption attempt on <uri> @v12 (shard 3/8): 61 owned fragment(s), 47 of them
  owed, 1,208,411 owed row(s), largest owed fragment 31,004 row(s), under
  selected-only scope and contract <digest>.
```

An unsharded run prints `(unsharded)` in the same position, so the two are
distinguishable in a log without knowing which was configured.

`CaptionRunResult` carries `shard_count` and `shard_index`, both null on an
unsharded run, so a campaign's summaries can be attributed to their slices after
the fact. They are run metadata and reach no table column.

The "nothing owed" warning is scoped the same way. When a selected-only job finds
no work it counts pending rows **within the fragments it owns**, and says so:
rows in other shards' fragments are not this job's to explain, and naming them
would send an operator looking for a fault in the wrong job.

### Driving a campaign

There is no campaign command, by design — a campaign is `count` launches of one
config differing only in `input.shard.index`.

```bash
# One config per index, identical but for input.shard.index. Indices are
# ZERO-BASED, so count 8 means 0 through 7 inclusive.
for index in $(seq 0 7); do
  yq ".input.shard = {\"count\": 8, \"index\": ${index}}" caption.yaml \
    > "caption-shard-${index}.yaml"
  cosmos-curator pipeline render "caption-shard-${index}.yaml"   # confirm the scope
  pixi run --as-is run-pipeline  "caption-shard-${index}.yaml"   # or submit to Slurm
done
```

Each shard issues its own `Update` transactions when it finishes, so captions
become visible shard by shard rather than all at once. That is a real difference
from an unsharded run and worth stating: [publication validates everything, then
commits in bounded
waves](#publication-validate-everything-then-commit-in-bounded-waves) **per
job**, and the invariant it guarantees — no partially-published fragment — holds
per fragment, which is what downstream readers actually depend on. A reader
mid-campaign sees some fragments captioned and others not, with no fragment
half-done.

A failed shard is re-launched with the same index. It resumes from its checkpoint
if its staged progress survived, and from its owed set if it did not; either way
it captions only what its own slice still owes.

To close a campaign out, launch every index again. A shard that owes nothing
returns before validating the model directory, resolving a workspace, or
initializing Ray, so a finished shard costs one driver plan and no GPU — which
makes "re-launch everything" both the cheapest check that a campaign is complete
and the action that completes it.

Nothing asserts that a campaign finished. No registry records which indices ran,
so launching 0, 1, 2 and 4 of 5 leaves shard 3's work owed with nothing reporting
it. Campaign bookkeeping belongs where jobs are launched.

## Publication: Validate Everything, Then Commit in Bounded Waves

Phase B reads the staged Parquet, folds duplicate `clip_id` rows, and validates
**before** touching Lance. Every operation below is per fragment. Equal
`clip_id` values in different fragments remain in different staged partitions,
coverage sets, and `fragment.update_columns` calls, so they neither fold together
nor cross-apply; `(fragment_id, clip_id)` is the publication identity even though
global `clip_id` uniqueness remains an upstream table-domain contract.

- duplicate rows for one `clip_id` are folded only if they are **identical** —
  conflicting duplicates are a hard error, never last-write-wins;
- no `clip_id` is owed **twice** — the update table carries one row per owed
  entry, so a repeat would reach `update_columns` as two right-side rows for one
  join key and it would publish an arbitrary one of them; the refusal names the
  fragment and both counts, the owed one and the distinct one;
- every owed `clip_id` has exactly one terminal row;
- no staged `clip_id` is absent from the owed set;
- every staged row carries the current `caption_contract_digest`.

An absent or empty partition reads as a typed zero-row table, so owed clips
reach the ordinary missing-row refusal rather than a staged-read failure.

Those last four are kept separate because they have different causes and only one
is fixed by re-running Phase A: a repeated owed clip means the producer's
uniqueness guarantee broke, a missing row means Phase A did not finish that clip,
an extra row means the staged tree belongs to a different attempt's row set, and a
foreign digest means it belongs to different semantics. The repeat is checked
first, because it makes every count after it misleading.

Two further checks then run **per row**, on the same walk that builds the update
table, so no row can reach `update_columns` unvalidated:

- **Terminal values are revalidated from the durable row.** `validate_terminal_values`
  re-derives the status/caption/error pairing off what was read back from Parquet
  rather than trusting that `terminal_row` enforced it when the row was made. The
  invariant is enforced at both ends because a row that violates it has no correct
  publication: there is no way to tell whether a status-less caption succeeded.
- **The model id must match this run's.** A staged row carrying a different
  `caption_model` is refused. The digest already covers the model, so this fires
  only when the two disagree — which means the stamp and the semantics the
  namespace pinned have come apart, and neither can be trusted to describe the
  other.

Both refuse as `CaptionPublicationError` naming the fragment and the clip. See
`prepare_fragment` in
[`publication.py`](../../../cosmos_curator/next/recipes/robot_action_caption/publication.py).

Each touched fragment is then prepared with `update_columns` in parallel Ray
tasks. Preparation writes data files; it does not commit. Refusing here rather
than asserting afterwards is the whole point: once the commit lands there is
nothing to refuse, and a caption is stochastic, so a wrongly published row cannot
be recomputed into the right one.

A worker reports a refusal as a **payload rather than a raise**. Ray Data replaces
whatever a UDF raises with a wrapper that is no longer an instance of its own
cause, so a raise would reach the driver as an untyped `RayError` and the operator
would lose the one thing that tells them what to fix. The driver re-raises it as
`CaptionPublicationError`, a `ValueError` subclass, so it joins the recipe's other
pre-commit refusals as one thing a caller handles. Each worker returns a single
JSON string column, which structurally cannot carry caption text back and keeps
driver memory O(touched fragments) rather than O(published rows).

Three guards then stand between preparation and the commit:

- **The prepared set must be exactly the planned set.** Preparing *fewer*
  fragments than planned would leave clips pending while the run reported success;
  preparing *more* means rows from a foreign attempt are about to be committed.
  `update_columns` would drop unmatched right rows without complaint either way,
  so neither is visible after the fact.
- **No fragment may be named twice.** Each fragment is a single work item, so a
  duplicate payload means a retried task's preparation was also collected, and
  committing two metadata versions of one fragment would make the outcome depend
  on collection order.
- **Embedded metadata must name its envelope fragment.** The expected-set check
  reads the payload's `fragment_id`, while the transaction consumes decoded
  `FragmentMetadata`. A mismatch is refused as `CaptionPublicationError` before
  transaction construction, so an expected envelope cannot smuggle another
  fragment's prepared descriptor into the commit.

Finally each `LanceOperation.Update` carries one wave of prepared fragments plus
`fields_modified`. `LanceOperation.Update` accepts a partial fragment list, which
is what makes a multi-fragment commit possible; the embeddings leg's
[`fill.py`](../../../cosmos_curator/next/recipes/embeddings/fill.py) already does
exactly this, and this recipe follows that precedent rather than inventing a
second commit shape. Columns outside the caption group survive because each
prepared `FragmentMetadata` enumerates *every* one of its fragment's column files,
not just the one `update_columns` wrote; `fields_modified` does not decide that —
it lists the modified field ids so Lance can drop those fragments from indices
covering them.

### Why publication commits in waves, and why not one commit per fragment

Validation is all-or-nothing; publishing is not. Every refusal runs over the
whole prepared set before a single transaction is issued, so a run that commits
anything has already accepted all of it — that ordering is what lets the commits
be split at all. What is then split is the writing: one bound,
`_PUBLICATION_WAVE_BOUND` (16), applies at both ends of the commit sequence, so
the prepared fragments are committed in never more than that many transactions
and never one carrying fewer than that many fragments. A publication too narrow
to fill two waves therefore still commits exactly once, and a campaign-scale one
commits a fixed number of times.

What splitting buys is that a landed wave is **committed**, and committed work is
reusable under any campaign shape, because the owed-work predicate reads the
caption columns and the contract digest and knows nothing about `input.shard`. A
publication interrupted after eight waves costs the next attempt only the
remaining eight — even if that attempt runs under a different `count`, where an
all-or-nothing commit would have left the whole corpus owed. What it does **not**
buy is any reuse of Phase A: publication begins only once inference has finished,
so the staged rows are at risk for the whole of Phase A either way.

The bound has both halves because a commit's cost is one manifest and a manifest
enumerates **every** fragment of the table, not the ones the commit touched —
measured at pylance 11.0.0 as 82 bytes per fragment, linear from 100 to 4,000.
So N commits over a table of F fragments write O(N × F) manifest bytes however
the work is divided:

- **Capping the count** is what makes splitting affordable. One commit per
  fragment is O(F²): for the ~31k fragments of a 250M-row table that is roughly
  73 GiB of manifests and 31k serialized commit round trips, against ~40 MiB and
  16 round trips under the cap.
- **Flooring the size** follows because a wave's cost does not shrink with its
  contents while the loss it prevents does, so splitting a narrow publication
  pays full price per piece for nothing. The floor binds *every* wave rather
  than only the first, which is why the split is derived as a wave count and
  then divided evenly: a fixed slice width divides with a remainder, and that
  remainder would be a final wave below the floor, paying a whole manifest for
  fragments its predecessor would have carried at the same price. So the
  prepared set takes as many waves as the floor affords — no more, and never one
  that ends in a short tail.

Commits after the first read against a version their predecessors have already
advanced past, and are not re-opened: measured at pylance 11.0.0, an `Update`
naming fragments no earlier wave touched lands against a stale read version.
That is the same property that lets two shards commit into one table.

The preparation cost is unchanged — O(touched fragments) before a single visible
byte changes, as parallel Ray tasks, and small next to inference.

`update_columns` **preserves unmatched left rows**: a row whose caption columns
were already populated and which is not in the staged set keeps its values. It is
the unmatched *right* rows — staged rows with no matching clip in the table — that
are dropped silently, which is why validation checks that no staged `clip_id` is
missing from the owed set rather than trusting the commit to complain.

### An ambiguous commit is verified, not trusted

Lance's commit raises `OSError` both for IO faults and for a commit conflict —
measured at the locked pylance 10.0.0 as a bare `OSError`, and at 11.0.0 as
`CommitConflictError`, which is itself an `OSError` — and **no exception type
tells you whether the commit landed**. Nor can the two be separated by type at
the locked version: a deterministic "invalid user input" rejection arrives as the
same bare `OSError` as a retryable conflict, and the only thing distinguishing
them is a human-readable message that no code may parse.
A response lost after the manifest was written raises exactly as
an outright rejection does. Re-raising unconditionally would send an operator to
re-publish a wave that already succeeded, and re-publishing under unseeded
generation writes different captions over good ones.

The table and its committed transaction history are the authorities, so the
recipe asks both. Each atomic publication stamps a fresh UUID under the
`publication_attempt` transaction property beside its durable `kind`; the
contract digest cannot substitute because it identifies semantics shared by
retries, not one physical attempt. On either exception the recipe re-opens the
table **at latest** — not
through the pinned handle, where every row of the wave is owed by definition
and the answer would always be "it failed" — then looks for the wave's fragments,
counts owed rows using this run's own predicate, and reads every transaction
newer than the pinned version through that observed latest version. This bounded
range is complete even if Lance rebases the update or a sibling commit follows
it: this attempt cannot have committed before its read version or after the
latest version just observed. See
[`_commit_verified`](../../../cosmos_curator/next/recipes/robot_action_caption/publication.py).
There are four outcomes:

- **Rows still owed.** The commit really did fail. A `ValueError` names the
  remaining count and the cause; nothing was published and the staged rows are
  intact.
- **No row owed and this attempt's marker is present.** The commit landed and
  the error was a lost response. The latest observed version is returned and the
  publication is treated as successful, with a WARNING recording that it got
  there through this path.
- **No row owed but this attempt's marker is absent.** Another writer may have
  completed the same rows. Canonical row state cannot attribute that work to this
  attempt, so publication raises `CaptionPublicationConflictError` instead of
  claiming success or prescribing destructive cleanup.
- **A fragment the wave targeted is gone.** Checked *before* either count,
  because a missing fragment makes both unanswerable. A successful
  `update_columns` rewrites a fragment's column files and keeps its id, so an
  absent id cannot mean "committed" — it means the table moved underneath the
  check, a concurrent compaction being the likely cause. Counting it as zero
  pending rows would therefore report a genuinely *failed* commit as landed. A
  `ValueError` names the absent fragments and the version they were absent at, and
  refuses the outcome as unverifiable rather than guessing; the staged rows are
  intact, and an operator resolves it by inspecting the table.

This is the one place a post-commit check is legitimate, and the distinction
matters: it does not assert that the publication was *correct* — that was refused
in advance, before any commit — it verifies both canonical state and ownership,
and says so when either is unprovable. The row check is O(fragments) filtered
counts on one column, the same per-fragment count planning already uses, never
O(rows) on the driver. The ownership check is O(versions committed since the
pinned read), not a scan of unrelated older history.

After a successful commit the namespace is removed. Cleanup is unconditional and
safe without a manifest check, because the namespace is derived rather than
claimed: anything under that path was staged by a run whose semantics are
identical to this one's, and this run has just published those very rows. That
rests on the [one-writer
precondition](#one-writer-at-a-time-per-namespace) rather than on anything the
path pins — a second concurrent attempt's staged rows would be deleted here too. A
cleanup failure is logged at WARNING and never fatal — a stale workspace costs
storage, while failing a run after a successful commit costs a
re-investigation. Cleanup runs
*only* after a commit: before one the staged rows are the sole copy of completed
GPU work.

### Why cleanup deletes through boto3

An S3 namespace is removed with botocore's `DeleteObjects`, not with PyArrow's
`S3FileSystem.delete_dir`, and the reason is a checksum disagreement between the
two clients that only one endpoint family notices.

`DeleteObjects` is the sole S3 delete the service model marks
`requestChecksumRequired` (`DeleteObject`, the single-key form, carries no
`httpChecksum` trait at all). A required checksum is sent regardless of
`AWS_REQUEST_CHECKSUM_CALCULATION`: `when_required` is *satisfied* by that
operation, not a way to opt out of it. The existing
`AWS_RESPONSE_CHECKSUM_VALIDATION=when_required` mitigation is a different
knob — it governs what the client verifies on responses.

Which algorithm each client picks is where they part company. PyArrow 25.0.1's
embedded AWS SDK for C++ sends CRC64-NVME, which Google Cloud Storage's
S3-compatibility layer does not recognize and reports as
`Missing required header for this request: Content-MD5 OR x-amz-checksum-crc32`.
botocore 1.43.86 sends CRC32 (`httpchecksum.DEFAULT_CHECKSUM_ALGORITHM`), which
that same message names as accepted. PyArrow exposes no per-client or
per-request control over its choice — `s3fs.cc` never mentions checksums, and
the only knobs the bundled SDK reads are the two `when_required` environment
variables — so the client is what had to change, not its configuration.

The consequence before this: every GCS run committed its captions and then
leaked its whole `results/` and `checkpoints/` prefix, since the failure lands
inside the deliberately nonfatal post-commit catch.

## Error Taxonomy

**A bad clip does not fail the run:** a row-local fault — an unusable `clip_uri`,
a clip object that is missing, zero-byte, or oversized, or an empty generation —
publishes one terminal `error` row for that clip, and the wave commits as normal.
A systemic fault, an exhausted transport budget, or an engine-side failure ends
the attempt instead and publishes nothing. The count of `error` rows surfaces as
`error_rows` in the run summary, and the table below gives the verdict for every
fault the recipe classifies.

The distinction that matters: one corrupt clip must not fail the run, and one
expired credential must not become millions of per-row errors. The current
authority on which side each fault lands is the recipe package docstring,
[`robot_action_caption/__init__.py`](../../../cosmos_curator/next/recipes/robot_action_caption/__init__.py).

The [`MediaReadError`](../../../cosmos_curator/next/utils/media_reader.py)
hierarchy encodes the trichotomy — `MediaObjectError`, `MediaTransportError`, and
`MediaSystemicError` — and the exception type *is* the classification, so nothing
downstream re-derives it.
[`media.py`](../../../cosmos_curator/next/recipes/robot_action_caption/media.py)
catches `MediaObjectError` and names the fault in a `media_error` column; it lets
`MediaTransportError` and `MediaSystemicError` propagate and end the attempt. The
caught fault is deliberately reduced to
`MediaObjectError: clip_id=<repr>` before either persistence or logging. The same
sanitized diagnostic becomes the published `caption_error` and appears in the
warning beside the clip id; the raw `clip_uri` and raw exception text are neither
persisted nor logged on this row-local path. Because `MediaObjectError` is the only
class caught there and has no subclasses, its name is a constant prefix on that
path. The class name is therefore published data: renaming it changes the stored
format even when the underlying fault is unchanged.

That redaction guarantee is scoped to the caught, row-local object-fault boundary.
`MediaTransportError` and `MediaSystemicError` are not converted into rows or
sanitized by `media.py`; they propagate to fail the attempt, and their native
reader diagnostics may contain the URI and underlying exception text.

### The taxonomy classifies a fault; it cannot bound its extent

The exception type encodes whether **retrying this object** could change the
answer. That is a per-object property, so it structurally cannot encode how many
objects share the condition — and extent is the second thing an operator cares
about. An absent clip object is correctly row-local, because no retry of *that*
object would help; but media deleted after `robot-action-split` wrote `clip_uri`,
a lifecycle rule expiring it, or a `clips.lance` restored without its media makes
that true of **every** clip. Each one then costs one terminal `error` row, and the
wave commits.

Publication surfaces that rather than refusing it. `error_rows` on the run summary
is what separates a corpus of captions from a corpus that errored identically on
every clip, and publication deliberately does **not** gate on the error share —
see [The summary reports the verdict
split](#the-summary-reports-the-verdict-split) for why no threshold exists.

Recovery is not a re-run either, and this is the invariant the rest of the
taxonomy argues from: **a published verdict is terminal under the contract that
produced it.** A published `error` row carries the current digest, so the pending
predicate no longer selects it and an ordinary re-run does not retry it. The only
ways back are a contract change — which moves the digest and therefore recaptions
the *whole* corpus, not the failed rows alone — or clearing the caption columns by
hand.

The authority on which side each fault lands remains the
[package docstring](../../../cosmos_curator/next/recipes/robot_action_caption/__init__.py),
which carries this caveat beside the classification it qualifies.

| Failure | Class | Retried by | Terminal row | Fails run |
| --- | --- | --- | --- | --- |
| Missing `clip_id` / `clip_uri`, a `clip_id` that is not exactly `string`, or a non-text `clip_uri` | structural, refused on the driver | no | none | yes |
| Model files absent or incomplete | systemic, refused by the registry driver preflight | no | none | yes |
| Absent prompt context column | narrows the run's contract | n/a | none | no |
| Non-text prompt context column | structural, refused on the driver before the widening | no | none | yes |
| Credentials rejected, bucket missing, account disabled, or a local path denied without proof the denial is this object's | systemic | no | none | yes |
| `clip_uri` is null on the row | row-local | no | `error` | no |
| Unsupported or malformed clip URI | row-local | no | `error` | no |
| Missing clip object | row-local | no | `error` | no |
| Zero-byte clip object | row-local | no | `error` | no |
| Clip exceeds `max_clip_bytes` | row-local | no | `error` | no |
| Transient storage 5xx or timeout | transport | yes, to `execution.media_attempts`, then at block level | none | yes, past the budget |
| Unrecognized backend read failure | transport | as above | none | yes, past the budget |
| Corrupt container, decode failure | execution, in the media preparation stage | Ray, bounded | none | yes, past the bound |
| Per-request engine failure | **systemic by choice** | Ray, bounded | none | **yes** |
| Empty generation | row-local | no | `error` | no |
| Engine or prepare actor death | execution | Ray, bounded | none | yes, past the bound |

A clip URI no backend claims — an unsupported scheme, or a location a syscall
refuses outright — is **row-local and terminal**. Nothing about one unusable URI
makes the other pending rows unreadable, so it costs one published `error` row
rather than the run. It is not retried either: the verdict is a property of the
URI, so a second attempt re-sends an identically doomed request. The reader
classifies it in
[`_classified`](../../../cosmos_curator/next/utils/media_reader.py), below the
credential, bucket, and account checks — that ordering is the contract, because a
fault true of *every* object must never be written out per object. Those rows
would be [terminal](#the-taxonomy-classifies-a-fault-it-cannot-bound-its-extent),
so the re-run that follows the fix would find nothing left to do.

The same determinism argument drives the retry policy: a decode failure or a
size violation cannot change on a second read, so retrying it only multiplies
the wasted transfer.

A **null `clip_uri`** is named as its own cause rather than handed to the reader.
Rendering a null location into a string yields `"None"`, which the reader then
refuses as a missing local object — and the published `media object is missing:
None` would equally describe a clip really named `None`. Instead the row carries
the fixed diagnostic `clip_uri is null: the source row names no media location`,
and its warning identifies the `clip_id` with repr-style quoting, matching the
other row-local media warnings. It is charged to the row because a
source table can hold one such row among millions of good ones.

### Why an exhausted transport budget fails the run

`MediaTransportError` sits on the raise side despite being the *retryable* class.
By the time it escapes, the reader has already spent its whole attempt budget
(`execution.media_attempts`, default 3), and a budget that runs out for one clip
is far more often a degraded read path than a genuinely unreadable object. Every
remaining clip would plausibly fail the same way, so one row per clip would bury
a degraded read path in data — and those `error` verdicts would be
[terminal](#the-taxonomy-classifies-a-fault-it-cannot-bound-its-extent). The
checkpoint is what makes stopping affordable: a later attempt keeps the captions
this one finished.

The same reasoning applies to a backend failure the reader does not recognize.
`_classified` reports an unknown fault as transport rather than as an object fault,
which is the safe direction for the *reader* — an unknown per-object hiccup costs
a few retries — and in this recipe it means such a fault ends the attempt rather
than publishing a verdict nobody has interpreted.

There is one amplification to be aware of when reading logs. The driver enables
Ray Data's transient-map-error retries, which match substrings against the
formatted exception text. A propagated `MediaTransportError` is outside the
row-local redaction boundary: `MediaObjectReader` includes the raw URI plus the
inner exception's class name and text in that diagnostic. It can therefore be
retried at the *block* level on top of the per-clip attempts, so the total read
attempts a single bad clip provokes is the product of the two budgets, not the
sum.

The block-level retry **replays the whole block**, not the failing row. Every
clip in that block is fetched again from scratch, including the ones that had
already succeeded within the attempt, because a map task is the unit of retry and
it holds no partial result. So the amplification is wider than it is deep: one
bad clip near the end of a block re-transfers every clip before it, once per
block attempt. That is the cost of the backstop, and it is bounded — both budgets
are finite, and nothing that reaches a terminal `error` row is retried at either
level, since a deterministic fault raises no exception the patterns match.

Opening the clip table is also driver work, but it is deliberately not normalized
into a recipe exception: `run_config()` calls `lance.dataset()` directly and
propagates the native Lance or storage-backend open exception unchanged.
Callers must not depend on one concrete exception type or message for a missing
dataset. Once the table is open, source-column, context, and
caption-group contract faults raise `ValueError` on the driver, not inside a Ray
task where the exception type would be erased and the message wrapped in a
`RayError`. Missing required model files raise `FileNotFoundError` from the
registry preflight when pending work reaches that check.

### An engine-side failure is fatal by design

`should_continue_on_error` stays **False** on the engine stage, and the current
processor contract is stronger than "the stage does not swallow the error": the
stage **re-raises** a per-row generation failure instead of emitting an
`__inference_error__` row, so there is no such thing as an engine-produced error
row. One clip the engine cannot answer for ends the attempt.

That is deliberate, and the reason is that the two cases are indistinguishable
from inside the stage. A request failure looks identical whether the input was
unusable or the engine is dying, so `True` buys a per-row error row for the first
case by silently absorbing the second — and a table of `caption_status = 'error'`
rows written by a broken engine is worse than a stopped run, because nothing
downstream can tell it from a table of genuinely uncaptionable clips. Re-running
a stopped attempt is cheap; identifying and clearing a silently corrupted
population is not.

This is the reasoning behind watching the error verdict at all, so it is worth
naming what makes that cheap: the run summary reports `error_rows` directly, so
an operator sees the shape of a published wave **without writing a query**. What
the pin buys is that an error row can only have come from a row-local media fault
or an empty generation — never from the engine — which is what makes the count
interpretable rather than merely alarming. The count is reported and never gated
on; see [The summary reports the verdict
split](#the-summary-reports-the-verdict-split).

What makes that affordable is **recovery, not tolerance**. The row checkpoint
bounds the loss to the clips whose staged rows had not been committed, so an
engine fault costs one shard of GPU time rather than the attempt's. Treat the
checkpoint wiring in
[`inference.py`](../../../cosmos_curator/next/recipes/robot_action_caption/inference.py)
as a correctness requirement of this choice: without it, the same pin turns every
engine fault into a full re-run of the phase, and the tempting remedy — flipping
the flag — is the other half of the same mistake.

The only reachable `__inference_error__` source is therefore `build_request`
routing a media fault into it, which is what makes a row-local media fault and a
row-local inference fault travel one channel and be published by one code path.
It is also why `build_request` pre-seeds the five output columns as NULL: Ray
**bypasses postprocess entirely** for a row carrying that column, so a bypassed
row's output *is* the dict `build_request` returned, and without the seed an error
row would reach the Parquet write five columns narrower than a success row. The
seed is NULL rather than a plausible-looking status because NULL is one arm of the
pending predicate — a row that somehow reached publication without passing through
`canonicalize` stays pending instead of carrying a verdict nothing computed. The
bypass mechanism is a Ray Data LLM processor contract preserved by this recipe's
engine wiring.

The two hooks share one explicit surviving key set. `read_generation` re-emits
exactly the keys `build_request` leaves in the processor data dict, and
`canonicalize` selects `STAGED_SCHEMA` rather than applying a generic cleanup.
Ray removes its own `__data` and `__idx_in_batch` controls; recipe identity and
checkpoint columns remain because the terminal sink needs them.

## Media Transport

Clips travel to the engine as base64 `data:video/mp4;base64,...` URLs. The
alternative — passing pre-decoded frames and skipping the engine's own decode —
loses on bytes. An encoded clip is a few megabytes; 32 decoded frames at typical
resolution are tens to hundreds of megabytes of raw pixels, and those bytes cross
the Ray object store between the media stage and the engine stage. Base64 costs a
4/3 expansion on a small number; frames cost tens of times the same number, plus
the object-store pressure that comes with it.

Shared memory and `memfd` do not apply: the two stages are different Ray actors,
potentially on different nodes, so there is no shared address space to exploit.

The media type is fixed at `video/mp4`: the upstream splitter writes MP4
exclusively, so any other container would change the split contract.

The media and sampling controls are digested:

| Value | Default | Where | Why |
| --- | --- | --- | --- |
| `video.fps` / `video.min_frames` / `video.max_frames` | unset | config | Optional all-or-none temporal group. When present, the loader owns timestamp selection, including variable-frame-rate inputs; when omitted, the backend owns temporal behavior. |
| `video.min_pixels` / `video.max_pixels` | unset | config | Optional all-or-none processor-native pair. When omitted, the backend or artifact owns pixel behavior. |
| `DO_SAMPLE_FRAMES` | `False` | `contracts.py` | Without it the model processor re-samples frames we already selected. A short clip is the visible case: it arrives with every frame retained and is then sampled again, so the frames the caption describes are not the frames we budgeted or digested. Not configurable: `True` has no correct setting here. |
| `media.max_clip_bytes` | 128 MiB | config | Refused off the store's own `stat()` **before** the transfer starts, and re-checked against the delivered length because that stat is advisory and a backend may omit it. Digested — [why](#result-identity-the-contract-digest). |
| `MAX_SUPPORTED_CLIP_BYTES` | 256 MiB | `config.py` | Hard ceiling on the configurable cap — see the arithmetic below. The floor is 1 byte by the field constraint; a cap below roughly 1 MiB cannot hold a decodable video at all. |

Each group is passed only when complete. An omitted temporal group emits an
empty `media_io_kwargs` mapping; an omitted pixel group emits an empty
`mm_processor_kwargs` mapping. The recipe does not infer either group from the
model alias.

### Why the byte cap has a ceiling

The bounds are arithmetic, not decoration. A clip encodes to base64 at
`ceil(n/3) * 4`, and that string is what travels to the engine, so in-flight host
memory per engine replica is `execution.inference_batch_size` x
`execution.max_concurrent_batches` of them — and this recipe's driver switches
Ray's high-memory detector off, so nothing warns about it.

The table uses the default geometry: batch size 16, eight concurrent batches,
and eight replicas on one node. Those first two multipliers are execution
parameters, so the table is a reference operating point rather than a runtime
bound:

| Cap | Encoded | Per replica at defaults | x8 replicas at defaults |
| ---: | ---: | ---: | ---: |
| 64 MiB | 85.3 MiB | 10.7 GiB | 85.3 GiB |
| 128 MiB | 170.7 MiB | 21.3 GiB | 170.7 GiB (default) |
| 256 MiB | 341.3 MiB | 42.7 GiB | 341.3 GiB (clip-cap ceiling) |

At 256 MiB that is one third of a 1 TB host in base64 strings alone, before
vLLM's retained original bytes and decoded frames. `MAX_SUPPORTED_CLIP_BYTES`
enforces only the per-clip cap: raising either execution multiplier or placing
more than eight replicas on a node can exceed this aggregate without a config
error. See the corresponding [limitation](#limitations).

These are worst cases that only materialize if the corpus holds clips near the
cap; the current upstream splitter writes clips of a few megabytes. That is a
**property of the input data**, not a guarantee this code makes: bounding the
aggregate is the producer's job, not captioning's.

The per-clip peak is also higher than the steady-state figure. Building the data
URL is the one point where the raw bytes, the encoded string, and the
concatenation result all coexist, so a clip briefly costs several times its own
size whatever the configured cap.

## Configuration

Strict, versioned Pydantic v2, resolved by
[`config.py`](../../../cosmos_curator/next/recipes/robot_action_caption/config.py).
`cosmos-curator pipeline schema robot-action-caption` publishes the model itself
and is authoritative.

The sections split along exactly one line, and the split is the point:

```text
  input            source location + execution scope
  output           staging location
  generation   ┐
  video        ├─  SEMANTIC — every field reaches the contract digest
  media        ┘
  execution        throughput — never reaches the digest
```

A field belongs in a semantic section if changing it changes what a run
publishes, and in `execution` otherwise. `execution`'s invariant is what makes
retuning a run safe: it never invalidates the completed GPU work a resume would
reuse. The three semantic sections have the opposite property on purpose, which
is why they are sections rather than fields tucked into `execution` — an operator
reading the config should not have to know which of two adjacent knobs recaptions
the corpus. The template says so in its own preamble, so the split is visible
before anyone reads this document. It also renders supported optional settings
as explicit `null` values so the complete editable surface remains discoverable.

A minimal config:

```yaml
schema_version: 1
kind: robot-action-caption

input:
  clips_lance_uri: s3://example-bucket/robot_clips/lance/clips.lance
  selected_only: true
  # shard is optional; omitting it means this one job owns every fragment.
  # shard:
  #   count: 8
  #   index: 0

execution:
  storage_profile: default
```

That is the entire surface:

| Field | Default | Meaning |
| --- | --- | --- |
| `input.clips_lance_uri` | required | The `clips.lance` to read **and widen**. Input and output are the same table. |
| `input.selected_only` | `true` | Input scope. When true, require exact-string `curate_selection_reason` and owe only rows equal to `selected`; other values and NULL remain untouched. Set false to caption every otherwise-pending row, including tables without the Curate column. Excluded from the caption contract digest **and** from workspace identity, so a widened attempt resumes a crashed narrower one instead of stranding its captions ([why](#why-input-scope-is-not-a-namespace-input)). |
| `input.shard` | absent | Execution scope across independently launched jobs. Absent means one job owns every fragment. When present, `count` and `shard.index` are both required — a half-configured pair would leave work claimed by no job — and `index` must be less than `count`, refused at parse time. Excluded from the contract digest, so re-sharding recaptions nothing; **included** in workspace identity, so concurrent shards do not share a staging directory ([Sharding a caption campaign](#sharding-a-caption-campaign)). |
| `output.staging_root_uri` | `<clips_lance_uri>__caption_staging` | Root of the recovery workspace. Derived on read from the **already-normalized** table URI, so one table reached by two spellings cannot produce two workspaces that cannot see each other's work. Configured local paths are canonicalized to absolute paths; the workspace boundary rejects any relative local value that reaches it. Absolute bare paths, absolute local `file://` URIs, and supported remote URIs such as `s3://` are valid. |
| `generation.temperature` | `0.7` | Sampling temperature. **Digested.** |
| `generation.top_p` | `0.8` | Nucleus sampling mass; must be in (0, 1]. **Digested.** |
| `generation.max_output_tokens` | `512` | Sampling cap, and the first of the two bounds `caption_status` measures against. **Digested.** |
| `generation.max_model_len` | `32768` | Engine context length, and the quieter second bound — it can cut a caption short without anything raising. Must exceed `max_output_tokens`, which the section validates. **Digested.** |
| `model.variant` | `qwen2.5_vl` | Nonblank key in the global `all_models.json` catalog. Registration supplies model ID, immutable revision, precision metadata, and the standard staged path; vLLM decides runtime compatibility. **Resolved identity is digested.** |
| `video.fps` / `video.min_frames` / `video.max_frames` | unset | Optional all-or-none temporal group. Values must be positive and frame bounds ordered. **Digested as the exact produced mapping.** |
| `video.min_pixels` / `video.max_pixels` | unset | Optional all-or-none processor-native pair. Values must be positive and ordered. **Digested as the exact produced mapping.** |
| `media.max_clip_bytes` | 128 MiB | Per-clip byte cap; a larger clip becomes a terminal `error` row. **Digested** — changing it recaptions the configured scope ([why](#result-identity-the-contract-digest)). Must be positive and at most 256 MiB. |
| `execution.storage_profile` | `default` | Non-empty credential profile for clip reads, staged writes, and the commit. Surrounding whitespace is rejected rather than stripped. |
| `execution.inference_concurrency` | `auto` | Maximum vLLM replicas; `auto` lets Ray scale with pending work and live GPUs. |
| `execution.inference_batch_size` | `16` | Rows per Ray Data vLLM batch. Larger batches can improve GPU utilization but increase the host-memory reservation of every media-carrying stage, reducing how many of its CPU workers fit on a node. Only the payload term scales: each reservation also carries a fixed resident cost, so halving the batch does not halve the reservation. Minimum 1. |
| `execution.max_concurrent_batches` | `8` | In-flight Ray batch calls per vLLM replica. Raises overlap and host-memory demand without changing the GPU count. Minimum 1. |
| `execution.tensor_parallel_size` | `1` | Tensor-parallel factor within each vLLM replica. Minimum 1. The replica GPU width is this value multiplied by `pipeline_parallel_size`; model compatibility and placement feasibility are unchecked here — see the corresponding [limitation](#limitations). |
| `execution.pipeline_parallel_size` | `1` | Pipeline-parallel factor within each vLLM replica. Minimum 1. Multiplies `tensor_parallel_size` into the replica GPU width and is execution-only. |
| `execution.max_num_batched_tokens` | unset | Positive vLLM scheduler token budget per step. Unset preserves vLLM's own default, which at the pinned version is **2048** — far below one multimodal prompt, so the prefill is spread over `ceil(prompt / 2048)` scheduler steps and this is a throughput knob on any large-prompt run rather than a rarely-needed override ([the default trap](#the-scheduler-token-budget-and-its-default)). Execution-only and excluded from caption identity. |
| `execution.gpu_memory_utilization` | unset | Fraction of each GPU vLLM may claim, in (0, 1]. Unset preserves vLLM's own default. Whatever the weights leave becomes KV cache, which fixes how many requests one replica can hold at once, so on a large model this is the knob that sets engine residency — vLLM prints the result as `Maximum concurrency for N tokens per request`. Raising it trades startup headroom for residency and can fail at CUDA-graph capture. Execution-only: it never changes a caption. |
| `execution.enable_prefix_caching` | `false` | Whether vLLM retains KV blocks for prefix reuse. Two-state, not tri-state: vLLM's own default is **on**, and inheriting it is the known-wrong value here, so there is no way to ask for it ([why](#prefix-caching-ships-off-and-the-default-is-a-measurement)). Execution-only: prefix reuse is semantically transparent, so it cannot change a caption. |
| `execution.media_attempts` | `3` | Transport read attempts per clip before the reader gives up. Minimum 1. The only lever over an intermittently failing object store. |
| `execution.media_fetch_workers` | `4` | Clips one media actor reads at once, and also its batch size. Minimum 1. Raises concurrency while block-bound and partitions the node's budget once slot-bound — see the [limitation](#limitations). Rejected on the driver if no live node can satisfy the resulting CPU and `curator_io` request together. |
| `execution.media_cpus_per_fetch` | `0.25` | CPU reservation per concurrent clip read. The actor multiplies it by `media_fetch_workers`, so widening an actor does not silently reduce its declared CPU share per read. Must be positive and finite. |
| `execution.lance_read_batch_size` | `256` | Rows per Lance scan batch before media fetching. Minimum 1. |
| `execution.parquet_rows_per_file` | `256` | Shared minimum and maximum staged-row hints passed to Ray's Parquet sink. Minimum 1. Ray's sink adopts the minimum as the write bundle's row floor, and one bundle is one checkpoint, so this is the recovery granularity — see [Staged Parquet, and why it is needed alongside the checkpoint](#staged-parquet-and-why-it-is-needed-alongside-the-checkpoint). Lowering it below `lance_read_batch_size` only splits files, because a bundle can never hold fewer rows than the block reaching the sink. |
| `execution.source_frame_pixels` | `921600` (1280×720) | Pixels in one decoded source frame. With `video.max_frames` it sizes the heap reservations the three media-carrying CPU stages declare to the scheduler, so it changes actor packing density and never a caption. Set it from the widest camera in the corpus: understating it packs more actors per node than their working set fits, and the symptom is an OOM kill rather than an error ([how the reservations are derived](#what-a-media-carrying-stage-reserves)). |
| `execution.progress` | `false` | Ray Data progress bars. |

`media_attempts` is the one media-adjacent knob that belongs under `execution`
rather than `media`, and the reason is the section split, not convenience: a
**spent** budget raises and ends the attempt rather than publishing an `error`
row (see [why that is](#why-an-exhausted-transport-budget-fails-the-run)).
Retrying more never changes what a caption *says* — only whether a flaky store
yields one at all — so it is a throughput lever and is excluded from the digest.
`max_clip_bytes` is the opposite: it changes what a run publishes (see
[Result Identity: the Contract Digest](#result-identity-the-contract-digest)),
which is why it sits in `media`.
Existing configurations that omit `input.selected_only` now use the safe
selected-only default. A standalone caption run or a pre-curation table must opt
into the former caption-all behavior with `selected_only: false`; the default
refuses a missing or mistyped Curate column before widening the table. If the
column exists but no selected pending row does while other pending rows remain,
the no-work path succeeds without acquiring the model, workspace, Ray, or GPU
and logs the hidden pending count plus both remedies.
`max_num_batched_tokens` is also execution-only: it controls whether and how
vLLM can schedule an otherwise identical request, not what a successful request
contains. A capacity-only retry therefore reuses completed rows.

**The model is selected by global registry alias.** `model.variant` is not an
architecture or processor family. Configuration validates only that the key
resolves globally. The pipeline passes `/config/models/<model_id>` to vLLM;
unsupported architecture, processor, dtype, quantization, or request geometry
fails in the real engine path rather than in a recipe-local compatibility table.
Weights must already be staged under `/config/models` on every inference worker;
the runtime does not download them.

**Engine and IO geometry belongs to `execution`.** Batch size, in-flight batch
depth, tensor-parallel size, per-fetch CPU share, Lance read batch size, and
staged-file row hints all depend on the selected model, GPU topology, node
resources, or storage service. They remain outside the caption digest, so an
operator can retune a resumed attempt without discarding completed stochastic
captions.

The media *batch* size is the exception, and it is instructive about the rule.
It is not a constant and not independently settable: it is `media_fetch_workers`,
because a batch that is not exactly the actor's fetch width either starves
threads or queues rows behind them. That knob became public precisely because
the code could *not* choose it better — the right width depends on per-clip
latency at the object store, and the one measured run found the compiled-in value
six times too narrow for the corpus (see [Limitations](#limitations)).

### Video sampling and override contract

`video: {}` requests no overrides. It emits empty temporal and pixel mappings,
so the selected backend or artifact owns both behaviors. A runtime config may
provide the complete temporal group (`fps`, `min_frames`, `max_frames`), the
complete pixel group (`min_pixels`, `max_pixels`), both groups, or neither. A
partial group is rejected. The recipe does not inspect the model identity,
derive controls from an alias, or classify the backend.

The Qwen2.5 runtime YAML in the benchmark runbook explicitly requests 3 FPS,
4-64 frames, and a 3,136/602,112 pixel pair. Under that run-specific temporal
group, representative upstream spans request:

| Clip duration | Requested samples | Result under 4–64 bounds |
| ---: | ---: | --- |
| 4 seconds | 12 | 12 frames |
| 10 seconds | 30 | 30 frames |
| 20 seconds | 60 | 60 frames |

The run-specific ceiling starts binding above roughly 21.3 seconds (`64 / 3`). The upstream
`robot-action-split` default maximum is 20 seconds, so the normal span remains
fully covered with four frames of headroom. Higher or lower ceilings remain
valid configuration: no 40/64/96-frame quality benchmark establishes one
universal optimum. Three FPS is therefore this run's reasoned
coverage/accuracy operating point, not a measured quality claim.

The runbook's Qwen2.5 ceiling is 768 aligned 28x28 cells per frame. At 20 seconds,
60 frames occupy at most 23,040 visual tokens. At the 64-frame ceiling the exact
visual bound is 24,576 tokens, leaving 8,192 tokens in the configured
32,768-token context before prompt text and generation. The 512-token generation
cap fits inside that residual, but the recipe declares no stable prompt
allowance and therefore does not promise that every possible rendered prompt
fits. The earlier 2,048-token planning allowance was a design estimate, not a
runtime contract.

The runbook separately sets the execution-only
`execution.max_num_batched_tokens=32768`. vLLM 0.27.1's Qwen2-VL startup
estimator profiles at most 14 video frames, while the media loader can honor a
larger configured frame ceiling. With the 602,112-pixel cap, startup otherwise
estimates 5,376 embeddings (`7 * 768`) even though a 28-frame 1080p-derived
request produces 10,080 (`14 * 720`). The in-process engine's inherited
2,048-token scheduler default therefore leaves the cache at
`max(2,048, 5,376) = 5,376`. The runbook override raises that cache boundary to
32,768, which covers this run's 24,576-token one-video visual ceiling without
changing `max_model_len`. `limit_mm_per_prompt` remains the truthful item count
of one; it is not inflated to compensate for per-item profiling.

`max_num_batched_tokens` defaults to null in the generic recipe and is omitted
from engine arguments in that case. Equality with `max_model_len` is not a
universal Qwen3 policy: increasing the scheduler budget raises batching and
encoder-cache capacity, can increase activation or cache pressure, and can
reduce concurrency or memory headroom. Operators set it only for a measured
model/runtime policy. Pinned vLLM permits a value below `max_model_len` while
chunked prefill is enabled; if a backend disables chunked prefill or imposes
other scheduler constraints, vLLM remains the validation authority.

#### The scheduler token budget and its default

Omitting the key does not mean "no setting". It means vLLM's own default, which
at the pinned version is **2048 tokens per step** — and that number is the
reason the knob is described above as a throughput lever rather than a rare
override. A prompt longer than the budget is prefilled over `ceil(prompt /
budget)` scheduler steps, and this recipe's prompts are long: a measured
8×H100 run carried **12,657 prompt tokens** producing **90 output tokens**, a
140:1 ratio that makes prefill essentially the whole cost of a request. At the
inherited 2048 each clip's prefill was chopped into roughly seven steps.

Nothing downstream can tell an inherited value from a chosen one once it is a
plain `int` inside the engine, and grepping this repository for the knob returns
only the forwarding branch — which reads as *not relevant* rather than as
*inherited*. So the resolved budget is reported at startup **with its
provenance**, naming vLLM as the source when the key is omitted; the value alone
would have been in that run's log and still been missed, because it was vLLM
printing it rather than Curator. See [What the startup log says about
capacity](#what-the-startup-log-says-about-capacity).

#### The GPU memory fraction, and why it is a config field

`execution.gpu_memory_utilization` exists because the recipe previously could not
express it *at all*: the knob passes through `engine_kwargs` unhindered, but no
config field and no call site supplied one, so the most consequential memory
number in the run was whatever vLLM's own default happened to be — 0.9 at the
pinned version, and vLLM has already moved that default once, from 0.9 to 0.92.
A dependency upgrade could therefore reshape the run's memory budget with no
config change, and a grep of this repository for the knob's name returned
nothing, which reads as *not relevant* rather than as *inherited*.

What makes it worth a field rather than a diagnostic is what the fraction buys
on a large model. The measured run loaded a dense bf16 checkpoint whose weights
alone are **51.08 GiB**; whatever the fraction leaves after that becomes KV
cache, and the cache is what fixes residency. vLLM printed the resulting ceiling
directly — `GPU KV cache size: 270,336 tokens` and `Maximum concurrency for
32,768 tokens per request: 8.25x` — against a measured time-weighted **7.34
requests Running** and **0.49 Waiting**. The engine was running at essentially
its residency ceiling with almost nothing queued behind it, which is the shape
that says the ceiling itself is the bound. Raising the fraction is the lever that
moves it; nothing else in the config can.

The field is optional, and `None` is load-bearing rather than a placeholder: it
is what keeps vLLM's own value in force. A recipe-side default would silently
override the dependency on every run that never mentioned the knob, which is the
same defect one level along from the one this field closes.

#### Prefix caching ships off, and the default is a measurement

Prefix reuse is *structurally unreachable* for this recipe's request shape rather
than merely unhelpful: `REQUEST_CONTENT_TYPES` renders `video_url` before `text`,
so every prompt opens with its own clip's vision tokens and two requests diverge
at the first token. No block is ever a shared prefix. The measured run bears it
out at **0.0% hit rate across all 11,188 engine statistics samples**, while the
retained blocks still counted against reported KV usage — inflating exactly the
figure an operator reads to judge residency — and the block hashing still ran
over every 12,657-token prompt.

So `execution.enable_prefix_caching` defaults to `false`, and that default is
where the reasoning lives: an operator reads a field description, not a call
site. The field is **two-state**, and that is the one thing about it worth
stating twice. Its neighbours `max_num_batched_tokens` and
`gpu_memory_utilization` are optional precisely so a run can decline to have an
opinion and inherit vLLM's; here the inherited value is **on**, which is the
known-wrong one, so there must be no way to select it. The key is therefore
written into `engine_kwargs` unconditionally rather than behind an
`is not None` guard.

What would make `true` worth measuring is rendering the instruction text
**first**, which would give every request a genuinely shared prefix. That is a
change to `REQUEST_CONTENT_TYPES` — which the caption contract digests, so it
cannot happen quietly and it recaptions the configured scope — and it is the
change to make, not this flag on its own.

#### The preparation stage is built with the engine's own context length

`PrepareMultimodalStageConfig.model_config_kwargs` carries `max_model_len`
alongside `media_io_kwargs`, and both are copied from the same engine mapping
the engine itself is built from. The preparation stage constructs its own vLLM
`ModelConfig`, so left unpassed it resolves the context length from the
checkpoint instead: the measured run had all 21 `PrepareMultimodalUDF` actors
logging `Using max model len 262144` while the engine ran at `32768`. Nothing
raises, because only the engine's limit is ever enforced — the two halves of one
request simply size for different windows, and the disagreement is visible
solely as that stray log line.

Declaring the pair in that runtime YAML prevents the staged artifact from
silently widening the pixel ceiling for that run. It does not make the numeric
values interpretation-stable: Transformers or vLLM can reinterpret the same
processor-native values after an upgrade. Omitting the complete group delegates
to backend or artifact behavior. Qwen2 treats the pair per frame, while
Qwen3-style whole-video processors trade temporal coverage against spatial
detail over the full temporal volume. The recipe does not enforce that
association, classify models into budget families, or parse artifacts; the real
vLLM path remains the compatibility and token-admission authority.

The real vLLM processor and engine own architecture, preprocessing, and token
admission. An incompatible model or excessive request fails loudly in Phase A.
Changing semantic video geometry changes the digest; retuning the execution-only
scheduler budget does not. Row checkpointing therefore lets a capacity-only
retry reuse already completed rows from the failed attempt.

### What a media-carrying stage reserves

Three CPU stages sit between the media reader and the engine — preparation,
chat template, and tokenize — and each declares a heap reservation to Ray's
scheduler. `memory` is a **placement hint Ray never enforces**, so a stage that
under-declares is not throttled; it is simply admitted at a density its working
set does not fit, and the node is OOM-killed rather than raising.
[`VllmProcessorConfig.stage_heap_reservations`](../../../cosmos_curator/next/recipes/robot_action_caption/engine.py)
derives two figures from three terms:

```text
reservation = resident + batches_in_flight × copies_held × batch_size × bytes_per_row
```

- **`batches_in_flight` is 2, and it is Ray's number rather than ours.**
  `ActorPoolMapOperator` resolves the pool's dispatch depth from the compute
  strategy, else the `DataContext`, else `max_actor_concurrency × 2`.
  `ray.data.llm`'s `build_cpu_stage_map_kwargs` supplies none of the three and
  sets no `max_concurrency`, so every CPU stage here lands on `1 × 2`. Nothing
  on the `_StageConfigBase` surface can lower it, which is why the reservation
  has to absorb it: an actor holds its next input while it works on the current
  one, so a figure sized for one batch under-declares by half.
- **`copies_held` differs by stage**, which is why there are two figures rather
  than one. A stage that only forwards the payload holds two copies — the
  unpickled input, and the output buffer pickle builds beside it, a round trip
  paid at every hand-off because Arrow cannot represent an ndarray nested in a
  struct. The preparation stage also decodes, adding the resized result beside
  the decoded frames, so it holds three.
- **`resident` is 3 GiB and does not scale with the batch**: interpreter,
  tokenizer, and the vLLM `ModelConfig` the preparation stage builds for itself.
  A purely proportional formula trends to zero exactly where this term is most
  of the footprint. The same figure is declared on both roles even though the
  `ModelConfig` is the preparation stage's alone — the measured spread between
  the roles is small beside the payload term, so on a forwarding worker the
  term is deliberately conservative rather than measured.

Worked at `source_frame_pixels: 921600` and `inference_batch_size: 16` (both
shipped defaults) with `video.max_frames: 64`, one row of media is 168.75 MiB
and one batch is 2.64 GiB, giving **18.82 GiB per preparation worker** and
**13.55 GiB per forwarding worker**. Leaving `video.max_frames` unset instead
falls back to the loader's own 32-frame default and halves the payload term,
giving 10.91 GiB and 8.27 GiB — and it also emits a warning, because that bound
holds only for a backend that honours `num_frames`. Whichever pair applies is
reported at startup; see [What the startup log says about
capacity](#what-the-startup-log-says-about-capacity).

Two consequences worth carrying into a sizing exercise. The figures are
*per worker*, and the three stages inherit the engine's replica range as their
own pool bounds, so the memory an allocation must place before any row moves is
each figure multiplied by that pool floor — not the figure alone. And because
`resident` does not scale, the payload term no longer cancels out of
`node_memory / reservation`: lowering `inference_batch_size` now genuinely
reduces the media in flight rather than only redistributing it across more
workers.

#### Both roles are sized from the pre-resize ceiling, and the forwarding pair over-reserves

`bytes_per_row` is one figure for both roles, and for the two forwarding stages
it is measurably too large. It is a ceiling on the payload **inside** the
preparation stage, before the processor resizes; the column that leaves that
stage has already been resized to the processor's pixel budget. On the measured
run the payload actually crossing the `ChatTemplate` and `Tokenize` boundaries
was **11.05 MiB per row** — agreeing to within 4% across three independent
readings, 171–177 MiB per 16-row block at each of those two stages and at the
engine — against the **168.75 MiB per row** both roles were sized from, a factor
of **15.3** on the per-row term.

That factor does not carry through whole, because `resident` does not scale with
it. Substituting the measured row into the same formula at the shipped batch of
16 gives `3 GiB + 2 × 2 × 16 × 11.05 MiB` = **3.69 GiB** per forwarding worker
against the **13.55 GiB** declared: over-declared **3.67×**, with the fixed 3 GiB
term absorbing the rest. Across that run's 28 forwarding workers it is 379.3 GiB
declared where 103.3 GiB would have covered it, so roughly **276 GiB was
reserved for payload that never arrived** — against 395.2 GiB legitimately
declared by the 21 preparation workers, on a cluster the run drove to 79.4% of
node memory. Meanwhile 91 of 192 CPUs sat idle, because the reservation rather
than the CPU budget capped how many of those cheap actors could be placed.

The consequence is stranded reservation, not overcommit, and that asymmetry is
the whole reason it is tolerable.

**It is deliberately not fixed, and the reason is that it cost nothing.** The
engine was the constraint on that run: it was never backpressured, and every
upstream stage was throttled behind it — fetch by `DownstreamCapacity` 88% of
samples, the four preparation stages by `ResourceBudget` 94%. Placing more of
the cheap CPU actors would have produced rows the engine could not consume any
faster. Splitting the figure also cannot be done *well* today: the honest
forwarding number is the resize ratio applied to the decode ceiling, and that
ratio is only derivable when `video.max_pixels` is set, which the measured run
left unset. A hand-set second field would be one corpus's measurement applied to
every later one, in the direction where being wrong is unrecoverable —
under-declaring is what OOM-kills a node, because `memory` is a placement hint
Ray never enforces.

So the direction of the error is the justification for leaving it: over-declaring
wastes budget, under-declaring kills the job. **Revisit it when the object store
or host memory becomes the binding constraint rather than the GPU** — the signal
is the engine stage no longer reporting itself throttle-free while the CPU stages
sit on `ResourceBudget`, or host memory refusing actors the CPU budget could
otherwise place. At that point derive the forwarding figure from
`video.max_pixels` so it stays a property of the configured policy rather than a
second number to keep true by hand.

The shape is derived; the constants are fitted to one measured run, in which a
`MapBatches(PrepareMultimodalUDF)` worker's resident set reached 16.47 GB
against a then-declared 5.66 GB, and that operator's pool of 230 workers
exhausted two nodes. Re-measure the in-flight factor after a Ray bump *or* a change to
`RAY_DATA_ACTOR_DEFAULT_MAX_TASKS_IN_FLIGHT_TO_MAX_CONCURRENCY_FACTOR`, which
overrides its default of 2 and so moves the factor without any version change —
only that factor can move, not the shape — and re-derive `resident` and
`copies_held` if a run is profiled at a second batch size, which is what would
separate the fixed term from the proportional one more sharply than a single
point can.

### Where a constant lives is decided by the digest

A constant the digest stamps lives in
[`contracts.py`](../../../cosmos_curator/next/recipes/robot_action_caption/contracts.py),
beside the digest that stamps it, and is imported from there by whichever module
consumes it — never restated locally. The direction is the whole guarantee: the
digest decides which published rows are still valid, so a value it stamps has to
be the *same object* the run used. Push one back down into its consumer and the
digest is stamping a copy; editing that copy would change every caption while the
digest stayed byte-identical, leaving two populations in one table with nothing to
tell them apart — the same trap as
[the comparison shape](#terminal_policy_version-covers-what-the-thresholds-cannot),
reached from a different direction.

That is why the following are `contracts.py` constants even though each reads like
it belongs to the module that uses it:

| Constant | Consumed by | Digest block |
| --- | --- | --- |
| `REQUEST_ROLE`, `REQUEST_CONTENT_TYPES` (`video_url`, `text`) | `build_request` in `request.py` | `request` |
| `MEDIA_TYPE` (`video/mp4`) | the data-URL prefix in `media.py` | `request` |
| `VIDEO_ITEMS_PER_PROMPT`, `PREPARE_MULTIMODAL` | the processor build in `inference.py` | `request` |
| `DO_SAMPLE_FRAMES` | the per-request `mm_processor_kwargs` flag in `build_request` | `video` |
| `TERMINAL_POLICY_VERSION` | nothing but the digest — it exists to be stamped | `terminal` |

`DO_SAMPLE_FRAMES` rides on the **request**, not on the engine build, and that
placement is the point rather than an oversight. Ray forwards a row's
`mm_processor_kwargs` into the vLLM prompt it constructs, so the flag reaches the
model processor from there; restating it in the processor build's
`mm_processor_kwargs` would put the same policy in two places, which is the copy
trap this section exists to forbid. `video.mm_processor_kwargs()` carries the
pixel pair and nothing else, and the digest stamps that mapping verbatim beside
the constant, so neither can move without moving the digest.

The same rule extends to the **configured** digest inputs. The run threads the
very `generation` and `video` sections it hashed into `build_request`,
`canonicalize`, and the processor build, and passes the hashed
`media.max_clip_bytes` into `ClipMediaFetch` — rather than letting any consumer
re-read or re-derive a value of its own. A digest input read in one place and
re-derived in another is the same copy trap reached through configuration instead
of through a constant.

The **engine context length** is the one most likely to be mistaken for a
throughput knob, and it is not: a context length that cuts a generation short
*without* raising changes what a caption says while nothing raises, so a row
produced under a different one was produced under different semantics. That is
why `generation.max_model_len` sits in a semantic section, and why editing it
recaptions the configured scope.

`REQUEST_CONTENT_TYPES` carries render **order**, not merely membership — the
model reads the clip before the instruction text, and swapping the two changes the
generated caption — which is why the digest stamps the list rather than a set.

`inference_concurrency` survives as a cap because a cluster's GPU budget is
genuinely an operator's decision. It is a *ceiling*, not a target:
`_inference_concurrency` derives the ceiling from pending work
(`ceil(pending_rows / batch_size)`, then clamped by the operator's cap if set) and
uses the live GPU count only to size the *initial* request. A GPU snapshot taken
before the pool starts is the wrong bound in both directions — on a cluster still
scaling up it reads near zero and pins the run to a fraction of the hardware it is
about to get, and on a large cluster it would license more replicas than there is
work to fill.

## Failure Modes and What a Resume Does

No failed attempt rolls anything back. The caption columns stay registered once
widened, and every staged row and checkpoint entry already written under the
derived namespace stays durable — so a retry on the same inputs re-resolves that
same namespace and re-infers only the clips it does not already hold there. Which
faults are refused before a GPU is allocated is listed under
[Launching](#launching); this table is what remains true once one is not.

| Last durable event | What the next invocation does |
| --- | --- |
| Failure before or during Phase A | Re-pins the same version (the columns already exist), resumes into the same namespace, and infers only the clips whose rows were not checkpointed. |
| Staged file written but its checkpoint still pending | Ray discards the pending output and replays that staged file. Bounded to one staged file's worth of GPU work. |
| Phase A finished, crash before publication | **The case this design exists for.** Every staged row is reused; nothing is inferred again; publication commits the lot. |
| A crash between publication waves | The waves that landed are canonical and their clips are complete under any later `count`. The next attempt re-prepares and republishes only the fragments no wave reached; nothing is re-inferred, and no fragment is half-published. |
| A validation failure in Phase B | Nothing is published. Fix the cause (a conflicting duplicate, a clip pending twice, a digest mismatch, a missing or extra staged clip) and re-run; the staged rows are intact. |
| Commit raised, the table shows no pending row in the wave, and this attempt's marker is in a newer transaction | The commit landed and the error was a lost response. The run reports the latest observed version with a WARNING; nothing to redo. |
| Commit raised and the table shows no pending row, but this attempt's marker is absent | Another writer may have completed the rows. The run raises a conflict error because it cannot attribute the canonical state to this attempt. |
| Commit raised and pending rows remain | Nothing was published. The staged rows are intact and a re-run publishes them without re-inferring. |
| Commit raised and a targeted fragment is absent | Whether it landed cannot be determined, so the run refuses as unverifiable rather than guessing. The staged rows are intact; inspect the table before re-running. |
| Commit succeeded, cleanup failed | The captions are canonical. Delete the workspace by hand: the namespace is version-independent, so a later attempt resolves to it again. Owing nothing, that attempt never opens it and the cost is storage alone; owing a row in a fragment that also holds the leftover staged rows, publication refuses before committing until the workspace is gone. |
| `PROMPT_VERSION` or any digested constant changed | A different namespace and a different contract. **Nothing is reused**, by design, and the rows captioned under the old contract become pending again and are overwritten. |
| An upstream schema change added or removed a supported context column | Same as above: the resolved subset moved, so the digest moved. The corpus is recaptioned. |
| Source version moved *without moving rows* (another leg's commit, this recipe's own widening) while a workspace holds progress | Resumes the **same** namespace and reuses every staged row. The checkpoint id names no location, and the version is not a namespace input. |
| Source **compacted** while a workspace holds progress | Resumes the same namespace, but the staged rows are keyed by retired fragment ids so none matches an owed row: the attempt re-infers its whole owed set. Nothing is corrupted, and the unmatchable partitions are dead weight until the namespace is deleted ([why](#no-recovery-across-compaction)). |
| A shard failed and is re-launched at the same index | Resumes its own namespace and infers only what its own slice still owes. Other shards are unaffected and were never waiting on it. |
| A shard's `count` changed mid-campaign | A different partition, so a different namespace. Committed captions are unaffected — a shard is not result identity — and that now includes every publication wave that landed before an interruption. What is stranded is progress the old count had only *staged*: a job that crashed during Phase A loses all of it. Choose `count` per campaign and keep it. |
| `input.selected_only` widened from `true` to `false` after a crash | Resumes the **same** namespace and infers only the clips that were not checkpointed, plus the newly in-scope ones. Scope is [not a namespace input](#why-input-scope-is-not-a-namespace-input). |
| `input.selected_only` narrowed from `false` to `true` after a crash | Resumes the same namespace, but publication refuses the staged clips the narrower attempt does not owe. Nothing is committed and the staged rows are intact; re-run widened to publish them. |
| `robot-action-split` appended rows after the pin | Those rows are outside this attempt and stay pending for the next one. |

Row-local media failures are **terminal data**, not transient: an ordinary re-run
under the same contract does not retry them, because the row now carries a status
and the current digest. See [The taxonomy classifies a fault; it cannot bound its
extent](#the-taxonomy-classifies-a-fault-it-cannot-bound-its-extent) for what it
takes to revisit them.

## Operator Runbook

### Launching

`robot-action-caption` is a pipeline kind, launched by the generic pipeline CLI
from a config file — there is no recipe-specific runner:

```bash
cosmos-curator pipeline template robot-action-caption > caption.yaml  # edit clips_lance_uri
cosmos-curator pipeline validate caption.yaml   # config faults only, no table read
cosmos-curator pipeline render   caption.yaml   # every resolved value, defaults included
pixi run --as-is run-pipeline caption.yaml      # execute
```

`validate` and `render` are config-only: they resolve the Pydantic model and read
nothing. Every data-contract check is the run's own driver work, and it happens
before a GPU is allocated — a missing `clip_id`/`clip_uri`, a `clip_id` stored as
anything but `string`, a non-text `clip_uri`, a supported context column present
but not holding text, or a partial caption column group all raise `ValueError`,
while opening the table itself propagates the native exception from
`lance.dataset()` or its storage backend unchanged, as detailed in
[Error Taxonomy](#error-taxonomy). Model-file availability has one registry-owned
driver preflight: when work is pending, the model directory must exist with
`config.json` and at least one recognized weight file before the workspace is
resolved; missing required files raise `FileNotFoundError`. A blank, unsupported,
relative, or ambiguous local staging root is refused before namespace
construction; an unwritable valid root is *not* caught by `resolve_workspace`,
which writes nothing, so the first permission failure it can produce is the staged
write inside Phase A.

The run prints two lines worth watching before any GPU work starts: the resolved
context columns (`resolved caption context columns [...] out of supported [...]`)
and the plan (`Caption attempt on <uri> @v<N> (<scope>): ... owned fragment(s),
... of them owed, ... owed row(s) ... and contract <digest>`). The first tells
you which optional columns this snapshot supplied, and the second gives you the
digest every query below needs. Its `<scope>` reads `unsharded` or
`shard <index>/<count>`, and **every count on that line is relative to what this
job owns** — see [Sharding a caption
campaign](#sharding-a-caption-campaign) for driving a multi-job campaign.

For multi-node capacity, submit the same driver command to a managed Ray cluster
as described in [Managed Ray Clusters on Slurm](curator-next-slurm-ray.md); the
[Curate runbook](../guides/curate-runbook.md) walks the same two launch paths in
detail.

### What the startup log says about capacity

Every capacity figure this phase depends on is resolved on the driver, before a
GPU is allocated, and each one is logged at `INFO`. The lines are meant to be
read together: the fetch-bound run in [Limitations](#limitations) is
diagnosable from them alone, without reconstructing any of it from source.

- **The source table's shape.** `Caption source s3://…/clips.refragmented.lance
  v23: 131,602 rows, 17 fragments (min=3,602 median=8,000 max=8,000 per
  fragment), 26 fields` — the geometry the run in [Limitations](#limitations)
  read, with the field count standing for however wide that table had been
  widened by then. Logged first, before Ray is initialized, because it describes
  the durable read and publication geometry: the fragment count bounds source
  scan tasks, and the largest fragment bounds one source task's scan tail. It no
  longer determines the media task count; lightweight rows are split into
  bounded execution blocks before fetch. Here the fragments are even — 8,000
  rows apiece, the `clips_per_publish_batch` default the producing split run
  wrote at, plus a short remainder. The distribution remains useful because the
  maximum and the mean can diverge, exposing source-scan skew even though fetch
  work is independently partitioned. Summarised from manifest metadata by
  [`report_table_shape`](../../../cosmos_curator/next/utils/lance_utils.py),
  which reads no column data, so it costs nothing against remote storage.

  This one line gains a suffix, at the same `INFO` level and in the same record,
  when at least half the fragments hold fewer rows than Lance's own small-file
  boundary of 1,024 — the point past which per-fragment manifest entry, file
  open, footer read and scanner setup stop being amortized. The suffix states
  that boundary and the cost of moving either way from it, and stops there: no
  instruction, and no operation named. That restraint is deliberate in both
  directions. A table refragmented to ~1,000-row fragments historically made
  this recipe's fetch-bound run finish by collapsing a single-actor straggler
  tail, but downstream row repartition now obtains that media parallelism
  without changing the table. Small fragments still carry storage overhead, and
  too few fragments can still limit source-scan startup; neither condition is a
  media-pool verdict anymore.
- **The node IO budget, and where it came from.** `Curator node IO budget
  resolved to 16 curator_io slot(s) per node from the built-in default
  (COSMOS_CURATOR_RAY_IO_SLOTS_PER_NODE is unset).` The provenance half is the
  point — a resolved `16` cannot say whether anyone chose it, and an un-passed
  `--ray-io-slots-per-node` implies a different fix from a deliberate one. Inside
  a Slurm job the line instead names the variable, because
  [`sbatch.sh.j2`](../../../cosmos_curator/client/slurm_cli/sbatch.sh.j2) exports
  it on every job (from the flag when passed, from the same built-in default
  otherwise), so "set" there is not evidence that an operator picked the number.
  Emitted by
  [`curator_io_slots_per_node`](../../../cosmos_curator/next/core/ray_runtime.py),
  so every recipe that reserves IO slots reports it.
- **What the cluster advertises.** `Ray cluster snapshot at driver start: 4 live
  node(s) advertising 384 CPU, 32 GPU, 64 curator_io in total. Nodes still
  joining are not counted.` — followed by one `node <address>: …` line each while
  the cluster is small enough for that to stay readable. It is a snapshot and
  says so: the Slurm supervisor releases the driver as soon as the *first* worker
  registers, so a four-node job can legitimately log fewer than four nodes here.
  The per-node breakdown is what exposes a heterogeneous cluster, where dividing
  a total by the node count is a fiction. Nothing waits for a complete picture —
  that would trade a diagnostic for a startup stall on every recipe.
- **How much of that table is still pending.** A resume of the run in
  [Limitations](#limitations) with sixteen of its seventeen fragments already
  captioned reads `Caption plan: 1 of 17 fragment(s) pending, holding 8,000
  row(s) with 8,000 in the largest. Non-strict row repartition will produce 32
  fetch work block(s) at 256 row(s) per block without coalescing fragments. The
  largest pending fragment still bounds one source scan task's tail, but no
  longer confines that fragment's media fetches to one actor.` This is the
  recipe-owned reading the shared helper cannot make: checkpoint filtering
  determines each fragment's pending row count, and non-strict repartition makes
  the downstream block count exactly `sum(ceil(pending_rows / R))`. The first
  attempt on the same table produces 527 fetch blocks from its 17 source
  fragments at `R=256`; the resume produces 32 from one source fragment. The
  largest figure is a *pending* row count, not the fragment's physical size, so
  it measures the source scan task's remaining tail while the derived block
  count measures media work.

  Deliberately `INFO`, however far the resulting block count sits below the
  allocated width. Whether that geometry is harmful also depends on per-clip
  service time and the cluster's idle policy, neither of which this phase owns
  or can infer at startup. The structural half that *is* decidable, a fetch
  supply too small to feed the engine floor, is compared on the line below in
  the unit the two sides share.
- **The fetch ceiling against the engine floor.** `Media fetch width 4 fits on 4
  of the 4 node(s) registered so far. The per-node budgets place 16 media
  actor(s) at once, holding 64 concurrent read(s), and the pool is capped there
  so it never requests an actor no node can host; the input block count caps it
  further.` then `Caption supply versus capacity, as two stocks measured at this
  instant and not as rates: media fetch holds at most 64 clip read(s) in flight —
  a ceiling, the lower of 64 from the live nodes' IO and CPU budgets and 2108
  from 527 input block(s) at width 4. The engine pool's floor of 32 replica(s)
  holds at least 1024 row(s) in flight to give each replica one full batch (32 x
  batch size 32); the pool may scale to 4113. Per replica that is 2.00 read(s) in
  flight behind a 32-row batch. How far the fetch side actually falls behind is
  set by the two stages' service times, which this run has not measured yet, so
  neither stock divided by the other is that factor. Concurrent reads cap the
  rate rows arrive at rather than the rows a replica can draw on - completed
  fetches queue ahead of the engine - so a supply under the floor's appetite is
  not itself starvation. If the block ceiling binds, lower
  execution.lance_read_batch_size below 256 to create more fetch work blocks
  without rewriting the Lance table.` The
  two sides are sized by unrelated inputs — the node budgets and repartitioned
  row blocks on one side, the pending row count on the other — so neither number
  alone says what the run was configured to hold. Both fetch bounds are printed
  rather than only the smaller: which one binds decides which lever is even
  relevant — a larger IO budget or a smaller row-block target. Both capacities are stated in rows in flight, the
  one unit they share; no rows-per-second figure is derived, because per-clip
  fetch and generation latencies are not known at startup. The denominator
  carries its own caveat — *of the 4 node(s) registered so far* — rather than
  borrowing the snapshot line's: the two are far enough apart that an operator
  quoting the fetch line alone would read a count of whichever nodes had joined
  as a count of the nodes the job was allocated.

  Two figures reported at one instant are both *stocks*, and dividing one by the
  other does not yield a shortfall factor however tempting the ratio looks: that
  factor is set by the two stages' service times per clip and per row, neither of
  which exists at startup. So the line names them as stocks, states them in the
  unit they share, and prints no multiplier. The line then says why the smaller
  number is not a shortage: concurrent reads bound the *rate* at which rows
  arrive, not the rows a replica can draw on, because completed fetches queue
  ahead of the engine. The same two numbers are also
  given **per replica**, because at cluster scale a pair like `64` against `1024`
  reads as merely large against merely larger, while `2.00 read(s) in flight
  behind a 32-row batch` is immediately legible. Two decimals rather than
  integer division, and deliberately: the extreme case is a floor holding more
  replicas than the supply has reads, which integer division renders as `0` — an
  absence of supply rather than the thin margin it is.

- **The scheduler token budget, and whether anyone chose it.** `Engine scheduler
  token budget: 2048 token(s) per step, from vLLM's own default because
  execution.max_num_batched_tokens is unset, not from a choice. A prompt longer
  than this takes more than one scheduler step to prefill, so a multimodal
  prompt of tens of thousands of tokens is spread over many steps at this
  value.` Reported either way, and the provenance half is the point for the same
  reason it is on the IO-budget line: an omitted key is a non-trivial dependency
  default, and `2048` against this recipe's ~12,657-token prompts is the
  difference between one prefill step and seven. When the YAML does set it the
  line instead reads `…, from execution.max_num_batched_tokens.`, so the two
  cases are told apart by the source clause rather than by the number — see [The
  scheduler token budget and its default](#the-scheduler-token-budget-and-its-default).
- **The resolved video policy the engine is built with.** `Caption video policy
  reaching the engine: media_io_kwargs={'video': {'fps': 3.0, 'min_frames': 4,
  'max_frames': 64}}, mm_processor_kwargs={'min_pixels': 3136, 'max_pixels':
  602112}.` The temporal group is nested under `video` because that is the shape
  the loader receives, and the processor mapping carries the pixel pair alone —
  `fps` is a loader concern and never appears there. Read from
  [the same two accessors](../../../cosmos_curator/next/recipes/robot_action_caption/config.py)
  the contract digest consumes, so the echo cannot describe a policy other than
  the one that ships. The digest proves two runs agree with each other; it cannot
  say *what geometry ran*, which is the question asked when a prompt overruns the
  context window, and recovering the pixel cap from the run otherwise means
  inverting the engine's own encoder-profiling arithmetic. A group this config
  declares nothing for is echoed as delegation — `unset, so the model artifact
  decides the pixel bounds` — rather than printed as `{}`, because an empty
  mapping and an inherited oversized cap look identical on the line and only the
  second is a problem.
- **A width that bought nothing.** At `WARNING`, and only when the configured
  width is above the default and holds no more reads than the default would:
  `Media fetch width 8 does not raise total read concurrency above the default
  width 4: 64 concurrent read(s) against 64. Each read reserves one curator_io
  slot, so a wider actor claims a larger share of the same per-node budget — 8
  actor(s) here against 16 at the default — instead of adding to it. Raise the
  budget with --ray-io-slots-per-node to raise the total; raise the width to
  shorten the tail one actor works through on its last block. This run raised
  the width alone, so it holds no more read capacity than the default does, in
  fewer and faster actors.` Because each concurrent read reserves one slot, the
  width *partitions* the node budget instead of enlarging it: placeable actors
  per node is `slots ÷ width`, so total reads is `(slots ÷ width) × width`, and
  `(16 ÷ 8) × 8 = (16 ÷ 4) × 4 = 16` under integer division. Raising the width
  alone is therefore inert on the total by construction, and a production run at
  width 8 bore that out: its throughput tracked the 64-read budget and did not
  move when the running actor count did.
  Nothing warns merely because the block count binds instead: a
  row-block-bound pool is a legitimate state calling for a smaller
  `execution.lance_read_batch_size`, and the line above already names which of
  the two bounds is active. When a width does not divide a node's budget evenly
  the remainder is stranded, so the ceiling line adds `The width does not divide
  every node's budget evenly, leaving 5 curator_io slot(s) unusable
  cluster-wide.`
- **What the media-carrying CPU stages reserve.** `Media-carrying stages reserve
  18.82 GiB per preparation worker and 13.55 GiB per forwarding worker, from a
  168.8 MiB per-row ceiling. Each figure covers every batch Ray Data keeps in
  flight per actor, not just one, plus a fixed resident cost that does not
  shrink with the batch — so halving execution.inference_batch_size does not
  halve it.` Two figures because the stage that decodes holds more copies than
  the two that only carry the result onward; [how both are
  derived](#what-a-media-carrying-stage-reserves). These are placement hints Ray
  never enforces, so the number to compare each against is node memory, not the
  other figure: a stage that under-declares is not throttled, it is packed at a
  density its working set does not fit and the node is OOM-killed rather than
  raising. The per-row ceiling is the half an operator checks against their own
  footage, since it is derived from `source_frame_pixels` and `video.max_frames`
  and understating either has no symptom until a worker dies. Absent entirely
  under a text-only configuration, where no stage carries a payload.
- **The resolved engine pool bounds.** `Built vLLM transform for … with an engine
  pool floored at 32 replica(s) and capped at 4113, each on 1 GPU(s); the floor
  pins 32 GPU(s) for the run's duration. Bounded restarts on stage(s) […]` Ray
  Data never scales an actor pool
  below `min_size`, so the floor is a reservation, not a starting point — and the
  GPU count it pins is the figure a cluster's idle-GPU reaper reports back. The
  recipe states the same number one layer up, where it is derived: `Caption
  inference will hold a floor of 32 replica(s) and scale to at most 4113, each on
  1 GPU(s), for 131602 pending row(s) against the 32.0 GPU(s) Ray reports free
  right now.` Both lines call element 0 a floor, because a run diagnosed from one
  of them must not have to reconcile it against the other.
- **What that floor cannot be told.** The floor line closes with `That GPU
  reading is a snapshot of whichever nodes have registered so far, and this floor
  is fixed for the run: nodes joining later let the pool scale toward its ceiling
  but never raise the floor, so a floor taken from a partial cluster under-uses
  the allocation until the run ends.` The exposure is real and structural: the
  Slurm supervisor releases the driver when the *first* worker registers, so a
  multi-node job can reach this derivation with one node in view. It is *stated*
  rather than *detected* because the driver cannot know what it is missing.
  `SLURM_JOB_NUM_NODES` is forwarded into the container, but the driver runs
  inside the head's own allocation, which asks for a single node; the accelerator
  nodes are a separate array allocation, and only a lane's own index travels with
  it. So the variable reads `1` in the driver however many lanes were submitted,
  and a warning keyed on it would be a permanent all-clear on exactly the path
  that has the race. Nothing waits for the remaining nodes either — whether to
  harden the derivation is a separate decision from making its exposure legible.
  Note that the inert-width warning above
  is unaffected: it compares two ceilings computed from the same node list, so a
  homogeneous cluster still assembling scales both together and the verdict
  survives, even though the absolute ceilings on that line do not.
- **A Ray actor-restart warning that no recipe setting can clear.** Ray logs
  `Actor with class name: 'MapWorker(MapBatches(ClipMediaFetch))' … has
  constructor arguments in the object store and max_restarts > 0. If the
  arguments in the object store go out of scope or are lost, the actor restart
  will fail`, citing
  [ray-project/ray#53727](https://github.com/ray-project/ray/issues/53727),
  which is open. The named stage is incidental: Ray emits this once per driver
  process, so it names whichever actor-backed stage was created first, and the
  exposure it describes is identical on every one of them. It is also *not* about
  the arguments this recipe passes, which are a profile name and three integers.
  Ray Data hands each map worker its serialized transformer and its
  `DataContext` as object references, so the warning's condition is met by the
  operator's own construction whatever a caller's arguments weigh. The only
  recipe-side lever in that condition is `max_restarts`, and setting it to zero
  trades a real recovery path for a quieter log — Ray Data's actor pool assumes
  in-place restarts and recovers a dead worker no other way. The bound stays
  finite and non-zero for that reason; see
  [`ACTOR_RESTART_BOUNDS`](../../../cosmos_curator/next/recipes/robot_action_caption/engine.py).

### What the run log says while the phase executes

The startup lines above describe what the phase was *configured* to hold. Three
more diagnostics describe what it actually did, and they exist because Ray's own
progress output cannot answer the questions a slow run raises. All three are
periodic or post-run rather than per-row: at this recipe's batch sizes a per-row
line is hundreds of thousands of lines, and every quantity here is only
meaningful over many samples anyway.

- **What the object store sustained.** Each `ClipMediaFetch` actor summarises its
  own transport every 100 batches: bytes read, transport-seconds against
  batch-wall-seconds, MiB/s, per-clip read p50/p90/max over its successful reads,
  and — the two figures no recipe-level clock could produce — the **attempts**
  those reads consumed and the seconds that **delivered nothing**. Attempts above
  delivered reads are retried transport faults and refused objects, so a store
  answering two requests in three is visible here and nowhere else: every `read()`
  call still returns bytes, so a rate alone reports a degrading store as a merely
  slow one. Nothing else in the phase measures the store, and the stage's *queue
  depth* cannot stand in for it — that depth is set by downstream backpressure,
  so a fetch-bound run and a fetch-idle one look identical without this. The two
  time figures are reported separately rather than differenced into a "CPU"
  number: reads overlap on the actor's thread pool, so wall time can fall either
  side of the summed transport time and their ratio is the overlap achieved, not
  a duty cycle. Only reads that **delivered bytes** reach the transport term and
  the quantiles; a failed attempt's duration is an attempt budget and is charged
  to the undelivered term instead, because folding it into the rate would make a
  degraded store look slow. Below two delivered reads the line says so rather
  than quoting quantiles — a p50 of `0.00s` reads as an instantaneous store,
  which is the opposite of what no measurement means.

  The measurement itself is taken one layer down, in
  [`MediaObjectReader`](../../../cosmos_curator/next/utils/media_reader.py).
  `read()` retries internally, so a clock wrapped around the call spans every
  failed attempt plus the backoff between them and reports the total as
  "transport": a clip delivered on its third attempt looked like one slow
  transfer, and the attempt count was not observable from outside the reader at
  all. The reader therefore times the delivering attempt alone, derives the rest
  of the call as a residual, counts every attempt whether or not it delivered,
  and accumulates all of it under its own lock — which is what makes the stage's
  concurrent thread-pool reads safe to measure without the stage collecting
  anything itself. `transport_stats()` hands back a frozen `MediaTransportStats`
  snapshot whose figures are cumulative since the actor built its reader, the
  same span the "after N batch(es)" count covers, so this recipe prints the
  snapshot as it stands and diffs nothing. What stays here is **presentation**:
  the 100-batch cadence, the quantile method, and the wording. The reader logs
  nothing at all — it is a shared utility under
  [`next/utils`](../../../cosmos_curator/next/utils/), and a cadence that suits a
  100-clip batch suits neither a single-object read nor a per-row one, so the
  interval is the caller's decision. This recipe is its only caller today; a
  second one gets the same figures by asking for a snapshot, with no change
  there.

  The snapshot keeps one float per delivered read for the actor's whole life,
  uncapped, and **this stage is the layer that bounds it** — only here is the
  rate that fills the sequence known. So long as the stage feeds a GPU-bound
  engine under backpressure its reads arrive at that engine's row rate, and a
  node added to the run adds a media actor and engine throughput together, so
  run duration is the only axis left and the allocation's time limit caps it:
  measured at the default `media_fetch_workers=4`, a nine-hour actor retained
  under a megabyte. Capping in the reader would be the wrong layer anyway — it
  makes a shared utility choose the summary it exists to leave open.
- **What the recipe's own two hooks cost.** `preprocess` and `postprocess` are
  the only seam this recipe owns inside the processor's stage graph — every other
  stage is Ray's, runs in its own actor, and emits no timing of its own. Each
  wrapped hook reports its cumulative and per-row cost every 2,000 rows, so a
  phase spending its time in request construction or result canonicalization is
  distinguishable from one spending it in the engine. A row that **raised** still
  contributes its cost: a hook that fails slowly is exactly the case worth
  seeing, and counting only the survivors hides it.
- **What each operator spent and what rate it achieved.** After the sink, the
  phase logs Ray Data's own statistics for the executed plan. Verified against
  Ray 2.58.0, that summary carries per-operator remote wall, CPU and UDF time,
  output rows and bytes *per block*, rows per task, tasks per node, and each
  operator's own throughput in rows/s, plus cluster and dataset spill totals and
  a dataset-level rows/s. The per-operator **rows/s** lines are what a bottleneck
  claim from this log may rest on, because a rate is the same quantity at every
  operator. What the summary does *not* carry is any queue depth at all — see
  [the gap that stays open](#limitations).

### Telling whether a run finished

**The committed Lance version is the sole completion signal.** There is no
`_SUCCESS` marker, no completion file, and nothing written after the commit. The
CLI prints one of two messages:

- `robot-action-caption: N caption row(s) canonical at v<VERSION> (E error, T truncated)`
- `robot-action-caption: no caption owed at v<VERSION>`

Here `N`, `VERSION`, `E`, and `T` are the corresponding
`CaptionRunResult` counts and ending version. The run summary carries exactly
`clips_lance_uri`, `source_version`, `ending_version`, `published_rows`,
`error_rows`, `truncated_rows`, `caption_contract_digest`, `shard_count`, and
`shard_index` — see
[`CaptionRunResult`](../../../cosmos_curator/next/recipes/robot_action_caption/pipeline.py).
The digest is the one name that differs between the two surfaces: the
`CaptionRunResult` attribute is `contract_digest`, while `as_summary()` publishes
it under `caption_contract_digest`, matching the Lance column it stamps.
`ending_version` is never null: a run that published
reports its commit, and a run that owed nothing reports the pinned version, which
already includes the widening commit.

The summary also carries `shard_count` and `shard_index`, both null on an
unsharded run, so a campaign's per-job summaries can be attributed to their
slices afterwards. They are run metadata: no table column records which job
captioned a row, because the shard is
[execution scope, not result semantics](#sharding-a-caption-campaign).

### The summary reports the verdict split

`error_rows` and `truncated_rows` are **subsets of `published_rows`**, not
additions to it, and they exist because neither verdict is distinguishable
per-row from the outcome an operator expects:

- A `truncated` row is shaped exactly like a `success` row on disk — non-null
  caption, null error — and satisfies the pending predicate identically. Nothing
  per-row separates a corpus of complete captions from one largely cut off.
- An `error` row is distinguishable per-row, but not per-*run*: without the
  count, `published_rows: 4000000` reads identically whether every row is a
  caption or [every row errored the same
  way](#the-taxonomy-classifies-a-fault-it-cannot-bound-its-extent).

The policy on both counts is deliberate and worth stating as policy: **count and
report, never gate.** No threshold, no config knob, no warning keyed on a share,
and no pre-commit refusal. A run whose every row is an error or a truncation
still commits. A guard would have to choose a threshold, and **no share is
universally wrong** — a corpus of genuinely unreadable media *should* publish
all-error, and the recipe cannot know the operator's context. So the recipe
reports the shape of the wave and leaves the only judgement that needs context to
the person who has it.

`success_rows` is deliberately **not** stored. It is
`published_rows - error_rows - truncated_rows`, and a third stored field could
disagree with its own summands — which is a worse failure than an arithmetic step
at the point of reading, because a count that reads as authoritative and is wrong
is not detectably wrong. Do not add it.

### Cross-checking the split against the table

The summary already reports the split, so the queries below are a **post-hoc
cross-check** rather than the way to obtain it — useful when reading a table
whose run summary is no longer to hand, or when confirming that what committed is
what the run said committed. Checking the table needs the digest, because
"pending" is contract-relative. Take it from the run summary:

```python
import lance

from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.next.recipes.robot_action_caption.table import caption_work_filter

digest = "..."  # caption_contract_digest from the run summary
model_key = "qwen2.5_vl"  # model.variant from the run configuration
model_id = resolve_registered_model(model_key).model_id
ds = lance.dataset("s3://bucket/robot_clips/lance/clips.lance")
print(ds.version)

# Owed under the configured scope.
print(ds.count_rows(filter=caption_work_filter(digest, model_id, selected_only=True)))

# Globally pending under that contract, including unselected rows.
print(ds.count_rows(filter=caption_work_filter(digest, model_id, selected_only=False)))

# Captioned under that contract, split by verdict.
# These two should reproduce the summary's error_rows and truncated_rows for a
# single-wave table; on a table several waves have published into they count the
# whole population under this contract rather than the last wave.
print(ds.count_rows(filter=f"caption_contract_digest = '{digest}' AND caption_status = 'error'"))
print(ds.count_rows(filter=f"caption_contract_digest = '{digest}' AND caption_status = 'truncated'"))
```

Use `caption_work_filter` rather than restating its SQL. It owns the explicit NULL
handling required by SQL three-valued logic, the complete terminal-value policy,
and safe model-id quoting; a hand-written status/digest subset can silently
under-count malformed current rows.

`caption_status IS NULL` on its own answers a different and still useful question:
how many rows *no* run has ever claimed. A non-zero count after a successful run
means rows were appended after the run pinned its version — expected while
`robot-action-split` is still appending.

Every count above is **table-wide**, which is the right scope for a cross-check
and the wrong scope for judging one job of a campaign. A shard's own owed count
is on its plan line. To reproduce it, count per fragment over the fragments that
shard owns rather than over the table:

```python
shard_count, shard_index = 8, 3
owed = sum(
    fragment.count_rows(filter=caption_work_filter(digest, model_id, selected_only=True))
    for fragment in ds.get_fragments()
    if fragment.fragment_id % shard_count == shard_index
)
```

The table-wide count is the sum of that over every index, which is the cheapest
way to confirm a campaign has nothing left owed.

### Resuming after a crash

Re-run the **same config against the same table**. Nothing else is required and
there is no resume flag: the namespace digest resolves to the same directory and
Ray's checkpoint filter drops the completed rows. On a sharded campaign, re-run
the failed shard with the **same** `input.shard.index` — a different index is a
different job with a different namespace, and it would re-infer.

A commit by another leg between the two attempts no longer prevents reuse. The
version is not a namespace input and the checkpoint id names no row location, so
a resume across it reuses everything. Two things still do silently prevent reuse
rather than failing, so check them if a resume re-infers work you expected it to
skip:

- **A compaction between the attempts.** It retires fragment ids, so the staged
  rows can no longer be matched to owed ones and the attempt re-infers its whole
  owed set ([why](#no-recovery-across-compaction)).
- **A changed `input.shard.count`.** That is a different partition and therefore
  a different namespace, so the previous count's progress is not visible.

In both cases the run's own log line names the workspace it resolved, so compare
it against the previous run's.

Do **not** change a digested constant to "retry with a better prompt" mid-recovery.
That is a different contract, so it resolves to a different namespace, re-infers
everything, and — because the old namespace is neither reused nor deleted — leaves
the previous attempt's staged rows behind.

### When publication refuses, and what to delete

Two of the pre-commit refusals name a path in their message, and they call for
**opposite** care. Both leave the staged rows intact and nothing published.

**Conflicting staged rows for one `clip_id`.** Two runs wrote the namespace at
once (see [One writer at a time per
namespace](#one-writer-at-a-time-per-namespace)) and generation is unseeded, so
these are two different answers rather than a retry artifact — there is no correct
one to publish. Every re-run resolves to the same namespace and refuses again, so
the run must be unblocked by hand.

Delete the **whole namespace** — the `workspace.root_uri` the message names —
never `results/` or a single `fragment_id=<id>` partition. The checkpoint beneath
the namespace still records those clips as complete, so a partition-only delete
makes the next run stage **nothing** for them, and publication then refuses from
the *missing-row* branch instead, whose message tells the operator to re-run Phase
A. That advice is correct for a genuine missing row and a dead end here: Phase A
skips those clips because the checkpoint says they are done. Deleting the
namespace discards the completed captions under it, and those clips are
recaptioned. That is the cost, and there is no cheaper remedy.

**Staged rows carrying a foreign contract digest.** The message names the staging
namespace and **deliberately prescribes nothing**. Do not delete on its advice,
because two causes produce it and their remedies are opposites:

- two contracts reached one staging root, in which case the tree holds real
  captions somebody paid for and deleting it is data loss; or
- something outside this recipe wrote into the namespace, in which case deleting
  is the repair.

Rows the namespace digest could not have produced do not identify which. Establish
that first — the runs' own log lines name the namespace each resolved — and only
then decide. A refusal message should carry exactly the certainty the guard has,
and this one knows only that the tree is not this run's.

**Staged rows for clips this attempt does not owe.** This one names no path, and
**nothing should be deleted for it**. Its ordinary cause is a *narrowed* resume:
an attempt that crashed under `input.selected_only: false` staged rows for clips
that a following `true` attempt no longer owes, and scope is [not a namespace
input](#why-input-scope-is-not-a-namespace-input), so the two share a directory.
Re-run with the wider scope and the same staged rows publish. Deleting the
namespace here would discard completed captions to work around a scope setting.

### Reading the staged workspace

A namespace holds `checkpoints/` (Ray's completed row ids) and `results/` (the
staged terminal rows, as Hive-style `fragment_id=<id>/` directories of Parquet).
When this workspace uses GCS through its S3/XML endpoint, driver and worker
filesystem construction follow the endpoint-scoped compatibility policy in
[Ray Data: S3-compatible object storage (GCS)](ray-data.md#s3-compatible-object-storage-gcs).
Useful readings:

- **Two sibling `<namespace digest>` directories** mean two attempts that cannot
  share rows ran against this table — they differ in source URI, contract digest,
  shard, staging protocol version, or staged schema fingerprint. On a sharded
  campaign this is the **expected** shape: one sibling per shard, all live at
  once. The digest is one-way, so the path cannot tell you *which* input differs;
  the runs' own log lines can, since each names its scope and the namespace it
  resolved.
- **`results/fragment_id=…` populated but no new table version** means Phase A got
  some way in and publication never happened. Re-run to finish it.
- **Staged rows are not canonical.** Never read captions from Parquet for
  anything downstream; if they are not in Lance, they are not published.
- **An orphaned namespace after a successful commit** is *not* inert, and this is
  the one case where that changed when the read version left the namespace digest.
  It only exists if cleanup failed, which is logged as a WARNING and never fails
  the run — but a later attempt on the same table, contract and scope resolves to
  that same namespace. It lands one of two ways. Owing nothing, it returns before
  the workspace is opened and never examines the leftover, so the cost is storage
  alone. Owing a row in a fragment that also holds the leftover staged rows,
  publication folds those extras and refuses *before* the commit with every
  staged row intact. Delete the namespace by hand; the published captions are
  canonical either way.
- **A namespace stranded by a contract-generation bump** is inert for a different
  reason: nothing ever resolves to it again. A generation bump — a new caption
  column, or any change to the prompt, the model pin, or a `generation` / `video`
  / `media` field — moves the contract digest, and a new column also moves the
  staged schema fingerprint. Either is a namespace input,
  so the next run lands somewhere new. Cleanup follows only a
  commit *under the namespace it published from*, so the previous generation's
  staged rows are never collected, however successfully the new generation
  finishes. Unlike the cleanup-failure case there is no WARNING, because nothing
  went wrong. Reaching a generation bump with staged rows outstanding is worth
  avoiding for the wasted GPU time, not for any risk it carries: the stale
  namespace cannot be read or corrupted by the new run. Delete it once no attempt
  under the old contract will be resumed.
- **A refused publication orphans storage too — inside the table, not the
  workspace.** Preparation writes one new column file per touched fragment before
  anything commits, and the driver raises on the first refusal any worker reports.
  Fragments not yet started are abandoned; fragments that already finished have
  already written their files, and with no commit there is no manifest that
  references them. Readers never see them, so this is a storage leak rather than a
  correctness problem, and it grows by about one column file per prepared fragment
  on every refused publication. The reclaim path is
  `dataset.cleanup_old_versions()` against `clips.lance`. Run it **deliberately**:
  it removes files no *retained* version references, so it also ends time travel to
  the versions it collects. Note also that a file referenced by no manifest at all —
  exactly these — is skipped until Lance can be sure it is not an in-flight write.
  That dependency-defined safety window is independent of the `older_than`
  argument, so a recent leak is not reclaimed immediately.
  `delete_unverified=True` removes that wait and with it the ability to tell a
  leaked file from a concurrent writer's in-progress one, so use it only when no
  other writer is touching the table — `robot-action-split` appending, or
  `embeddings` and `curate` widening, all look the same to it. A code-level fix
  (tracking prepared files and deleting them on the refusal path) was considered
  and judged not worth it: it would add a cleanup path that only ever runs after
  something has already gone wrong, to reclaim storage a maintenance command
  already reclaims. This note is the intended remedy.

## Limitations

- **Per-operator queued *rows* cannot be read from this run's artifacts, and that
  gap is open.** Verified against Ray 2.58.0: `Dataset.stats()` carries no queue
  depth of any kind, and the progress lines carry it only in **blocks**. A block
  is not a fixed amount of work — fusion, dynamic splitting and batch
  re-bundling all change rows per block along the plan — so two operators' block
  counts are quantities that merely share a name, and their difference means
  nothing. **That invalid comparison is what mis-attributed this very run's
  bottleneck**, crowning the batching stage over the engine that was actually
  the constraint. The only queue figure Ray publishes at all is the
  `data_operator_queued_blocks` Prometheus gauge, fed from live executor state
  ([`ray/data/_internal/stats.py`](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/data/_internal/stats.py)),
  which needs a metrics backend and is still in blocks. So no driver-log-only
  change closes this. Until it is closed, rest a starvation claim on a
  **per-consumer** figure whose numerator and denominator belong to the same
  operator — queued blocks divided by that operator's own replica count — or on
  the per-operator rows/s the post-run summary does provide.
- **The two forwarding stages reserve for a payload that no longer exists.** Both
  reservation roles are sized from the pre-resize decode ceiling, so on the
  measured run the `ChatTemplate` and `Tokenize` workers declared 13.55 GiB each
  where 3.69 GiB covered the column they actually carried — 3.67× over, and
  roughly 276 GiB of node memory reserved for payload that never arrived,
  capping those actors while 91 of 192 CPUs sat idle. Left as-is on purpose: it cost that run
  no throughput, because the engine was the constraint and every upstream stage
  was already throttled behind it, and the error is in the safe direction —
  over-declaring strands budget where under-declaring OOM-kills a node, since
  `memory` is a placement hint Ray never enforces. A correct second figure is
  also not derivable while `video.max_pixels` is unset, and a hand-set one would
  be one corpus's measurement applied to every later corpus. [The condition that
  makes it worth fixing, and how to derive it
  then](#both-roles-are-sized-from-the-pre-resize-ceiling-and-the-forwarding-pair-over-reserves).
- **Selection narrows rows, not necessarily fragments.** The selection predicate
  is pushed into Lance before Ray sees a row, but a sparse selected population
  can still touch many source fragments. One selected row in each of 100
  fragments still creates 100 source read tasks; the filtered rows are then
  repartitioned before media fetch. This deliberately does not rewrite the
  source: Lance fragment geometry remains the producer's durable storage
  concern, while bounded Ray blocks own downstream execution geometry.
- **One target caption contract per attempt.** The five columns are a single
  owned group, so one attempt cannot publish two models or prompt versions.
  Latest contract wins within its configured scope. Rows outside selected-only
  scope are intentionally untouched and can retain an older valid contract
  until they are selected or a broader run processes them; keeping two contracts
  side by side as durable named outputs would require separate column groups.
- **A digested change recaptions every row in the configured scope.** Because the pending
  predicate is digest-aware, moving the model pin, the prompt, or
  `max_clip_bytes` makes every already-captioned row pending again. Default mode
  processes the selected subset; unselected stale rows remain untouched until
  they are selected or the filter is disabled. There is no arbitrary slice
  filter, and no cost estimate is printed before the run commits to it — the plan
  line's owed row count is the only warning.
- **The digest's `model` block is a claim, not a measurement.** It stamps the
  model id, revision, and precision that `resolve_registered_model`
  ([`models/registry.py`](../../../cosmos_curator/models/registry.py)) reads out
  of the shipped registry entries in
  [`configs/all_models.json`](../../../cosmos_curator/configs/all_models.json).
  The same module's
  directory validator checks for `config.json` and a recognized weight
  file, but nothing hashes or parses those bytes. The one change the digest
  therefore cannot see is **weights replaced under a stable revision label**:
  the captions move, the digest does not, so the pending predicate keeps skipping
  the very rows the new weights would caption differently, and two populations
  of captions end up in one table with nothing on a row to tell them apart. This
  is accepted rather than closed — hashing a multi-gigabyte staged directory on
  every worker at every startup buys nothing that an operator convention does
  not. The convention: **treat a staged model directory as immutable, and bump
  the registry revision whenever its contents change.**
- **No recovery across compaction.** Nothing refuses it. Staged rows are keyed
  by the row's Lance address, so a compaction — which rewrites fragments into
  fresh ids — leaves the staged partitions unmatchable and the attempt re-infers
  its whole owed set.
  Progress is lost rather than corrupted — an unmatchable staged row cannot be
  published — but unlike the earlier version-digesting design the stale namespace
  is the *same* namespace, so the dead partitions sit beside the live ones until
  it is deleted. The rule is operational: do not compact or otherwise rewrite the
  source's fragment geometry while a workspace holds progress. Transparent
  recovery across that will not be built. A row moving *within* its fragment,
  which is what a metadata-only commit or a version bump does, costs nothing.
- **The namespace digest is not reversible.** With no manifest beside it, two
  sibling namespaces cannot be told apart from the filesystem alone; the runs'
  log lines are the only record of which inputs produced which directory. That is
  the accepted cost of having exactly one source of truth for compatibility.
- **The namespace carries no table identity.** `resolve_workspace` digests the
  canonical source URI, the contract digest, the shard, the staging protocol
  version, and the staged schema fingerprint — and the URI is a *path*, not an
  identity. Two different tables that occupy that path at different times (a table
  dropped and recreated, or one path reused across environments) consequently
  resolve to one namespace whenever their contract digests coincide. Removing the
  read version from the digest **widened** this exposure, and that is worth stating
  plainly rather than leaving for a reader to infer: the recreated table previously
  had to reach the same version number as well, and now does not.

  It is accepted on the same terms as before, because the collision is only
  reachable while a namespace is live — cleanup follows the commit — and what a
  colliding attempt can do is bounded by the checks that already exist. Staged
  rows carry `clip_id`, and publication refuses a staged set that does not match
  what the table owes *before* the commit, with every staged row intact. So the
  failure is a refusal an operator clears, not a caption published onto the wrong
  row. Closing it would mean digesting a per-table identity, and Lance exposes
  none that is durable: there is no dataset uuid on the manifest, and the first
  version's timestamp is not a substitute because version cleanup can delete
  version 1 outright. The discipline is therefore to give a recreated table a
  fresh path, or to point `output.staging_root_uri` somewhere the previous
  table's namespaces do not live.
- **A campaign's completeness is not asserted anywhere.** No registry records
  which shard indices ran, so launching indices 0, 1, 2 and 4 of 5 leaves shard
  3's work owed with nothing reporting it — each launched job succeeds and
  reports only its own slice. Completeness is *observable*, by re-launching every
  index and reading the owed counts, but nothing checks it. Campaign bookkeeping
  belongs where jobs are launched, for the same reason mutual exclusion does.
- **Shard balance is by fragment count, not by cost.** Modulo distributes
  fragment ids evenly and knows nothing about how many rows each fragment still
  owes, so a corpus whose remaining work is concentrated in a few fragments
  divides unevenly. Cost-balancing would need a coordinator or a manifest, which
  is the state this design deliberately lacks; where skew matters, raise `count`
  so each shard's worst-case share is smaller.
- **One writer at a time per namespace, documented and not enforced.** The path
  pins compatibility, not exclusivity. Note this is *per namespace*: concurrent
  shards are safe precisely because they resolve to different namespaces, so the
  limitation is about two jobs in **one** directory. Two runs resolving to one namespace
  concurrently generate different captions for the same clip under unseeded
  sampling, both append them into the same partition, and publication refuses the
  differing pair — permanently, because cleanup follows only a commit and every
  namespace input is stable across re-runs, so the leg stays blocked until an
  operator [deletes the whole
  namespace](#when-publication-refuses-and-what-to-delete) and pays for those
  clips again. Nothing detects it: no lease, lock file, claim record, PID check, or
  manifest, and none is planned. Mutual exclusion belongs where runs are launched
  (a Slurm job dependency, an NVCF concurrency limit, a monitoring loop that waits
  for its predecessor).

  This limitation and the previous one are **the same trade seen twice**. Deriving
  identity rather than claiming it is what removes per-row provenance validation
  and its failure modes, and the price is exactly the two guarantees a claim would
  have carried: *who else is writing this namespace*, and *whether the thing at
  this URI is still the same thing*. Both are pushed onto operator discipline,
  which is why both are limitations here rather than runtime checks.
- **There is no publish-only entry point.** `run_config` is the only way in, so a
  re-run that owes nothing but a republish still pays a full startup — model
  resolution, the staged-weight check, Ray init, and an engine pool that boots
  before the checkpoint filter can tell it there is no work. What the two-phase
  split buys is that the *inference* is not repeated, which is the expensive half;
  the cold start is the residual. Closing it would mean a publish-only entry point,
  which is a coherent possibility rather than a plan — so until one exists, treat a
  republish as a fresh run's startup cost.
- **The clip cap does not bound aggregate host memory.** The documented
  341.3 GiB base64 figure assumes `inference_batch_size: 16`,
  `max_concurrent_batches: 8`, and eight replicas on one node. Those execution
  parameters are intentionally tunable and no cross-field validator recomputes
  a node-memory ceiling; this recipe also disables Ray's high-memory detector.
  Size the product against the target node before raising either multiplier or
  packing more replicas.
- **A replica width is never checked for placement.** The configured
  `tensor_parallel_size * pipeline_parallel_size` reaches both the vLLM
  processor and the pool-floor derivation, so the GPU accounting is right.
  Above width one, Ray's vLLM stage reserves one GPU bundle per tensor/pipeline
  worker through a soft `PACK` placement group rather than on the Ray Data
  actor itself. The bundles can span nodes when one node cannot satisfy the
  width, but neither this recipe nor vLLM checks that the selected model and
  topology are viable before actors start. An unsatisfiable group remains
  `PENDING` instead of failing. Only TP=1, PP=1 has been exercised here.
  Engine-side `data_parallel_size` remains rejected because Ray Data already
  owns horizontal replication through `concurrency`.
- **Thinking suppression depends on the selected model's chat template.**
  Every request passes `enable_thinking: false` to the model-owned template and
  records that policy in the caption contract digest. Qwen3 templates honor the
  switch; templates without it accept and ignore the extra variable. The recipe
  deliberately does not parse `<think>` delimiters from generated text, so a
  checkpoint that ignores the switch can still publish reasoning traces.
- **No seeded generation, so no bitwise reproducibility.** Two runs over the same
  clip produce different captions. That is what makes recovery a correctness
  property rather than only a cost saving, and it is why conflicting staged
  duplicates fail loudly.
- **Truncation is inferred, not reported.** A generation that ends naturally on
  *either* bound — exactly at the output cap, or exactly at the context length net
  of the prompt — is labelled `truncated`. The conservatism is the same at both:
  the classifier cannot see why a generation stopped, only where it stopped.

  Revisit if a later engine version surfaces `finish_reason`, but expect **less**
  from that than it appears to offer. vLLM sets one `FinishReason.LENGTH` for both
  bounds, from a **single or-ed condition** (`num_tokens >= max_model_len or
  num_output_tokens >= max_tokens`) in its V1 stop check, and writes no
  `stop_reason` on that branch — both causes serialize to the identical `length`.
  The conflation is structural rather than an oversight: when a caller leaves
  `max_tokens` unset, vLLM sets it to the context length minus the prompt, so the
  two bounds are arithmetically the same number and attribution would be
  meaningless. They diverge only when a caller pins `max_tokens`, as this recipe
  does.

  So a surfaced `finish_reason` would decide **whether** a generation was cut off,
  never **which** bound cut it. Per-bound attribution would still need the token
  counts. On that trigger `caption_status` should **shrink** — a reason check plus
  the retained per-bound arithmetic — and must not be deleted: deleting it would
  silently drop context-bound attribution. Re-verify this engine contract after
  a vLLM dependency bump.
  Separately, a prompt that alone exceeds the context length is rejected at
  *admission* with an exception, so it never becomes a row and carries no reason
  at all.
- **Staged file geometry is best-effort.** `min_rows_per_file` and
  `max_rows_per_file` are hints in Ray's sink, so a pathological block
  distribution can still produce many small shards, and Phase B pays the listing
  cost.
- **The one measured run was fetch-bound, and badly.** On 4×8 H100 against a
  131,602-row table (17 fragments, 16 IO slots per node, serial per-actor
  fetching), 32 engine replicas delivered 7.1 rows/s while their own batch
  latency of 19.5–27.8 s per 32 rows put their combined capacity near 43 rows/s.
  Aggregate GPU utilization was about 17%, 10 of the 32 replicas never received a
  single request, and the scheduler's idle-GPU reaper cancelled the run. Treat
  `inference_concurrency` as *unjustified* until the fetch side is measured on the
  corpus at hand: the engine pool's initial size is derived from the GPU count, and
  Ray Data will not scale an actor pool below that floor, so any excess replica
  holds a GPU for the whole run. The startup log now states that floor, the GPUs
  it pins, and the fetch ceiling it has to be fed by, so this shape is visible
  before the first GPU-hour is spent — see
  [What the startup log says about capacity](#what-the-startup-log-says-about-capacity).
- **Source scanning remains fragment-shaped, but media fetch does not.** The
  read starts with one source task per pending Lance fragment, preserving pinned
  snapshot and publication geometry. Before media I/O, non-strict repartition
  splits each lightweight source block independently at
  `execution.lance_read_batch_size`; it does not coalesce adjacent fragments.
  The media actor ceiling is therefore the exact
  `sum(ceil(pending_rows_in_fragment / R))` fetch-block count rather than the
  pending fragment count. With the default `R=256`, the measured 131,602-row,
  17-fragment shape produces 527 fetch blocks, enough to reach the node
  placement cap without rewriting `clips.lance`. Fragment count and skew can
  still affect source-scan startup and tail, but no longer trap one fragment's
  media work on one actor.
- **The media pool is also capped at the actors the node budgets can place, and
  that cap is taken from a cluster that may still be assembling.** One
  concurrent read reserves one `curator_io` slot, so a node's slot budget divided
  by the fetch width is how many media actors it can host at once — width 16
  against 16 slots places exactly one per node. Left to autoscale, the pool
  requests actors toward the block count regardless, and every actor past that
  number stays *pending* for the whole run. Those phantom actors are not merely
  harmless: Ray Data's occupancy divides work in flight across every actor the
  pool has **requested**, so a fetch stage whose real actors are saturated
  reports slack it does not have — and on a fetch-bound run that is precisely the
  number an operator would use to rule this stage out. The pool is therefore
  capped at the placeable count, which makes the pool's own arithmetic correct
  rather than annotating a wrong figure. The cost is that the cap comes from the
  same driver-start snapshot as the engine floor, so a node registering later
  cannot be used by this pool; that is the existing snapshot exposure described
  below, not a new one, and it buys an occupancy whose denominator counts actors
  that exist. It also makes the startup actor figure narrower than it looks: it
  counts what the per-node budgets hold **simultaneously**, which is now also
  the pool's upper bound, and never a prediction of throughput.
- **Which ceiling binds decides what `media_fetch_workers` does.** One `curator_io`
  slot means one concurrent read, so a media actor reserves as many slots as it
  runs fetch threads, and concurrent reads are
  `min(fetch work blocks, placeable media actors) × width`. The two regimes
  behave differently, and reading only one of them is misleading:
  - *Block-bound* — the actor count is capped by the derived fetch-block count,
    and slots are spare. A smaller `execution.lance_read_batch_size` creates
    more execution blocks without changing Lance; a wider actor can also raise
    reads per block, but consumes proportionally more node slots.
  - *Slot-bound* — the budget is the binding constraint, and the width merely
    *partitions* it. A node advertising `DEFAULT_CURATOR_IO_SLOTS_PER_NODE = 16`
    ([`environment.py`](../../../cosmos_curator/core/utils/environment.py))
    sustains 16 concurrent fetches whether that is 16 single-threaded actors or 4
    four-threaded ones; what the width buys there is the same capacity for a
    quarter of the processes, storage clients, and CPU reservations. A width
    raised above the default into this regime is called out at `WARNING` on the
    driver, because the schedulability guard passes at exactly the width that
    gained nothing.

  Past the slot-bound point the only remedy is more capacity, which is fixed at
  node registration by `COSMOS_CURATOR_RAY_IO_SLOTS_PER_NODE` or the
  `--ray-io-slots-per-node` launcher option, and which no recipe config reaches.
  A width whose combined CPU and `curator_io` request fits no live node is
  rejected on the driver, because Ray leaves an unsatisfiable resource request
  pending forever rather than failing it. Passing that check only proves *one*
  node can host an actor, so the phase also logs how many nodes can — a width
  that fits on one node of a heterogeneous cluster confines the whole stage to
  it. Which regime a given run is in, and whether its fetch ceiling can feed the
  engine floor, is reported at startup; see
  [What the startup log says about capacity](#what-the-startup-log-says-about-capacity).
- **The engine floor is taken from a cluster that may still be assembling.** The
  Slurm supervisor starts the driver as soon as the first accelerator node
  registers, and the replica floor is derived from the GPUs visible at that
  moment and never revisited, so a job whose remaining nodes join a few seconds
  later runs to completion on a fraction of its allocation. The driver cannot
  detect this — the node count it would have to compare against is not reachable
  from where it runs — so the floor line states the exposure instead; see [What
  the startup log says about
  capacity](#what-the-startup-log-says-about-capacity). Removing it needs a
  launcher-side change (waiting for the expected lanes, or publishing the
  expected count to the driver), which is deliberately not attempted here.
- **The Ray Data LLM processor API is beta**, and the Lance behaviour this design
  leans on is version-sensitive. Take resolved dependency versions from the
  project lock files rather than from this document, and re-verify the
  `LanceOperation.Update`, `add_columns`, and `update_columns` claims after a
  Ray or Lance bump.
- **The video policy's engine-side path is verified only as far as Ray.**
  `media_io_kwargs["video"]` carries `fps`, `min_frames`, and `max_frames`, while
  the processor mapping carries a complete `min_pixels`/`max_pixels`
  pair. The transform copies the media mapping into the preparation stage's
  `ModelConfig`; end-to-end decoded-frame observations still require a real
  inference run. vLLM remains the authority if a selected artifact rejects those
  settings.

## Ownership Rules

Four ownership rules, all of which exist because a split owner fails silently
rather than loudly:

- **`__checkpoint_id` is spelled once**, in `datasource.py`, beside the scan that
  produces its values. The producing scan and Ray's checkpoint filter are two
  halves of one mechanism; if they disagree the run filters nothing and records
  nothing while appearing to work.
- **The `inference/<digest>` layout is computed once**, by a module-private helper
  in `workspace.py`. Nothing outside that module can construct the path, so the
  namespace a run stages into and the checkpoint tree written under it cannot
  drift apart.
- **Caption values have one author**, `canonicalize`. The media stage names faults
  and the engine hooks move keys; neither ever receives the digest or the model id,
  so a published row cannot be stamped by two code paths that disagree about what
  produced it.
- **A digested value is owned by `contracts.py`** and imported by its consumer,
  never restated there — see [Where a constant lives is decided by the
  digest](#where-a-constant-lives-is-decided-by-the-digest). A local copy leaves
  the digest stamping something other than what the run used, and editing the copy
  then changes every caption while the digest stays byte-identical.
