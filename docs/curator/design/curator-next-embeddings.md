# Curator Next — Embeddings (`cosmos_curator/next/embeddings`)

> **One table — read this first.** Embeddings persist **directly onto the
> existing `clips.lance`** as additive, nullable per-modality **column groups**
> (`embedding_<modality>_*`). There is **no** separate embedding table: a run
> neither creates nor reads any `clips.lance__emb_*` side table. The authoritative
> schema is the `*_GROUP_SCHEMA` definitions and the `EMBEDDING_COLUMN_GROUPS`
> registry in
> [`cosmos_curator/next/embeddings/schemas.py`](../../../cosmos_curator/next/embeddings/schemas.py);
> schema evolution and state validation are owned by
> [`columns.py`](../../../cosmos_curator/next/recipes/embeddings/columns.py)
> and the distributed column write, with its single commit, by
> [`fill.py`](../../../cosmos_curator/next/recipes/embeddings/fill.py).
> Each modality **adds** its group's columns to `clips.lance` in one metadata
> commit and **fills** them with a column-local atomic `LanceOperation.Update`; a
> group is absent from the table until its modality runs, so a consumer that wants
> a modality checks the **column's presence** (not merely NULL). `robot_action_split`
> keeps appending its narrow base rows, which read every present `embedding_*`
> field as NULL until the next embedding run fills it. Any pre-existing `__emb_*`
> side table from an earlier architecture is neither read nor written and may be
> deleted manually.

**Embeddings** is a generic Curator-Next capability: it turns a structured input
dataset — one keyed row per item — into one independent, key-aligned **feature
column group per modality**, added directly onto the source table with native Ray
Data. The package owns the modality abstraction (per-modality column-group schema,
applicability filter, and embedder), Ray Data execution, and the direct-column
persistence contract; those vectors are the raw material a later consumer compares
(clustering, de-duplication, balanced sampling, retrieval). This milestone
delivers **only** the embedding step.

```text
                      ┌─▶ modality A ─▶ + embedding_a_* columns ─┐
one keyed source ─────┼─▶ modality B ─▶ + embedding_b_* columns ─┼─▶ same table,
table (clips.lance)   └─▶ modality … ─▶ + embedding_…_* columns ─┘   one row per item
```

**Current implementation (egocentric robotics).** The shipped realization embeds
the clips `robot_action_split` produces into three modalities — text, image, and
action — plus a fitted action-PCA (Principal Component Analysis) basis, all
written back onto the one `clips.lance`:

```text
clips.lance ─┬─▶ [BGE text embedder]     ─▶ + embedding_text_*   columns
  (one row    ├─▶ [DINOv2 image embedder] ─▶ + embedding_image_*  columns
   per clip)  └─▶ [wrist-motion + PCA]    ─▶ + embedding_action_* columns
                                            (+ clips.lance__action_pca/<fingerprint>.npz)
```

See the [Glossary](#16-glossary) for PCA, BGE, DINOv2, ACT2, and the other terms
used below.

- Code: [`cosmos_curator/next/embeddings/`](../../../cosmos_curator/next/embeddings) ·
recipe [`cosmos_curator/next/recipes/embeddings/`](../../../cosmos_curator/next/recipes/embeddings)

---



## 1. Why embeddings at all

Embeddings turn items into vectors so a downstream consumer can measure a
**distance** between them — clustering, de-duplication, balanced sampling, and
retrieval all reduce to that distance, and a distance needs vectors. **In the
current implementation** the items are egocentric-manipulation clips: a raw
capture is heavily redundant (the same task is recorded thousands of times, and
naive sampling would over-represent whatever was captured most), so the
downstream goal is a **balanced, de-duplicated** training set.

Two clips can be "the same" along three independent axes, so one vector is not
enough:


| Axis       | Question it answers                       | Why it is independent                                                                                                                              |
| ---------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Text**   | *What task is this?*                      | A paraphrased instruction ("pick up the cup" vs "grasp cup") must land near itself; exact string grouping would treat synonyms as different tasks. |
| **Image**  | *What does the scene / object look like?* | Two clips with identical task text can be filmed in different places with different objects.                                                       |
| **Action** | *What did the hands actually do?*         | Two clips can share task **and** look alike yet be different motions — different hand, speed, or trajectory shape.                                 |


A later curation leg concatenates the three into one fused vector with fixed
weights, so the **widths and normalization conventions below are an interface**,
not tuning: changing one invalidates every stored vector.

The rest of this section explains *why three and not one*: what each modality
captures and ignores (§1.1), how they answer orthogonal questions (§1.2), how
they correlate (§1.3), worked examples where one vector alone is wrong (§1.4),
how downstream pipelines consume them (§1.5), and the fusion flow implemented by
the curation leg, intentionally not by this embeddings package (§1.6).

### 1.1 The three modalities in detail

Each modality is a deliberately **narrow** view of a clip: it captures one kind
of similarity and *intentionally ignores* the others, so the three are
complementary rather than redundant.

**Text —** `embedding_text_*` **— "What task is this?"**


|                |                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Consumes**   | `task_name` + `subtask_name` strings only (no media is read)                                                                   |
| **Captures**   | task semantics, subtask semantics, instruction meaning, language similarity, paraphrases (`"pick up the cup"` ≈ `"grasp cup"`) |
| **Ignores**    | visual appearance, robot motion, environment, execution style                                                                  |
| **Similarity** | semantic / linguistic — cosine on unit vectors                                                                                 |
| **Benefits**   | task-level grouping, balancing across tasks/subtasks, semantic retrieval ("find all pouring clips")                            |


**Image —** `embedding_image_*` **— "What does it look like?"**


|                |                                                                                                          |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| **Consumes**   | the first displayable frame of `clip_uri`                                                                |
| **Captures**   | scene appearance, object appearance, camera viewpoint, lighting, environment, visual context             |
| **Ignores**    | task semantics, motion over time, action trajectories                                                    |
| **Similarity** | visual — cosine on unit vectors                                                                          |
| **Benefits**   | appearance de-duplication, scene / domain balancing, visual retrieval ("find all clips in this kitchen") |


**Action —** `embedding_action_*` **— "How was it performed?"**


|                |                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| **Consumes**   | the shared ACT2 artifact (dual-wrist pose + ego-camera pose)                                            |
| **Captures**   | hand motion, wrist trajectories, manipulation style, motion dynamics, geometric trajectory similarity   |
| **Ignores**    | object appearance, scene, language                                                                      |
| **Similarity** | geometric / motion — Euclidean in the fitted PCA space                                                  |
| **Benefits**   | motion de-duplication, execution-style balancing, motion retrieval ("find all clips with this gesture") |




### 1.2 How the three complement each other

Each modality answers a different question about the **same** clip; together they
describe *what is happening*, *what it looks like*, and *how it was done*:

```text
Robot clip
    │
    ├── What is happening?    ──▶ Text   (task / subtask semantics)
    │
    ├── What does it look like? ──▶ Image  (scene / object appearance)
    │
    └── How was it performed?  ──▶ Action (wrist motion / dynamics)
```

Because the questions are orthogonal, the three vectors are produced
independently from three different inputs and only recombined downstream:

```text
                     One clip
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
     task text      first frame      wrist motion
        │               │                │
        ▼               ▼                ▼
  Text Embedding   Image Embedding  Action Embedding
   (2 × 384 unit)   (384 unit)       (97 raw PCA)
        │               │                │
        └───────────────┼────────────────┘
                        ▼
           Future multimodal representation
           (fused distance — see §1.6, not built here)
```



### 1.3 Correlations: when modalities agree and disagree

Two clips can match on one axis and differ on another; that independence is
exactly why one vector cannot stand in for the other two. Each row below is a
real pattern in egocentric manipulation data:


| Two clips have…                                | Text      | Image     | Action    | One-modality failure                                        |
| ---------------------------------------------- | --------- | --------- | --------- | ----------------------------------------------------------- |
| identical task, different objects              | same      | different | ~same     | text/action alone **wrongly merge**; image keeps them apart |
| same scene, different task                     | different | ~same     | task-blind | image alone **wrongly merges** two unrelated tasks; only text separates them |
| same motion, different task wording            | different | ~same     | ~same     | text alone **wrongly splits** near-duplicates               |
| same task **and** look, different manipulation | same      | ~same     | different | text+image alone **wrongly merge** distinct executions      |
| different wording, nearly identical action     | different | varies    | ~same     | text alone **wrongly splits**; action recovers the match    |
| different appearance, identical task           | same      | different | ~same     | image alone **wrongly splits** the same task                |


"~same" means close under that modality's distance; "different" means far.

**Read every Action cell as a claim about motion only.** Where a row's Action entry says
"~same", it is because the *gesture* is similar, never because the task label matches, and
"task-blind" marks a row where the two clips differ in task and the action vector simply
cannot see it. This is measured, not cautionary: on Mecka clips with
labels the action block scores a task-separation ratio of 0.990 and a same-task-closer AUC
of 0.543 — different-task pairs sit no farther apart than same-task pairs. The same run
scored AUC 0.853 for text and 0.879 for image over the same rows, so the apparatus can
find task structure where it exists. Gross dual-wrist kinematics do not distinguish
woodworking from `cleaning_shoes`: both are reach, grasp, and manipulate under a
head-mounted view. Action's value in this table is therefore entirely in the rows where
*execution* is the discriminator, and no consumer may use it to answer a task question.

### 1.4 Worked examples: why one embedding is not enough

Each example lists the three similarities and the behaviour a curation leg should
have — and how it would misbehave on a single modality.

1. **Same task, different object** — `"Pick up the mug"` over a white mug vs a
  blue mug. *Text*: same. *Image*: different. *Action*: nearly identical.
   → They are genuinely different training samples (object diversity); **image**
   keeps them apart. On text or action alone they collapse to one, discarding
   object variety.
2. **Same object / scene, different task** — same kitchen and coffee machine;
  clip A `"press the button"`, clip B `"open the drawer"`. *Text*: different.
   *Image*: ~same. *Action*: task-blind — pressing and pulling are both short
   single-arm reaches, and the measured task AUC of 0.543 says the vector does
   not separate them. → Two distinct tasks; **text** separates them, and text
   alone does. On image alone they merge, hiding a whole task; on action alone
   they also merge, which is why action is never the modality asked here.
3. **Same task, different execution** — `"Pour water"` done slowly two-handed vs
  quickly one-handed. *Text*: same. *Image*: ~same. *Action*: different. → Both
   are useful but distinct executions; **action** keeps the manipulation variety.
   On text+image alone they merge, biasing the set toward one motion style.
4. **Same execution, different language** — `"Pick up bottle"` vs `"Grab
  bottle"`, same gesture. *Text*: different strings but close after embedding;
   *Image*: ~same; *Action*: nearly identical. → Near-duplicates; **action** (and
   paraphrase-robust text) merge them. Exact-string grouping would wrongly split.
5. **Same motion, different scene** — the same reach-and-grasp filmed in two
  rooms. *Text*: ~same. *Image*: different. *Action*: nearly identical. → Scene
   diversity worth keeping; **image** separates them. On action alone they
   collapse, discarding environment variety.

The recurring lesson: **de-duplicating or balancing on any single modality either
merges clips that differ in an unseen axis or splits clips that are actually the
same** — only the combination is safe.

### 1.5 How downstream pipelines consume the embeddings

The embeddings are the raw material for a later curation leg (clustering,
de-duplication, balanced sampling, retrieval). The storage shape is chosen to
serve that consumer:

- **Co-located, per-modality column groups** — every modality's vectors live in
its own `embedding_<modality>_*` columns on the **same** `clips.lance` row (§3),
not in a separate table. A consumer reads only the modality columns it needs, and
a new modality is added later by adding its column group, without rewriting the
others.
- **Regenerated independently** — each modality re-runs on its own (a model swap
or a descriptor bump replaces one group via `--reset-group` plus a refill; the
other groups' columns are untouched).
- **Used singly or combined** — a pipeline may cluster on one modality (e.g.
appearance-only dedup) or fuse several (§1.6) for a task+look+motion distance.
- **No join to realign the modalities** — because all three groups sit on one
row, a clip's text, image, and action vectors are already aligned; there is no
`clip_id` inner-join step between modalities. A consumer that wants a modality
must test **column presence** first: a group's columns exist on the table only
after that modality has run (§3), so a modality that never ran has *no* such
column at all — distinct from a present-but-NULL row (a not-yet-embedded or
could-not-embed clip, §8). `clip_id` remains the stable per-(span, view)
identity (§2.1) for joining `clips.lance` against *other* keyed tables
downstream.



### 1.6 Conceptual multimodal fusion (intended, not implemented here)

The three vectors are designed to be fused into a single distance so that
"similar" means similar in task **and** look **and** motion. The intended flow:

```text
   Text        Image       Action
 (2×384 unit) (384 unit)  (97 raw PCA)
     │            │            │
     └────────────┼────────────┘
                  ▼
         Curation fusion  (per-modality normalize + weighted concat)
                  ▼
             Similarity   (one distance over the fused vector)
                  ▼
             Clustering
                  ▼
           De-duplication
                  ▼
          Balanced dataset
```

Fusion must respect the **scale contract** (§6): text/image are unit-norm
(cosine) while action is raw PCA coordinates (Euclidean), so a fusion step
normalizes each modality into a common space before weighting — which is why the
widths and normalizations here are a downstream **interface**, not free
parameters.

Two details of the sketch above are refined by the leg that actually consumes it
(`curator-next-curation.md`), and the difference matters to anyone reading this
section as a specification:

- **The fused distance feeds clustering and de-duplication, not balancing.** The
  `Balanced dataset` box is downstream of the fused vector but is not computed
  *from* it. Curation balances on the canonical task label together with a
  subtask cluster cell derived from `embedding_text_subtask`, not on raw
  `(task_name, subtask_name)` strings, deliberately keeping the label geometry
  separate from the fused locality geometry, because a locality cluster mixes
  many tasks and a task spans many clusters.
- **"Similar in task **and** look **and** motion" is carried by different blocks
  than the symmetry suggests.** The task part comes from text; the motion part
  comes from action; action supplies **no** task component at all (measured task
  AUC 0.543, §1.3). Fusing the three yields a redundancy distance, which is what
  clustering and dedup need — not a task-aware distance.

> **This milestone produces only the modality-specific embeddings.** It does
> **not** implement fusion, clustering, de-duplication, balanced sampling, or any
> curation. Those are the next leg; this document ends at the three embedding
> column groups plus the PCA basis.

---



## 2. Input formats



### 2.1 Where the source data comes from (field provenance)

The embed leg consumes the source clips Lance table that already exists — one
keyed row per item. In the current implementation that table is the clips table
`robot_action_split` wrote (one row per `(span, view)` clip), and the embed leg
never goes back to the raw dataset to reconstruct task/subtask text or re-cut
media (see §2.5). It also **writes back to this same table** (§3), so
`clips.lance` is at once the read source and the write target.

[`EMBED_SOURCE_ROW`](../../../cosmos_curator/next/embeddings/schemas.py) declares
the base columns the leg consumes; each modality's `source_columns` is a subset of
it, and a worker's fragment scan projects **only that subset** — never the whole
row. A renamed or retyped base column therefore surfaces as a scan failure in the
first worker that touches it, not as a silent wrong answer: the projection names
the columns explicitly and Lance rejects an unknown one.


| Column            | Type                | Feeds          | Where it originates (set by `robot_action_split`)                                                      |
| ----------------- | ------------------- | -------------- | ------------------------------------------------------------------------------------------------------ |
| `clip_id`         | `string` (non-null) | all legs (logical id) | SHA-256 digest of `(span_group_id[, view_name], video_bitrate)` — a hash, **not** a readable composite |
| `task_name`       | `string` (non-null) | text           | `meta/tasks.parquet` label for the span's `task_index`                                                 |
| `subtask_name`    | `string` (non-null) | text           | `meta/subtasks.parquet` label for the span's `subtask_index` (dominant signal)                         |
| `clip_uri`        | `large_string` (declared nullable; non-null in practice) | image    | `…/video/<view_name>/<clip_id>.mp4` — the **generated per-view clip**, not the original recording      |
| `action_data_uri` | `large_string` (declared nullable; non-null in practice) | action   | `…/action/<action_id>.bin` (legacy `.pickle` may exist from debug runs but the embed leg rejects it); `action_id` derives from the **span**, not the view         |
| `source_dataset`  | `string` (non-null) | — (declared base contract; no worker projects it — see below) | `robot_action_split` run config `input.source_dataset` (one dataset per extract run) |


`clip_id` is the stable per-clip identity carried on every base row, and it is
also the **join key** that routes a computed vector back to its `clips.lance` row
(§2.4, §3). `EMBED_SOURCE_ROW` declares four columns non-null (`clip_id`,
`task_name`, `subtask_name`, `source_dataset`) and the two URI columns nullable;
`robot_action_split` writes `clips.lance` via `CLIP_SCHEMA`, which is
**successes-only** — a row is committed to `clips.lance` only once its clip and
(when applicable) action artifact have actually been written, so `clip_uri` and
`action_data_uri` are non-null on every row a fresh `robot_action_split` run
produces. A failed clip never becomes a `clips.lance` row at all; it is recorded
instead in that run's `errors.json` alongside the media root, outside the Lance
table entirely. `EMBED_SOURCE_ROW` still *declares* the two URI columns nullable —
looser than the producer's real guarantee — so the embed leg keeps working
unchanged against an older table written before this schema tightened, and its
per-modality applicability predicates (§8), which test `IS NOT NULL`, stay
correct as defensive filters rather than becoming dead code: on a table produced
under `CLIP_SCHEMA` they simply always evaluate true. Extra columns on
`clips.lance` — including the `embedding_*` groups the recipe itself adds — are
irrelevant to a worker's scan, so widening the table never disturbs the read
path.

**One row = one** `(span, view)` **clip.** A **span** is one contiguous
`subtask_index` run of a single episode; its `span_group_id` is a hash of
`(source, episode, subtask_index, frame_start)` and is identical for every
camera. A multi-camera source emits one **view** — one clip — per camera.
`clip_id` folds `view_name` into its digest **only** when the source is
multi-view (`source_is_multiview`), so single-view datasets get exactly one clip
per span.

**Why several clips can share one** `action_data_uri`**.** The action artifact is
named from `action_id = f(span_group_id, action_format, source_dataset)`, which
does **not** include `view_name`. Wrist / hand / camera geometry is a property of
the span, not of any one camera, so **one span → one action artifact →
potentially several view clips**. The action leg relies on this: it computes each
span's descriptor once and fans it out to every view sharing the artifact (§5).

**No worker projects** `source_dataset`**.** It is a non-null column of
`EMBED_SOURCE_ROW` (the declared base contract, present on every `clips.lance`
row), but no modality's `source_columns` includes it, so no worker's fragment
scan reads it. Action applicability is
**Mecka-only and purely structural**: a clip is applicable iff it carries a
non-empty `action_data_uri` (§8). There is no dexterous / `source_dataset`
registry gate — ACT2 `.bin` artifacts are self-describing, so the reader resolves
dexterity from the artifact header's `spec_name`, and a non-mecka or malformed
artifact is rejected per row by the extractor's geometry checks (a NULL group,
retried next run) rather than filtered out by a dataset name.

### 2.2 Action artifact (ACT2 binary)

`action_data_uri` points at a self-describing **ACT2 binary** (`.bin`). The embed
leg rejects legacy `.pickle` artifacts without worker-side `pickle.loads`; use
ACT2 for production and for any table the action leg will read. ACT2 is decoded by
[`action_binary.decode_action_bin`](../../../cosmos_curator/next/media/action_binary.py);
its header carries `spec_name`, which gates the mecka wrist alignment (§5). One
artifact is shared by **every view** of a span, so the action leg de-duplicates
on `action_data_uri`.

A mecka payload provides, per frame:


| Key                                                 | Shape     | Meaning                                          |
| --------------------------------------------------- | --------- | ------------------------------------------------ |
| `hand_left_cam`, `hand_right_cam`                   | `(T, 63)` | 21-joint hand skeleton (camera frame), flattened |
| `hand_left_cam_rotation`, `hand_right_cam_rotation` | `(T, 84)` | 21-joint quaternions (xyzw), flattened           |
| `camera_position`                                   | `(T, 3)`  | ego-camera translation                           |
| `camera_rotation`                                   | `(T, 4)`  | ego-camera quaternion (xyzw)                     |




> **Provenance note.** These pose fields are *deterministic annotation* shipped
> with the dataset — never model- or VLM-generated. See §5 ("Provenance: wrist
> localization is deterministic annotation, not model output") for why the
> geometry is annotation-sourced and what model output is (and is not) used for.

A mecka payload also carries a **per-clip `intrinsics` tail** of shape `(8,)` —
camera intrinsics stored once for the whole clip, after the frame records. The
descriptor does **not** consume it, but it is why the ACT2 layout has a
`per_clip_offset` at all, and an artifact rebuilt from the source table without
it fails `encode_action_bin`'s exact-field-set check.

**Offset arithmetic (mecka), worked by hand.** With `N` frames:

```text
per-frame record = 63 + 63 + 84 + 84 + 3 + 4          = 301 float32 = 1204 bytes
header           = frame_data_offset                   = 1024 bytes
frame block      = N * 1204 bytes                       (frame-major, fixed stride)
per_clip_offset  = 1024 + N * 1204                      (intrinsics tail starts here)
intrinsics tail  = 8 float32                            = 32 bytes
total payload    = 1024 + N * 1204 + 32 = 1056 + N * 1204 bytes
```

`_parse_header` recomputes `per_clip_offset == frame_data_offset + num_frames *
frame_record_size_bytes` and rejects a self-inconsistent header, so this
arithmetic is auditable by hand against any real file.

### 2.3 Image and text inputs

- **Image**: `clip_uri` is a video; the leg decodes only the **first displayable
frame** via the sensor library (`CameraSensor` + one-timestamp `SamplingSpec`),
which handles B-frame presentation order correctly. The object is **fetched into
memory with one sequential read** and the decoder works from that buffer, never
from a live remote stream: on an object store a seek is a fresh ranged request, so
a seek-heavy index build pays a network round trip per seek and costs far more than
downloading the whole object once (see
[sensor-library-cloud-storage.md](../guides/sensor-library-cloud-storage.md)).
Buffering is pixel-identical — it changes only where the decoder's bytes come from.
Because a clip is therefore one bounded sequential fetch, the frame reader reads up
to `image.read_concurrency` clips at a time through a thread pool and the embedder
then runs the backbone once over the survivors; the reads come back **keyed by
input row index**, so the leg's order-preserving contract (§7) holds independently
of completion order.
- **Text**: `task_name` / `subtask_name` strings, whitespace-collapsed before
embedding so spacing variants map to one canonical model input.



### 2.4 The complete read path

Each modality reads `clips.lance` **independently** and at a pinned version, but
the read is **not** a driver-side scan: the driver derives three values from the
manifest, and the rows are read inside the workers.

1. **URI** — `config.clips_lance_uri` (+ `storage_profile`, resolved into Lance
  storage options). The same URI is the write target (§3).
2. **Derive the work** — the first lines of
  [`fill_embedding_group`](../../../cosmos_curator/next/recipes/embeddings/fill.py)
   read the **manifest only**: the dataset's version, the fragment id list
   (`dataset.get_fragments()`), and the row predicate
   (`columns.pending_filter`). No row is scanned and no row is counted, so this
   costs one metadata read and is O(fragments). There is no planning *object* and
   no planning *step* — the three values are locals inside the fill.
3. **Distribute fragment ids** — the fill turns the id list into a Ray Data
  dataset (`ray.data.from_items`) and maps it over an actor pool. The unit of
   work is a **fragment id**, so nothing but integers crosses the driver boundary.
4. **Scan inside the worker** — each worker re-opens the table at the pinned
  version, takes **its own** fragment, and scans it with
   `columns=source_columns` and `filter=row_filter` — only that modality's source
   columns, and only the rows it still owes work on (applicability AND primary
   vector IS NULL, §7). It never projects an `embedding_*` vector column.
5. **Optional dev cap** — `config.max_fragments` truncates the **fragment list**
  (not the row set); `[:None]` is the identity slice, so an uncapped run pays
   nothing for the knob.

Because the fragment is the work unit, the table's fragment geometry is this
leg's parallelism ceiling, and the run logs it once before the first fill:
`embedding source s3://…/clips.lance v41: 8,412,900 rows, 1,052 fragments
(min=4,900 median=8,000 max=8,000 per fragment), 31 fields`. The
fragment count bounds how many workers can run at all, and the largest fragment
bounds how long the last one keeps the others waiting — so a table that is coarse
or lopsided is visible before any GPU time is spent on it rather than inferred
afterwards from a long tail. The line is summarised from manifest metadata by
[`report_table_shape`](../../../cosmos_curator/next/utils/lance_utils.py) and
reads no column data. Read after `ensure_embedding_columns`, so its version and
field count describe the schema the fills actually scan.

That same helper appends a note to the same record — never a second one, and
never at a level above `INFO` — when at least half the fragments hold fewer rows
than Lance's own small-file boundary of 1,024: the point past which per-fragment
manifest entry, file open, footer read and scanner setup stop being amortized.
The note states that boundary and the cost of moving either way from it, and
stops there. It issues no instruction and names no operation, because whether the
layout is actually wrong depends on how wide the reader is, which the storage
helper cannot know — a layout it reports on can be the right one, and coarsening
it costs read parallelism and lengthens the tail the last block runs alone. For
the same reason the converse fault, too few fragments for the cluster's width,
draws nothing from the helper at all.

Pinning the version is **not** a concurrency mechanism — the table is assumed
quiescent for the duration of a run (§7). It is there for two ordinary reasons:
every worker must scan one schema and one row set, and Lance needs a base version
to build a valid `Update` transaction.

There is **no** shared read and **no** driver-side `materialize()` of the corpus.
The action leg's PCA fit is the one place the driver reads rows itself, and it
reads a **single narrow column** (`scan_column`, §5).

```text
robot_action_split
        │  append base rows (embedding_* fields read as NULL)
        ▼
   clips.lance
        │
        │  driver, per modality: fill_embedding_group derives
        │                        pinned version + row filter
        │                        + fragment ids   [:max_fragments]
        ▼
   from_items([fragment ids]) ─▶ actor pool
                                   │  each worker, on its own fragment:
                                   │  fragment.scanner(columns = <modality inputs>,
                                   │                   filter  = applicable
                                   │                             AND primary IS NULL)
                                   ▼
                                 embed ─▶ update_columns (§3)

   text  ──▶  image  ──▶  action      (each its own fill and pool; sequential, §10.1)
```



### 2.5 What the embed leg does NOT read

The embed leg operates purely on `robot_action_split`'s durable outputs. It does
**not** read any of the following — each was consumed upstream during extraction:


| Not read by the embed leg                           | Consumed upstream by                          |
| --------------------------------------------------- | --------------------------------------------- |
| the original episode / long-recording video         | `robot_action_split` smart-cut                |
| the raw LeRobot `data/chunk-*/*.parquet` frame rows | `robot_action_split` discovery                |
| `meta/episodes/*` offsets                           | `robot_action_split` discovery                |
| `meta/tasks.parquet` / `meta/subtasks.parquet`      | `robot_action_split` discovery (label lookup) |
| the original chunk directories                      | `robot_action_split` discovery / cut          |


`task_name` and `subtask_name` are **already-resolved strings** on the clips row;
the text leg never re-derives them from a `task_index` / `subtask_index`. The
only media the embed leg opens are the **generated** per-clip MP4 (`clip_uri`,
image leg) and the **generated** per-span action artifact (`action_data_uri`,
action leg).

---



## 3. Output format: `embedding_*` column groups on `clips.lance`

Embeddings are **additive, nullable column groups written directly onto
`clips.lance`** — the same row a base field lives on carries that clip's vectors.
There is no separate embedding table and no `clip_id` copy: each modality owns a
private `embedding_<modality>_*` namespace, adds those columns in one metadata
commit, and fills them with a column-local atomic commit (§7). The authoritative
definitions are the `*_GROUP_SCHEMA` schemas and the `EMBEDDING_COLUMN_GROUPS`
registry in
[`schemas.py`](../../../cosmos_curator/next/embeddings/schemas.py); everything
below is derived from them.

Vectors are stored as **`fixed_size_list<float32, dim>`**: the width is a schema
guarantee (a mixed-width model swap fails loud at write), and it is the type a
Lance vector index requires. The list child is left nullable so an all-NULL
column survives the metadata-only `add_columns`. Every embedding field is
**nullable**, and it carries **no field-level metadata**.

Ownership and scale are expressed structurally instead. A column's owning modality
is its `embedding_<modality>_*` name prefix, and the authoritative mapping from
modality to columns is the
[`EMBEDDING_COLUMN_GROUPS`](../../../cosmos_curator/next/embeddings/schemas.py)
registry, which a consumer imports rather than re-deriving from the schema. The
scale contract is a property of the modality (§6), so it belongs with the
modality's definition, not repeated on each field. Field metadata was dropped
because it is a second, silently-divergent source of truth: it does not survive an
`add_columns` round trip uniformly, nothing validated it, and a stale
`embedding.scale` on a column is strictly worse than no annotation at all.

### Per-modality independent group presence

A group's columns exist on `clips.lance` **only after its modality has run**. Each
group is validated **independently** as one of two states — *absent as a whole*
(none of its columns exist) or *present and exactly matching its own schema* (all
of its columns exist with the right type and nullability). A partial or
wrong group is a corrupt schema and fails on the driver (§7). One group present while
another is absent is valid and expected: a `text`-only run adds only the three
text columns and leaves the image and action columns off the table entirely.

Because a not-yet-run modality has **no** column at all, a consumer that wants a
modality must test **column presence**, not merely NULL: `embedding_image IS NULL`
is meaningful only once the image group has been added.

### 3.1 Text — `embedding_text_*` (`TEXT_GROUP_SCHEMA`)


| Column                    | Type                          | Notes                                        |
| ------------------------- | ----------------------------- | -------------------------------------------- |
| `embedding_text_subtask`  | `fixed_size_list<float32,384>`| subtask vector, **L2-normalized**; the group's **primary vector** (its NULL state marks the row pending) |
| `embedding_text_task`     | `fixed_size_list<float32,384>`| task vector, **L2-normalized**               |
| `embedding_text_model_id` | `string`                      | producer id, e.g. `BAAI/bge-small-en-v1.5` (the group's provenance column) |


Two vectors are stored rather than one blended string so the subtask signal (the
dominant curation term) stays clean while task-level grouping is still available
without re-embedding (§4.3). All three fields transition together: a complete text
group has both vectors and the model id, so a row with one text vector but not the
other is **corrupt**, not pending (§7).

### 3.2 Image — `embedding_image_*` (`IMAGE_GROUP_SCHEMA`)


| Column                     | Type                          | Notes                                        |
| -------------------------- | ----------------------------- | -------------------------------------------- |
| `embedding_image`          | `fixed_size_list<float32,384>`| image vector, **L2-normalized**; the group's **primary vector** |
| `embedding_image_model_id` | `string`                      | producer id, e.g. `facebook/dinov2-small` (provenance column) |


The image group is applicable only where `clip_uri` is present (non-null and
non-empty, §8); an applicable clip whose media cannot be decoded fills as an
all-NULL image group (retried next run), preserving one output per selected row.

### 3.3 Action — `embedding_action_*` (`ACTION_GROUP_SCHEMA`)


| Column                                 | Type                         | Notes                                          |
| -------------------------------------- | ---------------------------- | ---------------------------------------------- |
| `embedding_action`                     | `fixed_size_list<float32,97>`| action vector, **raw PCA coordinates** (NOT unit-norm); the group's **primary vector** |
| `embedding_action_descriptor_version`  | `string`                     | descriptor-semantics fingerprint, e.g. `dual-wrist-v3` (provenance) |
| `embedding_action_pca_fingerprint`     | `string`                     | content fingerprint of the PCA basis that produced the vector (provenance) |

`embedding_action_pca_fingerprint` identifies **which** basis produced a vector:
`descriptor_version` alone cannot, since two bases fit from the same descriptors
are incomparable under one version. Both provenance columns are checked to hold at
most one distinct value across the group, so a table accidentally built from two
bases (or two descriptor versions) is detected before a fill (§7). Rows reference
the basis by fingerprint; the basis bytes live in a content-addressed artifact
(§3.4).


### 3.4 Action PCA artifact (`clips.lance__action_pca/<fingerprint>.npz`)

The action group stores only the **fingerprint** of its basis; the basis bytes
live in an **immutable, content-addressed** `.npz` object in a sibling directory:

```text
clips.lance__action_pca/
    <sha256-fingerprint>.npz     one immutable object per fitted basis
```

Each `.npz` holds `mean` `(600,)`, `components` `(97, 600)`, and provenance
(`descriptor_dim`, `n_components`, `n_fit_rows`, `descriptor_version`,
`fit_timestamp`, `explained_variance_ratio`). The object is named by a SHA-256 of
its own `mean` + `components` bytes + `descriptor_version`, so two artifacts share
a name iff they would project identically. It is self-describing, so a load can
reject an incompatible basis (missing key, `descriptor_dim` / `n_components` /
shape / version mismatch) with a driver-side `ValueError`;
`explained_variance_ratio` is optional on load (older 7-key artifacts read it as
`nan`), and a fit below the retained-variance floor warns but is not rejected.

Content addressing decouples the artifact write from the Lance commit: the basis
is persisted *before* the commit, and each row references it by fingerprint, so a
failed commit leaves only a harmless unreferenced object — never new basis bytes
paired with old action vectors. An append reuses the fingerprint the group's rows
already carry; a reset group fits a new basis under a new fingerprint (§5, §11).
There is no mutable "current PCA" pointer.

### 3.5 One clip, end to end (worked example)

Follow a single `(span, view)` clip. Its `clip_id` is a SHA-256 digest (shown
truncated as `a1b2c3d4…`, **not** a readable name). Its embeddings are **columns
on the same row**, not rows in three tables.

**Base row (written by** `robot_action_split`**), embedding columns still NULL /
absent:**

```jsonc
{ "clip_id":         "a1b2c3d4…",
  "task_name":       "Make coffee",
  "subtask_name":    "Pick up capsule",
  "clip_uri":        "s3://…/video/observation.images.head/a1b2c3d4….mp4",
  "action_data_uri": "s3://…/action/9f8e7d….bin" }   // shared by every view of the span
```

**What each modality reads and writes (onto the same row):**

```text
task_name    "Make coffee"      ─▶ text model  ─▶ embedding_text_task    (384-d unit)
subtask_name "Pick up capsule"  ─▶ text model  ─▶ embedding_text_subtask (384-d unit)
clip_uri     a1b2c3d4….mp4      ─▶ first frame ─▶ vision model ─▶ embedding_image (384-d unit)
action_data_uri 9f8e7d….bin     ─▶ dual-wrist 600-d ─▶ PCA ─▶ embedding_action (97-d raw)
```

**The same row after all three modalities ran (embedding columns shown; base
fields omitted for brevity):**

```jsonc
{ "clip_id": "a1b2c3d4…",
  "embedding_text_subtask": [0.031, -0.088, ...],   // 384 floats, ||v|| = 1
  "embedding_text_task":    [0.012,  0.104, ...],   // 384 floats, ||v|| = 1
  "embedding_text_model_id": "BAAI/bge-small-en-v1.5",
  "embedding_image": [-0.05, 0.07, ...],            // 384 floats, ||v|| = 1
  "embedding_image_model_id": "facebook/dinov2-small",
  "embedding_action": [1.83, -0.44, ...],           // 97 floats, raw PCA coords
  "embedding_action_descriptor_version": "dual-wrist-v3",
  "embedding_action_pca_fingerprint": "9c1f0a…" }
```

If a modality could not embed this clip (e.g. an unreadable MP4 or a non-mecka
artifact), that group's columns stay NULL on this row and a downstream read that
requires a non-null vector filters it out (§8). If a modality never ran, its
columns are **absent** from the table entirely (§3, "per-modality independent
group presence").

### 3.6 Reading an embedding column group back

Every vector is stored as Arrow `fixed_size_list<float32, dim>`. A consumer (e.g.
the fuse / cluster leg) reconstructs a dense `(n, dim)` matrix like this, first
guarding that the group's columns **exist** and then filtering the still-NULL
(not-yet / could-not embed) rows:

```python
import lance
import numpy as np

ds = lance.dataset("s3://bucket/run/clips.lance")

# A modality's columns exist only after it has run; check presence before NULL.
if "embedding_image" not in ds.schema.names:
    raise SystemExit("the image modality has not run on this table yet")

# Read only rows that have actually been embedded (vector IS NOT NULL).
table = ds.to_table(columns=["clip_id", "embedding_image"], filter="embedding_image IS NOT NULL")
clip_ids = table.column("clip_id").to_pylist()
# fixed_size_list<float32,384> -> (n, 384): compact the chunks first, then flatten
# the child buffer. A sliced list array's .values returns the FULL underlying
# buffer unless combine_chunks() compacts it, which would silently over-read.
flat = table.column("embedding_image").combine_chunks().values.to_numpy(zero_copy_only=False)
vectors = flat.reshape(len(clip_ids), -1).astype(np.float32)  # (n, IMAGE_DIM)
```

The text group has **two** vector columns (`embedding_text_subtask`,
`embedding_text_task`); read each the same way. The action vector
(`embedding_action`) is 97-d and **not** unit-normalized (raw PCA coordinates),
unlike the L2-normalized text / image vectors. No column carries an
`embedding_dim` — the width is fixed by the `fixed_size_list` type.

---



## 4. How the text and image legs work

Both are **pure-compute callables**, not Ray Data stages. The worker
(`_FragmentWorker`, §7) constructs the embedder **once per actor** — so the model
loads once, with heavy imports deferred into the constructor — and calls it per
scanned batch. Each `__call__` takes an Arrow batch of the modality's
`source_columns` and returns exactly the group's columns, one row per input row in
input order. Neither leg keeps cross-batch state, and neither knows anything about
Lance, fragments, or keys.

### 4.1 Text leg — `task_name` + `subtask_name` → two text vectors

`SentenceTransformerTextEmbedder` reads `task_name` and `subtask_name` from the
worker's fragment scan and emits **two** 384-d vectors per clip. It is **key-free**:
it never reads `clip_id` and never emits it. The batch it returns is exactly the
text group's columns, one row per input row **in input order** — the filler
re-attaches the key positionally from the same scanned batch (§7).

```text
row: (task_name, subtask_name)
          │            │
     format_task()  format_subtask()      ← collapse whitespace, None → ""
          │            │
        encode(normalize_embeddings=True)      (two batched encode calls)
          │            │
          ▼            ▼
   embedding_text_task  embedding_text_subtask   → embedding_text_* columns on
    (384-d unit)         (384-d unit)              clips.lance (+ embedding_text_model_id)
```

Step by step:

1. **Format** (`text/formatter.py`, pure): `format_task` / `format_subtask`
  collapse every run of whitespace (tabs, newlines, doubled spaces) to a single
   space and strip the ends; `None` (an absent field) formats to `""`. The task /
   subtask strings arrive from many upstream sources with incidental spacing, and
   the text model is whitespace-sensitive — two clips with the same instruction but different
   spacing would otherwise land at slightly different points in the embedding
   space. Formatting makes the model input a deterministic function of the words
   alone (e.g. `"open  the\tdrawer"` and `"open the drawer"` produce the identical
   vector).
2. **Encode**: the batch's formatted subtask strings and task strings are encoded
  in two `SentenceTransformer.encode(..., normalize_embeddings=True)` calls, so
   each vector is **L2-normalized** (cosine distance downstream is a plain dot
   product).
3. **Write**: `text_columns_batch` builds the text group batch =
  `(embedding_text_subtask, embedding_text_task, embedding_text_model_id)` — one row
   per scanned row, in scan order — which the worker joins to `clip_id` and writes
   into the `embedding_text_*` columns of its fragment (§7).

**Why two vectors, not one blended string.** `subtask_name` is the fine-grained
instruction and the **dominant** term in the downstream curation distance;
`task_name` is its parent, used for task-level grouping. They are stored
separately — not blended into one string — so the two semantic levels stay
independently comparable and re-weightable. The full rationale, trade-offs,
downstream use cases, and examples are in §4.3. Each vector embeds a different,
independent string (`task_name` vs `subtask_name`).

**No row ever fails.** `task_name` / `subtask_name` are non-null on the source
contract, and an empty instruction still embeds to a valid (content-free) unit
vector — so the text **leg** is 1:1: every scanned row fills a complete group
(`filled == selected`, no per-row NULLs). The row predicate is version-pinned to the
rows whose primary vector is still NULL, so a re-run fills only the not-yet-embedded
rows (§7). Blank (empty / whitespace-only) instructions all embed to the **same**
point, which reads as a false "exact duplicate" downstream; the leg keeps them by
design but **counts and logs** them per batch so the condition is observable. A
zero-row batch returns an empty group batch (mirroring the image / action legs)
rather than depending on the Ray batcher never delivering an empty block.

### 4.2 Image leg — `clip_uri` → one image vector

`HfVisionImageEmbedder` reads `clip_uri` and emits one 384-d vector per
**readable** clip. Like the text embedder it is key-free and order-preserving: one
output row per input row, so an unreadable clip yields an **all-NULL row** rather
than a dropped one.

```text
row: (clip_uri)                          ── ClipFrameReader.read_many(uris);
        │                                   up to read_concurrency rows at a time
   open stream (lazy smart_open params) → one sequential read → in-memory buffer
        │
   read_first_frame(buffer)  (CameraSensor + 1-timestamp SamplingSpec)
        ▼
   first displayable RGB frame (H,W,3)   — None ⇒ row's group stays all-NULL
        │                                  ── returned keyed by input row index

   AutoImageProcessor → vision model → pooled output → L2-normalize
        ▼
   embedding_image (384-d unit)   → embedding_image_* columns on
                                    clips.lance (+ embedding_image_model_id)
```

Step by step:

1. **Read one frame** (`image/frame_reader.py`): open `clip_uri` with
  `smart_open`, pull the object into memory with **one sequential read** — never
   staged to a local temp file, and never handed to the decoder as a live remote
   stream — then decode only the **first displayable** frame from that buffer via
   `CameraSensor` + a one-timestamp `SamplingSpec`. Using the sensor library (rather than a hand-
   rolled decode) gets correct presentation-order handling for B-frame streams —
   the first *decoded* frame is not necessarily the first frame a viewer sees.
   The result is an `(H, W, 3)` `uint8` RGB array; a non-3-channel or absent
   frame yields `None`, so that clip's image group is emitted as **all-NULL** (its
   vector and model id both NULL) rather than dropped — preserving one output row
   per scanned row, which is what lets the filler align the group positionally —
   and is retried on the next run. The storage client and its `smart_open` transport params are resolved
   **once per backend** (scheme + bucket), outside the per-clip drop handler, so
   a systemic misconfiguration — a wrong storage profile / endpoint, or a backend
   `smart_open` cannot bridge — fails the whole leg loudly instead of being
   logged-and-dropped for every clip and silently leaving every vector NULL.
   Each `read` still passes the current `clip_uri` into `get_smart_open_params`
   on cache miss so mixed-backend tables do not reuse the first URI's client.
   The reader owns the concurrency of its own transport: `read_many` takes one
   batch's URIs, resolves every backend they name on the calling thread, fans the
   reads out over a bounded pool sized by its own `read_concurrency` field, and
   returns `{input row index: frame}`. Keying the result by input position rather
   than returning a sequence is what makes a frame-to-row misalignment
   unrepresentable — a row that named no media, or whose clip was dropped, is
   simply an absent key. The embedder therefore holds no pool and no read width of
   its own; it forwards the configured width into the reader it constructs and
   scatters the returned frames to their rows.
2. **Embed**: the frame(s) go through `AutoImageProcessor` and the vision model
   (DINOv2 by default); the leg takes the pooled output selected by the spec's
   `VisionPooling` (for DINOv2, `pooler_output` — the pooled head over the CLS
   token after the final layernorm, not the raw CLS hidden state) and
   **L2-normalizes** it. The image-processor `backend` is pinned to
   `"torchvision"` so preprocessing (resize / resample / mean-std normalization)
   is a fixed function of `model_id` rather than of whichever backend happens to
   be importable in the runtime environment; a PIL-only environment now fails
   loudly at construction rather than silently producing subtly different vectors
   under an identical `model_id`.
3. **Write**: `image_columns_batch` builds the image group batch =
  `(embedding_image, embedding_image_model_id)` — one row per scanned row in scan
   order, with an all-NULL group for a failed decode — which the worker joins to
   `clip_id` and writes into the `embedding_image_*` columns of its fragment (§7).

**Why the reads are concurrent inside the GPU actor, not a separate CPU stage.**
The obvious-looking alternative is a CPU read stage feeding a GPU embed stage, so
the GPU is not held while a clip downloads. Per clip the leg is
**network-latency bound**: the download dominates the read and it releases the
GIL, so concurrent reads inside one actor attack that term directly — which is
why `read_concurrency` exists at all. The decode those reads feed is
**GIL-serialized**, and the preprocess and forward are serial in the actor.

A separate CPU stage would attack the decode term instead, and genuinely could,
because separate stages are separate processes and therefore separate GILs. It is
nonetheless rejected here, on cost rather than on effect: it would cost a
materialized hand-off of decoded frames between stages plus a regroup before the
per-fragment write, and raising the table's fragment count buys the *same*
per-process division of the decode floor with no new architecture (§7). Note also
that most of the preprocess + forward term is CPU preprocessing
(`AutoImageProcessor` resize / normalize), which a GPU stage **relocates rather
than removes**.

**Three ceilings bound `read_concurrency`**, and the knob moves none of them:

| Ceiling | Mechanism |
|---|---|
| `read_concurrency` ≤ `batch_size` | the reader draws its clips from one scan batch, so the excess workers would never get work — enforced by a `model_validator`, not merely documented |
| decode throughput | GIL-serialized, so extra readers overlap the fetch wait and not the decode |
| storage connection pool | the actor's readers share one storage client, sized by `BaseClientConfig.max_concurrent_threads` |

**Why the default is already past the knee.** Because the width widens only the
download while the decode floor stays put, there is a crossover width above which
the decode is the larger of the two terms, and the shipped default sits past it.
Doubling the knob therefore shortens a term that is no longer the binding one:
raising it was considered, modelled against an observed production run, and
dropped — not left untried.

That crossover is a property of **one deployment**, not of the code: it is set by
the storage endpoint's latency and the typical per-clip object size. Treat the
default as tuned for the endpoint it was measured against, and re-measure before
re-tuning if either of those changes materially.

**Sizing `num_cpus`.** For the same reason the reservation is a **fixed** budget
that does *not* track `read_concurrency`: an actor's concurrent readers contend
for one interpreter instead of each occupying a core, so the reservation covers
one decode stream plus a slice for the serial preprocess and forward. Raising the
two together is **not** required.

The lever that does divide the decode floor is **actor count**: separate actors
are separate processes and therefore separate GILs. It is not a knob here — it is
bounded by the fill's work-item count, which is the table's fragment geometry up
to a block cap ("How many fill workers actually run", §7).

**Why the first frame, and why DINOv2.** A clip is one continuous subtask from a
head-mounted camera, so its frames are highly correlated; one frame answers "does
this look like a different place / object" at a fraction of the decode + inference
cost. DINOv2's self-supervised features are organized around visual structure
rather than text describability, so this axis does not re-encode (and
double-count) the text signal.

**Failures.** A clip whose media is missing or undecodable fills as an all-NULL
image group on its row (logged, retried next run), preserving one output per
scanned row; a systemic misconfiguration (bad `storage_profile` /
endpoint) fails the leg loudly instead — see §8.

### 4.3 Task vs subtask: why two text embeddings

The text leg stores **two** vectors per clip — `embedding_text_subtask` and
`embedding_text_task` — rather than one. This is a deliberate design choice, not just
a mirror of the two source columns.

**Conceptual model — Task ▸ Subtask ▸ Clip.** A clip is one *subtask*, but every
subtask belongs to a larger *task*:

```text
Task: "Make coffee"
  ├── Pick up cup
  ├── Insert capsule
  ├── Close machine
  ├── Press button
  └── Pour coffee        ← each leaf is one clip's subtask
```

The clip *is* "Insert capsule", but it also *belongs to* "Make coffee". Those are
two different — both useful — notions of similarity, so the leg records both.

**Why not concatenate into one string?** Embedding `"Make coffee. Insert capsule."` once would collapse the two levels into a single point that can never
be pulled apart again. Storing two vectors avoids five irreversible losses:


| A single blended vector cannot…                     | Two separate vectors can              |
| --------------------------------------------------- | ------------------------------------- |
| measure task- and subtask-similarity **separately** | compare each level on its own         |
| **weight** the levels differently downstream        | e.g. `0.7·subtask + 0.3·task`         |
| search by **task only**                             | retrieve "all `Make coffee` clips"    |
| search by **subtask only**                          | retrieve "all `Insert capsule` clips" |
| cluster on **one level independently**              | cluster subtasks *within* a task      |


A blended vector's geometry is also dominated by whichever phrase is longer or
more distinctive — an uncontrollable mix — whereas separate encodings keep each
signal clean.

**What each single level alone would lose.**

- *Only* `embedding_text_task`*:* every clip of "Make coffee" gets ~the same vector, so
"Insert capsule", "Press button", and "Pour coffee" become indistinguishable.
Fine-grained de-duplication and clustering are impossible — the five subtasks
cannot be told apart.
- *Only* `embedding_text_subtask`*:* the five subtasks separate cleanly, but the fact
that they share the parent task "Make coffee" is lost. Task diversity, task
balancing, and hierarchical sampling become impossible — nothing records that
these clips belong to one activity.

Keeping both is what lets a downstream leg operate at *either* level, or both.

**Downstream use cases.**


| Downstream need                                                             | Uses                     |
| --------------------------------------------------------------------------- | ------------------------ |
| task balancing · task statistics · task diversity                           | `embedding_text_task` only    |
| duplicate detection · fine clustering · semantic search · subtask balancing | `embedding_text_subtask` only |
| "clips in task = Make coffee **and** subtask = Insert capsule"              | both                     |
| cluster within each task, then de-duplicate only *inside* the task          | both                     |
| balance subtasks while preserving task diversity                            | both                     |

The curation leg uses `embedding_text_subtask` as one block of its fused
clip-similarity vector for clustering and de-duplication. It also uses
`embedding_text_task` to merge near-duplicate task labels during fairness grouping.
See [Curator Next Curation](curator-next-curation.md#grouping-canonical-labels-then-a-similarity-merge).
The two columns are still stored separately rather than pre-combined because the
needs in the table above are not curation-specific, and a downstream leg may weight
each block independently.

**Independent weighting, without re-embedding.** Because the vectors are stored
apart, a future leg chooses a strategy at consumption time — for example:

```text
similarity = 0.7 · d(subtask_a, subtask_b) + 0.3 · d(task_a, task_b)
   — or —
cluster by subtask, then sample uniformly by task
```

Changing that weighting, or switching from subtask-only to a blend, requires
**no re-run of the embedding pipeline** — only the two vectors already on disk. A
single blended embedding would force a re-embed for every such change.

**What is stored.** The `embedding_text_*` group on each clip's `clips.lance` row
(see §3.1):

```text
embedding_text_subtask · embedding_text_task · embedding_text_model_id
```

- both vectors belong to the **same row** and the **same group**;
- both come from the **same model** (`embedding_text_model_id`) in a **single
pass** over the batch (two `encode` calls in one actor);
- they are stored as separate columns only because they represent **different
semantic levels** — not because they are different data or different models.

Both are 384-d and L2-normalized (§3.1, §6).

**Examples.**

1. *Siblings under one task — "Prepare breakfast"* with subtasks Open fridge /
  Take milk / Close fridge / Pour milk. The four `embedding_text_task`s are
   ~identical (one activity), so a task-level view groups them; the four
   `embedding_text_subtask`s are distinct, so a subtask-level view separates each
   manipulation. Both relationships are available at once.
2. *A shared subtask across different tasks — "Pick up sponge".*
  ```text
   Task "Clean table"  ·  Subtask "Pick up sponge"
   Task "Wash dishes"  ·  Subtask "Pick up sponge"
  ```
   The two `embedding_text_subtask`s are ~identical (same fine action) while the two
   `embedding_text_task`s differ (different activities). Storing both preserves *both*
   truths: a subtask-only view can merge them (e.g. motion-agnostic dedup of the
   pick-up gesture), while a task-only view keeps them apart (two different
   chores). One blended vector could represent only one of these relationships.

**Future extensibility.** Storing two levels now enables later consumers without
a re-embed: hierarchical clustering, hierarchical retrieval, curriculum learning
(easy→hard by task then subtask), balanced sampling, task-level analytics,
subtask-level analytics, and multimodal fusion (§1.6). **These are future
consumers — this milestone only produces the two text vectors; it performs none
of them.**

---

## 5. The action descriptor (`dual-wrist-v3`)

The action leg is the highest-risk part because it depends on the exact shape of
someone else's pose arrays. The wrist-pose derivation follows the **reference
hand-pose pipeline** verbatim (the authoritative consumer of this pose data),
which is why `dual-wrist-v3` differs from a naive camera-frame descriptor. The
descriptor is `2 arms × 50 samples × (3 position + 3 rotation) = 600`.

```text
hand_*_cam[i]          → (21,3) skeleton; select wrist (joint 0) → (N,3)
hand_*_cam_rotation[i] → (21,4) xyzw;     select wrist (joint 0) → (N,4)
  drop no-hand / non-finite frames (shared hand+camera validity mask)
  wrist_cam    = pose(pos, quat)                  # wrist in the CAMERA frame
  cam_pose     = pose(camera_position, camera_rotation)
  if mecka:  wrist_cam = wrist_cam @ WRIST_FRAME_ALIGN_MECKA
  wrist_static = cam_pose @ wrist_cam             # compose out head motion
  wrist_rel    = inv(wrist_static[0]) @ wrist_static   # frame-0 anchored
  pos: center + unit-ball scale             (offset/scale invariant)
  rot: rotation matrix → rotvec → unwrap    (3-component; the per-frame unwrap
                                             keeps the track continuous across
                                             the ±π branch cut)
  arc-length resample to 50 stations        (speed/pause invariant)
```

**Worked shape trace** (one clip, `N = 120` raw frames, 2 of them "no-hand"):

```text
per arm:
  hand_*_cam            (120, 63)  -> reshape (120, 21, 3) -> wrist (120, 3)
  hand_*_cam_rotation   (120, 84)  -> reshape (120, 21, 4) -> wrist (120, 4)
  validity mask drops 2 no-hand frames            -> (118, 3) + (118, 4)
  compose + frame-0 anchor + center/scale + rotvec-> pos (118, 3), rot (118, 3)
  arc-length resample to 50 stations              -> (50, 3) + (50, 3) = (50, 6)
both arms:  (50, 6) + (50, 6) -> flatten + concat -> (600,)
```

Two arms x 50 stations x 6 channels = 600. The frame count `N` and the surviving
count vary per clip; the **output width is always 600** because the resample
fixes the station count. A clip with fewer than `_MIN_VALID_FRAMES` survivors on
either arm is rejected (`TOO_FEW_VALID_FRAMES`) rather than resampled from too
few points (§8).

Each step removes a specific nuisance variation:

- **Wrist selection** summarizes where the hand went without finger articulation.
Reading the flat arrays as if already a single pose is a silent wrong-width
bug, so the 21-joint shape is validated, not assumed.
- **Drop unusable frames first** — non-finite coordinates or near-zero-norm
quaternions (the "no hand detected" encoding). `scipy.from_quat` silently
normalizes, so a zero-norm quaternion left in would become a full-magnitude
rotation of noise. Dropping must precede centering/scaling, which are global.
- **Mecka wrist alignment** (`WRIST_FRAME_ALIGN_MECKA`) re-bases the wrist-local
frame to the reference convention; applied only for the mecka spec, gated by
the ACT2 header's `spec_name`. Its exact matrix values are pinned by a test.
- **Compose out the camera** so head motion does not masquerade as hand motion;
the axis measures the **hands**, not the head (contrast with the extract-side
scalar head-motion score, which is metadata only).
- **Frame-0 anchoring** makes the descriptor invariant to the absolute world pose
at the clip's start.
- **Center + unit-ball scale** makes position measure trajectory *shape*, not
where in the room it happened; a motionless track is returned centered rather
than divided by a near-zero radius (which would fabricate noise).
- **Rotvec, not quaternions** — quaternions double-cover SO(3), so a straight-
line quaternion distance is not a meaningful orientation distance; a rotation
vector is a proper 3-component form. Rot6d/rot9d are not used here; rotvec is sufficient for curation — §13.
- **Rotvec unwrapping (the `v2 → v3` change)** — `as_rotvec` returns an angle in
`[0, π]`, so a track that rotates past π flips sign and jumps by ~2π at the
branch cut. Because the arc-length resample parametrizes over all six channels
*jointly*, a single such jump can dominate the segment sum and push most
stations into a representation artifact — corrupting the position channels too.
Choosing, at each frame, the antipodal representative nearest the previous frame
(both denote the same rotation) keeps the descriptor a continuous function of the
motion. This is a **semantics** change, not a width change, which is exactly what
`DESCRIPTOR_VERSION` exists to catch.
- **Arc-length resample** makes the axis about shape: a slow and a fast execution
of the same gesture produce the same 50 stations. The station count fixes the
width, so it is contract, not a free parameter.
- **Both arms required** — a clip missing either wrist track is rejected, never
half-filled; a partial vector would compare as if the missing arm were
motionless, a false similarity nothing downstream could detect.

`DESCRIPTOR_VERSION` (`dual-wrist-v3`) fingerprints the descriptor **semantics**,
not its width. It must be bumped whenever any step above changes meaning — e.g.
reordering arms, switching resample interpolation, or (as in `v2 → v3`) adding
rotvec unwrapping — because that is the only signal that distinguishes "same
width, different meaning" for a persisted PCA basis (`action/pca.py` refuses a
basis whose recorded version differs from the running code).

### 5.1 What the descriptor removes vs. keeps

The chain is a sequence of deliberate invariances: each step erases one nuisance
factor that would otherwise make two identical gestures read as different. Read
this table as the *contract* — anything in the left column is intentionally **not**
recoverable from the descriptor.

| Nuisance factor                | Treatment                          | Which step |
| ------------------------------ | ---------------------------------- | ---------- |
| absolute world position        | removed                            | frame-0 anchor |
| absolute initial orientation   | removed                            | frame-0 anchor |
| head / camera (ego) motion     | removed                            | compose with camera pose |
| execution speed / pauses       | reduced                            | arc-length resample |
| trajectory translation & scale | normalized                         | center + unit-ball |
| quaternion sign (double cover) | removed                            | rotvec + unwrap |
| finger articulation            | discarded                          | wrist-joint selection |

What **remains meaningful** after all of the above — i.e. what the descriptor
distance actually compares:

- **trajectory shape** — the geometric path each wrist traces, independent of
where in the room or how fast it happened;
- **relative wrist orientation** — how the wrist re-orients along that path;
- **left/right differences** — the two arms are kept separate (first 300 dims
left, last 300 right), so a one-handed vs two-handed gesture is distinguishable;
- **manipulation-motion geometry** — the combined dual-wrist gesture that makes
"similar hand motion" map to "close vector" for clustering / dedup.

### 5.2 Why rotation vectors, not quaternions — the 180° double-cover problem

The one step above that looks arbitrary until you hit it is `rotation matrix →
rotvec`. A new engineer reasonably asks *"the poses arrived as quaternions — why
not just compare those?"* The answer is a geometric fact about rotations that
trips up even experienced engineers, so it is worth stating explicitly.

**The double cover.** The space of 3-D rotations, `SO(3)`, has three degrees of
freedom, but a unit quaternion has four components constrained to the unit sphere
`S³`. That sphere covers `SO(3)` **twice**: a quaternion `q` and its negation
`−q` denote the **exact same rotation**. There is no canonical sign — both are
equally valid, and a producer (or the same producer across frames) may hand you
either one.

**Why smooth motion can look discontinuous.** Because `q` and `−q` are the same
rotation, a sign can flip between two frames of a perfectly smooth motion.
Compared component-wise, `q` and `−q` are maximally far apart (their difference
has norm 2), so a sign flip reads as an enormous orientation change even though
the wrist did not move discontinuously at all.

**The 180° sweep.** The effect is not merely a producer quirk — it becomes
*unavoidable* the moment a **relative** rotation sweeps past 180°.
`Rotation.as_rotvec` returns `θ·n̂` with the angle `θ` canonicalized to `[0, π]`,
and a rotation by 181° about axis `n̂` is the same rotation as 179° about `−n̂`.
So the rotvec's sign flips exactly at the crossing. Follow one wrist turning
steadily about a fixed axis `n̂`:

| physical sweep | orientation (smooth?) | `as_rotvec` returns | jump? |
| -------------- | --------------------- | ------------------- | ----- |
| 0°   | identity | `0`         |                    |
| 45°  | +45°  | `+45°·n̂`  |                       |
| 90°  | +90°  | `+90°·n̂`  |                       |
| 135° | +135° | `+135°·n̂` |                       |
| 179° | +179° | `+179°·n̂` |                       |
| 181° | +181° | `−179°·n̂` | **← ~2π sign flip**   |
| 225° | +225° | `−135°·n̂` |                       |
| 270° | +270° | `−90°·n̂`  |                       |

The physical orientation is monotone and smooth throughout; only the
*representation* jumps by ~2π at the 180° boundary. Nothing about the motion
changed — just the label printed for it.

**Practical consequences** if that jump is left in the descriptor:

- **Trajectory comparison** — two identical sweeps that cross 180° at slightly
different frames get large, purely artificial distances.
- **Interpolation / resampling** — this pipeline resamples uniformly in arc
length over all six channels *jointly*, so a single ~2π rotation jump can be
longer than the entire real trajectory; almost every station then lands on the
discontinuity instead of the motion, which corrupts the position channels too
(they share the same parametrization).
- **Distance computation** — an L2 distance is dominated by the branch-cut
artifact rather than the gesture.
- **PCA** — the artifact injects a large spurious variance direction, so
components end up encoding *where the branch cut fell* rather than how the hand
moved: wasted low-dimensional budget and an unstable basis across re-fits.
- **Descriptor stability** — near-duplicate clips can land far apart purely on
which side of 180° the sensor noise put them, defeating similarity search / dedup.

**Why rotvec fixes it.** A rotation vector is a **minimal 3-component** encoding
(one per DOF), so — unlike the 4-component unit quaternion living on a constrained
sphere — PCA and Euclidean distance operate on an unconstrained space with no
redundant, sign-ambiguous dimension. After the per-frame **unwrap** (below) there
is **no sign ambiguity**, trajectories are **smooth** for interpolation / distance
/ PCA, and the encoding is **geometrically legible**: direction = rotation axis,
magnitude = rotation angle.

**How the implementation handles it** (`_unwrap_rotvecs`). The code enforces
*inter-frame continuity* rather than trusting the raw canonical form. The two
representatives of one rotation are the rotvec `r` and its antipodal twin
`r·(1 − 2π/‖r‖)` (the same rotation taken the "long way"); at each frame the code
keeps whichever is nearer the previous frame, letting the magnitude grow past π so
the track stays continuous. Two related choices matter for correctness:

- Near-zero-norm quaternions are dropped **before** `from_quat`, because SciPy
silently re-normalizes and would otherwise turn a "no hand" (~0-norm) quaternion
into a full-magnitude rotation of noise.
- The unwrap is skipped below a tiny angle, where the antipodal formula
`1 − 2π/‖r‖` diverges as `‖r‖ → 0` and the frame is already continuous.

**Residual limitations (the representation is not magic):**

- Unwrapping guarantees continuity only when consecutive valid frames are close
(adjacent relative rotations well under 180° apart). A track whose survivors are
sparse or spliced across large masked gaps can be unwrapped the wrong way — one
reason sparse tracks are rejected upstream by the validity-fraction gate.
- An **isolated** rotation of exactly 180° is genuinely ambiguous (`+π·n̂` and
`−π·n̂` are the same rotation and equidistant); the choice is arbitrary but
harmless at the continuity limit.
- Linear interpolation of rotvecs during resampling is **not** the `SO(3)`
geodesic (SLERP). This is deliberate: the descriptor is a fixed-width similarity
summary, not a trajectory to be replayed, and switching to SLERP would change
every vector and require a `DESCRIPTOR_VERSION` bump.

Conceptually:

```text
        physical wrist rotation (smooth, continuous)
                        │
                        ▼
             continuous orientation in SO(3)
                        │
        ┌───────────────┴───────────────────┐
        ▼                                    ▼
  quaternion (S³, 2:1 cover)          rotation vector (minimal 3-D)
        │                                    │
   q and −q are the SAME rotation      axis = n̂, angle = ‖r‖
        │                                    │
   sign can flip on smooth motion      as_rotvec canonicalizes θ ∈ [0, π]
        │                                    │
   ~2π jump when a sweep crosses 180°   ── unwrap: pick the antipodal twin
        │                                     nearest the previous frame
        ▼                                    │
   artificial distance / unstable PCA        ▼
                                       smooth track → stable descriptor
```

Note that a rotation vector does **not** eliminate the double cover topologically
— axis-angle / exponential coordinates keep a residual double representation
exactly at `θ = π` (and every `SO(3)` representation in four or fewer Euclidean
dimensions is discontinuous somewhere). What the unwrap buys is a **track** that
is continuous frame-to-frame, which is all the descriptor needs.

### 5.3 Provenance: wrist localization is deterministic annotation, not model output

The wrist geometry this descriptor consumes is **deterministic per-frame pose
annotation** carried inside the ACT2 artifact (`hand_left_cam` /
`hand_right_cam`, `hand_*_cam_rotation`, `camera_position` / `camera_rotation`;
§2.2), decoded by `decode_action_bin`. It is never produced by a caption, a
language model, or a vision-language model (VLM). The instruction text the
**text leg** embeds (`task_name` / `subtask_name`, §4.1) is a *separate,
semantic* channel: it says *which task* the span performs and never feeds the
geometry. The two channels are computed independently and never cross.

```text
ACT2 pose annotation --decode_action_bin--> dual_wrist_motion_descriptor --> PCA --> action embedding   (GEOMETRY)
task / subtask label --format_task/subtask--> BGE --------------------------------> text   embedding   (SEMANTICS)
```

**Why the geometry is not model-generated.** A VLM is strong at *semantics* —
"the person reaches toward the cup" — but the descriptor needs answers a caption
does not reliably carry: *which* wrist (left vs right), *where exactly* the
wrist is in 3-D, and *in which frame*. Model-generated localization can be
semantically plausible yet geometrically wrong. Because this axis is a
high-precision signal, it is sourced from annotation **by construction**: there
is no model-localization path in the code that could silently become geometric
ground truth. This is a data-quality / geometric-accuracy decision, not a
preference for hand-written text.

**Why a wrong localization is expensive here.** Every step in the derivation
above assumes the input is a physically meaningful wrist track:

- a small position error changes the resampled trajectory *shape* the axis is
  built to measure;
- a left/right swap silently mislabels the two 300-d halves — the both-arms
  requirement rejects a *missing* arm, not a *swapped* one;
- a hallucinated pose passes the finite / non-zero-quaternion validity mask and
  is then treated as real motion;
- the error propagates deterministically: descriptor -> PCA basis -> every
  downstream Euclidean distance, clustering, and dedup decision.

**What model output *is* acceptable for.** Semantic text. The `task_name` /
`subtask_name` instructions — and, in the upstream reference pipeline, Qwen3-VL
video captions — feed only the **text** leg's BGE embedding (§4.1). Using a
model there is fine: that channel measures *task* similarity, not geometry, and
a paraphrase or a slightly noisy caption shifts a clip within text space without
ever touching a wrist coordinate.

**When could a model replace the deterministic localization?** Only after an
evaluation on this dataset family shows the model clears thresholds derived from
the descriptor's own sensitivity (the thresholds are not invented here):

- **left/right hand identity** — exact; a swap corrupts the vector and nothing
  downstream can detect it;
- **wrist localization error** — small enough, *after* the descriptor's center
  + unit-ball scale, that the resampled trajectory shape stays within the PCA
  basis's tolerance;
- **temporal consistency** — no per-frame jitter that arc-length resampling
  would amplify into false stations;
- **occlusion robustness** — stable through the hand-occluded frames the
  validity mask currently drops;
- **camera / view robustness** — consistent across the multi-view clips that
  share one span's single action artifact (§2.1);
- **failure detection / calibration** — a trustworthy "no reliable hand" signal
  so a bad frame is dropped (as the validity mask does today) rather than
  hallucinated.

Until such an evaluation exists, wrist localization stays annotation-sourced.

### 5.4 From 600 numbers to 97

Before the fit mechanics, it is worth being explicit about *what the 600 numbers
are* and *why they collapse to so few*.

The descriptor's width is pure bookkeeping of the chain above — not a tuned
hyperparameter:

```text
  per wrist, per station:  [ pos x,y,z | rot x,y,z ]            =   6
  × 50 arc-length stations along the trajectory                 = 300
  × 2 wrists (left, right), concatenated                        = 600
```

So `DESCRIPTOR_DIM = 600` is exactly `2 × 50 × 6`, fixed by the geometry. Change a
factor (more stations, a third channel, one arm) and the width changes with it.

Those 600 axes are nowhere near independent. A wrist trajectory is *smooth*, so
neighbouring stations move together; the two hands are often coordinated in a
bimanual task; and within a station the three position and three rotation
channels drift slowly. Most of the descriptor's variance therefore lives in a
much smaller subspace, so a single table-wide PCA keeps ~90% of it in far fewer
dimensions:

```text
  600 raw dims (2 × 50 × 6)          ──PCA──▶          97 coordinates
                                                       (~90% of the variance)

  redundant by construction:                  compact & consistent:
    • adjacent stations move together           • ~1/6 the width
    • left / right co-vary (bimanual)           • Euclidean distance ≈ motion difference
    • pos & rot channels drift slowly           • one fixed, versioned basis for all rows
```

Reducing with one shared basis buys three things at once: it **shrinks storage
and every downstream distance** (97 floats, not 600); it yields a space where
**plain Euclidean distance already approximates motion difference**, so consumers
do not each invent their own reduction; and it keeps that reduction **consistent
and versioned** rather than re-derived per reader.

Why *97* specifically? It is the component count that reaches the
**~90%-variance design target** on a reference population — a tuned constant
(`ACTION_DIM`), not a magic number, recorded in the basis so it cannot drift
silently. Because mean-centering costs one degree of freedom, fitting 97
components needs **strictly more than 97 rows** (≥ 98); fewer would fit a
rank-deficient basis whose last components are null-space noise while the
explained-variance figure misleadingly reads 100%.

### 5.5 The PCA basis

`fit_action_pca` computes a mean and the top 97 right singular vectors via numpy
SVD (no scikit-learn) and logs achieved explained variance against a rough 90%
target. It rejects a fit that cannot be honest:

- **Too few rows** — the sample must be strictly larger than the component count
(98 ≥ minimum). Mean-centering costs a degree of freedom, so a sample of
exactly 97 rows fits a rank-96 basis whose 97th component is null-space noise,
while the explained-variance figure misleadingly reads 100%.
- **Zero total variance** (all descriptors identical) or **effective rank below
97** — the retained tail would be noise, not motion.

**Load-by-fingerprint or fit-then-save (basis selection is a predicate on the
group, not a file flag).** Because every basis is an immutable, content-addressed
artifact (§3.4) and every action row records the
`embedding_action_pca_fingerprint` it was projected under, the action leg decides
what to do from the group's own state:

- **Group already holds embedded rows** (an incremental append) — it reads the
single surviving fingerprint (`validate_group_state` has proven there is at most
one) and **loads that exact basis**, re-projecting the new descriptors into the
identical coordinate system. This is a semantic no-op for the existing rows and is
logged so an operator expecting a refit sees why they did not get one.
- **Group is empty** (a first fit, including the state left by `--reset-group`) —
it **fits a fresh basis** from the run's descriptor sample and persists it
content-addressed (`save_if_absent` writes `<fingerprint>.npz`, reads it back, and
validates it), then projects. The whole group is then computed under the new
fingerprint.

`max_fragments` does not bias the fit. It caps the **fragments the fill visits**,
while the candidate scan reads the whole table's `action_data_uri` column, so a
truncated run fits from the same population an untruncated one would.
A recorded `descriptor_version` differing from the running code's is stale: the
leg refuses to load and directs the operator to `--reset-group action` (§11.1).
There is **no** "delete the `.npz`" step and no mutable current-basis file —
emptying the group is what refits, because content addressing makes an append
reuse the existing fingerprint and only an empty group fit a new one.

**How the fit sample is drawn (URI-first, no full-corpus pass).** A fit needs a
*representative* sample, not every descriptor. The driver therefore samples
**before** extracting anything:

```text
scan action_data_uri only  ──▶  rank each by sha256(uri)  ──▶  keep the
(narrow column, predicate       (deterministic, no RNG,        smallest-rank
 pushed into Lance)              no seed to record)            2 x sample_size
                                                                    │
                            fit on the survivors  ◀── extract only those  ◀──┘
                            (>= ACTION_DIM + 1)       (bounded actor pool)
```

Ranking by `sha256(uri)` makes the sample a deterministic function of the corpus:
re-running picks the same spans without storing a seed. De-duplicating on
`action_data_uri` means a multi-view span contributes **one** vote to the basis, not
one per camera. The candidate set is deliberately an **oversample** (twice the
requested size): sampling precedes extraction, so an unreadable or
geometrically-rejected artifact removes a candidate from the fit, and the extra
margin absorbs that loss without a second pass. A fit left with `ACTION_DIM` or
fewer distinct descriptors raises rather than persisting a rank-deficient basis
(§5.4).

**Operationally — first run vs later runs.** The basis is fit **once** on an empty
group and then loaded and reused by fingerprint, so every action vector (across
runs) lives in the same coordinate system:

```text
first action run (empty group)              later runs (group has a fingerprint)
------------------------------              ------------------------------------
action artifacts                            read the group's fingerprint
   │ 600-d descriptors                           │ load <fingerprint>.npz
   │ sample (dedup on action_data_uri)           │
   ▼ fit PCA ──▶ save <fingerprint>.npz          ▼
project all descriptors ──▶ 97-d            project new descriptors ──▶ 97-d
```

That reuse is exactly why an append must not silently re-fit the basis: new
vectors would land in a different coordinate system from the rows already written,
and no shape check would catch it. Content addressing enforces this — an append
resolves the fingerprint the rows already carry, never a newly fit one.

### 5.6 What an action artifact costs, and the two ceilings above it

The action leg looks compute-bound and is not. Measured over 256 production
artifacts, one artifact costs a **mean 548 ms**, split:

| Term | Median | Mean | Share of the mean |
|---|---|---|---|
| read (network) | 492.95 ms | 512.43 ms | **94.3 %** |
| decode (ACT2 header + arrays) | 0.10 ms | 0.10 ms | ~0 % |
| geometry (`dual_wrist_motion_descriptor`) | 32.18 ms | 31.09 ms | 5.7 % |

The shares are of the mean, which is the figure a walltime is planned against; the
medians are lower because the size distribution has a long tail (below). Do not mix
the two columns — a share taken against the median total would read ~93.9 %. The
three terms sum to 543.62 ms against the 548 ms measured end to end (the rate
quoted below is 1/0.548); the ~4 ms difference is per-artifact overhead outside the
three timed terms and is not attributed further.

Artifacts are small — 145.5 kB min, 247.9 kB median, 705.4 kB max — so 248 kB in
493 ms is not a bandwidth figure. It is **latency**, and it decomposed into two
round trips plus a short transfer, because the repo's S3 helper reached for
boto3's managed transfer even on objects far below the multipart threshold. That
helper now serves a sub-threshold object from a single `get_object`
(`download_object_as_bytes`), which removed one of the two round trips.

The remaining wait is what `action.read_concurrency` overlaps. **Threads, not
processes**: the fetch releases the GIL, so N in-flight fetches cost N payload
buffers rather than N cores, and shipping decoded arrays across a process
boundary would cost more than the geometry they feed.

**Ceilings bound the leg, and the read width moves only past the first two.** The
per-actor ceiling splits by pass, because a fill and the PCA pass source their work
items differently.

| Ceiling | Mechanism | Lever |
|---|---|---|
| storage connection pool | the actor's readers share one client per backend, sized by `BaseClientConfig.max_concurrent_threads` (100), so a width of 32 does not queue behind it | raise the client's pool size |
| per **actor** | the ~31 ms geometry is GIL-held numpy, so it serializes however wide the reads are | none — this is the floor |
| per **fill** | a fill's work item is a Lance fragment, so actor count is the table's fragment count | fragment geometry, owned by the producer (§7) |
| per **PCA pass** | it builds work items from a URI list, not from fragments, so its pool is bounded by the CPUs allocated | allocate more CPUs |

Read the ceilings against the measurement: threaded **reads alone** reach
32.5 rows/s, but the full read-plus-geometry pipeline stalls at 18.6 rows/s on one
worker. The gap is the geometry. So widening the reads past the point where the
fetch is no longer the larger term buys nothing, and the only thing that divides
the geometry is more **processes** — separate actors have separate GILs.

That is also why the two passes behave so differently on the same table, and why
conflating them wastes work. The PCA candidate pass is **already pool-scaled**: on
a 96-CPU allocation it reached 110 rows/s across its pool, and even on a 24-CPU
node its 38.50 rows/s against a 1.82 rows/s per-worker rate implies ~21 concurrent
actors. A fill over a **single-fragment** table, by contrast, is one actor no
matter how large the cluster is. A resource line read at *completion* reports
post-scale-down values and must never be used to infer the pool size during a run
— misreading one is what once made the PCA pass look starved.

**The default of 32** matches the image leg's and sits where the measured curve
bends: 1 thread 1.9 rows/s, 16 threads 16.4, 32 threads 18.6. It may not exceed
`action.batch_size` (default 256), since the reader draws its URIs from one scan
batch; a wider value is rejected at config assembly rather than silently clamped.

**Determinism is unaffected, but the basis fingerprint is not stable across a
re-fit.** The fit sample is chosen by smallest `sha256(action_data_uri)` rank, a
pure function of the URI, so the selected **set** is identical at any read width or
partitioning. The sample's row **order** does follow arrival, and float64 summation
is not associative, so a re-fit permutes the last bits and the content-addressed
fingerprint changes completely. Measured 2026-08-23: the same table refit under a
different actor count moved from `7e54dacc506c` to `371c3b20dc92` with *identical*
fit statistics (97 components, 95.9 % variance, effective rank 594). That is a new
name for an equivalent basis. The operator hazard is not the new name — it is
refitting while rows already reference an older one, which content addressing
already prevents (§5.5).

---

## 6. The scale contract

The modalities are **deliberately not stored at the same scale**:


| Group  | Stored as               | Compared by            |
| ------ | ----------------------- | ---------------------- |
| Text   | unit-norm               | cosine (dot product)   |
| Image  | unit-norm               | cosine (dot product)   |
| Action | **raw** PCA coordinates | Euclidean in PCA space |


Text/image direction carries the meaning and magnitude is a model artifact, so
they are normalized. The action vector's magnitude *is* meaningful (a coordinate
in a fitted basis where distance already corresponds to motion difference), so
normalizing it would discard the difference between a small and a large gesture.
This is fixed **at write time**, not left to the reader — otherwise a different
caller could silently reweight the fused distance while every shape check still
passed. Changing a normalization convention invalidates every stored vector,
exactly like changing a curation weight.

A consumer that normalizes on read is making a deliberate scale conversion, and
curation does exactly that: it L2-normalizes the action block into the cosine
space so the fused vector stays unit-norm. That converts the question from "same
motion size" to "same motion pattern" and is documented as such in
`curator-next-curation.md`. Nothing is lost for *task* purposes by that
conversion, because the raw vector has no task signal to lose. The measurement is
taken in this table's own space — `distance_space: euclidean_pca`, magnitude
intact — and still returns a task-separation ratio of 0.990 and AUC 0.543 (§1.3).
Normalization only ever discards information, so a direction-only view cannot
acquire a task signal the full vector never had.

**What the action vector can and cannot be asked**, stated once so no consumer
has to re-derive it:

| Question | Supported | Why |
|---|---|---|
| "did these two clips move alike?" | yes | the basis is fitted so Euclidean distance approximates motion difference (§5) |
| "is this clip a near-duplicate recording of that one?" | yes, as one term of the fused distance | motion agreement is a necessary part of redundancy |
| "how large was the gesture?" | yes, from the raw magnitude | magnitude is preserved at write time, which is why this table says raw |
| "which task is this?" | **no** | ratio 0.990, AUC 0.543 — a coin flip |
| "are these two clips the same task?" | **no** | same as above; use canonicalized `task_name` (not the action vector) |

---



## 7. Persistence and re-run semantics

Embeddings own only the `embedding_*` column groups on `clips.lance`; they never
mutate a base column and never create a side table. Schema evolution and state
validation are owned by
[`columns.py`](../../../cosmos_curator/next/recipes/embeddings/columns.py);
deriving the work, writing it, and committing it are owned by
[`fill.py`](../../../cosmos_curator/next/recipes/embeddings/fill.py). "Planning"
is not a separate responsibility anywhere — it is three local values at the top of
`fill_embedding_group`.

Every modality runs the same sequence. Text and image take it with no branch at
all; action takes one explicit driver branch before its fill (see **Action
pre-step** below).

- **Ensure schema** — `ensure_embedding_columns` adds the missing fields of only
the enabled groups in one `add_columns` metadata commit. A group already present
must match its own schema exactly; a partial or wrong group raises. A text-only
run adds only the three text columns and never creates image/action columns. That
`add_columns` is a **commit**, so a first run advances the table's version even
when every fill then finds nothing pending: the run reports that version and
names the fill's absence, rather than reporting the table as unchanged. It is the
run's *first* commit, so any fill that does commit supersedes it as the reported
version (§11.3).
- **Validate group state** — `validate_embedding_group` enforces that the group's
complete rows carry at most one producer identity per provenance column, and —
where the modality configures an expected producer — that it is that one. A stale
producer (e.g. a different model id) fails and directs the operator to
`--reset-group`. Action passes no expected provenance: its producer identity is
the PCA fingerprint its own rows carry, so only the single-producer rule applies
and the basis is then read back from that fingerprint (§5). A mixed NULL /
non-NULL *partial* row is not checked because it is unreachable: one
`update_columns` call writes every field of a row's group together. Proving a
column holds a single producer needs every non-null value looked at — no Lance
fragment statistic or index metadata answers it — so the check is expressed as a
`SELECT DISTINCT ... LIMIT` and Lance does that pass over one encoded column,
returning a bounded result; the driver never holds a value per row. The same
bounded read (`lance_utils.distinct_non_null_values`) is what the Curate leg's
preflight uses to refuse *fusing* a group that carries two identities, since one
table can be filled by two runs whose configs differed and only a later reader of
the whole column can see it — see
[Producer identity](curator-next-curation.md#producer-identity-is-part-of-the-source-contract).
- **Fill** — `fill_embedding_group` derives the version, the fragment ids and the
pending predicate (`applicable AND primary vector IS NULL`) from the manifest —
a **metadata read**: no row is scanned and no row is counted on the driver — and
runs one actor pool over those ids. Each worker scans its own fragment's pending
rows, calls the embedder (which returns exactly one row per scanned row, in scan
order, an all-NULL group for a per-row failure), attaches `clip_id` positionally,
and writes the group's columns back into that fragment.
- **Commit** — one `LanceOperation.Update` naming only the fragments that were
written (see "Atomic column write" below). A run commits nothing when no fragment
had a pending row, or when the table has no fragments at all. Committing nothing
*while a fragment failed* is not one of those cases: it raises (see "Total
outage" below).

**Action pre-step.** Before the action fill exists, the driver binds the run's one
PCA basis with
[`resolve_action_pca`](../../../cosmos_curator/next/recipes/embeddings/action_pca.py)
(§5). This is an explicit `if` in
[`embed.py`](../../../cosmos_curator/next/recipes/embeddings/embed.py), not a
lifecycle hook the other modalities also carry. A `None` return is how a run with
no action data is skipped: with no clip carrying an `action_data_uri` there is
nothing to load and nothing to fit, so the run reports a skipped modality
**without ever starting an actor pool**. Because the basis is a required argument
of `build_action_fill`, "workers exist but no basis is bound" is not a
constructible state.

### NULL state semantics

There are no status columns. A row's state for one modality is read from its group
columns and the modality's applicability predicate:

| State | Group columns | Meaning |
|---|---|---|
| not applicable | all NULL | the predicate is false for this row (e.g. no `clip_uri` for image) |
| applicable, pending | all NULL | not embedded yet (a fresh row, or a prior per-row failure) |
| failed this run | all NULL | a per-row decode / geometry failure; retried next run |
| successful | all non-NULL | a complete group with a single producer identity |
| stale | all non-NULL | complete but the provenance differs from the configured producer; a run fails and requires `--reset-group <modality>` |

A **mixed** NULL / non-NULL group on one row is not a state the write path can
produce: a row's whole group is written by a single `update_columns` call, so it
transitions atomically. There is no partial-row check because there is no partial
row to find.

Pending and failed are intentionally the **same** persisted NULL state; per-row
failure reasons are logged and counted in the run summary, not stored as durable
error columns. A modality that has never run has **no** column at all (a distinct
condition from a present NULL — §3).

### Incremental fill and group reset

Incremental is the **only** run mode. There is no rebuild option: a run embeds
only applicable rows whose primary vector is still NULL, and already-complete rows
are untouched. New base rows appended after the pinned version stay NULL and are
picked up next run.

Replacing a group is therefore a separate **maintenance operation**, not a run
mode. An ordinary run is config-driven (§9); a replacement instead goes through
the reset-only
[maintenance runner](../../../cosmos_curator/next/recipes/embeddings/examples/run_embedding_pipeline.py),
whose `--reset-group <modality>` calls
[`drop_embedding_group`](../../../cosmos_curator/next/recipes/embeddings/columns.py)
for each named group, re-adds the columns empty, and exits without embedding
anything; the next ordinary run refills the group from scratch. Making it a
separate invocation is deliberate: a config field could be left in a YAML file and
silently re-executed by a scheduled run, and an accidental group replacement burns
a full corpus of GPU time.

Three measured behaviors of the drop are worth knowing before you use it:

- **Dropping an absent group is a no-op.** `drop_embedding_group` narrows to the
fields the schema actually carries and skips the call entirely when none are
present, so it returns `0` and the table's version is **unchanged** — a reset of a
modality that never ran leaves no trace in the version history. (Lance's own
`drop_columns([])` does not raise but does commit a version; the guard is what
keeps the no-op off the history.)
- **A partially present group drops only what is there.** `drop_columns` rejects a
name the schema does not carry and then drops *nothing*, so passing a partial
group whole would fail the entire reset instead of clearing what exists.
- **Lance reuses field ids after a drop / re-add cycle.** The re-added columns can
carry the same field ids the dropped ones had. Field ids are therefore not a
stable identity for a group across a reset; column names are.

The consequence to plan around is the **all-NULL window**: between the reset and
the refill the group is present and entirely NULL, which every consumer reads as
"not embedded yet". A partial refill is safe rather than mixed, because the drop
detaches the old data file — the fragments a refill never reaches read NULL, not
pre-reset values.

### Atomic column write (no staging tree, no tombstones)

The fill writes each group **directly into `clips.lance`** with no application
staging layer and no shuffle:

1. The driver turns the planned fragment ids into work items
(`ray.data.from_items`) and maps them over an actor pool, one fragment per
`__call__`. Nothing but integers leaves the driver.
2. Each worker opens the table at the pinned `read_version`, takes its fragment,
and **streams**: scan a batch of pending rows → embed it → hand it straight to
`update_columns`. Peak worker memory is one batch of vectors, not one fragment's
worth.
3. `LanceFragment.update_columns(reader, left_on="clip_id")` writes a **new column
file** holding only the group's columns for that fragment's existing rows, and
returns the fragment's updated **metadata** plus the field ids it rebound. The
worker sends the driver **one JSON string per touched fragment** — `_RESULT_SCHEMA`
is a single `large_string` column, so it cannot carry column data back at all
without changing type — which is what makes driver state O(touched fragments)
rather than O(rows) a structural guarantee instead of a convention.
4. The driver commits every written fragment in **one** `LanceOperation.Update`
against the pinned version, so the group's vectors and provenance become visible
together, atomically. `fields_modified` scopes the commit to this group's field
ids, and `transaction_properties` tags it (`kind=curator-next-embedding`, plus the
group name) so the commit is attributable in the table's version history.

**Why this never creates a tombstone.** Both write steps are additive at the
storage layer:

- `add_columns` with an all-nullable schema is **metadata-only**. Lance records the
new fields in the manifest and reads them as NULL for every existing row; no data
file is rewritten and no row is replaced.
- `update_columns` **rebinds field ids to a new file** rather than replacing rows.
The new file holds the group's columns for the fragment's existing rows in their
existing offset order; row addresses, row count, and every other column's file are
untouched. Because no row is superseded, Lance writes **no deletion vector**. The
previous column file stays on disk referenced only by older versions.

`update_columns` joins with **left-outer** semantics, so a fragment row absent from
the update table keeps its previous value. That is what makes it safe to write only
the *pending subset* of a fragment. A fragment with nothing pending is not
rewritten at all: the worker probes the scanner for a first batch before calling
`update_columns`, so an unchanged rerun of a fully embedded table commits nothing.

**Why `clip_id` is a sound join key here.** It is an ordinary persisted column, so
it survives the update (unlike a virtual row address, which is a physical
coordinate). The upstream leg makes no uniqueness guarantee, and Lance resolves a
duplicate key by taking one matching row's value, so this path **assumes rows
sharing a `clip_id` agree** — same source columns, hence the same computed value and
the same applicability. Under that assumption every duplicate receives the value it
would have computed for itself. Rows that share a key but disagree on their source
columns are malformed input: the write path neither detects nor reconciles them,
because doing so would cost a de-duplicating shuffle on every run to guard against
data the producer is not supposed to emit.

### How many fill workers actually run

The fill's concurrency is set by its **work-item count**, not by the actor pool's
size and not by the cluster's GPU count alone. `ray.data.from_items` emits one
block per work item up to a 200-block ceiling, and one block is walked by one
actor:

| Fragments visited | Blocks | Concurrent fragments |
|---|---|---|
| N ≤ 200 | N | N — one fragment per task, so a slow fragment delays only itself |
| N > 200 | 200 | 200 — each block carries ⌈N / 200⌉ fragment ids that one actor walks serially |

So the ceiling is `min(fragments visited, 200, free cluster slices)`. The concrete
consequences an operator hits:

- **400 fragments, 64 GPUs** → 64 concurrent image actors: blocks (200) exceed the
  cluster's slices, so the cluster is the binding term.
- **1 fragment, 64 GPUs** → **one** actor. Adding GPUs changes nothing, because
  there is only one work item. This is the failure mode to check first when a leg
  is slow with idle GPUs; the fix is the producer's fragment geometry, not a config
  knob here.

Every run logs the fragment count it found and the ceiling it derived, so the
number that actually capped the run is in the log rather than inferred from GPU
utilisation.

`override_num_blocks=len(fragment_ids)` would lift the 200-block ceiling. It is
deliberately **not** passed, and the trade-off is accepted rather than overlooked:
at the deployed 8-GPU shape the cluster term binds long before the block ceiling,
so the argument would add a knob with no effect while taking over Ray Data's own
block sizing. The cost of that choice is that a table with more than 200 fragments
cannot exceed 200 concurrent actors — revisit when one modality is given more than
200 usable cluster slices.

### Why not a higher-level Lance API

Per-fragment `update_columns` plus one `LanceOperation.Update` is the most
elaborate write in Curator Next, so the first question a reader should ask is why
a one-call API does not do the job. Every candidate was measured against the same
requirement — write **new values into existing columns of existing rows**, from
many workers, without moving vectors through the driver:

| Candidate | Why it cannot be used here |
|---|---|
| `merge_insert(clip_id).when_matched_update_all()` | Two independent disqualifiers. It takes a driver-side `RecordBatchReader`, so feeding it from Ray streams **every vector in the corpus through the driver process**; and it implements an update as delete-plus-append (`REWRITE_ROWS`), writing a deletion vector per touched fragment and reordering rows on a table `robot_action_split` owns and other legs read |
| `LanceDataset.merge(table, key)` | Add-only. A column that exists on both sides raises (`OSError: Column ... exists in both sides`) |
| `fragment.merge_columns` + `LanceOperation.Merge` | Add-only, demands the **complete** fragment list rather than the touched subset, and is marked "Internal API" upstream |
| `dataset.update()` | Takes SQL expressions, so it cannot write a vector |
| `add_columns(ReaderLike)` | Add-only and positional |
| `ray.data.Dataset.write_lance` | An append sink, not a column update — and broken on the current Ray / pylance pin |

The chosen path is the only one that is simultaneously column-local, atomic,
partially fillable, and distributed: a per-fragment
`LanceFragment.update_columns` keyed on `clip_id`, committed once as a
`LanceOperation.Update` naming only the touched fragments, as detailed under
[Atomic column write](#atomic-column-write-no-staging-tree-no-tombstones). The
table above is the write-primitive comparison in full; the upstream reference is
[Lance data evolution](https://lance.org/guide/data_evolution/).

### The quiescent-source contract

`clips.lance` is assumed **complete and quiescent** for the duration of a run: no
concurrent `robot_action_split` append, and no second embedding writer. This is a
first-class assumption of the design, not a detail of the commit — it is why the
pinned version is a correctness convenience (one schema per run, and a base
version for the `Update`) rather than a concurrency mechanism, and why the recipe
carries no coordination protocol. The concurrency subsection below says what
actually happens if the contract is broken in each direction.

### Fragment-level failures are skipped, not fatal

A fragment that raises for **any** reason is skipped rather than failing the
modality: the worker logs a WARNING naming the fragment id and the exception,
emits an error payload instead of its metadata, and the run commits every fragment
that did succeed. The per-modality summary line reports how many were skipped.

The reason to prefer this to failing fast is asymmetric cost. The alternative
aborts the whole modality on any single fragment failure and **discards every
other fragment's completed work** — on a GPU workload, potentially hours — for a
fault that is often transient.

No retry machinery is needed, because a skipped fragment was never written. Any
rows it still owed stay pending, so the ordinary pending predicate re-selects them
and the **next run refills them with no operator action**. Note that "skipped"
does not imply "had work": the fault window opens before the worker can tell
whether the fragment held a pending row at all. A skipped fragment
also contributes nothing to `selected` or `filled`: from the table's point of view
the run never visited it. Fragment-level failure therefore behaves exactly like
the per-row failure semantics of §8, instead of being a special catastrophic case.

Because a skipped fragment contributes to no row count, the count of skipped
fragments is itself carried on `ModalityResult.skipped_fragments` rather than only
logged. That is not a reporting nicety: it is the **only** field that distinguishes
a run which lost work from one which had none to do, and the total-outage refusal
below is expressed on it.

One consequence is accepted deliberately, recorded here so it is not rediscovered
as a bug: **intra-run retry becomes inter-run retry.** Because the worker catches
the failure, Ray never observes a task failure and its own task retry never fires.
A fault Ray would have retried within the run is retried by the *next* run
instead. A retry costs a re-scan of one fragment either way, but a run's
wall-clock no longer absorbs transient faults.

### Total outage: where the skip policy ends

The skip spends one fragment's work to keep the rest of the run's. That is a trade
only while there **is** a rest. A run that wrote **no** fragment while at least one
failed has kept nothing, left the group exactly as it was, and — because a skipped
fragment contributes to no row count — reports `selected=0, filled=0,
committed_version=None`: byte-for-byte the result of a run over an
already-complete table.
[`FillContractError`](../../../cosmos_curator/next/recipes/embeddings/fill.py)
refuses that case, on the driver, before any commit.

The discriminator is the **skip count**, not a ratio, because the two outcomes
differ in *kind* rather than in degree:

| written | skipped | Verdict |
|---|---|---|
| ≥ 1 | 0 | ordinary success |
| ≥ 1 | ≥ 1 | success; the skipped rows stay pending and refill next run — unchanged skip policy |
| 0 | 0 | success: nothing was owed (an idempotent re-run, or a table with no fragments) |
| 0 | ≥ 1 | **`FillContractError`** — nothing written while a fragment failed |

The last row deliberately claims less than "every fragment that had work failed",
because the driver cannot establish that. A fragment can fail while opening the
dataset or pulling its first batch — before it could report whether it had a
pending row — and a fragment with nothing pending returns no result at all. So a
**complete** table with one faulting fragment lands in that row too: zero written,
one skipped, nothing owed. Refusing it is still right (an unreadable data file is
a real fault, and a re-run either clears it or confirms it), but the message says
only what is known and does not blame lost work. The cost is that an idempotent
re-run is no longer *unconditionally* green: it is green unless a fragment faults,
and under a `max_fragments: N` cap a single fault is enough because `N` is the
whole visited set. The alternative — having workers report "I had pending rows"
so the guard fires only on a *known* loss — was rejected as strictly worse: an outage
that kills every fragment at open time reports no pending rows either, so the
guard would go silent on the most severe case it exists for.

Three shapes were rejected:

- **A skip-ratio threshold** (fail above *N%* skipped) is a knob whose only
  defensible value is "all of them", and a knob with one sensible value should not
  exist. An earlier iteration of this recipe carried a 50% action *drop*-ratio
  warning for the same reason and it was deleted, because the expectation is that
  every row is written.
- **Keying on the commit alone** ("a run that committed nothing fails") breaks
  idempotency outright: an unchanged re-run over a complete table legitimately
  commits nothing and must stay green. Distinguishing "nothing to do" from
  "everything failed" is the entire point, and only the skip count does it — it
  preserves idempotency for every re-run in which nothing faults, which the
  commit-alone rule does not.
- **Comparing skips against the fragments *visited*** (`skipped == len(fragments)`)
  looks equivalent but is strictly weaker and fails silently where it matters most:
  a fragment with nothing pending reports no result at all, so on a table that is
  99% embedded the skip count can never reach the visited count — the guard would
  go quiet on exactly the incremental top-up runs it exists for.

The outage is raised as `FillContractError`, the same type as a violated invariant,
and **the two prognoses are carried by the message text, not by the type**. They
genuinely differ — a contract violation means the run's own inputs are wrong and no
retry can fix it, whereas an outage is usually the environment failing every
fragment alike, where re-running is the first remedy — but that guidance only ever
reached an operator through the message, since the generic runtime reports every
exception a run raises identically as a `runtime` failure (below) and nothing in
the tree branched on the narrower type. A second type distinguishable only by
reading the source was judged not to earn its place; the outage message
therefore opens with "Re-run first", and the invariant messages name the input to
fix. The cost of the collapse is accepted openly: a caller that one day needs to
retry only outages would have to re-introduce the distinction rather than find it
hidden behind a sentinel field or a subclass by another name.

One systemic class never reaches this guard at all: a broken projection or row
filter. Those are the run's shared scan inputs, and the driver refuses them upfront
rather than letting every fragment skip on them (below). That includes the case
worth naming, because it looks like it should mis-fire here: a **group whose
columns were never added**. The pending filter names the group's primary vector, so
`explain_plan` rejects it with "No field named …" before any actor exists — a
config fault is never reported with the outage message's "re-run first" advice.

### `FillContractError`: what stops the run instead

[`FillContractError`](../../../cosmos_curator/next/recipes/embeddings/fill.py) is
the one exception the skip does not swallow. Its contract is deliberately broad:
**a failure the run must stop for rather than skip past**. That covers the total
outage of the previous section, plus a violated invariant of the fill itself — a
property of the run's own inputs, which no retry can fix — as opposed to the
environment and data faults a fragment is skipped for. Four checks raise it: the
driver's precondition check on the run's shared scan inputs (below), an embedder
that returned a different number of rows than it was given, an embedder whose
columns do not cast to the stored schema, and a fragment id that is absent at the
pinned version (the derived fragment list and the table disagree).

**Wrong row count and wrong schema are one fault, so they are treated alike.** Both
mean the embedder is broken and will fail every fragment identically; the schema
half surfaces only at the cast to the stored types, where a missing, renamed, extra
or *reordered* column raises a plain `ValueError` and an uncastable dtype or wrong
vector width raises `ArrowNotImplementedError` / `ArrowTypeError` — measured on
pyarrow 24, and worth stating because none of those is the `ArrowInvalid` one would
guess. Those three types are caught individually rather than through their common
`pa.ArrowException` base, which would also catch `ArrowMemoryError`: an allocation
failure during the cast is transient, and promoting it here would discard every
other fragment's work. Letting the schema half fall through to the
broad per-fragment handler still failed the run, via the outage guard, but replaced
one precise sentence with *N* skip warnings and a generic outage message.

**The inputs every fragment shares are checked once, on the driver.** The
projection and the row filter do not vary by fragment, so a broken one — a source
column that is missing, renamed, or absent from the projection; a predicate that
does not parse — is not a fragment fault at all: it would fail identically on every
fragment, and under the skip policy the modality would report a successful run over
an untouched group. `fill_embedding_group` therefore resolves the scan plan before
any actor is created, and raises. `LanceScanner.explain_plan` is what makes that
affordable: it resolves and optimizes the scan — which is where an unknown column or
an unparsable predicate is rejected — **without reading a row**, so a re-run with
nothing pending still reads nothing. It is also what leaves the per-fragment handler
free to be broad: by the time it runs, the faults still reachable there are
genuinely local ones.

Both obvious alternatives are wrong, which is why the distinction is a named
exception rather than an exception-type filter:

- **Excluding `ValueError` broadly from the skip** would make an ordinary
  *storage* `ValueError` fatal — defeating the skip policy exactly when it matters
  most, since transient object-store faults are the failures it exists for.
- **Excluding nothing** would let a broken embedder skip every fragment and leave
  an empty group. The run would still fail — the outage guard below catches
  "nothing written while fragments failed" — but it would be reported as a
  transient outage whose advice is to re-run, when no re-run can help until the
  embedder is fixed. The cost is misdiagnosis, not silence.

**It travels back to the driver as a payload, not as an exception.** Ray Data
replaces whatever a `map_batches` UDF raises with a `UserCodeException` wrapper and
does not carry the original object across the task boundary, so a worker's raise
reaches the driver as a `RayTaskError` that is *not* an instance of its own cause —
measured, not assumed. The worker therefore logs the violation at ERROR level and
emits a distinct **fatal** payload, and `fill_embedding_group` raises
`FillContractError` on the first such payload it sees, while the results are still
streaming in, so the fragments not yet started are abandoned rather than computing a
result nothing will commit.

**The classification does not depend on which batch violated the contract.** Only a
fragment's first scanned batch is embedded on the worker's own stack; Lance pulls the
rest itself, through an Arrow C data interface that carries an error *message* but not
the Python exception object — measured, not assumed: a `FillContractError` raised
there re-emerges from `update_columns` as a bare `RuntimeError`. Unhandled, that would
be read as an ordinary fault and the fragment would be SKIPPED, so a broken embedder
would skip every fragment and be misreported as a transient outage — but only on a
table whose fragments exceed one scan batch, which is every production table and no
small test fixture. The worker
therefore records the violation as the exception leaves the batch generator and
re-raises it on its own stack once `update_columns` returns, which is what keeps
fatal-versus-skip a property of the violation rather than of the scan geometry.

Being raised on the driver is what makes the type useful: it stops the modality
**before** any commit, so the previous version stays current and the failure is
reported by the runtime rather than discovered later by a reader. It subclasses
`ValueError` alongside the recipe's other pre-commit refusals (a stale group, a
missing basis, a failed commit), which keeps the family one catchable type — but
no exit code depends on that base any more. A run goes through the generic
[`run-pipeline`](../../../cosmos_curator/client/pipeline_cli/pipeline_runtime.py)
runtime, which catches every exception the run raises alike: with `--json` it
prints `{"ok": false, "error": "runtime", "message": …}` and exits **2**, and
without it the exception re-raises as an ordinary traceback. (A *config* fault is
distinct: it surfaces from `prepare_run` before Ray starts, so it never reaches
this code, and is reported as `error: "invalid"` — also exit **2**.) The one
surviving
`except ValueError -> exit 1` belongs to the reset-only maintenance runner, and a
reset runs no fill, so it covers only a missing or unreadable clips table.

### Provenance and staleness

Because each group records its producer identity (`embedding_text_model_id` /
`embedding_image_model_id` for the model modalities; `embedding_action_descriptor_version`
+ `embedding_action_pca_fingerprint` for action), a model or basis change is
detectable before a fill. Re-embedding a group under a new checkpoint is
`--reset-group <modality>` followed by an ordinary run. For action, a
`descriptor_version` bump is a stale group: the load refuses it and directs the
operator to reset, after which the next run extracts descriptors and fits a fresh
content-addressed basis together (§5); there is no separate "delete the basis"
step.

### Failure recovery, concurrency, and maintenance

- **Per-row failures** stay NULL and retryable; the next run's row predicate picks
them up.
- **Fragment-level failures** are skipped, not fatal (above): the surviving
fragments still commit, and the skipped ones refill on the next run.
- **Systemic failures** (a bad schema, a stale group, a missing PCA artifact, a
total action outage, a violated fill contract) raise on the driver **before** the
commit, so the previous committed version stays current and no partial group is
published.
- **A duplicated fragment result is refused.** Each fragment is a single work item,
so two results for one fragment mean a retried task's write was also counted;
committing two metadata versions of one fragment would make the outcome depend on
ordering, so `_collect` raises instead.
- A task or commit failure can leave **uncommitted, unreferenced** Lance column
files (orphans) inside the table directory. They are invisible to readers (absent
from the manifest) and are reclaimed only by the operator's own maintenance.
**The recipe never compacts, cleans, or deletes the shared table** — it owns
columns, not the table's lifecycle.
- **Concurrency** — the contract is quiescence (above), but the commit is
nonetheless tolerant of the one way it is likely to be broken by accident. The
recipe passes no `max_retries` to `lance.LanceDataset.commit`, so Lance's own
default (20 on the pinned pylance) applies and a commit that loses a race is
**rebased** onto the newer manifest. That is desirable rather than merely
tolerated: the transaction names specific fragments and specific field ids, so a
producer's append of *new* fragments cannot conflict with it. The replayed
`Update` still lands this run's fills, the appended rows keep their NULL group
values, and the next ordinary run fills them through the pending path. Two
concurrent *embedding* writers on one group would be a genuine conflict — they
collide on the same field ids — but that is what the quiescence contract forbids;
if it happens anyway, exhausting the rebases surfaces as a `ValueError` naming the
group and the version, not as a merged result.

---



## 8. Error handling and applicability

- A **per-row failure** (decode, deserialize, bad shape, model error, missing
media) leaves that row's whole group **all-NULL** on `clips.lance` — vector and
provenance together — with a logged count, and it is retried on the next run. No
partial or zero-filled vector is ever written: a zero vector would pass every
shape check and quietly corrupt a downstream centroid, and a half-written group is
rejected as corrupt (§7).
- A **malformed URL** is a per-row failure too, including one whose authority the
URI parser rejects outright. The per-backend cache key both media-reading legs
derive from `clip_uri` / `action_data_uri`
([`backend_key`](../../../cosmos_curator/core/utils/storage/storage_utils.py))
falls back to that URL's raw scheme-and-authority text instead of raising, so one
unparseable row costs its own row rather than the batch it arrived in. The key stays
sound: two URLs share it only when that raw text is identical, so no backend's
client is ever reused to serve another backend's reads.
- A clip can therefore have a complete group for some modalities and an all-NULL
group for others; a later read that requires a non-null vector omits it —
expected, not a bug. (A modality that never ran has **no** column at all — distinct
from a present NULL, §3.)
- **Applicability predicates** are pushed into the Lance scan so a modality selects
exactly the rows it will attempt:

  ```text
  text   := true                                            (every row)
  image  := clip_uri IS NOT NULL AND clip_uri != ''
  action := action_data_uri IS NOT NULL AND action_data_uri != ''
  ```

  On a table produced by `robot_action_split`'s current `CLIP_SCHEMA` (§2.1) both
  URIs are non-null on every row, since a failed clip is never committed to
  `clips.lance` in the first place — these predicates are then always true and
  cost nothing beyond the scan. They stay in place as a defensive contract check
  against any table where that guarantee doesn't hold (an older table, or a
  future producer), rather than being narrowed to assume it.

- **Action is Mecka-only and purely structural.** A clip is applicable iff it
carries a non-empty `action_data_uri`; there is **no** dexterous / `source_dataset`
registry gate. A row whose artifact turns out non-mecka or malformed is not
excluded from the denominator — it is decoded and rejected per row by the
extractor's geometry checks, leaving an all-NULL action group (retried next run).
Those geometric rejections are summarized once per batch, each reason carrying a
count of rejected **artifacts** **plus a bounded sample of the redacted artifact
URIs behind it**, so the artifact to inspect is named and not merely counted. The
unit is the artifact rather than the clip because the extractor's per-batch memo
derives a span's artifact once and one artifact can leave several view rows
pending; a clip key would name only one of them, and the driver's PCA-fit pass has
no clip key at all (§10.2).
A run that fails **every** selected action row **and** finds the action group
otherwise empty raises (a total outage, indistinguishable from a
misconfiguration); a partial per-row loss is left to the next run's pending
predicate with no threshold and no warning (§11.3).

`check_action_outcome` is the **per-row** total-outage check and is disjoint from
the per-fragment refusal of §7, not covered by it: it fires only once fragments
*were* written and committed — which is what makes `selected` non-zero — and every
row inside them still embedded to NULL. The two conditions are mutually exclusive
on `written == 0` versus `selected > 0`, which is a statement about counts and not
about exception types, so collapsing §7's refusal into `FillContractError` leaves
that proof untouched. Its "re-run first, transient S3 errors cause this" guidance
applies to both.

---



## 9. Configuration

[`EmbeddingPipelineConfig`](../../../cosmos_curator/next/recipes/embeddings/config.py)
(pydantic, `frozen`, `strict`, `extra="forbid"`). Embedding-specific only — no
clustering / curation knobs.


| Field                      | Default                                                 | Purpose                                                 |
| -------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| `schema_version`           | *(required)*                                            | config schema generation (`1`); pinned rather than defaulted, so a file written against a later generation is refused instead of being read under this generation's field meanings |
| `kind`                     | *(required)*                                            | `embeddings` — the discriminator the generic CLI routes on; it must match the registered kind name ("Running it", below) |
| `clips_lance_uri`          | *(required)*                                            | the shared clips table — **both** the read source and the write target |
| `storage_profile`          | `default`                                               | table / media / artifacts (one profile)                 |
| `modalities`               | `(text, image, action)`                                | `Modality` `StrEnum`; de-duplicated, first-seen order recorded. Execution is a fixed `text -> image -> action` cascade, so the tuple only records which modalities are enabled, not the order they run |
| `text`                     | `num_gpus=0.25, batch_size=256`                        | text resources (`TextEmbeddingConfig`). `batch_size` is **rows per fragment-scan batch** handed to the embedder, and is also passed on as the embedder's inner encode batch — one knob, so a scan batch and an encode batch cannot silently diverge |
| `image`                    | `num_gpus=1.0, num_cpus=2.0, batch_size=64, memory_gb=8, read_concurrency=32` | image resources (`ImageEmbeddingConfig`). `read_concurrency` sets how many clips one actor fetches at once; it widens the download only, since the decode it feeds is GIL-serialized, and 32 already sits at the knee where that decode floor overtakes the download term (§4.2). It may not exceed `batch_size` — enforced by a `model_validator`, not merely documented. `num_cpus` is declared rather than defaulted to prevent a silent oversubscription, and does **not** scale with `read_concurrency` for the same GIL reason. Only the image modality declares a `memory_gb` footprint (its decoded-frame batch is the binding constraint); the text modality relies on Ray Data's default |
| `action`                   | `num_cpus=1.0, batch_size=256, read_concurrency=32, pca_sample_size=50000`  | wrist-motion resources + PCA sample (`ActionEmbeddingConfig`, `pca_sample_size >= ACTION_DIM + 1`). `read_concurrency` sets how many distinct artifacts one actor fetches at once; it widens the network wait only, since the wrist geometry it feeds is GIL-serialized, so an actor's throughput ceiling does not move with it (§5.6). It may not exceed `batch_size` — enforced by a `model_validator`, not merely documented. Dividing the per-actor ceiling takes more actors, which for a fill is the table's fragment count (§7) |
| `model_weights_path`       | placeholder bucket                                     | base for `download_models`                              |
| `max_fragments`            | `None`                                                 | dev cap on the **fragments** each modality visits; does not narrow the PCA candidate scan. Fragment-granular rather than row-granular because the unit of work *is* a fragment: a row cap could not be honoured across independent workers without a shared counter |

There is no `rebuild` field. Replacing a group is the `--reset-group` maintenance
operation (§7), not a run mode, so a destructive replacement cannot be left
in a YAML file and re-executed by a scheduled run. The config model is
`extra="forbid"`, so an old file still carrying `rebuild:` (or the removed
`text.encode_batch_size:`) fails loudly at assembly rather than being ignored.


There is no worker-count or write-concurrency knob. Each modality's pool floors at
one actor and grows to whatever the cluster slices and the work-item count allow
("How many fill workers actually run", §7), so a `column_write_concurrency`-style
setting would only duplicate what the resource request and the table's fragment
geometry already express. `batch_size` is the only fill-shaping knob, and it shapes
**rows per scan batch**, not fragments per task: a Ray work item is pinned at
exactly one fragment, which bounds a worker's in-flight memory to one fragment's
batch. Up to the 200-block ceiling that also gives one block per fragment, so a
slow fragment delays only itself; above it a block carries several fragment ids
that one actor walks serially.


There is deliberately **no** `source_lance_uri`, no side-table URIs, and no
PCA-artifact URI: the recipe reads and writes the one `clips.lance` and always
derives the action-PCA directory (`clips.lance__action_pca/`) from it. There is
also **no** `action_format` knob — production artifacts are ACT2 `.bin`; a legacy
`.pickle` is rejected by the action reader without deserializing it on workers.

### Running it

A run is a config file plus the generic Curator Next runtime — there is no
long-form flag surface for it. The recipe is registered as pipeline kind
`embeddings`
([`pipeline_kind.py`](../../../cosmos_curator/next/recipes/embeddings/pipeline_kind.py)),
and one resolver
([`resolve_config`](../../../cosmos_curator/next/recipes/embeddings/config.py))
serves `validate`, `render`, `schema`, and the run, so the four surfaces cannot
disagree about what a file means. The generic contract is
[pipeline-configs.md](pipeline-configs.md).

```bash
# discover: an editable YAML template (--json adds required-field metadata),
# then every field with its default
cosmos-curator pipeline template embeddings
cosmos-curator pipeline schema embeddings

# host-side pre-flight: validate, and print the resolved config that will run
cosmos-curator pipeline validate embeddings.yaml --set max_fragments=2
cosmos-curator pipeline render embeddings.yaml

# execute inside the runtime environment: bare, local Docker, managed Ray
pixi run --as-is run-pipeline embeddings.yaml
cosmos-curator local launch -- pixi run --as-is run-pipeline /config/embeddings.yaml
cosmos-curator slurm ray submit CLUSTER_CONFIG -- pixi run --as-is run-pipeline RECIPE_CONFIG

# maintenance only, never a run mode (§7): reset a group, then re-run to refill
python -m cosmos_curator.next.recipes.embeddings.examples.run_embedding_pipeline \
    --clips-lance-uri s3://bucket/run/clips.lance --reset-group text
```

`--set PATH=VALUE` is accepted by `validate`, `render`, and the run alike. An
assignment is applied to the loaded mapping **before** validation, so an
overridden value is checked by exactly the rules a written one is, and each value
is parsed as YAML rather than kept as text — which this strict model needs: a
quoted `"2"` is rejected for `max_fragments`. A config launched under Docker or
managed Ray, and every path inside it, must be written as paths the runtime
environment can see (`/config/...` for the default local mount).

---



## 10. Execution model in depth

### 10.1 Three independent modalities, run sequentially

`run_embedding_pipeline` runs the enabled modalities **one after another** on the
driver — text, then image, then action — not concurrently. "Independent" here
means *decoupled and separately re-runnable*, **not** simultaneous:

- Each modality is itself a **whole-cluster** Ray Data job that already
  parallelizes across all workers and saturates the GPUs. Running two model
  modalities at once would contend for the same GPUs and memory — lowering
  throughput and risking OOM, not raising it. The parallelism that matters is
  **within** a modality, not across them.
- **One commit per group keeps the version history legible.** Each modality's fill
  ends in a single `Update` scoped to its own field ids, so a sequential run leaves
  one attributable commit per modality. Concurrency here would buy nothing: the
  groups are disjoint, so the writes do not compete for correctness, only for the
  same GPUs.
- The action modality resolves its **PCA basis on the driver** before any compute
  starts (§5), so its fill cannot begin until that decision is made.
- Each modality re-opens the latest committed version, so image and action see the
  columns text just added; and sequential execution keeps failure isolation and
  the per-modality fill logs (§7) readable one modality at a time.

### 10.2 No corpus materialization

Nothing materializes the corpus. Deriving the work reads the manifest, and each
worker scans its own fragment (§2.4), so the largest thing in memory at any moment
is one batch of one fragment's rows.

The action leg's PCA fit is the only step that could plausibly need the whole
corpus, and it avoids it by **sampling on URI before extracting**: a narrow
`action_data_uri` scan, ranked by `sha256(uri)`, extracts only a bounded candidate
set (§5). The fit therefore reads a bounded number of artifacts, and the fill
decodes each artifact inside the worker that owns its row.

The alternative — extract every descriptor, then sample — needs the full corpus
resident to pick the sample, and makes the fit a barrier the whole leg waits behind.
Sampling first costs one property: because some sampled artifacts fail to read, the
fitted population, and therefore the fingerprint, differs from what a full-corpus
sample would have produced. Bases are immutable and content-addressed, so that is a
*different* basis, not a corrupted one.

### 10.3 Where models load

Every modality runs through the same `_FragmentWorker` actor pool
(`ActorPoolStrategy(min_size=1)`). The worker constructs its embedder in `__init__`,
so a model loads **once per actor** — never per fragment, per batch, or per clip.
The pool pins no maximum: Ray places at most one actor per free resource slice, so
the cluster's GPU (or CPU) count is one of the ceilings, with no driver-side
capacity check. The other is the work-item count (§7), which is what makes a
fragment-poor table starve a large pool.

| Modality | Resource request | Why |
|---|---|---|
| text (BGE) | `num_gpus=0.25` | the model is small enough to pack several actors per GPU, and loading it is the expensive part — hence one load per actor |
| image (DINOv2) | `num_gpus=1.0`, `num_cpus=2.0`, `memory=8 GiB` | a whole GPU per actor plus a declared host-memory footprint, because the decoded-frame batch is the binding constraint. The CPU reservation covers one GIL-serialized decode stream plus the serial preprocess and forward — a fixed budget rather than one that grows with `read_concurrency` (§4.2) — and is declared rather than left to Ray's default so an unreserved actor cannot oversubscribe the node |
| action | `num_cpus=1.0` | ACT2 decode plus numpy geometry, no GPU; the small PCA basis is pickled into the actor by value |

Under `ActorPoolStrategy` the UDF must be a **callable class**, not a bound method —
Ray Data rejects the latter outright. That applies to the fragment worker and
equally to the action leg's PCA candidate extractor, which is why the extractor is a
class with a `__call__` and is handed to `map_batches` as a class plus
`fn_constructor_kwargs`.

### 10.4 Performance choices

- **One frame per clip** — a clip is one continuous subtask, so its frames are
  highly correlated; one frame answers "different place / object" at a fraction
  of the decode + inference cost (§4.2).
- **Batched inference** — `batch_size` amortizes GPU kernel launch and Python
  overhead across many clips.
- **Action descriptor computed once per `action_data_uri`** and fanned out to a
  span's views via a per-batch memo (§2.1, §5) — a multi-view span is decoded and
  geometry-processed once, not once per camera.
- **PCA fit sample de-duplicated on `action_data_uri`** so a multi-view span
  contributes one vote to the basis, not N (§5).
- **Action artifact reads overlapped, and served by one request** — the per-artifact
  cost is ~94 % network, so `action.read_concurrency` hides the wait while
  `download_object_as_bytes` drops a round trip on any object below the multipart
  threshold. At or above that threshold the same branch *adds* one round trip (a
  probe GET whose body is discarded unread before the managed transfer re-resolves
  the size), which large-object callers such as the video stages pay. Neither
  touches the GIL-held geometry, which is the per-actor floor (§5.6).
- **A fragment with nothing pending is never rewritten** — the worker probes its
  scanner before writing, so an unchanged re-run over a fully embedded table reads
  metadata and stops (§7).

---

## 11. Re-run, reset, and failure scenarios

A quick reference consolidating §5, §7, and §8.

### 11.1 Re-run scenarios

| Scenario | Do | Effect |
|---|---|---|
| second run, nothing new | run again | every applicable row's primary vector is already non-NULL, so the selection is empty, no fragment is rewritten, no commit; idempotent |
| new clips appended to `clips.lance` | run again | the new rows read their `embedding_*` columns as NULL, so the next run selects and fills only them; action loads the existing basis by fingerprint |
| partial failure last run (NULL groups, skipped fragments, or a modality that raised before commit) | run again | already-complete rows are skipped; only the still-NULL rows fill. No operator action is needed to retry a skipped fragment (§7) |
| model swap (text / image checkpoint) | `--reset-group <modality>`, then run | validation would otherwise refuse the group as stale; the reset empties it and the run recomputes it, recording the new `*_model_id` |
| descriptor-semantics change (e.g. `dual-wrist-v2` → `dual-wrist-v3`) | `--reset-group action`, then run | an empty group is what makes the leg fit a fresh content-addressed basis, so the group and its basis are recomputed together; a stale version is otherwise refused at load (§5) — no "delete the basis" step |
| change fusion weights / normalization downstream | *not a re-embed* | applied at consumption by the curation leg; the stored scale is fixed (§6) |

### 11.2 Why replacement is a separate invocation

There is no `rebuild` run mode; §7 describes `--reset-group` in full. The
short version, and the reason it reads as an extra step rather than a config
field: a field left in a YAML file can be silently re-executed by a scheduled run,
and an accidental group replacement burns a full corpus of GPU time. Requiring two
deliberate invocations makes that cost impossible to pay by accident.

The `rebuild` field's other behaviors do not need replacing. A reset already
NULLs every row, so the "recompute rows that are no longer applicable" case
collapses into the ordinary pending path, and `--reset-group` composes with a
`max_fragments` cap (a capped refill simply leaves the rest pending) where
`rebuild` had to forbid the combination.

### 11.3 Failure taxonomy

| Scope | Cause | Result | Reported as |
|---|---|---|---|
| one row | missing / undecodable media (image); unreadable / malformed / non-mecka artifact (action) | that row's whole group left **all-NULL**; retried on the next run (the pending predicate still selects NULL rows). This does **not** make the run incomplete: the row was attempted and the NULL group was committed — contrast the skipped fragment below, which was never attempted | logged count; counted in `failed` |
| one fragment | any exception inside a worker that is not a `FillContractError` — a transient object-store fault, an unreadable data file — **while at least one other fragment was written** | the fragment is **not written** and is excluded from the commit; its rows stay pending and the next run refills them. The commit stands, but the run then **raises**, so it is not reported as finished | WARNING per fragment, plus `ModalityResult.skipped_fragments`; the run ends in `IncompleteRunError` — a `RuntimeError`, deliberately outside the `FillContractError` family below, raised *after* the commit by the pipeline-kind adapter (§7, §15) |
| one modality does nothing | nothing pending in the fragments the run visited, the table has no fragments, or no clip carries action data — **and no fragment failed** | no commit **from the fill**; on a modality's first run the schema widening (§7) has already committed | `committed_version=None`, `skipped_fragments=0`. The run-level `ending_version` is the widening's version when it added columns **and no other modality committed** — `clips.lance at v<N> (columns added, no fill committed)`; a sibling modality that did commit supersedes it, and `null` (`clips.lance unchanged`) is reported only when the run committed nothing at all |
| whole modality fails loudly | a `FillContractError` — nothing written while at least one fragment failed; an embedder that changed a batch's row count or returned columns that do not cast to the stored schema; a fragment absent at the pinned version; plus a missing / renamed / mistyped source column; a stale or multi-producer group needing `--reset-group`; a missing referenced PCA basis; the same fragment reported twice; every action row failing with the group otherwise empty (per-row total outage); a commit that could not be rebased | raises on the driver; **no fill commits**, so no vectors land and the last data-bearing version stays current — though a first run's schema widening (§7) has already committed the empty columns, which a re-run then finds present and adds nothing to. One exception: the per-row action total outage is checked *after* its own commit, so that group's version stands | exception (`ValueError`, or its `FillContractError` subclass) |

The **text modality never fails a row** — `task_name` / `subtask_name` are
non-null and an empty string still embeds to a valid vector (§4.1).

Note what separates row three from the total-outage case in row four: it is
the **skip count**, not the commit. Both commit nothing, and a skipped fragment
contributes to no row count, so `selected` and `filled` are zero in both. Only
`skipped_fragments` tells a run that lost work from a run that had none to do —
which is why the distinction is carried by the exit code and not only by a log
pattern (§7).
Rows one and two both *do* commit: a failed row is still written (as all-NULL),
and row two is by definition a run where another fragment succeeded.

---

## 12. Future work: an optional video / temporal modality (Cosmos-Embed1)

This section is **forward-looking design, not a committed feature**. It records
what a fourth *video / temporal* modality would look like, why it is
deliberately absent today, when it would earn its place, and — if it is ever
added — the model ([Cosmos-Embed1](#16-glossary)) and interface it would use.
Nothing here ships in this milestone.

### 12.1 Why there is no video modality today

The three current axes already cover *task*, *appearance*, and *motion* (§1.1).
Two of them make a temporal video embedding redundant for the default egocentric
path:

- **Motion is already owned by the action leg**, and owned *better*. The action
  descriptor is a deterministic, geometrically-grounded encoding of dual-wrist
  pose from the ACT2 artifact (§5, §5.3) — not a model guess. A video model
  would re-encode that same motion signal statistically, at far higher decode +
  inference cost, and its output would **correlate with the action axis**,
  eroding the orthogonality the whole design rests on (§1.1–§1.4).
- **Appearance is already owned by the image leg.** A clip is one continuous
  subtask from a head-mounted camera, so its frames are highly correlated; the
  first frame answers "different place / object?" cheaply (§4.2).

A video embedding would therefore correlate with *both* existing axes at once.
That is exactly why multi-frame / video embedding was rejected as the image
modality's implementation: a temporal model is **not** a drop-in replacement for
DINOv2 on the `embedding_image_*` group. Any video axis must be *additive*, not a
substitution.

### 12.2 When a video modality would earn its place

A temporal axis adds signal the three current axes cannot, in these cases:

| Situation | Why the three axes fall short | What a video axis adds |
|---|---|---|
| **Action artifact absent / non-dexterous** | a clip with no `action_data_uri` is not applicable to the action modality (§8), so it has no action vector and motion similarity is unavailable | the only motion-aware axis left for those datasets |
| **Cross-corpus dedup / retrieval** | the classic split-annotate pipeline already stores clip-level video vectors (`ce1_embd_*` / `iv2_embd`, see `docs/curator/reference/video-pipelines.md`) | a *comparable* clip-level video vector so egocentric clips can be joined against a web-video corpus embedded the same way |
| **Long or multi-subtask clips** | one frame under-represents appearance that changes over time | a temporal summary across the clip's frames |

In every case it is a **new, opt-in modality** — a new `embedding_video_*` column
group with its own dim, provenance, and lifecycle. The per-modality column-group
storage shape (§3) already supports this: adding the video
group leaves the `embedding_text_*`, `embedding_image_*`, and `embedding_action_*`
columns untouched, and a consumer tests the video group's presence exactly as it
does the others (§3).

### 12.3 Interface impact if it is added

An `embedding_video_*` column group would follow the same add-and-fill contract as
the others (§7), with these differences a consumer and the curate leg must absorb:

- **Width.** `fixed_size_list<float32, VIDEO_DIM>` where `VIDEO_DIM` is CE1's
  variant width (256 or 768) — it breaks the current uniform 384-d text/image
  width, but the scale contract (§6) already normalizes each modality
  independently, so a fourth width is fine. CE1's video vectors are unit-norm
  (cosine), like text and image — unlike action's raw-PCA Euclidean space.
- **Multi-frame read.** Unlike the image leg's single-frame decode (§4.2), a
  video model needs N frames (CE1 = 8), so the frame reader would use a
  multi-timestamp `SamplingSpec` instead of the one-timestamp read, at higher
  decode cost.
- **Applicability.** Applicable where `clip_uri` is present (non-null and
  non-empty), exactly like the image group (§8); a clip with no media leaves its
  video group all-NULL.
- **Fusion.** The (future) curate leg (§1.6) must weight a fourth axis; because
  it correlates with image and action (§12.1), its fusion weight would typically
  be **low or conditional** (e.g. only for the §12.2 datasets), not equal.

### 12.4 It stays off by default

Even if implemented, the video axis is **opt-in**. The three-axis design remains
the recommended egocentric-curation path because it keeps the "one independent
question per modality" invariant (§1.1) clean. The video axis exists only for the
narrow cases in §12.2, and its correlation with the existing axes is a feature
trade-off the operator must choose deliberately, not a default.

---

## 13. Future work: rot6d / rot9d rotation channels (action leg)

This section is **forward-looking design, not a committed feature**. It records
why the action descriptor uses **rotation vectors** today, what **rot6d** and
**rot9d** mean in the downstream training stack, when rot6d might
earn an experiment, and what switching would cost. Nothing here ships in this
milestone.

### 13.1 Why rotvec today

The `dual-wrist-v3` descriptor (§5) encodes each resampled wrist station as
three position channels plus **three rotvec channels** (after per-frame unwrap).
That choice is tuned for **curation geometry**, not for model-training action
chunks:

- **Minimal degrees of freedom.** A wrist orientation has three rotational
  degrees of freedom; rotvec is a 3-component encoding of `SO(3)` after unwrap,
  without quaternion double-cover ambiguity (§5).
- **Sufficient for dedup.** Nearest-neighbor search on PCA-reduced action
  embeddings only needs a stable, monotone-ish distance on *motion shape*.
  Euclidean distance on unwrapped rotvec tracks, resampled to fixed stations,
  is enough for that purpose; rot6d buys no extra orthogonality against text or
  image.
- **Fixed 600-d budget.** Two arms × 50 stations × (3 pos + 3 rot) = 600.
  Replacing rot with rot6d would widen the raw descriptor unless station count
  or position channels are cut elsewhere.

### 13.2 What rot6d and rot9d are

Cosmos training stacks store wrist rotations for **action prediction**, not for Lance dedup. There the rotation format is configurable:

- **rot6d** — the first two rows of the 3×3 rotation matrix stacked as six
  continuous values (Zhou et al.); common in causal / chunk action configs.
- **rot9d** — all nine matrix entries; default in some posttrain paths.
- **axis-angle / rotvec** — the same 3-vector as here (`pose_utils` names it
  `axisangle`); not the default training tensor in most of those configs.

Those formats optimize **differentiability and continuity inside a learned
action head**, not the deterministic fixed-length descriptor this embed leg
writes. Curation does not need bit-identical parity with a training tensor
unless an operator explicitly wants export alignment (§13.3).

### 13.3 When rot6d would earn its place

Rot6d remains **off by default**. Consider a `dual-wrist-v4` (or similar) only
if evidence shows rotvec is the bottleneck:

| Trigger | Rationale |
|---|---|
| **Frequent π-singularity artifacts** | unwrap (§5) removes most ±π flips, but angle π still has a residual two-fold ambiguity; if many clips sit near that singularity, rot6d's matrix parameterization can behave more smoothly under Euclidean distance |
| **Ablation on action NN quality** | offline study (e.g. held-out duplicate labels) shows rot6d-PCA beats rotvec-PCA on motion-similarity tasks that matter for dedup |
| **Training-stack alignment** | an operator wants action embeddings whose raw channels match imaginaire4's rot6d action chunks for direct fine-tuning or joint retrieval — cosmetic for dedup, operational for export |

Each case is an **opt-in descriptor bump**, not a silent swap: `DESCRIPTOR_VERSION`
must change, PCA refits on the new width (~900-d if rot channels go 3→6 with
the same 50 stations and two arms), and action rows re-embed from scratch
(`--reset-group action`, then a run — §11).

### 13.4 Why not rot9d

Nine channels per station restate the same `SO(3)` element with redundancy and
orthogonality drift unless re-projected every frame. Within a fixed descriptor
budget, rot6d is the usual continuous compromise; rot9d is unlikely to improve
curation distances enough to justify 50% more rotation channels versus rot6d.

### 13.5 It stays rotvec by default

The three-modality design keeps action as deterministic geometry (§1.1). Rotvec
+ unwrap is the committed path for this milestone because it is **enough for
motion dedup** at the lowest channel cost. Rot6d is documented so a future
change is deliberate — versioned, re-fit, re-embedded — not discovered by
accident in a downstream notebook.

---

## 14. FAQ

- **Why three embedding modalities?** Task, appearance, and motion are
  independent similarity axes — §1.
- **Why two text embeddings?** Task vs subtask are different semantic levels kept
  separately comparable — §4.3.
- **Do the two text vectors embed the same data?** No — two independent strings,
  same model, one pass — §4.1.
- **Why only one image frame?** A clip is one correlated subtask; one frame is
  enough and far cheaper — §4.2 / §10.4.
- **Why is there no video embedding? Could we add Cosmos-Embed1?** A temporal
  video vector would re-encode motion the action leg already owns and appearance
  the image leg already owns, correlating with both. It is not a replacement for
  either leg; it is possible only as an opt-in *additive* fourth modality
  (Cosmos-Embed1 preferred over InternVideo2) for datasets without action
  artifacts or for cross-corpus dedup — §12.
- **Why PCA, and why 97 dimensions?** Reduce the redundant 600-d descriptor to a
  ~90%-variance basis where Euclidean ≈ motion difference — §5.
- **Why rotvec instead of rot6d for the action leg?** Minimal 3-DOF encoding
  after unwrap is enough for motion dedup; rot6d/rot9d are for downstream
  training continuity (imaginaire4 action chunks), not required for Lance
  similarity — §5 / §13.
- **Why per-modality column groups instead of separate tables?** Co-locating each
  modality on the one `clips.lance` row gives independent presence and
  regeneration with no modality-to-modality join — §3.
- **How do the modalities realign?** They share one row, so a clip's text / image /
  action vectors are already aligned — there is no realignment step. `clip_id` is
  both the row's identity and the key each fill joins on — §2.1, §3, §7.
- **Can I regenerate only one modality?** Yes — `modalities` selects which run;
  each fills its own `embedding_<modality>_*` group — §9.
- **Can I re-run safely?** Yes — a re-run fills only still-NULL rows, which is
  also how a skipped fragment is retried; replacing a whole group is the separate
  `--reset-group` operation — §7, §11.
- **A fragment failed. What do I do?** Nothing — the surviving fragments committed
  and the next ordinary run refills the skipped rows. Read the WARNING lines to
  decide whether the cause was transient — §7.
- **The run failed with "no fragment was written while N fragment(s) failed". What
  now?** Re-run it first. Nothing was committed, so the group is untouched, and a
  transient storage fault fails fragments alike with no in-run retries. If it
  persists, the per-fragment WARNING lines name the actual cause. Note this also
  fires when the table was *already* complete and a fragment merely faulted — the
  message does not claim work was lost, only that nothing was written — §7.
- **Why are some clips NULL / missing in one modality?** A modality leaves the
  row's group all-NULL when it cannot embed it (missing media / bad artifact) or
  the row is inapplicable; a modality that never ran has **no** column at all; a
  read filters on the non-null vector — §3 / §8.
- **Can I run without a GPU?** The model modalities need staged weights and are
  meant for GPU (CPU works but is slow); the action modality is CPU-only.
- **What if I delete a PCA `.npz`?** Bases are immutable and content-addressed, so
  deleting one that live rows reference makes the next action run fail with a
  clear missing-basis error. To refit, run `--reset-group action` and then re-run:
  an empty group fits a fresh basis under a new fingerprint and recomputes the
  group with it — §5 / §11.

---

## 15. Limitations

- **Three distinct states, read carefully.** A modality's group can be *absent*
(its columns do not exist because the modality never ran — test **column
presence**), *present but NULL* (not-yet or could-not embed — filter
`<vector> IS NOT NULL` for embedded rows only, §3.6), or *complete*. A plain read
returns present-but-NULL rows too; absence is a schema fact, not a row value.
- **Single-writer per group.** The design's contract is that `clips.lance` is
quiescent for a run (§7). Two embedding runs filling the same group are therefore
unsupported, and Lance would collide on the same field ids anyway. A concurrent
base-row *Append* is the one violation that is benign: `fields_modified` scopes
the commit to the group's own field ids, so Lance rebases the update onto it and
the appended rows simply read NULL until the next run.
- **A partial fragment loss commits the survivors and then fails the run.** The
commit itself stands, by design (§7): the fragments that were written are durable,
the skipped rows stay pending, and the next run refills them. But the adapter
([`recipes/embeddings/pipeline_kind.py`](../../../cosmos_curator/next/recipes/embeddings/pipeline_kind.py))
raises `IncompleteRunError` instead of returning, so
[`run-pipeline`](../../../cosmos_curator/client/pipeline_cli/pipeline_runtime.py)
exits non-zero and a scheduled job cannot record the attempt as finished while work
is still owed. There is deliberately no skip-ratio threshold — one skipped fragment
is enough.
The cost is the **structured summary**: a raise is reported as a fault (§7), so the
payload is replaced rather than joined by it. The exception's message *is* the run's
own summary line, so `ending_version`, the per-modality `filled/selected`, and the
skip total survive as prose — but a caller that wanted them as fields has to parse
that line, and the table URI and per-modality `committed_version`s are not in it.
The raise also fires when the skipped fragments owed no rows at all, because a
fragment can fault before its scan could say (§7) — the same unknowability that
makes the total-outage guard claim no lost rows. Since the fill is idempotent, the
remedy is the same cheap one either way: run it again.
- **A faulting fragment on an already-complete table fails the run.** The outage
guard cannot tell a fragment that failed *with* pending rows from one that failed
before it could say, so a re-run over a fully embedded table raises instead of
exiting 0 if any fragment faults (§7 explains why claiming less is the right
trade). The fault is real, so the exit is not spurious — but it is reported as
"nothing written while a fragment failed", not as lost work.
- **An outage in one modality hides the earlier modalities' summary.** The driver
returns `EmbeddingRunResult` only after every requested modality, so a raise in the
image leg discards the accounting for a text commit that already landed durably.
The commit itself is safe and visible in the table's version history; only the
summary line is lost. Pre-existing for a violated invariant, but an outage is
environmental and so more likely to strike mid-run.
- **Maintenance is operator-owned.** The recipe never compacts, cleans, or
restores `clips.lance`; orphaned uncommitted column files from a failed run are
reclaimed only by the operator's own version cleanup (§7). A whole-table
restore to roll back a bad group would also drop later base appends and other
modality commits — reset the group and refill forward instead.
- The hand skeleton joint count (21) is **dataset-specific**; a different
skeleton would reject every clip. Parametrize only when such a dataset appears.
- Reads and writes go to **one** `storage_profile`; a split read/write profile is
not modeled in this milestone.

---



## 16. Glossary


| Term                                    | Meaning                                                                                                                                                                                                                                                                |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PCA**                                 | Principal Component Analysis — a linear dimensionality reduction that projects vectors onto the orthogonal directions of greatest variance (fit here via SVD). The action leg uses it to reduce the 600-d wrist descriptor to 97-d while keeping ~90% of the variance. |
| **SVD**                                 | Singular Value Decomposition — the matrix factorization used to compute the PCA basis (`numpy.linalg.svd`); the top singular vectors become the PCA components.                                                                                                        |
| **Explained variance**                  | The fraction of the sample's total variance captured by the retained components; the fit targets a rough 90%.                                                                                                                                                          |
| **Embedding**                           | A fixed-length numeric vector representing a clip along one modality, so that "similar" clips are "close" under a distance.                                                                                                                                            |
| **Modality**                            | One of the three independent similarity axes: `text`, `image`, `action`.                                                                                                                                                                                               |
| **Redundancy axis**                     | A dimension along which two clips can be "the same" (same task text / same appearance / same hand motion); curation de-duplicates along all three.                                                                                                                     |
| **BGE**                                 | BAAI General Embedding — the small (384-d) sentence-embedding model (`BAAI/bge-small-en-v1.5`) used for the text leg.                                                                                                                                                  |
| **DINOv2**                              | A self-supervised vision model (`facebook/dinov2-small`, 384-d) used for the image leg; its features describe visual structure rather than text-alignment.                                                                                                             |
| **Cosmos-Embed1 (CE1)**                 | An NVIDIA video-text embedding model (`cosmos_curator/models/cosmos_embed1.py`) that maps an 8-frame clip and text into one shared space; variants `224p` / `336p` / `448p` give 256-d or 768-d vectors. Used by the classic split-annotate pipeline; the preferred model **if** an optional video modality is ever added here — §12.                     |
| **InternVideo2 (IV2)**                  | A video-text embedding model (`cosmos_curator/models/internvideo2_mm.py`, 4 frames, 512-d) also offered by the classic pipeline; not the preferred model for a future video axis (Cosmos-Embed1 is) — §12.                                                              |
| **Video / temporal modality**           | A hypothetical fourth similarity axis that would embed several frames of a clip over time. Deliberately absent (it correlates with both the image and action axes); documented as opt-in future work — §12.                                                              |
| **CLS token**                           | The transformer's pooled / class-token output taken as the whole-image embedding; the image leg uses DINOv2's `pooler_output`.                                                                                                                        |
| **L2-normalized / unit-norm**           | A vector scaled to length 1 (Euclidean / L2 norm = 1); cosine distance between unit vectors is a plain dot product.                                                                                                                                                    |
| **Cosine distance**                     | `1 - cos(a, b)`; direction-based similarity used to compare the (unit-norm) text and image vectors.                                                                                                                                                                    |
| **Rotvec**                              | Rotation vector — a rotation as a 3-component axis-angle vector (axis scaled by angle); a minimal 3-component alternative to the 4-component quaternion. It removes the quaternion sign ambiguity in normal use (after the per-frame unwrap), but is not double-cover-*free*: a residual double representation remains exactly at angle π (see §5).                                                                                                          |
| **rot6d**                               | Six-dimensional continuous rotation encoding (first two rows of the 3×3 matrix, Zhou et al.); Optional future alternative to rotvec in the action descriptor — §13.                                                                                                              |
| **rot9d**                               | Nine-dimensional flattened rotation matrix; Not recommended for the embed descriptor — §13.                                                                                                                             |
| **Quaternion (xyzw)**                   | A 4-component rotation representation; `xyzw` is the scalar-last ordering (SciPy's default), used with no reordering.                                                                                                                                                  |
| **Double cover**                        | The 2:1 property that a quaternion `q` and its negation `−q` denote the same `SO(3)` rotation; the source of the ~2π sign-flip artifact around a 180° sweep that the rotvec + unwrap step removes (see §5).                                                              |
| **Descriptor**                          | The raw 600-d dual-wrist motion vector, before PCA reduction (see §5).                                                                                                                                                                                                 |
| `dual-wrist-v3`                         | The current `DESCRIPTOR_VERSION` string fingerprinting the descriptor's *semantics* (v3 adds rotvec unwrapping over v2); a persisted PCA basis records it and refuses reuse on mismatch.                                                                                |
| **ACT2**                                | The self-describing binary action-artifact format decoded by `action_binary.decode_action_bin`; its header carries `spec_name`.                                                                                                                                        |
| **mecka**                               | The ACT2 `spec_name` for the dexterous dual-wrist datasets this leg can embed (see `ACTION_BINARY_SPEC_BY_DATASET`); gates the `WRIST_FRAME_ALIGN_MECKA` alignment.                                                                                                    |
| **LeRobot**                             | The upstream episodic robot-dataset layout that `robot_action_split` cuts into the clips table.                                                                                                                                                                        |
| **span / view**                         | A `span` is one cut segment of an episode; a multi-camera source yields several `view`s per span, each a distinct `clip_id` but sharing one action artifact.                                                                                                           |
| `clip_id`                               | The stable per-(span, view) identity minted by `robot_action_split` (a SHA-256 digest of `span_group_id`, optional `view_name`, and bitrate — not a readable string). Embeddings carry it on each row, join each fill on it, and consumers use it to join `clips.lance` to *other* keyed tables. Uniqueness is not guaranteed upstream; the write path does not depend on it (§7).                                                         |
| **Lance**                               | The columnar table format `clips.lance` is stored in; embeddings are added to it as nullable column groups via metadata-only `add_columns` plus fragment-local `update_columns` published in one `LanceOperation.Update`.                                                |
| **Arrow /** `fixed_size_list<float32>`  | Apache Arrow in-memory format; embeddings are stored as `fixed_size_list<float32, dim>` columns whose width is a schema guarantee (no per-row `embedding_dim`).                                                                                                         |
| **Column group**                        | A modality's `embedding_<modality>_*` columns (vectors + provenance) added directly onto `clips.lance` as one atomic unit: added together in one `add_columns`, filled together in one `LanceOperation.Update`, and validated together (absent-as-a-whole or present-and-complete).                                                                                        |
| **Fragment**                            | Lance's unit of physical storage: a slice of the table's rows with its own per-column data files. It is also this leg's **unit of work** — a worker is handed a fragment id, and scans, embeds, and writes that fragment alone.                                          |
| **Pending row**                         | An applicable row whose group primary vector is still NULL — the single definition of "work this modality still owes", expressed by `columns.pending_filter` and used unchanged by every run. |
| **Group reset**                         | Dropping a group's columns and re-adding them empty (`--reset-group` -> `drop_embedding_group`), which makes every row pending again so the next ordinary run refills the group from scratch — §7, §11. |
| **Fill contract violation**             | An invariant of the fill itself that no retry can fix: an embedder that returned the wrong row count or columns that do not cast to the stored schema, or a fragment absent at the pinned version. Raised as `FillContractError` and, unlike an ordinary fragment failure, it stops the run — §7. That type also carries the total outage below, its contract being "a failure the run must stop for rather than skip past"; which of the two occurred, and whether to re-run or fix an input, is in the message — §7. |
| **Total outage**                        | A run that wrote **no** fragment while at least one failed: the per-fragment skip kept nothing, so the group is untouched and the counts are those of a run that owed no work. Raised as `FillContractError`, the same type as a violated invariant: the "re-run first" prognosis lives in the message, not in the type — §7. The per-**row** analogue (fragments committed, but every row in them NULL) is `check_action_outcome` — §8. |
| **Tombstone / deletion vector**         | Lance's record of rows logically deleted from a fragment. This leg creates none: `add_columns` is metadata-only and `update_columns` rebinds field ids to a new column file without replacing any row (§7).                                                             |
| `map_batches` **/** `ActorPoolStrategy` | Ray Data's batch transform and its actor-pool compute strategy. Under `ActorPoolStrategy` the UDF must be a **callable class**, not a bound method — the class is constructed once per actor, which is how each actor loads its model once.                             |


