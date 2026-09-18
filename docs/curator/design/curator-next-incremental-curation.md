# Curator Next: Incremental Curation

## Status

This design records the original `video-split` publication model and the proposed curation extensions. That recipe has
been removed. Current row growth and source checkpoints belong to [video-cut](curator-next-video-cut.md); clip media,
column enrichment, and span checkpoints belong to [video-transcode](curator-next-video-transcode.md). See those designs
and [video-caption](curator-next-video-captioning.md) for the implemented publication and recovery boundaries.

## Summary

Maintain one canonical Lance table with one row per clip. The table grows in two independent directions:

| Kind of growth | Owner | Operation |
| --- | --- | --- |
| More rows and fragments | Video splitting | Append newly created clips. |
| More columns | Other curation streams | Add nullable fields, then fill existing fragments. |

Splitting is the only initial operation that changes row count. It uploads clip media, coalesces terminal records into
bounded publication batches, and immediately commits the successful clip rows from each batch as a Lance fragment.
Recovery reconciles the input sources and expected clip IDs with the clip rows already committed.

Every later curation stream preserves the rows and fragment boundaries established by splitting. A stream first adds
its result fields to the schema. It then computes and commits those fields one canonical fragment at a time. A committed
fragment update is both published output and a recovery checkpoint.

This model does not require the final schema to be known when the corpus is created. New curation streams add new
columns whenever they are introduced.

## Canonical Clip Table

The canonical table contains one row per clip. Its initial schema only needs stable clip identity, the source and clip
geometry needed for reconciliation, and references to durable media:

```text
record_schema_version
media_contract_version
source_id
source_uri
source media metadata
start_ns
end_ns
clip_id
clip_uri
clip media metadata
```

`source_id` must identify immutable source content under the producer's source contract. `video-split` derives it from
the normalized source URI and requires the bytes at that URI to remain immutable. A producer that accepts mutable URIs
must instead incorporate a version ID, content digest, or another exact revision identifier into `source_id` or store a
separate `source_revision`.

`clip_id` is deterministic for `source_id`, the requested clip geometry, and the media-producing contract. Retrying a
split with the same inputs must derive the same clip ID and media location regardless of run order, worker placement,
or fragment packing.

Curation fields are added later, for example:

```text
caption__qwen_v1
embedding__siglip_v1
filter__aesthetic_v1
```

The names are illustrative. Each curation stream exclusively owns one or more fields. If a model, prompt,
preprocessing rule, output type, or other result-defining behavior changes, the new computation writes new versioned
fields instead of silently changing the meaning of existing data.

A nullable result can use null to mean pending and a non-null value to mean complete. When an empty output,
inapplicability, or terminal failure must be distinguished, the result should be a struct containing a state, value,
and optional error. Every applicable row should reach a terminal non-null outcome before its fragment is committed.

Lance schema evolution supports adding nullable fields without materializing values for existing rows. See the
[Lance data evolution guide](https://lance.org/guide/data_evolution/).

## Row Growth: Splitting

The splitting stream is the sole producer of clip rows and new canonical fragments. There is one logical splitter at a
time. Ray workers may process sources and stage fragment files concurrently, but one coordinator owns canonical table
commits; ordinary Lance appends do not enforce `clip_id` uniqueness between concurrent splitters.

For each run, it:

1. resolves the current source videos under the producer's immutable-source contract;
2. opens and validates the canonical table, or atomically creates an absent table with the splitting-owned schema and
   zero rows;
3. reads committed clip IDs and the source metadata needed to reconstruct split plans;
4. reconstructs expected deterministic clip IDs from committed metadata when the source is already known;
5. skips a source without reading it when all expected clip IDs are committed;
6. downloads and probes an unknown source, or downloads a known partial source, and processes only missing clips;
7. writes each clip MP4 to its deterministic durable location;
8. coalesces terminal clip and error records into bounded publication batches; and
9. stages and commits the successful clip rows from each batch to Lance with `Append` as soon as it is ready.

The schema-only bootstrap uses create-only semantics and never overwrites an existing table. If another creator wins a
race, the splitter reopens the table and validates its splitting-owned fields instead. Every clip fragment, including
the first one, uses `Append`. The empty bootstrap remains valid if the first run later fails or produces no clip rows.
Sources omitted from a later input selection do not delete previously committed clips.

The initial `clips_per_publish_batch` default is 100,000 terminal records. Sparse error records participate in that
boundary, so each fragment contains at most 100,000 clips and may be smaller when a batch also contains errors. The
splitter also commits a smaller final fragment when it catches up to the current source set or shuts down cleanly.

Media is made durable before its metadata row is committed. If the splitter fails before the Lance commit, a retry may
find unreferenced media objects, but it derives the same IDs and object locations and safely recreates or reuses them.
If the commit succeeded, reconciliation finds the clip rows and skips the work.

Before each append, the splitter opens the latest table and checks the candidate fragment's clip IDs. If every ID is
already present, the descriptor is a successful replay and the append is skipped; if none is present, it proceeds.
The same check resolves an ambiguous commit response before retry. Partial presence violates the expected atomic,
single-writer protocol and is reported rather than blindly appending the whole fragment.

Sources with at least one committed clip carry enough canonical source metadata to reconstruct their expected clip IDs
without rereading the source. The initial implementation re-evaluates sources that produce no clips or fail before
producing a clip because they have no canonical clip row. If that becomes expensive, splitting can add a small durable
source-outcome table. That state is local to split reconciliation; it does not change the canonical clip-table or
enrichment protocols.

## Column Growth: Curation Streams

Captioning, embedding, scoring, filtering, OCR, quality measurement, and similar operations add information to existing
clips. They do not append replacement rows or create a second canonical table.

A new stream follows this protocol:

1. Add all of the stream's nullable result fields to the canonical schema in one schema transaction.
2. Open the latest table and enumerate its fragments.
3. For each fragment whose result is still pending:
   1. read the exact clip IDs and input columns for that fragment;
   2. compute one terminal result for every applicable clip;
   3. validate that the results cover the same clip IDs exactly once;
   4. write the new column data aligned with the fragment's existing rows; and
   5. atomically commit the fragment update.
4. Reopen the latest table and continue until every fragment observed by the run is complete.

The update writes only fields owned by the stream. It preserves row count, row order, fragment boundaries, and columns
owned by other streams. Multiple result fields from one stream, such as a value and status, are committed together.

The canonical fragment is the initial recovery boundary:

| Last durable event | Recovery behavior |
| --- | --- |
| Computation started, but the fragment update was not committed | Recompute the fragment. |
| Column files written, but no transaction commit | Recompute or reuse them, then commit. |
| The fragment update was committed | Detect the terminal results and skip the fragment. |
| Splitting appended another fragment | See null result fields and process the new fragment on the next pass. |

There is no permanent whole-corpus completion state. A curation stream can be caught up to a particular Lance version;
new split fragments make it pending again.

The intended Lance primitive is a fragment-scoped column update committed as a table transaction, rather than a row
append or replacement of the whole table. The implementation must qualify the exact APIs and conflict behavior against
the supported Lance version and storage backends. The [Lance transaction
specification](https://lance.org/format/table/transaction/) describes the atomic version and conflict model.

## Optional Fine-Grained Checkpointing

Recomputing an entire fragment is the deliberately simple starting point. It should be sufficient for embeddings and
other moderately priced work if the canonical fragment size is reasonable.

For expensive work such as VLM captioning, a stream may later use a temporary Lance checkpoint table:

1. compute clips in smaller batches;
2. commit each batch to the temporary table keyed by `clip_id`;
3. once every clip in a canonical fragment has a terminal checkpoint result, validate exact coverage;
4. materialize the aligned final column data for that canonical fragment; and
5. commit one update to the canonical table.

After the canonical update commits, its values are authoritative and the corresponding temporary checkpoints may be
garbage-collected. Checkpoint batch size is stream-specific: captioning may use small batches, while embedding may use
larger ones. Adding this optimization does not change the canonical schema, fragment boundaries, or completion rules.

## Concurrency and Maintenance

Splitting has one logical writer, and each result field has one logical writer. Different curation streams may compute
concurrently, and splitting may continue to append fragments while existing fragments are enriched.

Schema registration and fragment commits still enter one Lance version history. A writer that loses an optimistic
commit race reopens the latest version and retries or restages its update using the already computed results. Expensive
inference should not be repeated merely because the table advanced. The first implementation may serialize these short
canonical commits if that is simpler; this does not require serializing computation.

Splitting owns the initial canonical schema. Later streams may extend it with nullable fields. A split fragment can
continue to contain only splitting-owned fields; Lance presents absent enrichment fields as null. A schema change that
races with an append may be retried as a short control-plane operation.

Compaction and other operations that rewrite fragment layout should not run while fragment-scoped updates are in
flight. They can initially be scheduled between curation runs. If online compaction is added later, writers must detect
changed fragments and restage their already computed results by `clip_id`.

## Covered Workloads

The column-growth protocol covers any operation that ultimately publishes one result per clip, including:

- captions, embeddings, OCR, classifications, scores, and filter decisions;
- deduplication or clustering jobs that perform global computation but publish a group or decision per clip; and
- later versions of an existing computation, published into new fields.

Filtering records a value or decision; it does not delete rows from the canonical corpus. Export chooses which fields
and decisions to apply.

If a future transformation creates additional durable entities or otherwise changes row cardinality, it follows a
row-growth protocol like splitting: deterministic identities, durable payloads before metadata, append-only fragments,
and reconciliation on restart. That is a separate extension from ordinary per-clip enrichment.

## Implementation Sequence

The first increment defines the minimal clip schema and deterministic identities, then implements splitting with
bounded fragment appends and source-to-clip reconciliation. Later increments will:

1. implement dynamic nullable-field registration and a fragment-scoped column update primitive;
2. use that primitive for captioning, then embedding and filter values;
3. measure fragment replay cost before implementing temporary checkpoint tables; and
4. add coordinated compaction only after append and column-update recovery are qualified.

The implementation should demonstrate that:

- an absent destination is bootstrapped as an empty table with the splitting-owned schema;
- every split fragment uses `Append`, and a clean final fragment may be smaller;
- a source spanning a committed fragment boundary is reconciled at clip granularity after restart;
- a split restart appends no duplicate clips and recreates only missing work;
- a fully committed source is skipped without downloading it;
- an ambiguous split commit is resolved from its candidate clip IDs before any retry;
- a curation restart skips committed fragments and replays only an uncommitted fragment;
- a field can be introduced after clips already exist;
- clips appended after field registration begin with null and are discovered by the stream;
- fragment updates preserve identities, row order, boundaries, and unrelated fields; and
- a commit conflict can be retried from computed results without rerunning inference.

## Related Design Notes

- [Cosmos Curator Next](curator-next.md)
- [Curator Next Video Cut](curator-next-video-cut.md)
- [Curator Next Video Transcode](curator-next-video-transcode.md)
- [Ray Data Design](ray-data.md)
- [Managed Ray Clusters for Curator Pipelines on Slurm](curator-next-slurm-ray.md)
