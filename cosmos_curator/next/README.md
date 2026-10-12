# Curator Next

Curator Next is the starting point for new physical AI data curation workflows. It combines
maintained reference recipes with a shared toolkit, using Ray Data for execution and Lance for canonical tabular outputs.

Explore the [recipes](recipes/) and use `cosmos-curator pipeline --help` for config tooling. The
[pipeline kind registry](../pipelines/builtin_pipeline_kinds.py) lists the recipes exposed through the generic CLI,
alongside explicitly marked legacy entries.

`cosmos_curator.next` is an incubation namespace: schemas, identities, configs, and import paths can change without a
backward-compatibility guarantee. Once components are proven across workflows, they graduate into namespaces named for
durable responsibilities through an explicit transition plan.

Coding agents and contributors starting a new recipe should first read the
[Curator Next agent guidelines](AGENTS.md). They cover shared component adoption, recipe contracts, and managed Ray
qualification criteria.

For checkpointed enrichment, start with the
[adoption example](../../docs/curator/design/curator-next-checkpoint-recovery.md#adopting-the-reference-pattern).
`video_embed` demonstrates independent row processing; `video_transcode` demonstrates source grouping and durable payload
writes. The shared [`enrichment`](enrichment/__init__.py) package owns the run sequence, checkpoint scopes, publication,
and recovery proof, while each recipe's Ray processing graph remains explicit.

See [Cosmos Curator Next](../../docs/curator/design/curator-next.md) for the design direction.
