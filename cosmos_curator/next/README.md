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

See [Cosmos Curator Next](../../docs/curator/design/curator-next.md) for the design direction.
