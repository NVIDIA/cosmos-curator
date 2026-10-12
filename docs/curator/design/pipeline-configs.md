# Schema-Validated Pipeline Configs

## Summary

Pipeline configuration files should become the primary input contract for Ray Data Curator pipelines, starting with
video split. They should define versioned, schema-validated pipeline intent that can be rendered, validated, inspected,
and eventually consumed by execution and run-reporting tools.

The first implementation should target the Ray Data video split pipeline. The schema should describe canonical
`video_split` intent and execute on Ray Data in v1. The Xenna-based split pipeline is out of scope for this design and
should remain unchanged.

This design is scoped to pipeline configs only. It should preserve room for future run artifacts, evidence packs, and
agent-facing orchestration, but those are not required for the first implementation.

## Goals

- Make JSON/YAML pipeline configs the preferred way to run Ray Data split.
- Define versioned schemas with strict validation, clear typed errors, and generated JSON Schema.
- Support deterministic config resolution from defaults, presets, user config, and small command-line overrides.
- Produce a canonical resolved config before execution.
- Keep the user-facing CLI small: template, validate, render, schema, and preset inspection.
- Start with Ray Data video split as the canonical config-backed pipeline.
- Keep the schema organized around pipeline concepts rather than historical flag names.
- Make the first runnable config discoverable from CLI help without requiring users or agents to read source.
- Support both human YAML authoring and JSON-oriented automation without splitting them into different workflows.

## Non-Goals

- Building a new orchestration system, chatbot, or run artifact store.
- Building a second long-form command-line interface for Ray Data split.
- Modeling every existing CLI flag as a permanent config field.
- Supporting arbitrary DAG composition in the first schema.
- Replacing Slurm, NVCF, local launch, or OSMO execution interfaces.

## Proposed Model

The config path should make typed config the center of the Ray Data pipeline contract:

```text
JSON/YAML config + optional --set overrides
                   |
                   v
          parse + raw validation
                   |
                   v
       defaults + preset resolution
                   |
                   v
            resolved config
                   |
                   v
     resolved-schema validation
                   |
                   v
       Ray Data pipeline execution
```

A pipeline config has two important forms:

1. **User config**: the JSON/YAML file a user writes. It may rely on packaged defaults, presets, and aliases.
2. **Resolved config**: the fully expanded canonical JSON object used for execution. It has explicit defaults, no
   unresolved presets, no aliases, and a fixed schema version.

Execution should use the resolved config, not the raw user config. This makes runs reviewable, reproducible, and easier
for tools to reason about.

Example user config shape:

```yaml
schema_version: 1
kind: video_split

input:
  video_path: /data/videos
  limit: null

split:
  method: transnetv2

caption:
  enabled: true
  model: qwen
  preprocess_mode: model
  preset: balanced

output:
  clip_path: /data/curated
  metadata_format: json

execution:
  preset: local_gpu
```

The exact fields should be chosen during Ray Data split schema work. The important point is that the schema is organized
around pipeline concepts rather than historical flag names. `kind: video_split` should describe the canonical pipeline
intent; Ray Data is the only supported v1 implementation and does not need to be selected in user config.

## Schema Strategy

Use Pydantic v2 for the schema layer.

Reasons:

- It provides typed Python models and strict validation from JSON/YAML data.
- It supports JSON Schema generation for docs, tooling, and agent-facing validation.
- It supports discriminated unions for stage, model, and output variants.
- It provides useful error locations for unknown fields, invalid values, and invalid combinations.
- It can emit canonical dictionaries for resolved config rendering.

Use Pydantic's schema customization hooks deliberately: field descriptions, examples, and `json_schema_extra` should
make generated JSON Schema useful to editors and agents, not merely structurally valid. JSON Schema should remain the
formal validation surface, but it should not be the only onboarding surface.

Pydantic should not become the whole architecture. Keep separate layers:

```text
raw JSON/YAML
  -> parse + raw validation
  -> defaults and preset resolution
  -> resolved canonical config
  -> resolved-schema validation
  -> pipeline execution
```

Raw validation should reject malformed input and unknown fields that can be checked before resolution. Required-value
and cross-field validation should run against the resolved config so defaults and presets can satisfy fields. Deprecated
fields may be accepted only when the schema explicitly maps them to new fields and emits a warning.

## Config Resolution

Resolution order should be deterministic:

1. Load packaged pipeline defaults and the packaged preset registry.
2. Parse the user config enough to identify the pipeline kind and any preset references.
3. Apply packaged pipeline defaults.
4. Apply the selected preset fragments.
5. Apply explicit user config fields.
6. Apply small CLI overrides such as `--set caption.enabled=false` or `--set split.limit_clips=10`.

The resolver should output canonical JSON and include enough metadata to explain where meaningful values came from.
That provenance can be minimal at first, but `render` should make it clear what will run before execution starts.

Presets should be named config fragments, not hidden execution branches. Inline preset references in user config select
fragments from the preset registry during resolution. Explicit user config fields override values supplied by selected
presets, and the resolved config should show the final values explicitly.

## Discoverability and Ease of Use

Config files introduce a new user contract, so the design should optimize the first successful run as much as the
fully reproducible expert run. Users and agents should not have to infer "what should I write?" from Pydantic internals
or from a large JSON Schema document.

The CLI should expose two related artifacts:

1. **Editable template**: one smallest runnable config per pipeline kind, relying on packaged defaults.
2. **Resolved config**: the canonical expanded JSON used for execution.

Templates and presets serve different roles:

- **Templates** are whole starting config documents printed by `cosmos-curator pipeline template`. Each kind exposes one
  template; whole-template profiles are not part of the CLI contract.
- **Presets** are named section-local config fragments applied during resolution. A preset is referenced from inside a
  config section, such as `split.preset: fixed_stride_10s`, and is not runnable by itself.

Users may edit the printed template to add preset references and lower the
input limit. For example, a cheap first `video_split` run can select:

```yaml
input:
  limit: 2
split:
  preset: fixed_stride_10s
caption:
  preset: off
```

During `render` or execution, the resolver expands `fixed_stride_10s` into explicit split settings. Explicit user fields
still win over values supplied by the preset.

YAML should be the default authoring format for human-facing template output because it is easier to read and edit.
JSON should be available for every command through `--json` so agents, tests, and scripts can consume outputs without
scraping prose. This is a convention, not a hard split: humans may request JSON and agents may emit YAML when writing
files for users.

Agent-facing outputs should expose required author inputs explicitly. The raw user schema may keep sections optional so
defaults and presets can satisfy them, but tools still need to know that a useful run requires values such as
`input.video_path` and `output.clip_path`. Template JSON should therefore include both the example config and
structured required-field metadata.

Expert users should not pay an onboarding tax. The same CLI should let them skip templates and use `validate`, `render`,
`schema`, `presets`, and `--set` directly.

## CLI Contract

The pipeline config CLI should be small and JSON-friendly. The initial namespace is intentionally generic so future
config-backed pipelines do not need their own top-level command:

```text
cosmos-curator pipeline template <kind> [--json]
cosmos-curator pipeline validate <config>
cosmos-curator pipeline render <config> [--set path=value]
cosmos-curator pipeline schema <kind>
cosmos-curator pipeline presets list
cosmos-curator pipeline presets show <name>
```

Human-readable output is useful, but every command should support `--json`. Tools should not need to scrape prose. The
`template` command should print editable YAML by default and machine-readable metadata with `--json`, including the
pipeline kind, description, required author fields, and template config.

Ray Data split knobs should be added to the typed config first. The Ray Data split module entry point should accept a
config file and small `--set` overrides only, avoiding a second long-form flag surface.

The `validate`, `render`, `schema`, and `presets` commands are host-side config tooling. Execution should happen in the
same runtime environment as the pipeline itself. For local Docker, that means the existing launcher remains the outer
execution surface and the default Pixi environment runs the pipeline task inside the container:

```bash
cosmos-curator local launch -- \
  pixi run --as-is run-pipeline /config/video_split.yaml
```

The `run-pipeline` Pixi task is the run-only shortcut for the config-backed runtime command. Config files and every path
inside them must be written as paths visible to the runtime environment, such as `/config/...` for the default local
workspace mount or an explicit container path from `--extra-volumes`.

## First Implementation: Ray Data Video Split

Ray Data video split is the first target because it is the intended canonical substrate for new pipeline execution.

Implementation approach:

1. Define Pydantic models for a v1 `video_split` config backed by Ray Data execution.
2. Add packaged defaults and a small set of presets.
3. Add host-side `template`, `validate`, `render`, `schema`, and `presets` commands plus a runtime Pixi task.
4. Refactor Ray Data split assembly and execution to consume the resolved typed config directly.
5. Add tests for template output, validation errors, preset expansion, canonical rendering, JSON Schema generation, and
   config-to-Ray Data execution mapping.

The resolved config is the contract. Any command-line entry point for Ray Data split should stay small: accept a config
file, support small overrides, and avoid growing a second long-form flag surface.

This gives the project a clean example of how new Ray Data pipelines should be structured:

- typed config in;
- one minimal template per kind discoverable from the CLI;
- resolved config rendered before execution;
- JSON Schema generated from the same models used by execution;
- Ray Data primitives and execution settings derived directly from the resolved config.

## Composability Constraint

The first schema should remain a concrete Ray Data-backed video split config, but it should be built from typed
fragments: input, split, transcode, caption, output, and execution. This keeps the first implementation focused while
avoiding a monolithic config shape that cannot evolve. Future fragments such as embedding and filtering can be added
when the Ray Data pipeline supports them.

Do not start with an arbitrary DAG. An ordered pipeline with typed fragments is enough for v1.

## Success Criteria

- A user can run Ray Data video split from a schema-versioned YAML config.
- A new human user can print the kind's YAML template from CLI help and run it after editing only paths.
- An AI agent can discover supported kinds, required author fields, examples, schemas, presets, and validation errors
  through JSON outputs.
- `validate` catches unknown fields, invalid values, and invalid combinations before execution.
- `render` produces canonical JSON that can be reviewed and rerun.
- Preset expansion is visible in rendered output.
- Ray Data split consumes the resolved typed config directly.
- The v1 schema covers common Ray Data split runs without carrying every existing CLI flag.
- JSON Schema can be generated for the supported pipeline config.

## Open Questions

- What is the smallest useful v1 Ray Data video split schema that covers common OSS runs?
- Which presets should be packaged first?
- How much provenance should `render` include for defaults and presets?
- Where should generated JSON Schema files live, if they are checked in?
