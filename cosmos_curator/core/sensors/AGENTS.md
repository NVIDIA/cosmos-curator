# Sensor Library — Agent Guidelines

## Self-containment

No module under `cosmos_curator.core.sensors` may import anything from
outside this package -- not `cosmos_curator.next`, not `cosmos_curator.core.utils`,
not `cosmos_curator.pipelines`, nothing. This is a real, enforced rule, not
a preference: `tests/cosmos_curator/core/sensors/test_import_boundary.py`
walks every `.py` file under this directory and fails the build on any
`cosmos_curator.*` import that isn't `cosmos_curator.core.sensors.*` itself.
No exceptions are configured, and none should be added without first
changing that test deliberately.

**Why this matters in practice**: `core/sensors/mp4/` (generic ISO-BMFF box
parsing) is the concrete example -- it lives here because it has no
dependency outside this package. MP4 source discovery (finding candidate
files under an S3 prefix) lives in
`cosmos_curator.next.recipes.mp4_index.discovery` instead, because it
depends on `cosmos_curator.core.utils.storage` -- the shared client layer
every other `next` recipe already uses to list S3 objects. A module that
needs something outside this package doesn't get an exception carved out
for it; it stays where that dependency is allowed.

Third-party and stdlib imports are unrestricted -- only `cosmos_curator.*`
imports are in scope for this rule.

## Layer direction

`cosmos_curator.next`/`cosmos_curator.pipelines` may depend on
`cosmos_curator.core.sensors`; the reverse never happens, matching every
other real dependency direction in this codebase. Both `cosmos_curator.next`
and `cosmos_curator.core.sensors` are incubating -- neither promises a
stable package path yet -- but the *direction* itself is not incubating;
it holds regardless of what moves where inside either tree.
