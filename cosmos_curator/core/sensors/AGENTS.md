# Sensor Library — Agent Guidelines

## Package Boundary

Within `cosmos_curator`, sensor modules may import only from `cosmos_curator.core.sensors`.
The boundary is enforced by
[`test_import_boundary.py`](../../../tests/cosmos_curator/core/sensors/test_import_boundary.py).
Keep code that needs other Curator packages outside the sensor library; do not weaken the test to accommodate it.

For example, generic MP4 parsing lives in `core/sensors/mp4/`, while source discovery lives in
`next/recipes/mp4_index/discovery.py` because it uses shared storage clients.

Stdlib and third-party imports are outside this package-boundary rule; the data-integrity kernel has additional
restrictions below.

## Data-Integrity Kernel

`data_integrity/` stays free of CLI, table-storage, and cloud-client concerns. It may not import `argparse`, `lance`,
`pyarrow`, `boto3`, `botocore`, `smart_open`, or `cosmos_curator.core.sensors.scripts`. These restrictions are enforced by
[`test_kernel_boundary.py`](../../../tests/cosmos_curator/core/sensors/data_integrity/test_kernel_boundary.py).
Workflow concerns belong in `cosmos_curator.next.recipes.data_integrity`.

## Layer Direction

`cosmos_curator.next` and `cosmos_curator.pipelines` may depend on the sensor library; the reverse is forbidden.
Next and sensors have provisional package paths during incubation, but that dependency direction holds across moves.
