## Tests

This directory contains all tests for the curator and clients.

The test suites have separate markers and execution environments:

| Suite | Marker | Local command | CI job |
| --- | --- | --- | --- |
| Unit | Neither runtime marker | `pixi run unit-test` | `unit_tests`, shell runners |
| CPU integration | `@pytest.mark.integration` | `pixi run --as-is -e default integration-test` | `integration_tests`, Kubernetes |
| GPU/model | `@pytest.mark.env("default")` or another required env | `pixi run --as-is -e default gpu-test` | `gpu_tests`, Slurm |

Use `pixi run unit-test-fast` to run unit tests with four workers. Both unit-test
tasks automatically select the `dev` environment.

Tests that start a real Ray cluster, including Ray Data operations that initialize
Ray implicitly, must be excluded from the unit suite. Use `integration` for CPU
Ray and runtime-service tests. Tests requiring GPUs or staged model weights use
the env marker instead. Put the marker on each test or class, or use `pytestmark`
for a module containing only that kind of test. Marking a fixture does not mark
the tests that request it.

The Kubernetes integration job reuses the slim image and runs serially so each
pytest worker does not start its own cluster. It uses the `k8s-cpu` runner pool
with per-job requests for 8 CPUs and 32 GiB RAM, and has a 30-minute timeout.
These tests use CPU execution and synthetic/local data, with no model download.
Both runtime jobs publish JUnit and coverage reports; `combine_coverage` combines
them with unit coverage.

Both runtime test tasks collect `pipelines`, `models`, and `next`. They avoid
client and benchmark tests whose imports require development dependencies absent
from runtime environments. The runtime image also omits `cosmos_curator.client`
itself. In modules collected by either task, keep client imports inside the unit
tests that need them so runtime collection does not require the deployment client.

Both jobs run automatically on merge requests. `optional-integration-tests` makes
only the Kubernetes integration job optional and manual; `optional-gpu-tests`
does the same for the Slurm GPU job. Both still run automatically on merge trains.

Pure functions and mocked Ray plans belong in the CPU suite. Request `ray_local`
only for Curator Next integration tests that need a cluster, and use
`curator_io_cluster` when stages reserve `curator_io` resources. Tests that exercise
fresh-cluster recovery should own their clusters in subprocesses so they do not
shut down another test's shared cluster.
