# Release Versioning Design

## Summary

Cosmos Curator release versions should come from Git tags, not from a manually maintained version string in
`pyproject.toml`.

The release process starts in the internal GitLab repository and publishes through GitHub:

1. Develop and merge changes to the internal GitLab main branch.
2. Prepare a release by adding a matching `CHANGELOG.md` heading, for example `## [2.0.1]`.
3. Merge the changelog update to the internal GitLab main branch.
4. Tag the release commit locally, for example `v2.0.1`.
5. Push the internal GitLab main branch to GitHub `main`.
6. Push the release tag to GitHub.
7. GitHub Actions creates the public GitHub release from the tag.
8. The GitHub-to-GitLab mirror brings the public `main` state and release tag back into GitLab.

Given this flow, the release tag is the only durable version. GitLab package builds, GitHub release notes, and Python
package metadata should all derive from Git state.

## Decision

Use tag-derived dynamic versioning for Python package metadata and tag-validated changelog extraction for GitHub
releases.

The target state is:

- `pyproject.toml` uses `dynamic = ["version"]` instead of a static `[project].version` value.
- `setuptools-scm` derives package versions from Git tags.
- Pixi remains the dependency manager, while package builds use the PyPA `build` frontend.
- GitLab CI fetches enough Git history and tags for `setuptools-scm` to compute versions.
- The GitLab package job runs the package build directly and does not rewrite package metadata.
- The GitHub release workflow validates that the pushed tag has a matching `CHANGELOG.md` section.
- Missing changelog content blocks the GitHub release.

GitLab CI's role is to make tags and history available to the build. It should not choose or rewrite the package version.

## Git Tag Contract

Release tags use a leading `v`, for example `v2.0.1`.

Consumers normalize the tag by removing one leading `v`:

- `v2.0.1` becomes `2.0.1`.
- `2.0.1` stays `2.0.1`.

The normalized version must match the changelog heading exactly:

```markdown
## [2.0.1]
```

The internal GitLab repository should contain the same release tags as GitHub. Release tags are created locally and
pushed to GitHub first; the GitHub-to-GitLab mirror is expected to pull those tags into GitLab. This lets GitLab CI derive
package versions from the latest reachable release tag on the internal GitLab main branch, including development builds after
a release.

## GitLab Package Builds

The previous package job mutated project metadata before building:

```bash
VERSION=$(grep -m 1 'version = ' pyproject.toml | cut -d'"' -f2)
PKG_VERSION="${VERSION}.dev${TIMESTAMP}"
poetry version "${PKG_VERSION}"
poetry build --no-interaction
```

This depended on a manually updated static version. It also allowed the package version to drift from the release tag.

The target behavior is to build from a checkout with enough history and tags for `setuptools-scm`:

```yaml
variables:
  GIT_DEPTH: "0"
```

```bash
pixi run build
```

If the job keeps a shallow checkout, it must explicitly unshallow and fetch tags before building.

With `setuptools-scm`, builds from a tagged commit produce the release version:

```text
v2.0.1 -> 2.0.1
```

Builds after a release tag produce a development version derived from the latest reachable tag, tag distance, and commit.
This gives GitLab CI packages monotonically ordered development versions without choosing the next release number by hand.

For package registry uploads, configure `setuptools-scm` to avoid local version metadata in published package versions:

```toml
[tool.setuptools_scm]
local_scheme = "no-local-version"
```

## GitHub Release Workflow

The GitHub release workflow should treat the tag as the release identity and `CHANGELOG.md` as the release note source.

Expected behavior:

1. Read the release tag from `github.ref_name` or the manual `workflow_dispatch` input.
2. Normalize the tag by stripping one leading `v`.
3. Require an exact changelog heading for the normalized version.
4. Extract that changelog section.
5. Fail if the heading is missing.
6. Fail if the extracted release notes are empty.
7. Create the GitHub release with the extracted notes.

The workflow should not create releases with placeholder notes such as:

```text
No changelog entry found for 2.0.1
```

A missing changelog section means the release is incomplete and should be fixed before publishing.

For manual `workflow_dispatch` runs, the workflow should ensure it reads `CHANGELOG.md` from the requested tag commit,
not from an unrelated branch checkout.

## Source Archives

GitHub releases include automatically generated source archives. These are created by GitHub as release artifacts, but
they are not a documented Cosmos Curator installation path. The documented user flow uses a Git checkout with submodules.

Because the GitHub archives do not behave like a normal Git checkout with `.git` metadata available, they are not part of
the package-versioning contract. If source-archive installation becomes a supported user flow later, add archive metadata
support or publish built package artifacts from a real tagged checkout.

## Implemented State

The implementation follows this design:

- `pyproject.toml` uses `dynamic = ["version"]` and `setuptools-scm`.
- GitLab package builds use `GIT_DEPTH: "0"`, fetch tags, activate the Pixi dev environment, and run
  `python -m build`.
- GitHub release creation fails if the tag does not have a matching changelog section or the extracted notes are empty.
- Development builds after `v2.0.0` resolve to `2.0.1.devN` versions from Git tag distance.

## Follow-Up Checks

- Confirm the GitHub-to-GitLab mirror copies release tags, not only branch heads.
- After the next release tag is pushed to GitHub, verify the tag is visible from GitLab before relying on it for package
  version derivation.
