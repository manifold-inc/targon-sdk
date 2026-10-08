# Releasing the SDK

The SDK packages are versioned together but published independently.

## Release prerequisites

Before tagging:

1. Merge the release PR into `main`.
2. Confirm Go, Python, and TypeScript checks are green.
3. Run `make test`, `make lint`, and `make build`.
4. Confirm package versions and the changelog agree.

The `release` GitHub environment must be configured with:

- PyPI trusted publishing for the `publish.yml` workflow.
- `NPM_TOKEN`, scoped to publish `@targon/sdk`.

Protect the environment with required reviewers.

## Release-candidate versions

The `4.0.0-rc.1` package versions are:

- Python: `4.0.0rc1`
- npm: `4.0.0-rc.1`
- Go module: `github.com/manifold-inc/targon-sdk/libs/go/v4`

## Component tags

Create tags from the merged release commit:

```bash
git switch main
git pull --ff-only

git tag -a python/v4.0.0-rc.1 -m "Python SDK v4.0.0-rc.1"
git tag -a typescript/v4.0.0-rc.1 -m "TypeScript SDK v4.0.0-rc.1"
git tag -a libs/go/v4.0.0-rc.1 -m "Go SDK v4.0.0-rc.1"

git push origin \
  python/v4.0.0-rc.1 \
  typescript/v4.0.0-rc.1 \
  libs/go/v4.0.0-rc.1
```

The component-tag workflows verify the tag against package metadata before
publishing. npm prereleases use the `next` distribution tag. Go is published
when the module tag becomes available through the Go module proxy.

## Local package checks

### Python

```bash
cd libs/python
python -m build
python -m twine check dist/*
```

### TypeScript

```bash
cd libs/typescript
npm ci
npm run build
npm test
npm pack --dry-run
```

### Go

```bash
cd libs/go
go test ./...
go vet ./...
test "$(go list -m)" = "github.com/manifold-inc/targon-sdk/libs/go/v4"
```

## Verify published packages

Install each release candidate into a clean temporary project and rerun the
smallest create/exec/delete smoke flow:

```bash
pip install --pre "targon-sdk==4.0.0rc1"
npm install @targon/sdk@4.0.0-rc.1
go get github.com/manifold-inc/targon-sdk/libs/go/v4@v4.0.0-rc.1
```

## Promote to final

After the release-candidate soak:

1. Change Python to `4.0.0`.
2. Change npm to `4.0.0`.
3. Change the Go user-agent version to `4.0.0`.
4. Replace release-candidate install examples in SDK and public docs.
5. Add the dated `4.0.0` changelog heading.
6. Repeat package checks and tag with the same component prefixes.
