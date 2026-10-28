<h1 align="center">
  <img src="https://targon.com/targon-logo.svg" alt="Targon" width="32" style="vertical-align: bottom;">
  Targon SDK
</h1>

<p align="center">
  <strong>Build and deploy on GPUs — in seconds, not hours.</strong>
</p>

<p align="center">
  | <a href="#install-the-cli"><b>Install</b></a> |
  <a href="https://github.com/manifold-inc/targon-sdk/tree/main/examples"><b>Examples</b></a> |
  <a href="https://docs.targon.com"><b>Documentation</b></a> |
</p>

<p align="center">
  <a href="https://github.com/manifold-inc/homebrew-tap"><img alt="Homebrew" src="https://img.shields.io/badge/homebrew-targon-orange"></a>
  <a href="https://docs.targon.com"><img alt="Docs" src="https://img.shields.io/badge/docs-targon.com-56D4DD"></a>
  <a href="https://pypi.org/project/targon-sdk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/targon-sdk"></a>
  <a href="https://crates.io/crates/targon-sdk"><img alt="crates.io" src="https://img.shields.io/crates/v/targon-sdk"></a>
  <a href="https://www.npmjs.com/package/@targon/sdk"><img alt="npm" src="https://img.shields.io/npm/v/@targon/sdk"></a>
  <a href="https://pkg.go.dev/github.com/manifold-inc/targon-sdk/libs/go/v4"><img alt="Go Reference" src="https://pkg.go.dev/badge/github.com/manifold-inc/targon-sdk/libs/go/v4.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
</p>

## Install the CLI
```bash
brew tap manifold-inc/tap
brew install targon
```
```bash
targon --version
targon auth login
```
Build from source:
```bash
make cli
```

## SDK v4

The v4 release candidate exposes typed VM, Bare Metal, and Sandbox APIs. Go and
Python provide all three typed workload services; TypeScript provides the
Sandbox API.

Package sources:

- [Go](libs/go)
- [Python](libs/python)
- [TypeScript](libs/typescript)

In Go and Python, `Create` is the default operation: it registers, deploys,
and optionally waits for a typed workload. Advanced Go and Python callers can
use separate `Register` and `Deploy` operations. Common Workloads list, get,
state, events, and delete operations are also Go/Python APIs.

TypeScript currently exposes Sandbox only. It does not expose VM, Bare Metal,
the common Workloads service, or separate `Register`/`Deploy` operations.

Migrate generic VM or Bare Metal creation to `VMs.Create` in Go /
`client.vms.create()` in Python or `BareMetal.Create` in Go /
`client.bare_metal.create()` in Python. Migrate Sandbox creation to
`Sandboxes.Create`, `client.sandboxes.create()`, or
`client.sandboxes.create()` in TypeScript.

Organization discovery/management, personal API tokens, and inventory are
user-scoped and organization-optional. Typed workloads, projects, SSH keys,
members, service tokens, wallet, and credits are organization-bound; use
`ForOrg` in Go or `for_org` in Python to create an organization-bound client
view.

Rental and Volume APIs are unavailable in v4 and have no current replacement.

See the package README in each directory, the
[sandbox cookbook](cookbook/README.md), and the
[public Sandbox SDK documentation](https://docs.targon.com/sdk/sandboxes) for
installation and usage. The cookbook is the place to copy programmatic
create, exec, file, freeze, fork, and teardown flows.

Release maintainers should follow [RELEASING.md](RELEASING.md) for package
checks, GitHub environment secrets, component tags, and registry publication.

## Contributing

We welcome contributions! Please see our contributing guidelines before submitting PRs.
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request


## Changelog
See [CHANGELOG.md](CHANGELOG.md) for release notes.


## License
Apache 2.0 — see [LICENSE](LICENSE) for details.
