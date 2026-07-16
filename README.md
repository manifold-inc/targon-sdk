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
  <a href="https://www.npmjs.com/package/@targon/sdk"><img alt="npm" src="https://img.shields.io/npm/v/@targon/sdk"></a>
  <a href="https://pkg.go.dev/github.com/manifold-inc/targon-sdk/libs/go"><img alt="Go Reference" src="https://pkg.go.dev/badge/github.com/manifold-inc/targon-sdk/libs/go.svg"></a>
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
