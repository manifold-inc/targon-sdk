<h1 align="center">
  <img src="https://targon.com/targon-logo.svg" alt="Targon" width="32" style="vertical-align: bottom;">
  Targon SDK
</h1>

<p align="center">
  <strong>Build and deploy serverless Python on GPUs — in seconds, not hours.</strong>
</p>


<p align="center">
   | <a href="#installation"><b>Install</b></a> |
   <a href="#getting-started"><b>Getting Started</b></a> |
   <a href="https://github.com/manifold-inc/targon-sdk/tree/main/examples"><b>Examples</b></a> |
   <a href="https://docs.targon.com/sdk/app"><b>Documentation</b></a> |
</p>


<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-stable-brightgreen">
  <a href="https://pypi.org/project/targon-sdk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/targon-sdk?color=blue&label=PyPI"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9+-orange">
  <a href="https://github.com/manifold-inc/targon-sdk/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/manifold-inc/targon-sdk"></a>
</p>



## About

Targon SDK is a Python framework for building and deploying serverless applications on the Targon platform. Define your app, decorate your functions, and deploy — the SDK handles containers, scaling, and infrastructure.

- **Zero infrastructure** — No containers to build, no clusters to manage
- **GPU-first** — H100, H200, and more. Request with `resource="h200-small"`
- **Web endpoints** — `@targon.fastapi_endpoint()` turns functions into APIs
- **Scales to zero** — Pay only when your code runs
- **Custom images** — Build containers in Python with `pip_install()`, `env()`, and more

> **Stability:** The Targon SDK follows semantic versioning. Breaking changes
> are only introduced in major releases.

## Installation

**Requires Python 3.9+**

```bash
pip install targon-sdk
```

**Install from source**
```bash
git clone https://github.com/manifold-inc/targon-sdk.git
cd targon-sdk
pip install -e .
```

## Getting Started

```python
import targon

app = targon.App("hello-world", image=targon.Image.debian_slim())

@app.function(resource=targon.Compute.CPU_SMALL)
def greet(name: str) -> str:
    return f"Hello, {name}!"

@app.local_entrypoint()
def main():
    print(greet.remote("World"))
```

```bash
# Authenticate
targon setup

# Run remotely
targon run hello.py

# Deploy as a service
targon deploy hello.py
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
