# Targon SDK

Python SDK for building and deploying serverless applications on the Targon platform. It includes the command-line client, decorator-friendly runtime APIs, and publishing utilities.

---

## Requirements

- Python **3.9+**
- Targon account with API access  
- Optional: Docker (for advanced image customization)

---

## Installation

### From PyPI

```bash
pip install targon-sdk
```

### From source

```bash
git clone https://github.com/manifold-inc/targon-sdk.git
cd targon-sdk
pip install -e .
```

---

## Quick Start

1. **Configure credentials**

   ```bash
   targon setup
   # Follow the prompts to store your API key securely
   ```

2. **Define an app**

   ```python
   # my_app.py
   import targon
   import subprocess

   app = targon.App("my-first-app")

   @app.function()
   @targon.web_server(port=8000)
   def serve():
       subprocess.Popen("python -m http.server 8000", shell=True)

   @app.local_entrypoint()
   def main():
       print("Hello from Targon!")
   ```

3. **Deploy**

   ```bash
   targon deploy my_app.py
   ```

4. **Iterate locally**

   ```bash
   targon run my_app.py --message "hello"
   ```

---

## CLI Overview

| Command | Description |
| ------- | ----------- |
| `targon setup` | Store or update API credentials. |
| `targon deploy <file.py>` | Build and deploy an app module. |
| `targon run <file.py>` | Execute a local entrypoint in an ephemeral session. |
| `targon app list` | List deployed/running apps. |
| `targon app functions <app_id>` | Inspect functions for a given app. |
| `targon app delete <app_id>` | Delete an app and its deployments. |

---

## Examples

Visit `examples/` for ready-to-run templates:

- `gettin_started/getting_started.py` – minimal hello world workflow.
- `gettin_started/web_endpoint_simple.py` – FastAPI endpoint via `@targon.fastapi_endpoint`.
- `web/web_endpoint_asgi.py` – full ASGI application deployment.
- `llm/vllm_example.py` – vLLM inference service running behind `@targon.web_server`.
- `gen-ai/` – generated media workloads.

Deploy an example directly:

```bash
targon deploy examples/gettin_started/getting_started.py
```

---

## Development Notes

- The SDK mirrors common serverless-style patterns: decorator-based registration, cloudpickle serialization, and runtime-provisioned images.
- Core packages used at runtime:
  - `cloudpickle` for function transport
  - `grpcio` / `grpcio-tools` for RPC communication
  - `aiohttp` for async HTTP interactions
- Tests live under `tests/`. Run them with `pytest` (optional dependency).
- When contributing protocol changes, regenerate stubs with the matching `grpcio-tools` version to avoid runtime mismatches.

---

## Support & Feedback

Open an issue or reach out to `dev@manifold.inc` for questions, feature requests, or bug reports. Contributions are welcome!  
