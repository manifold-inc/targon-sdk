# Targon Python SDK

Python SDK to interact with Targon workloads.

## Installation

```bash
pip install targon-sdk
```

## Quickstart

```python
from targon import Client, Resources
from targon.client import CreateWorkloadRequest

client = Client.from_env()

request = CreateWorkloadRequest(
    name="my-workload",
    image="nginx:latest",
    resource_name=Resources.CPU_SMALL,
)
workload = client.workload.create(request)
client.workload.deploy(workload.uid)
```

The Python SDK uses the v3 organization-scoped API. Set credentials and an
organization directly:

```python
client = Client(api_key="...", org="acme")
```

or resolve them from the environment and the active Targon CLI profile:

```bash
export TARGON_API_KEY="..."
export TARGON_ORG="acme"
```

Organization precedence is:

1. `Client(..., org="acme")` or `Client.from_env(org="acme")`
2. `TARGON_ORG`
3. The active profile selected with `targon org use <slug>`

An organization is only required when an org-scoped client is used.
Inventory, organization listing, and personal API tokens remain available
without one.

## Organizations and tokens

```python
client.orgs.list()
client.members.list(role="OWNER")
wallet = client.wallet.get()
credits = client.credits.get()

# Personal token owned by the authenticated user.
personal_token = client.api_tokens.create("laptop")

# Service token owned by the selected organization.
service_token = client.service_tokens.create("ci")
```

For multi-organization applications, create an org-bound view without
mutating the original client:

```python
personal = client.for_org("personal")
personal.workload.list()
```

Custom API hosts and transport options can be set on the root client:

```python
client = Client(
    api_key="...",
    org="acme",
    base_url="https://api.example.test",
    timeout=60,
    verify_ssl=True,
)
```

## Migration from 0.1

- API calls now use `/tha/v3`.
- Workloads, projects, volumes, SSH keys, wallet, credits, members, and
  service tokens require an organization.
- `client.user` and API-key rotation were removed. Use `client.wallet`,
  `client.credits`, and `client.api_tokens`.
- `VolumeClient.delete_deployment` was removed because v3 has no equivalent.

## Development

```bash
pip install -e ".[dev]"
make lint
```

## License

Apache 2.0 — see [LICENSE](../../LICENSE) for details.
