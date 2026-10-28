# Targon Python SDK

Python SDK to interact with Targon workloads.

## Installation

```bash
pip install --pre "targon-sdk==4.0.0rc1"
```

## Quickstart

```python
from targon import Client, Resources, VMCreateParams

client = Client.from_env(org="acme")

vm = client.vms.create(
    VMCreateParams(
        name="my-vm",
        image="ubuntu-24.04",
        resource_name=Resources.CPU_SMALL,
        password="change-me",
    )
)
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

User-scoped services do not require an organization:

- `client.orgs` for organization discovery and management
- `client.api_tokens` for personal API tokens
- `client.inventory` for available capacity

Typed workloads (`client.vms`, `client.bare_metal`, and `client.sandboxes`),
projects, SSH keys, members, service tokens, wallet, and credits are
organization-bound.

## Organizations and tokens

```python
orgs = client.orgs.list()
acme = client.for_org("acme")
members = acme.members.list()
credits = acme.credits.get()

# Personal token owned by the authenticated user.
personal_token = client.api_tokens.create("laptop")

# Service token owned by the selected organization.
service_token = acme.service_tokens.create("ci")
```

The `acme` view is organization-bound without mutating the original client.

## Typed workloads

Use the typed client that matches the compute you are creating:

```python
import os

from targon import BareMetalCreateParams, Resources, VMCreateParams

vm = acme.vms.create(
    VMCreateParams(
        name="my-vm",
        image="ubuntu-24.04",
        resource_name="cpu-small",
        password="change-me",
    )
)

metal = acme.bare_metal.create(
    BareMetalCreateParams(
        name="gpu-host",
        image="ubuntu-24.04",
        resource_name=Resources.H100_SMALL,
        region=os.environ["TARGON_REGION"],
        ssh_keys=("key-uid",),
    )
)
```

For advanced workflows, split creation into registration and deployment:

```python
registered = acme.vms.register(
    VMCreateParams(
        name="my-vm",
        image="ubuntu-24.04",
        resource_name="cpu-small",
        password="change-me",
    )
)
vm = registered.deploy()
```

The legacy generic `client.workload.create()`/`deploy()` flow migrates to
`client.vms.create()` or `client.bare_metal.create()`; Sandbox workloads
migrate to `client.sandboxes.create()`. Rental and Volume APIs are unavailable
in v4 and have no current replacement.

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

## Sandboxes

Sandboxes and templates are fetched, frozen resources with explicit read-only
fields. Their methods stay bound to the client that fetched them:

```python
from targon import SandboxCreateParams, SandboxTemplateStatus

templates = client.sandboxes.templates.list(
    status=SandboxTemplateStatus.READY,
)
template = templates.items[0]

# Service API
sandbox = client.sandboxes.create(
    SandboxCreateParams(
        name="my-sandbox",
        template=template,
        ttl_sec=3600,
        idle_timeout_sec=300,
    )
)

# Equivalent resource API
other = template.create_sandbox(
    "my-other-sandbox",
    ttl_sec=3600,
    idle_timeout_sec=300,
)
```

Template resources also provide `refresh()`, `update(...)`, and `delete()`.
Only a bound, READY template resource can create a sandbox; raw template IDs
are not accepted.

File and terminal operations are grouped under bound capabilities:

```python
sandbox.files.write("/tmp/hello.txt", "hello")
contents = sandbox.files.read("/tmp/hello.txt", as_text=True)

terminal = sandbox.terminals.create(cols=100, rows=30)
with sandbox.terminals.connect(terminal.id) as connection:
    connection.send(b"pwd\n")
```

For applications that operate by workload ID, the equivalent service APIs
remain available as `client.sandboxes.files` and
`client.sandboxes.terminals`.

The former direct resource aliases (`read_file`, `write_file`,
`list_terminals`, `create_terminal`, `delete_terminal`, and
`connect_terminal`) have been removed in favor of these capability objects.
`SandboxTemplate.create_sandbox` now accepts only the resource form shown
above; use `client.sandboxes.create(SandboxCreateParams(...))` for the service
form.

## Migration from v3

- Replace generic workload creation and deployment with the typed VM, Bare
  Metal, or Sandbox clients shown above.
- Common workload list, get, state, events, and delete operations remain on
  `client.workload`.
- Organization discovery, personal API tokens, and inventory remain
  user-scoped and do not require an organization. Bind all other services to
  an organization with `client.for_org(...)`.
- Rental and Volume APIs, including volume attachment, were removed without a
  replacement.

## Development

```bash
pip install -e ".[dev]"
make lint
```

## License

Apache 2.0 — see [LICENSE](../../LICENSE) for details.
