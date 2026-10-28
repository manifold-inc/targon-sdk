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
        resource_name=Resources.H100_SMALL,
        ssh_keys=("ssh-key-uid",),
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

from targon import (
    BareMetalCreateParams,
    BMConfigInput,
    PortConfig,
    Resources,
    VMConfigInput,
    VMCreateParams,
)

vm = acme.vms.create(
    VMCreateParams(
        name="my-vm",
        image="ubuntu-24.04",  # see acme.vms.images()
        resource_name=Resources.H100_SMALL,
        vm_config=VMConfigInput(hostname="my-vm"),
        ssh_keys=("ssh-key-uid",),
        # Ports are TCP/UDP only, unique per port/protocol, and 22 is reserved for SSH.
        ports=(PortConfig(port=8080, protocol="TCP"),),
    )
)

# Bare Metal capacity is regional; pick an available resource and region.
capacity = next(
    (
        item
        for item in acme.inventory.list(
            inventory_type="bm", region=os.environ.get("TARGON_REGION")
        )
        if item.available > 0 and item.region is not None
    ),
    None,
)
if capacity is None:
    raise SystemExit("no Bare Metal inventory available")

metal = acme.bare_metal.create(
    BareMetalCreateParams(
        name="gpu-host",
        image="ubuntu-24.04",  # see acme.bare_metal.images()
        resource_name=capacity.name,
        region=capacity.region.uid,
        ssh_keys=("ssh-key-uid",),
        bm_config=BMConfigInput(hostname="gpu-host"),
    )
)
```

VM registration accepts an optional `vm_config=VMConfigInput(hostname=...)`;
there is no VM password. Bare Metal hostnames are sent as
`bm_config.hostname`. Both accept an optional `reservation_uid`. Blank
optional strings such as `project_id=""` are omitted from the request.
`bare_metal.create()` and `bare_metal.deploy()` wait up to
`DEFAULT_BARE_METAL_WAIT_TIMEOUT` (30 minutes) by default; VM and Sandbox
waits default to 5 minutes.

`VM` and `BareMetal` resources expose workload fields as read-only properties
(`vm.uid`, `vm.state`, `vm.vm_config`, `metal.bm_config`, ...) and keep the
full response on `.workload`.

Typed operations follow backend support:

| Client | Operations |
| --- | --- |
| `client.vms` | `register`, `deploy`, `create`, `get`, `update`, `reboot`, `get_logs`, `stream_logs`, `images` |
| `client.bare_metal` | `register`, `deploy`, `create`, `get`, `reboot`, `power_on`, `power_off`, `reinstall`, `images` |
| `client.workload` | `get`, `list`, `delete`, `get_state`, `get_events`, `wait_until_ready`, `verify`, `attach_ssh_key`, `detach_ssh_key` |

`client.inventory.list()` accepts `inventory_type` (`vm` or `bm`), `gpu`, and
`region`. Deprecated `rental`, `storage`, and `serverless` types fail
validation.

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
- Common workload list, get, state, events, delete, `wait_until_ready`,
  `verify`, and SSH key attach/detach remain on `client.workload`.
- `ssh_keys` is a sequence (tuple or list) of SSH key UID strings, for example
  `ssh_keys=("ssh-key-uid",)`. A bare string, bytes, set, or generator is
  rejected with a `ValidationError` rather than being split into characters, and
  blank or non-string entries are rejected too. This applies to VM, Bare Metal,
  and Sandbox create/update params.
- `wait_until_ready` fails only on `error`, `suspended`, and `deleted`.
  `stopped` and `frozen` can still transition, so they keep waiting until the
  timeout.
- `VmConfig(password=...)` was removed. VMs use SSH keys and an optional
  `VMConfigInput(hostname=...)`.
- `client.workload.get_logs()`/`stream_logs()` moved to
  `client.vms.get_logs()`/`stream_logs()` because logs are VM-only.
  `client.workload.vm_images()` moved to `client.vms.images()`; Bare Metal
  images are `client.bare_metal.images()`.
- `client.workload.reboot()` moved to `client.vms.reboot()` and
  `client.bare_metal.reboot()`.
- `WorkloadResponse.region`/`hostname` were replaced by
  `vm_config.hostname` and `bm_config.region`/`bm_config.hostname`.
- Generic `update`, `suspend`, `exec`, and `exec_stream` were removed. Sandbox
  exec lives on `client.sandboxes`.
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
