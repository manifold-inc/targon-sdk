# Targon Go SDK

Go SDK to interact with Targon workloads.

## Installation

```bash
go get github.com/manifold-inc/targon-sdk/libs/go/v4@v4.0.0-rc.1
```

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"

	targon "github.com/manifold-inc/targon-sdk/libs/go/v4"
)

func main() {
	client, err := targon.NewFromEnv()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	ctx := context.Background()
	vm, err := client.VMs.Create(ctx, targon.VMCreateParams{
		Name:         "my-vm",
		Image:        "ubuntu-24.04",
		ResourceName: targon.H100Small,
		SSHKeys:      []string{"ssh-key-uid"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("created VM", vm.UID)
}
```

`VMs.Create`, `BareMetal.Create`, and `Sandboxes.Create` register, deploy, and
optionally wait for a typed workload. `Workloads` remains the common service
for operations every workload type supports: get, list, state, events,
`WaitUntilReady`, `Verify`, SSH key attach/detach, and delete.

The Go SDK uses the v3 organization-scoped API. Set credentials and an
organization directly:

```go
client, err := targon.New(targon.Config{
	APIKey: "...",
	Org:    "acme",
})
```

or resolve them from the environment and the active Targon CLI profile:

```bash
export TARGON_API_KEY="..."
export TARGON_ORG="acme"
```

Organization precedence is:

1. `Config.Org` or `Client.ForOrg("acme")`
2. `TARGON_ORG`
3. The active profile selected with `targon org use <slug>`

User-scoped services do not require an organization:

- `Orgs` for organization discovery and management
- `APITokens` for personal API tokens
- `Inventory` for available capacity

Typed workloads (`VMs`, `BareMetal`, and `Sandboxes`), projects, SSH keys,
members, service tokens, wallet, and credits are organization-bound. Use
`ForOrg` to create an organization-bound view without mutating the original
client:

```go
orgs, err := client.Orgs.List(ctx, targon.Page{})
if err != nil {
	log.Fatal(err)
}
fmt.Println("organizations", len(orgs.Items))

acme, err := client.ForOrg("acme")
if err != nil {
	log.Fatal(err)
}
members, err := acme.Members.List(ctx, "", "", targon.Page{})
if err != nil {
	log.Fatal(err)
}
credits, err := acme.Credits.Get(ctx)
if err != nil {
	log.Fatal(err)
}
fmt.Println("members", len(members.Items), "credits", credits)
```

## Typed workloads

Use the typed service that matches the compute you are creating:

```go
vm, err := client.VMs.Create(ctx, targon.VMCreateParams{
	Name:         "my-vm",
	Image:        "ubuntu-24.04", // see client.VMs.Images
	ResourceName: targon.H100Small,
	VMConfig:     &targon.VMConfigInput{Hostname: "my-vm"},
	SSHKeys:      []string{"ssh-key-uid"},
	// Ports are TCP/UDP only, unique per port/protocol, and 22 is reserved for SSH.
	Ports: []targon.PortConfig{{Port: 8080, Protocol: "TCP"}},
})

// Bare Metal capacity is regional; pick an available resource and region.
items, err := client.Inventory.List(ctx, targon.InventoryListParams{
	Type: "bm", Region: os.Getenv("TARGON_REGION"),
})
if err != nil {
	return err
}
var capacity *targon.Inventory
for i := range items {
	if items[i].Available > 0 && items[i].Region != nil {
		capacity = &items[i]
		break
	}
}
if capacity == nil {
	return errors.New("no Bare Metal inventory available")
}

metal, err := client.BareMetal.Create(ctx, targon.BareMetalCreateParams{
	Name:         "gpu-host",
	Image:        "ubuntu-24.04", // see client.BareMetal.Images
	ResourceName: capacity.Name,
	Region:       capacity.Region.UID,
	BMConfig:     &targon.BMConfigInput{Hostname: "gpu-host"},
	SSHKeys:      []string{"ssh-key-uid"},
})
```

VM registration accepts an optional `VMConfig.Hostname`; there is no VM
password. Bare Metal hostnames are sent as `bm_config.hostname`. Both accept an
optional `ReservationUID`. `BareMetal.Create` and `BareMetal.Deploy` wait up to
`targon.DefaultBareMetalWaitTimeout` (30 minutes) unless `WaitOptions.Timeout`
is set; VM and Sandbox waits default to 5 minutes.

Typed operations follow backend support:

| Service | Operations |
| --- | --- |
| `VMs` | `Register`, `Deploy`, `Create`, `Get`, `Update`, `Reboot`, `Logs`, `StreamLogs`, `Images` |
| `BareMetal` | `Register`, `Deploy`, `Create`, `Get`, `Reboot`, `PowerOn`, `PowerOff`, `Reinstall`, `Images` |
| `Workloads` | `Get`, `List`, `Delete`, `GetState`, `GetEvents`, `WaitUntilReady`, `Verify`, `AttachSSHKey`, `DetachSSHKey` |

`Inventory.List` accepts `Type` (`vm` or `bm`), `GPU`, and `Region`.
Deprecated `rental`, `storage`, and `serverless` types fail validation.

Common workload responses mirror the API: `ReservationUID`, `CreatedBy`, and
type-specific `VMConfig{Hostname}` or `BMConfig{Region, Hostname}`. There are no
top-level `Region` or `Hostname` fields.

### Migrating from v3

| v3 | v4 |
| --- | --- |
| `Workloads.Create` + `Workloads.Deploy` | `VMs.Create` or `BareMetal.Create` |
| `Workloads.Create` with type `SANDBOX` | `Sandboxes.Create` |
| `VMConfig{Password}` | removed; use SSH keys and optional `VMConfigInput{Hostname}` |
| `Workloads.GetLogs` / `StreamLogs` | `VMs.Logs` / `VMs.StreamLogs` (logs are VM-only) |
| `Workloads.VMImages` | `VMs.Images`; Bare Metal uses `BareMetal.Images` |
| `Workloads.Reboot` | `VMs.Reboot` or `BareMetal.Reboot` |
| `Workloads.WaitUntilReady(ctx, uid, timeout, interval)` | `Workloads.WaitUntilReady(ctx, uid, WaitOptions{...})` |
| `Inventory.List(ctx, type, gpu)` | `Inventory.List(ctx, InventoryListParams{...})` |
| `Workloads.Update`, `Suspend`, `Exec`, `ExecStream` | removed; Sandbox exec lives on `Sandboxes` |
| Rental, Volumes, `AttachVolume`, `DetachVolume` | removed; no current API equivalent |

## Sandboxes

Sandbox creation requires a fetched, ready template resource. The template
resource remains bound to its client and can create sandboxes directly:

```go
template, err := client.Sandboxes.Templates.Get(ctx, "sbt-example")
if err != nil {
	log.Fatal(err)
}

sandbox, err := template.CreateSandbox(ctx, targon.SandboxCreateParams{
	Name: "dev-box",
})
if err != nil {
	log.Fatal(err)
}
fmt.Println("created sandbox", sandbox.UID)
```

Template resources returned by `List`, `Get`, `Update`, and sandbox `Publish`
support `Refresh`, `Update`, `Delete`, and `CreateSandbox`:

```go
description := "Base image for development sandboxes"
template, err = template.Update(ctx, targon.UpdateSandboxTemplateParams{
	Description: &description,
})
if err != nil {
	log.Fatal(err)
}
```

## Development

```bash
make test
make lint
```

## License

Apache 2.0 — see [LICENSE](../../LICENSE) for details.
