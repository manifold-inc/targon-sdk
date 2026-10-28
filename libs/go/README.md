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
		ResourceName: targon.CPUSmall,
		Password:     "change-me",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("created VM", vm.UID)
}
```

`VMs.Create`, `BareMetal.Create`, and `Sandboxes.Create` register, deploy, and
optionally wait for a typed workload. `Workloads` remains the common service
for list, state, events, and delete operations.

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
region := os.Getenv("TARGON_REGION")

vm, err := client.VMs.Create(ctx, targon.VMCreateParams{
	Name:         "my-vm",
	Image:        "ubuntu-24.04",
	ResourceName: targon.CPUSmall,
	Password:     "change-me",
})

metal, err := client.BareMetal.Create(ctx, targon.BareMetalCreateParams{
	Name:         "gpu-host",
	Image:        "ubuntu-24.04",
	ResourceName: targon.H100Small,
	Region:       region,
	SSHKeys:      []string{"key-uid"},
})
```

For advanced workflows, split creation into registration and deployment:

```go
registered, err := client.VMs.Register(ctx, targon.VMCreateParams{
	Name:         "my-vm",
	Image:        "ubuntu-24.04",
	ResourceName: targon.CPUSmall,
	Password:     "change-me",
})
if err != nil {
	log.Fatal(err)
}
vm, err = registered.Deploy(ctx, targon.WaitOptions{})
```

The legacy generic `Workloads.Create`/`Workloads.Deploy` flow migrates to
`VMs.Create` or `BareMetal.Create`; Sandbox workloads migrate to
`Sandboxes.Create`. Rental and Volume APIs are unavailable in v4 and have no
current replacement.

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
