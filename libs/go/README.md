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
	wl, err := client.Workloads.Create(ctx, targon.CreateWorkloadRequest{
		Name:         "my-workload",
		Image:        "nginx:latest",
		ResourceName: targon.CPUSmall,
	})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := client.Workloads.Deploy(ctx, wl.UID); err != nil {
		log.Fatal(err)
	}
	fmt.Println("created", wl.UID)
}
```

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

An organization is only required when an org-scoped client is used.
Inventory, organization listing, and personal API tokens remain available
without one.

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
