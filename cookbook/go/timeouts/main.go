package main

import (
	"context"
	"fmt"
	"log"

	"github.com/manifold-inc/targon-sdk/cookbook/go/internal/box"
	targon "github.com/manifold-inc/targon-sdk/libs/go/v4"
)

func main() {
	ctx := context.Background()
	client, err := box.Client()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	sandbox, err := box.Create(ctx, client, "ttl")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	ttl, idle := 1800, 600
	updated, err := sandbox.Update(ctx, targon.SandboxUpdateParams{
		SandboxConfig: &targon.SandboxConfigInput{
			TTLSec:         &ttl,
			IdleTimeoutSec: &idle,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	if updated.SandboxConfig == nil {
		log.Fatal("sandbox_config missing after update")
	}
	gotTTL, gotIdle := 0, 0
	if updated.SandboxConfig.TTLSec != nil {
		gotTTL = *updated.SandboxConfig.TTLSec
	}
	if updated.SandboxConfig.IdleTimeoutSec != nil {
		gotIdle = *updated.SandboxConfig.IdleTimeoutSec
	}
	fmt.Printf("ttl_sec=%d idle_timeout_sec=%d\n", gotTTL, gotIdle)
}
