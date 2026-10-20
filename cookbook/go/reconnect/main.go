package main

import (
	"context"
	"fmt"
	"log"

	"github.com/manifold-inc/targon-sdk/cookbook/go/internal/box"
)

func main() {
	ctx := context.Background()
	client, err := box.Client()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	sandbox, err := box.Create(ctx, client, "reconn")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	fmt.Println("persist this uid:", sandbox.UID)

	// A later process constructs its own client and calls Get(uid).
	// List only returns summaries; Get returns a sandbox you can exec.
	again, err := client.Sandboxes.Get(ctx, sandbox.UID)
	if err != nil {
		log.Fatal(err)
	}
	result, err := again.Exec(ctx, "hostname", 30)
	if err != nil {
		log.Fatal(err)
	}
	if result.Code != 0 {
		log.Fatal(result.Stderr)
	}
	fmt.Print(result.Stdout)
}
