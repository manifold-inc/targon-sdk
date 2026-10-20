package main

import (
	"context"
	"fmt"
	"log"
	"time"

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

	sandbox, err := box.Create(ctx, client, "fork")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	childName := box.Name("child")
	child, err := sandbox.Fork(ctx, targon.ForkSandboxParams{
		Name: &childName,
		Wait: targon.WaitOptions{Timeout: 10 * time.Minute, PollInterval: 2 * time.Second},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer child.Delete(ctx)

	fmt.Println("child", child.UID)
	result, err := child.Exec(ctx, "echo forked", 60)
	if err != nil {
		log.Fatal(err)
	}
	if result.Code != 0 {
		log.Fatal(result.Stderr)
	}
	fmt.Print(result.Stdout)
}
