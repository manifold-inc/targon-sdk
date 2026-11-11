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

	sandbox, err := box.Create(ctx, client, "exec")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	result, err := sandbox.Exec(ctx, "python --version", 60)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("code=%d timed_out=%v\n", result.Code, result.TimedOut)
	fmt.Println("stdout:", result.Stdout)
	fmt.Println("stderr:", result.Stderr)
	if result.Code != 0 || result.TimedOut {
		log.Fatal("command failed")
	}
}
