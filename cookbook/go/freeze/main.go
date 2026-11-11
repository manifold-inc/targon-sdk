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

	sandbox, err := box.Create(ctx, client, "freeze")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	wait := targon.WaitOptions{Timeout: 5 * time.Minute, PollInterval: 2 * time.Second}
	frozen, err := sandbox.Freeze(ctx, wait)
	if err != nil {
		log.Fatal(err)
	}
	state, err := frozen.GetState(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("frozen status:", state.Status)

	running, err := frozen.Thaw(ctx, wait)
	if err != nil {
		log.Fatal(err)
	}
	result, err := running.Exec(ctx, "echo thawed", 60)
	if err != nil {
		log.Fatal(err)
	}
	if result.Code != 0 {
		log.Fatal(result.Stderr)
	}
	fmt.Print(result.Stdout)
	state, err = running.GetState(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("thawed status:", state.Status)
}
