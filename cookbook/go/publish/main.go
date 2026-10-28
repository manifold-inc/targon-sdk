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

	sandbox, err := box.Create(ctx, client, "pub")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	template, err := sandbox.Publish(ctx, targon.PublishSandboxParams{
		Name:        box.Name("tpl"),
		DisplayName: "Cookbook snapshot",
		Description: "Created by cookbook/go/publish",
		Wait:        targon.WaitOptions{Timeout: 10 * time.Minute, PollInterval: 2 * time.Second},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer template.Delete(ctx)
	fmt.Printf("template %s status=%s\n", template.UID, template.Status)
}
