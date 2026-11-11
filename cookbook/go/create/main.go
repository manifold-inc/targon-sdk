package main

import (
	"context"
	"fmt"
	"log"
	"time"

	targon "github.com/manifold-inc/targon-sdk/libs/go/v4"
)

func main() {
	ctx := context.Background()
	client, err := targon.NewFromEnv()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	templates, err := client.Sandboxes.Templates.List(ctx, targon.ListSandboxTemplatesParams{
		Status: targon.SandboxTemplateStatusReady,
	})
	if err != nil {
		log.Fatal(err)
	}
	if len(templates.Items) == 0 {
		log.Fatal("no READY sandbox template is available")
	}
	template := &templates.Items[0]
	fmt.Printf("using template %s (%s)\n", template.UID, template.Name)

	ttl, idle := 900, 300
	sandbox, err := client.Sandboxes.Create(ctx, targon.SandboxCreateParams{
		Name:     fmt.Sprintf("cb-create-%d", time.Now().Unix()%1_000_000),
		Template: template,
		SandboxConfig: &targon.SandboxConfigInput{
			TTLSec:         &ttl,
			IdleTimeoutSec: &idle,
		},
		Wait: targon.WaitOptions{
			Timeout:      10 * time.Minute,
			PollInterval: 2 * time.Second,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		fmt.Println("deleting", sandbox.UID)
		if err := sandbox.Delete(ctx); err != nil {
			log.Fatal(err)
		}
	}()

	state, err := sandbox.GetState(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("created %s status=%s\n", sandbox.UID, state.Status)
}
