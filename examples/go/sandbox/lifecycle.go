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

	fmt.Println("Creating sandbox")
	s, err := client.Sandboxes.Create(ctx, targon.SandboxCreateParams{
		Name:     fmt.Sprintf("go-example-%d", time.Now().Unix()%1_000_000),
		Template: &templates.Items[0],
		Wait: targon.WaitOptions{
			Timeout:      5 * time.Minute,
			PollInterval: 2 * time.Second,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	deleted := false
	defer func() {
		if !deleted {
			_ = s.Delete(ctx)
		}
	}()
	fmt.Printf("Sandbox created (%s)\n", s.UID)

	response, err := s.Exec(ctx, "python --version", 60)
	if err != nil {
		log.Fatal(err)
	}
	if response.Code != 0 {
		fmt.Printf("Error: %d %s\n", response.Code, response.Stderr)
	} else {
		fmt.Print(response.Stdout)
	}

	fmt.Println("Removing sandbox")
	if err := s.Delete(ctx); err != nil {
		log.Fatal(err)
	}
	deleted = true
	fmt.Println("Sandbox removed")
}
