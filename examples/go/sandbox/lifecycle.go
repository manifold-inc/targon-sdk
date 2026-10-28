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

	if err := run(ctx, client); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, client *targon.Client) error {
	templates, err := client.Sandboxes.Templates.List(ctx, targon.ListSandboxTemplatesParams{
		Status: targon.SandboxTemplateStatusReady,
	})
	if err != nil {
		return err
	}
	if len(templates.Items) == 0 {
		return fmt.Errorf("no READY sandbox template is available")
	}

	fmt.Println("Registering sandbox")
	s, err := client.Sandboxes.Register(ctx, targon.SandboxCreateParams{
		Name:     fmt.Sprintf("go-example-%d", time.Now().Unix()%1_000_000),
		Template: &templates.Items[0],
	})
	if err != nil {
		return err
	}
	deleted := false
	defer func() {
		if !deleted {
			_ = s.Delete(ctx)
		}
	}()
	fmt.Printf("Sandbox registered (%s)\n", s.UID)

	s, err = s.Deploy(ctx, targon.WaitOptions{
		Timeout:      5 * time.Minute,
		PollInterval: 2 * time.Second,
	})
	if err != nil {
		return err
	}
	fmt.Printf("Sandbox running (%s)\n", s.UID)

	response, err := s.Exec(ctx, "python --version", 60)
	if err != nil {
		return err
	}
	if response.Code != 0 {
		fmt.Printf("Error: %d %s\n", response.Code, response.Stderr)
	} else {
		fmt.Print(response.Stdout)
	}

	fmt.Println("Removing sandbox")
	if err := s.Delete(ctx); err != nil {
		return err
	}
	deleted = true
	fmt.Println("Sandbox removed")
	return nil
}
