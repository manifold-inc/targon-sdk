// Package box is the shared create/delete setup for the sandbox cookbook.
package box

import (
	"context"
	"fmt"
	"time"

	targon "github.com/manifold-inc/targon-sdk/libs/go/v4"
)

func Client() (*targon.Client, error) {
	return targon.NewFromEnv()
}

func Name(action string) string {
	// 1-32 lowercase letters, digits, or hyphens. Cannot start or end with '-'.
	return fmt.Sprintf("cb-%s-%d", action, time.Now().Unix()%1_000_000)
}

func ReadyTemplate(ctx context.Context, client *targon.Client) (*targon.SandboxTemplate, error) {
	page, err := client.Sandboxes.Templates.List(ctx, targon.ListSandboxTemplatesParams{
		Status: targon.SandboxTemplateStatusReady,
	})
	if err != nil {
		return nil, err
	}
	if len(page.Items) == 0 {
		return nil, fmt.Errorf("no READY sandbox template is available")
	}
	return &page.Items[0], nil
}

func Create(ctx context.Context, client *targon.Client, action string) (*targon.Sandbox, error) {
	template, err := ReadyTemplate(ctx, client)
	if err != nil {
		return nil, err
	}
	ttl, idle := 900, 300
	return client.Sandboxes.Create(ctx, targon.SandboxCreateParams{
		Name:     Name(action),
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
}
