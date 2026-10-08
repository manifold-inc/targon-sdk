package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	targon "github.com/manifold-inc/targon-sdk/libs/go/v4"
)

func main() {
	client, err := targon.New(targon.Config{
		APIKey: os.Getenv("TARGON_API_KEY"),
		Org:    os.Getenv("TARGON_ORG"),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	ctx := context.Background()
	wl, err := client.Workloads.Create(ctx, targon.CreateWorkloadRequest{
		Name:         "my-workload",
		Image:        "nginx:latest",
		ResourceName: targon.CPUSmall,
		Ports:        []targon.PortConfig{{Port: 80}},
		Envs:         []targon.EnvVar{{Name: "ENV", Value: "production"}},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Created workload %s (%s)\n", wl.UID, wl.Name)

	deploy, err := client.Workloads.Deploy(ctx, wl.UID)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Deploying workload %s...\n", deploy.UID)

	state, err := client.Workloads.WaitUntilReady(ctx, wl.UID, 5*time.Minute, 0)
	if err != nil {
		fmt.Printf("Workload did not become ready: %v\n", err)
		return
	}
	fmt.Printf("Status: %s (%d/%d ready)\n", state.Status, state.ReadyReplicas, state.TotalReplicas)
	for _, u := range state.URLs {
		fmt.Printf("  port %d -> %s\n", u.Port, u.URL)
	}

	logs, err := client.Workloads.GetLogs(ctx, wl.UID, targon.LogOptions{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("--- logs ---")
	fmt.Println(logs)
}
