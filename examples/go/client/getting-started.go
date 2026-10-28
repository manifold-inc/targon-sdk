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
	if err := runVM(ctx, client); err != nil {
		log.Fatal(err)
	}
	if err := runBareMetal(ctx, client); err != nil {
		log.Fatal(err)
	}
}

func runVM(ctx context.Context, client *targon.Client) error {
	vm, err := client.VMs.Register(ctx, targon.VMCreateParams{
		Name:         "example-vm",
		Image:        os.Getenv("TARGON_VM_IMAGE"),
		ResourceName: targon.CPUSmall,
		Password:     os.Getenv("TARGON_VM_PASSWORD"),
		Ports:        []targon.PortConfig{{Port: 22}},
	})
	if err != nil {
		return err
	}
	defer deleteWorkload(ctx, client, vm.UID)
	fmt.Printf("Registered VM %s\n", vm.UID)

	vm, err = vm.Deploy(ctx, targon.WaitOptions{Timeout: 5 * time.Minute})
	if err != nil {
		return err
	}
	fmt.Printf("VM %s is running\n", vm.UID)
	return nil
}

func runBareMetal(ctx context.Context, client *targon.Client) error {
	metal, err := client.BareMetal.Register(ctx, targon.BareMetalCreateParams{
		Name:         "example-bare-metal",
		Image:        os.Getenv("TARGON_BM_IMAGE"),
		ResourceName: os.Getenv("TARGON_BM_RESOURCE"),
		Region:       os.Getenv("TARGON_REGION"),
		SSHKeys:      []string{os.Getenv("TARGON_SSH_KEY_UID")},
	})
	if err != nil {
		return err
	}
	defer deleteWorkload(ctx, client, metal.UID)
	fmt.Printf("Registered Bare Metal %s\n", metal.UID)

	metal, err = metal.Deploy(ctx, targon.WaitOptions{Timeout: 30 * time.Minute})
	if err != nil {
		return err
	}
	fmt.Printf("Bare Metal %s is running\n", metal.UID)
	return nil
}

func deleteWorkload(ctx context.Context, client *targon.Client, uid string) {
	if err := client.Workloads.Delete(ctx, uid); err != nil {
		log.Printf("cleanup %s: %v", uid, err)
	}
}
