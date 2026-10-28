package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	targon "github.com/manifold-inc/targon-sdk/libs/go/v4"
)

// Required: TARGON_API_KEY, TARGON_ORG, TARGON_SSH_KEY_UID.
// Optional: TARGON_VM_IMAGE, TARGON_BM_IMAGE, TARGON_REGION, TARGON_BM_RESOURCE.
// Images default to the first entry from the image catalog. The Bare Metal
// region and resource default to the first available bm inventory entry.
func main() {
	sshKey := strings.TrimSpace(os.Getenv("TARGON_SSH_KEY_UID"))
	if sshKey == "" {
		log.Fatal("set TARGON_SSH_KEY_UID to an SSH key UID")
	}
	client, err := targon.New(targon.Config{
		APIKey: os.Getenv("TARGON_API_KEY"),
		Org:    os.Getenv("TARGON_ORG"),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	ctx := context.Background()
	if err := runVM(ctx, client, sshKey); err != nil {
		log.Fatal(err)
	}
	if err := runBareMetal(ctx, client, sshKey); err != nil {
		log.Fatal(err)
	}
}

func runVM(ctx context.Context, client *targon.Client, sshKey string) error {
	image := os.Getenv("TARGON_VM_IMAGE")
	if image == "" {
		images, err := client.VMs.Images(ctx)
		if err != nil {
			return err
		}
		if len(images) == 0 {
			return errors.New("no VM images available")
		}
		image = images[0].Name
	}

	// SSH is reachable on the VM's public IP and SSH port; port 22 cannot be
	// forwarded.
	vm, err := client.VMs.Create(ctx, targon.VMCreateParams{
		Name:         "example-vm",
		Image:        image,
		ResourceName: targon.H100Small,
		VMConfig:     &targon.VMConfigInput{Hostname: "example-vm"},
		SSHKeys:      []string{sshKey},
		Ports:        []targon.PortConfig{{Port: 8080, Protocol: "TCP"}},
		Wait:         targon.WaitOptions{Timeout: 10 * time.Minute},
	})
	if err != nil {
		return err
	}
	defer deleteWorkload(ctx, client, vm.UID)
	fmt.Printf("VM %s is running\n", vm.UID)
	return nil
}

func runBareMetal(ctx context.Context, client *targon.Client, sshKey string) error {
	region := os.Getenv("TARGON_REGION")
	resource := os.Getenv("TARGON_BM_RESOURCE")
	if resource == "" {
		items, err := client.Inventory.List(ctx, targon.InventoryListParams{Type: "bm", Region: region})
		if err != nil {
			return err
		}
		for _, item := range items {
			if item.Available > 0 && item.Region != nil {
				resource, region = item.Name, item.Region.UID
				break
			}
		}
		if resource == "" {
			return errors.New("no Bare Metal inventory available")
		}
	}
	if region == "" {
		return errors.New("set TARGON_REGION when TARGON_BM_RESOURCE is set")
	}

	image := os.Getenv("TARGON_BM_IMAGE")
	if image == "" {
		images, err := client.BareMetal.Images(ctx)
		if err != nil {
			return err
		}
		if len(images) == 0 {
			return errors.New("no Bare Metal images available")
		}
		image = images[0].Name
	}

	// Create waits up to targon.DefaultBareMetalWaitTimeout (30 minutes).
	metal, err := client.BareMetal.Create(ctx, targon.BareMetalCreateParams{
		Name:         "example-bare-metal",
		Image:        image,
		ResourceName: resource,
		Region:       region,
		BMConfig:     &targon.BMConfigInput{Hostname: "example-bare-metal"},
		SSHKeys:      []string{sshKey},
	})
	if err != nil {
		return err
	}
	defer deleteWorkload(ctx, client, metal.UID)
	fmt.Printf("Bare Metal %s is running in %s\n", metal.UID, region)
	return nil
}

func deleteWorkload(ctx context.Context, client *targon.Client, uid string) {
	if err := client.Workloads.Delete(ctx, uid); err != nil {
		log.Printf("cleanup %s: %v", uid, err)
	}
}
