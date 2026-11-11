package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	targon "github.com/manifold-inc/targon-sdk/libs/go/v4"
)

func main() {
	ctx := context.Background()
	client, err := targon.NewFromEnv()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	const prefix = "cb-"
	cursor := ""
	deleted := 0
	for {
		page, err := client.Sandboxes.List(ctx, targon.ListSandboxesParams{
			Page: targon.Page{Limit: 100, Cursor: cursor},
		})
		if err != nil {
			log.Fatal(err)
		}
		for _, item := range page.Items {
			if !strings.HasPrefix(item.Name, prefix) {
				continue
			}
			fmt.Printf("deleting %s (%s)\n", item.UID, item.Name)
			if err := client.Sandboxes.Delete(ctx, item.UID); err != nil {
				log.Fatal(err)
			}
			deleted++
		}
		if page.NextCursor == nil || *page.NextCursor == "" {
			break
		}
		cursor = *page.NextCursor
	}
	fmt.Printf("deleted %d sandbox(es)\n", deleted)
}
