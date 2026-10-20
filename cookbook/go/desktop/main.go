package main

import (
	"context"
	"fmt"
	"log"

	"github.com/manifold-inc/targon-sdk/cookbook/go/internal/box"
)

func main() {
	ctx := context.Background()
	client, err := box.Client()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	sandbox, err := box.Create(ctx, client, "desk")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	desktop, err := sandbox.GetDesktop(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("desktop available=%v port=%d listening=%v\n", desktop.Available, desktop.Port, desktop.Listening)
	if desktop.WSURL != "" {
		fmt.Println("desktop ws url:", desktop.WSURL)
	}

	ticket, err := sandbox.MintAccessTicket(ctx, 60)
	if err != nil {
		log.Fatal(err)
	}
	// ticket.Ticket is a secret. Print only when it expires.
	fmt.Println("access ticket expires at", ticket.ExpiresAt)
}
