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

	sandbox, err := box.Create(ctx, client, "term")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	session, err := sandbox.CreateTerminal(ctx, 80, 24)
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.DeleteTerminal(ctx, session.ID)
	fmt.Println("terminal", session.ID)

	conn, err := sandbox.ConnectTerminal(ctx, session.ID, targon.ConnectTerminalOptions{})
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	if err := conn.Write(ctx, []byte("echo cookbook-terminal\n")); err != nil {
		log.Fatal(err)
	}
	readCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	frame, err := conn.Read(readCtx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s", frame)
}
