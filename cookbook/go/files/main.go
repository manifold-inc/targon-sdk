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

	sandbox, err := box.Create(ctx, client, "files")
	if err != nil {
		log.Fatal(err)
	}
	defer sandbox.Delete(ctx)

	if err := sandbox.WriteFile(ctx, "/tmp/hello.txt", []byte("hello from the cookbook\n")); err != nil {
		log.Fatal(err)
	}
	result, err := sandbox.Exec(ctx, "cp /tmp/hello.txt /tmp/hello.out && wc -c /tmp/hello.out", 60)
	if err != nil {
		log.Fatal(err)
	}
	if result.Code != 0 {
		log.Fatal(result.Stderr)
	}
	text, err := sandbox.ReadFile(ctx, "/tmp/hello.out")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(result.Stdout)
	fmt.Printf("text: %s\nbytes: %d\n", text, len(text))
}
