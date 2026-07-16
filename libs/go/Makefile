.PHONY: build test lint fmt clean

build:
	go build ./...

test:
	go test ./...

lint:
	go vet ./...
	gofmt -l .

fmt:
	gofmt -w .

clean:
	go clean
