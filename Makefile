.PHONY: build run test lint fmt clean pre-push ci

# Binary name
BINARY=go-ai-agent

# Build the application
build:
	go build -o bin/$(BINARY) ./cmd/server

# Run the application
run:
	go run ./cmd/server

# Run all tests
test:
	go test -v -race -cover ./...

# Run short tests only
test-short:
	go test -v -short ./...

# Run linter
lint:
	golangci-lint run

# Format code
fmt:
	go fmt ./...
	goimports -w .

# Clean build artifacts
clean:
	rm -rf bin/
	go clean

# Pre-push checks (run before pushing to remote)
pre-push: fmt lint test-short

# Full CI simulation
ci: lint test

# Install development dependencies
deps:
	go install golang.org/x/tools/cmd/goimports@latest
	go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest

# Tidy go modules
tidy:
	go mod tidy

# Generate (placeholder for future code generation)
generate:
	go generate ./...
