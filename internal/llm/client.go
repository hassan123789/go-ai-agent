package llm

import (
	"context"
	"io"
)

// Message represents a chat message.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// Role represents the role of a message sender.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ChatRequest represents a request to the LLM.
type ChatRequest struct {
	Messages    []Message
	MaxTokens   int
	Temperature float32
	Stream      bool
}

// ChatResponse represents a response from the LLM.
type ChatResponse struct {
	Content      string
	FinishReason string
	Usage        Usage
}

// Usage contains token usage information.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// StreamChunk represents a single chunk in a streaming response.
type StreamChunk struct {
	Content      string
	FinishReason string
	Done         bool
	Error        error
}

// Client defines the interface for LLM providers.
type Client interface {
	// Chat sends a chat completion request and returns the response.
	Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error)

	// ChatStream sends a streaming chat completion request.
	ChatStream(ctx context.Context, req *ChatRequest) (<-chan StreamChunk, error)

	// Close releases any resources held by the client.
	Close() error
}

// StreamReader is an interface for reading streaming responses.
type StreamReader interface {
	Read() (StreamChunk, error)
	Close() error
}

// streamReaderAdapter adapts an io.ReadCloser to StreamReader.
type streamReaderAdapter struct {
	reader io.ReadCloser
}

func (s *streamReaderAdapter) Close() error {
	return s.reader.Close()
}
