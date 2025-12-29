package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/sashabaranov/go-openai"
)

// StrictToolDefinition extends ToolDefinition with strict mode support.
type StrictToolDefinition struct {
	ToolDefinition
	Strict bool `json:"strict"`
}

// StreamingToolCall represents a tool call being streamed.
type StreamingToolCall struct {
	ID             string `json:"id"`
	Name           string `json:"name"`
	ArgumentsDelta string `json:"arguments_delta"`
	ArgumentsFull  string `json:"arguments_full"`
	IsComplete     bool   `json:"is_complete"`
}

// ToolStreamChunk represents a streaming chunk with tool call info.
type ToolStreamChunk struct {
	StreamChunk
	ToolCalls []StreamingToolCall `json:"tool_calls,omitempty"`
}

// ChatWithToolsStreamRequest is the request for streaming tool calls.
type ChatWithToolsStreamRequest struct {
	Messages    []Message
	Tools       []ToolDefinition
	StrictMode  bool
	MaxTokens   int
	Temperature float32
}

// RetryConfig configures retry behavior.
type RetryConfig struct {
	MaxRetries     int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	BackoffFactor  float64
}

// DefaultRetryConfig returns sensible retry defaults.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:     3,
		InitialBackoff: 1 * time.Second,
		MaxBackoff:     30 * time.Second,
		BackoffFactor:  2.0,
	}
}

// ChatWithToolsStream streams a chat completion with tool support.
func (c *OpenAIClient) ChatWithToolsStream(ctx context.Context, req *ChatWithToolsStreamRequest) (<-chan ToolStreamChunk, error) {
	messages := convertMessages(req.Messages)
	tools := convertToolsWithStrict(req.Tools, req.StrictMode)

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = c.defaultMax
	}

	temperature := req.Temperature
	if temperature <= 0 {
		temperature = 0.7
	}

	stream, err := c.client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    messages,
		Tools:       tools,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Stream:      true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create stream: %w", err)
	}

	ch := make(chan ToolStreamChunk)
	toolCallBuffer := make(map[int]*StreamingToolCall)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		for {
			response, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				// Send final tool calls
				finalCalls := make([]StreamingToolCall, 0)
				for _, tc := range toolCallBuffer {
					tc.IsComplete = true
					finalCalls = append(finalCalls, *tc)
				}
				if len(finalCalls) > 0 {
					ch <- ToolStreamChunk{
						StreamChunk: StreamChunk{Done: true},
						ToolCalls:   finalCalls,
					}
				} else {
					ch <- ToolStreamChunk{StreamChunk: StreamChunk{Done: true}}
				}
				return
			}
			if err != nil {
				ch <- ToolStreamChunk{StreamChunk: StreamChunk{Error: err, Done: true}}
				return
			}

			if len(response.Choices) == 0 {
				continue
			}

			choice := response.Choices[0]

			// Handle regular content
			chunk := ToolStreamChunk{
				StreamChunk: StreamChunk{
					Content:      choice.Delta.Content,
					FinishReason: string(choice.FinishReason),
					Done:         choice.FinishReason != "",
				},
			}

			// Handle tool calls
			for _, tc := range choice.Delta.ToolCalls {
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}

				if toolCallBuffer[idx] == nil {
					toolCallBuffer[idx] = &StreamingToolCall{
						ID:   tc.ID,
						Name: tc.Function.Name,
					}
				}

				toolCallBuffer[idx].ArgumentsDelta = tc.Function.Arguments
				toolCallBuffer[idx].ArgumentsFull += tc.Function.Arguments

				// Add current state to chunk
				currentCalls := make([]StreamingToolCall, 0, len(toolCallBuffer))
				for _, stc := range toolCallBuffer {
					currentCalls = append(currentCalls, *stc)
				}
				chunk.ToolCalls = currentCalls
			}

			ch <- chunk
		}
	}()

	return ch, nil
}

// ChatWithRetry executes a chat request with automatic retry.
func (c *OpenAIClient) ChatWithRetry(ctx context.Context, req *ChatRequest, retryConfig RetryConfig) (*ChatResponse, error) {
	var lastErr error
	backoff := retryConfig.InitialBackoff

	for attempt := 0; attempt <= retryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
			backoff = time.Duration(float64(backoff) * retryConfig.BackoffFactor)
			if backoff > retryConfig.MaxBackoff {
				backoff = retryConfig.MaxBackoff
			}
		}

		resp, err := c.Chat(ctx, req)
		if err == nil {
			return resp, nil
		}

		lastErr = err

		// Check if error is retryable
		if !isRetryableError(err) {
			return nil, err
		}
	}

	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

// ChatWithToolsRetry executes a chat with tools request with retry.
func (c *OpenAIClient) ChatWithToolsRetry(ctx context.Context, req *ChatWithToolsRequest, retryConfig RetryConfig) (*ChatWithToolsResponse, error) {
	var lastErr error
	backoff := retryConfig.InitialBackoff

	for attempt := 0; attempt <= retryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
			backoff = time.Duration(float64(backoff) * retryConfig.BackoffFactor)
			if backoff > retryConfig.MaxBackoff {
				backoff = retryConfig.MaxBackoff
			}
		}

		resp, err := c.ChatWithTools(ctx, req)
		if err == nil {
			return resp, nil
		}

		lastErr = err

		if !isRetryableError(err) {
			return nil, err
		}
	}

	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

// isRetryableError determines if an error should trigger a retry.
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()

	// Rate limit errors
	if contains(errStr, "rate limit", "429", "too many requests") {
		return true
	}

	// Server errors
	if contains(errStr, "500", "502", "503", "504", "server error") {
		return true
	}

	// Timeout errors
	if contains(errStr, "timeout", "deadline exceeded") {
		return true
	}

	// Connection errors
	if contains(errStr, "connection reset", "connection refused", "EOF") {
		return true
	}

	return false
}

// contains checks if s contains any of the substrings.
func contains(s string, substrings ...string) bool {
	for _, sub := range substrings {
		if len(sub) > 0 && len(s) >= len(sub) {
			for i := 0; i <= len(s)-len(sub); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
		}
	}
	return false
}

// convertToolsWithStrict converts tools with strict mode.
func convertToolsWithStrict(tools []ToolDefinition, strictMode bool) []openai.Tool {
	result := make([]openai.Tool, len(tools))
	for i, t := range tools {
		params := t.Function.Parameters
		if strictMode && params != nil {
			// Add strict mode requirements
			params["additionalProperties"] = false
		}

		result[i] = openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  params,
				Strict:      strictMode,
			},
		}
	}
	return result
}

// StructuredOutput defines a schema for structured responses.
type StructuredOutput struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Schema      map[string]any `json:"schema"`
	Strict      bool           `json:"strict"`
}

// ChatWithStructuredOutput requests a response matching the schema.
func (c *OpenAIClient) ChatWithStructuredOutput(ctx context.Context, req *ChatRequest, output StructuredOutput) (*ChatResponse, error) {
	messages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = openai.ChatCompletionMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = c.defaultMax
	}

	temperature := req.Temperature
	if temperature <= 0 {
		temperature = 0.7
	}

	// Build schema for response format
	schemaBytes, _ := json.Marshal(output.Schema)

	resp, err := c.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		ResponseFormat: &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:        output.Name,
				Description: output.Description,
				Schema:      json.RawMessage(schemaBytes),
				Strict:      output.Strict,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("structured output request failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices in response")
	}

	return &ChatResponse{
		Content:      resp.Choices[0].Message.Content,
		FinishReason: string(resp.Choices[0].FinishReason),
		Usage: Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}, nil
}

// ErrorHandler provides structured error handling.
type ErrorHandler struct {
	OnRateLimit    func(err error) error
	OnServerError  func(err error) error
	OnTimeout      func(err error) error
	OnInvalidInput func(err error) error
	OnDefault      func(err error) error
}

// DefaultErrorHandler returns a basic error handler.
func DefaultErrorHandler() *ErrorHandler {
	return &ErrorHandler{
		OnRateLimit: func(err error) error {
			return fmt.Errorf("rate limited, please retry later: %w", err)
		},
		OnServerError: func(err error) error {
			return fmt.Errorf("server error, please retry: %w", err)
		},
		OnTimeout: func(err error) error {
			return fmt.Errorf("request timed out: %w", err)
		},
		OnInvalidInput: func(err error) error {
			return fmt.Errorf("invalid input: %w", err)
		},
		OnDefault: func(err error) error {
			return err
		},
	}
}

// Handle classifies and handles an error.
func (h *ErrorHandler) Handle(err error) error {
	if err == nil {
		return nil
	}

	errStr := err.Error()

	if contains(errStr, "rate limit", "429") {
		if h.OnRateLimit != nil {
			return h.OnRateLimit(err)
		}
	}

	if contains(errStr, "500", "502", "503", "504", "server error") {
		if h.OnServerError != nil {
			return h.OnServerError(err)
		}
	}

	if contains(errStr, "timeout", "deadline exceeded") {
		if h.OnTimeout != nil {
			return h.OnTimeout(err)
		}
	}

	if contains(errStr, "invalid", "bad request", "400") {
		if h.OnInvalidInput != nil {
			return h.OnInvalidInput(err)
		}
	}

	if h.OnDefault != nil {
		return h.OnDefault(err)
	}

	return err
}
