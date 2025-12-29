package llm

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/sashabaranov/go-openai"
)

// OllamaClient implements the Client interface using Ollama's OpenAI-compatible API.
// Ollama runs LLMs locally and exposes an OpenAI-compatible endpoint.
type OllamaClient struct {
	client     *openai.Client
	model      string
	defaultMax int
}

// OllamaConfig contains configuration for the Ollama client.
type OllamaConfig struct {
	// BaseURL is the Ollama API endpoint (default: http://localhost:11434/v1)
	BaseURL   string
	Model     string
	MaxTokens int
}

// Common Ollama model names
const (
	OllamaLlama3_2   = "llama3.2"
	OllamaLlama3_1   = "llama3.1"
	OllamaLlama3     = "llama3"
	OllamaMistral    = "mistral"
	OllamaCodeLlama  = "codellama"
	OllamaGemma2     = "gemma2"
	OllamaQwen2_5    = "qwen2.5"
	OllamaDeepSeekR1 = "deepseek-r1"
	OllamaPhi3       = "phi3"
)

// NewOllamaClient creates a new Ollama client.
func NewOllamaClient(cfg OllamaConfig) (*OllamaClient, error) {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}

	model := cfg.Model
	if model == "" {
		model = OllamaLlama3_2
	}

	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	// Create OpenAI client configured for Ollama
	config := openai.DefaultConfig("")
	config.BaseURL = baseURL

	client := openai.NewClientWithConfig(config)

	return &OllamaClient{
		client:     client,
		model:      model,
		defaultMax: maxTokens,
	}, nil
}

// Chat sends a chat completion request and returns the response.
func (c *OllamaClient) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
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

	resp, err := c.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	})
	if err != nil {
		return nil, fmt.Errorf("chat completion failed: %w", err)
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

// ChatStream sends a streaming chat completion request.
func (c *OllamaClient) ChatStream(ctx context.Context, req *ChatRequest) (<-chan StreamChunk, error) {
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

	stream, err := c.client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Stream:      true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create stream: %w", err)
	}

	ch := make(chan StreamChunk)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		for {
			response, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				ch <- StreamChunk{Done: true}
				return
			}
			if err != nil {
				ch <- StreamChunk{Error: err, Done: true}
				return
			}

			if len(response.Choices) > 0 {
				choice := response.Choices[0]
				ch <- StreamChunk{
					Content:      choice.Delta.Content,
					FinishReason: string(choice.FinishReason),
					Done:         choice.FinishReason != "",
				}
			}
		}
	}()

	return ch, nil
}

// Close releases any resources held by the client.
func (c *OllamaClient) Close() error {
	// Ollama client doesn't have explicit cleanup
	return nil
}

// ListLocalModels lists all locally available Ollama models.
// This is a convenience method specific to Ollama.
func (c *OllamaClient) ListLocalModels(ctx context.Context) ([]string, error) {
	models, err := c.client.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	names := make([]string, len(models.Models))
	for i, model := range models.Models {
		names[i] = model.ID
	}
	return names, nil
}
