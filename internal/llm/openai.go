package llm

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/sashabaranov/go-openai"
)

// OpenAIClient implements the Client interface using OpenAI's API.
type OpenAIClient struct {
	client     *openai.Client
	model      string
	defaultMax int
}

// OpenAIConfig contains configuration for the OpenAI client.
type OpenAIConfig struct {
	APIKey    string
	Model     string
	MaxTokens int
}

// NewOpenAIClient creates a new OpenAI client.
func NewOpenAIClient(cfg OpenAIConfig) (*OpenAIClient, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("API key is required")
	}

	client := openai.NewClient(cfg.APIKey)

	model := cfg.Model
	if model == "" {
		model = openai.GPT4oMini
	}

	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	return &OpenAIClient{
		client:     client,
		model:      model,
		defaultMax: maxTokens,
	}, nil
}

// Chat sends a chat completion request and returns the response.
func (c *OpenAIClient) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
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
func (c *OpenAIClient) ChatStream(ctx context.Context, req *ChatRequest) (<-chan StreamChunk, error) {
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
func (c *OpenAIClient) Close() error {
	// OpenAI client doesn't have explicit cleanup
	return nil
}
