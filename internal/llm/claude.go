package llm

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// ClaudeClient implements the Client interface using Anthropic's Claude API.
type ClaudeClient struct {
	client     *anthropic.Client
	model      anthropic.Model
	defaultMax int
}

// ClaudeConfig contains configuration for the Claude client.
type ClaudeConfig struct {
	APIKey    string
	Model     string
	MaxTokens int
}

// Claude model constants for convenience
const (
	ClaudeOpus45   = string(anthropic.ModelClaudeOpus4_5_20251101)
	ClaudeOpus4    = string(anthropic.ModelClaudeOpus4_20250514)
	ClaudeSonnet45 = string(anthropic.ModelClaudeSonnet4_5_20250929)
	ClaudeSonnet4  = string(anthropic.ModelClaudeSonnet4_20250514)
	ClaudeSonnet37 = string(anthropic.ModelClaude3_7SonnetLatest)
	ClaudeHaiku45  = string(anthropic.ModelClaudeHaiku4_5_20251001)
	ClaudeHaiku35  = string(anthropic.ModelClaude3_5HaikuLatest)
	ClaudeHaiku3   = string(anthropic.ModelClaude_3_Haiku_20240307)
)

// NewClaudeClient creates a new Claude client.
func NewClaudeClient(cfg ClaudeConfig) (*ClaudeClient, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("API key is required")
	}

	client := anthropic.NewClient(
		option.WithAPIKey(cfg.APIKey),
	)

	model := anthropic.Model(cfg.Model)
	if cfg.Model == "" {
		model = anthropic.ModelClaude3_5HaikuLatest
	}

	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	return &ClaudeClient{
		client:     &client,
		model:      model,
		defaultMax: maxTokens,
	}, nil
}

// Chat sends a chat completion request and returns the response.
func (c *ClaudeClient) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	messages := make([]anthropic.MessageParam, 0, len(req.Messages))
	var systemPrompt string

	for _, msg := range req.Messages {
		switch msg.Role {
		case RoleSystem:
			systemPrompt = msg.Content
		case RoleUser:
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		case RoleAssistant:
			messages = append(messages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = c.defaultMax
	}

	params := anthropic.MessageNewParams{
		Model:     c.model,
		MaxTokens: int64(maxTokens),
		Messages:  messages,
	}

	// Add system prompt if present
	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemPrompt},
		}
	}

	// Set temperature if provided
	if req.Temperature > 0 {
		params.Temperature = anthropic.Float(float64(req.Temperature))
	}

	resp, err := c.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("chat completion failed: %w", err)
	}

	// Extract text content from response
	var content string
	for _, block := range resp.Content {
		if block.Type == "text" {
			content += block.Text
		}
	}

	return &ChatResponse{
		Content:      content,
		FinishReason: string(resp.StopReason),
		Usage: Usage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
	}, nil
}

// ChatStream sends a streaming chat completion request.
func (c *ClaudeClient) ChatStream(ctx context.Context, req *ChatRequest) (<-chan StreamChunk, error) {
	messages := make([]anthropic.MessageParam, 0, len(req.Messages))
	var systemPrompt string

	for _, msg := range req.Messages {
		switch msg.Role {
		case RoleSystem:
			systemPrompt = msg.Content
		case RoleUser:
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		case RoleAssistant:
			messages = append(messages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = c.defaultMax
	}

	params := anthropic.MessageNewParams{
		Model:     c.model,
		MaxTokens: int64(maxTokens),
		Messages:  messages,
	}

	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemPrompt},
		}
	}

	if req.Temperature > 0 {
		params.Temperature = anthropic.Float(float64(req.Temperature))
	}

	stream := c.client.Messages.NewStreaming(ctx, params)

	ch := make(chan StreamChunk)

	go func() {
		defer close(ch)

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "content_block_delta":
				if event.Delta.Type == "text_delta" {
					ch <- StreamChunk{
						Content: event.Delta.Text,
						Done:    false,
					}
				}
			case "message_stop":
				ch <- StreamChunk{
					Done: true,
				}
				return
			case "message_delta":
				if event.Delta.StopReason != "" {
					ch <- StreamChunk{
						FinishReason: string(event.Delta.StopReason),
						Done:         true,
					}
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- StreamChunk{
				Error: err,
				Done:  true,
			}
			return
		}

		// Ensure we send a done signal
		ch <- StreamChunk{Done: true}
	}()

	return ch, nil
}

// Close releases any resources held by the client.
func (c *ClaudeClient) Close() error {
	// Anthropic client doesn't have explicit cleanup
	return nil
}
