package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// ToolCall represents a tool call made by the LLM.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolDefinition defines a tool that can be called by the LLM.
type ToolDefinition struct {
	Function FunctionDefinition `json:"function"`
	Type     string             `json:"type"`
}

// FunctionDefinition defines a function for the LLM.
type FunctionDefinition struct {
	Parameters  map[string]any `json:"parameters"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
}

// ChatWithToolsRequest represents a request with tool definitions.
type ChatWithToolsRequest struct {
	Messages    []Message
	Tools       []ToolDefinition
	MaxTokens   int
	Temperature float32
}

// ChatWithToolsResponse represents a response that may contain tool calls.
type ChatWithToolsResponse struct {
	ToolCalls    []ToolCall
	Content      string
	FinishReason string
	Usage        Usage
}

// ToolMessage represents the result of a tool call to be sent back to the LLM.
type ToolMessage struct {
	ToolCallID string
	Content    string
}

// ToolClient extends Client with function calling capabilities.
type ToolClient interface {
	Client

	// ChatWithTools sends a chat completion request with tool definitions.
	ChatWithTools(ctx context.Context, req *ChatWithToolsRequest) (*ChatWithToolsResponse, error)

	// ChatWithToolResults continues a conversation after tool execution.
	ChatWithToolResults(ctx context.Context, req *ChatWithToolsRequest, toolResults []ToolMessage) (*ChatWithToolsResponse, error)
}

// ChatWithTools sends a chat completion request with tool definitions.
func (c *OpenAIClient) ChatWithTools(ctx context.Context, req *ChatWithToolsRequest) (*ChatWithToolsResponse, error) {
	messages := convertMessages(req.Messages)
	tools := convertTools(req.Tools)

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
		Tools:       tools,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	})
	if err != nil {
		return nil, fmt.Errorf("chat with tools failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices in response")
	}

	choice := resp.Choices[0]
	return &ChatWithToolsResponse{
		Content:      choice.Message.Content,
		ToolCalls:    convertToolCalls(choice.Message.ToolCalls),
		FinishReason: string(choice.FinishReason),
		Usage: Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}, nil
}

// ChatWithToolResults continues a conversation after tool execution.
func (c *OpenAIClient) ChatWithToolResults(ctx context.Context, req *ChatWithToolsRequest, toolResults []ToolMessage) (*ChatWithToolsResponse, error) {
	messages := convertMessages(req.Messages)

	// Add tool result messages
	for _, result := range toolResults {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:       openai.ChatMessageRoleTool,
			Content:    result.Content,
			ToolCallID: result.ToolCallID,
		})
	}

	tools := convertTools(req.Tools)

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
		Tools:       tools,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	})
	if err != nil {
		return nil, fmt.Errorf("chat with tool results failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices in response")
	}

	choice := resp.Choices[0]
	return &ChatWithToolsResponse{
		Content:      choice.Message.Content,
		ToolCalls:    convertToolCalls(choice.Message.ToolCalls),
		FinishReason: string(choice.FinishReason),
		Usage: Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}, nil
}

// convertMessages converts our Message type to OpenAI's format.
func convertMessages(msgs []Message) []openai.ChatCompletionMessage {
	result := make([]openai.ChatCompletionMessage, len(msgs))
	for i, msg := range msgs {
		result[i] = openai.ChatCompletionMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		}
	}
	return result
}

// convertTools converts our ToolDefinition to OpenAI's format.
func convertTools(tools []ToolDefinition) []openai.Tool {
	result := make([]openai.Tool, len(tools))
	for i, tool := range tools {
		// Convert parameters map to json.RawMessage
		paramsBytes, _ := json.Marshal(tool.Function.Parameters)

		result[i] = openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  json.RawMessage(paramsBytes),
			},
		}
	}
	return result
}

// convertToolCalls converts OpenAI's ToolCall to our format.
func convertToolCalls(calls []openai.ToolCall) []ToolCall {
	result := make([]ToolCall, len(calls))
	for i, call := range calls {
		result[i] = ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: call.Function.Arguments,
		}
	}
	return result
}

// HasToolCalls returns true if the response contains tool calls.
func (r *ChatWithToolsResponse) HasToolCalls() bool {
	return len(r.ToolCalls) > 0
}

// IsComplete returns true if the response is complete (no more tool calls needed).
func (r *ChatWithToolsResponse) IsComplete() bool {
	return r.FinishReason == "stop" && len(r.ToolCalls) == 0
}
