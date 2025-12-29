package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"

	"github.com/hassan123789/go-ai-agent/internal/llm"
	"github.com/hassan123789/go-ai-agent/internal/tools"
)

// ReActAgent implements the ReAct (Reasoning + Acting) pattern.
// It interleaves reasoning (Thought) and acting (Action/Observation) steps
// to accomplish tasks using available tools.
//
// Reference: Yao et al., 2022 - "ReAct: Synergizing Reasoning and Acting in Language Models"
// https://arxiv.org/abs/2210.03629
type ReActAgent struct {
	llm    *llm.OpenAIClient
	tools  *tools.Registry
	config Config
}

// NewReActAgent creates a new ReAct agent with the given LLM client and tools.
func NewReActAgent(llmClient *llm.OpenAIClient, toolRegistry *tools.Registry, config Config) *ReActAgent {
	if config.MaxIterations <= 0 {
		config.MaxIterations = 10
	}
	if config.SystemPrompt == "" {
		config.SystemPrompt = defaultSystemPrompt
	}

	return &ReActAgent{
		llm:    llmClient,
		tools:  toolRegistry,
		config: config,
	}
}

// defaultSystemPrompt is the default system prompt for the ReAct agent.
const defaultSystemPrompt = `You are a helpful AI assistant that can use tools to help answer questions.

When you need to use a tool, call the appropriate function. After receiving the tool's response,
use that information to formulate your final answer.

Think step by step:
1. Analyze the user's question
2. Determine if you need to use any tools
3. If yes, call the appropriate tool with the correct parameters
4. Use the tool's output to formulate your answer
5. Provide a clear, helpful response

Be concise and accurate in your responses.`

// Run processes a query and returns the final response.
func (a *ReActAgent) Run(ctx context.Context, query string) (*Response, error) {
	return a.RunWithHistory(ctx, nil, query)
}

// RunWithHistory processes a query with conversation history.
func (a *ReActAgent) RunWithHistory(ctx context.Context, history []Message, query string) (*Response, error) {
	// Build initial messages
	messages := a.buildMessages(history, query)

	// Build tool definitions
	toolDefs := a.buildToolDefinitions()

	var allSteps []Step
	var totalUsage Usage

	// ReAct loop
	for i := 0; i < a.config.MaxIterations; i++ {
		if a.config.Verbose {
			log.Printf("[ReAct] Iteration %d/%d", i+1, a.config.MaxIterations)
		}

		// Call LLM with tools
		resp, err := a.llm.ChatWithTools(ctx, &llm.ChatWithToolsRequest{
			Messages: a.toLLMMessages(messages),
			Tools:    toolDefs,
		})
		if err != nil {
			return nil, fmt.Errorf("LLM call failed: %w", err)
		}

		// Accumulate usage
		totalUsage.PromptTokens += resp.Usage.PromptTokens
		totalUsage.CompletionTokens += resp.Usage.CompletionTokens
		totalUsage.TotalTokens += resp.Usage.TotalTokens

		// Check if we have tool calls
		if resp.HasToolCalls() {
			// Process each tool call
			for _, toolCall := range resp.ToolCalls {
				// Record action step
				actionStep := Step{
					Type:      StepTypeAction,
					ToolName:  toolCall.Name,
					ToolInput: toolCall.Arguments,
				}
				allSteps = append(allSteps, actionStep)

				if a.config.Verbose {
					log.Printf("[ReAct] Action: %s(%s)", toolCall.Name, toolCall.Arguments)
				}

				// Execute the tool
				result, err := a.executeTool(ctx, toolCall)
				if err != nil {
					return nil, fmt.Errorf("tool execution failed: %w", err)
				}

				// Record observation step
				observationStep := Step{
					Type:       StepTypeObservation,
					ToolName:   toolCall.Name,
					ToolOutput: result,
				}
				allSteps = append(allSteps, observationStep)

				if a.config.Verbose {
					log.Printf("[ReAct] Observation: %s", result)
				}

				// Add assistant message with tool call
				messages = append(messages, Message{
					Role:    "assistant",
					Content: formatToolCallMessage(toolCall),
				})

				// Add tool result message
				messages = append(messages, Message{
					Role:    "tool",
					Content: result,
				})
			}
		} else {
			// No tool calls - we have the final answer
			if a.config.Verbose {
				log.Printf("[ReAct] Final answer: %s", resp.Content)
			}

			return &Response{
				Output: resp.Content,
				Steps:  allSteps,
				Usage:  totalUsage,
			}, nil
		}
	}

	return nil, errors.New("max iterations exceeded without reaching a final answer")
}

// buildMessages constructs the initial message list.
func (a *ReActAgent) buildMessages(history []Message, query string) []Message {
	messages := make([]Message, 0, len(history)+2)

	// System prompt
	messages = append(messages, Message{
		Role:    "system",
		Content: a.config.SystemPrompt,
	})

	// History
	messages = append(messages, history...)

	// Current query
	messages = append(messages, Message{
		Role:    "user",
		Content: query,
	})

	return messages
}

// buildToolDefinitions converts the tool registry to LLM tool definitions.
func (a *ReActAgent) buildToolDefinitions() []llm.ToolDefinition {
	toolList := a.tools.List()
	defs := make([]llm.ToolDefinition, len(toolList))

	for i, tool := range toolList {
		params := tool.Parameters()
		defs[i] = llm.ToolDefinition{
			Type: "function",
			Function: llm.FunctionDefinition{
				Name:        tool.Name(),
				Description: tool.Description(),
				Parameters: map[string]any{
					"type":       params.Type,
					"properties": params.Properties,
					"required":   params.Required,
				},
			},
		}
	}

	return defs
}

// toLLMMessages converts agent messages to LLM messages.
func (a *ReActAgent) toLLMMessages(messages []Message) []llm.Message {
	result := make([]llm.Message, len(messages))
	for i, msg := range messages {
		result[i] = llm.Message{
			Role:    llm.Role(msg.Role),
			Content: msg.Content,
		}
	}
	return result
}

// executeTool executes a tool call and returns the result.
func (a *ReActAgent) executeTool(ctx context.Context, toolCall llm.ToolCall) (string, error) {
	tool := a.tools.Get(toolCall.Name)
	if tool == nil {
		return "", fmt.Errorf("tool %q not found", toolCall.Name)
	}

	result, err := tool.Execute(ctx, toolCall.Arguments)
	if err != nil {
		return "", fmt.Errorf("tool %q execution error: %w", toolCall.Name, err)
	}

	return result.String(), nil
}

// formatToolCallMessage formats a tool call for the message history.
func formatToolCallMessage(toolCall llm.ToolCall) string {
	data, _ := json.Marshal(map[string]any{
		"tool_call": map[string]any{
			"id":        toolCall.ID,
			"name":      toolCall.Name,
			"arguments": toolCall.Arguments,
		},
	})
	return string(data)
}
