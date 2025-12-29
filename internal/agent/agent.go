// Package agent provides AI agent implementations that can reason and act
// using tools to accomplish tasks.
package agent

import (
	"context"
)

// Agent defines the interface for AI agents.
// An agent can process queries and return responses, potentially using tools.
type Agent interface {
	// Run processes a query and returns the final response.
	Run(ctx context.Context, query string) (*Response, error)

	// RunWithHistory processes a query with conversation history.
	RunWithHistory(ctx context.Context, history []Message, query string) (*Response, error)
}

// Message represents a message in the conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Response represents the result of an agent run.
type Response struct {
	// Output is the final answer from the agent.
	Output string `json:"output"`

	// Steps contains the reasoning and action steps taken.
	Steps []Step `json:"steps,omitempty"`

	// Usage contains token usage information.
	Usage Usage `json:"usage"`

	// Metadata contains additional agent-specific information.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Step represents a single step in the agent's reasoning process.
type Step struct {
	// Type is the step type: "thought", "action", or "observation".
	Type string `json:"type"`

	// Content is the content of the step.
	Content string `json:"content"`

	// ToolName is the name of the tool called (for action steps).
	ToolName string `json:"tool_name,omitempty"`

	// ToolInput is the input to the tool (for action steps).
	ToolInput string `json:"tool_input,omitempty"`

	// ToolOutput is the output from the tool (for observation steps).
	ToolOutput string `json:"tool_output,omitempty"`
}

// Usage contains token usage information for the agent run.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StepType constants for the ReAct loop.
const (
	StepTypeThought     = "thought"
	StepTypeAction      = "action"
	StepTypeObservation = "observation"
)

// Config contains configuration for agents.
type Config struct {
	// SystemPrompt is the system prompt for the agent.
	// If empty, a default ReAct prompt is used.
	SystemPrompt string

	// MaxIterations is the maximum number of reasoning loops.
	// Prevents infinite loops. Default is 10.
	MaxIterations int

	// Verbose enables detailed logging of agent steps.
	Verbose bool
}

// DefaultConfig returns the default agent configuration.
func DefaultConfig() Config {
	return Config{
		MaxIterations: 10,
		Verbose:       false,
	}
}
