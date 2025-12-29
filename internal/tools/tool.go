// Package tools provides the interface and implementations for AI agent tools.
// Tools are external capabilities that an AI agent can invoke during task execution,
// following the ReAct (Reasoning + Acting) pattern.
package tools

import (
	"context"
	"encoding/json"
)

// Tool represents an external capability that an AI agent can invoke.
// Each tool has a name, description, and a set of parameters defined as JSON Schema.
type Tool interface {
	// Name returns the unique identifier for this tool.
	// This is used by the LLM to specify which tool to call.
	Name() string

	// Description returns a human-readable description of what this tool does.
	// This helps the LLM understand when to use this tool.
	Description() string

	// Parameters returns the JSON Schema for the tool's input parameters.
	// This tells the LLM what arguments the tool expects.
	Parameters() ParameterSchema

	// Execute runs the tool with the given arguments and returns the result.
	// The arguments are passed as a JSON string from the LLM.
	Execute(ctx context.Context, arguments string) (Result, error)
}

// ParameterSchema defines the JSON Schema for tool parameters.
// This follows the OpenAI function calling specification.
type ParameterSchema struct {
	Type       string                    `json:"type"`
	Properties map[string]PropertySchema `json:"properties,omitempty"`
	Required   []string                  `json:"required,omitempty"`
}

// PropertySchema defines a single property in the parameter schema.
type PropertySchema struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
}

// Result represents the output of a tool execution.
type Result struct {
	// Metadata contains additional information about the execution.
	Metadata map[string]any `json:"metadata,omitempty"`

	// Output is the main result content to be returned to the LLM.
	Output string `json:"output"`

	// Error contains any error message if the tool execution failed.
	Error string `json:"error,omitempty"`
}

// Success creates a successful result with the given output.
func Success(output string) Result {
	return Result{Output: output}
}

// SuccessWithMetadata creates a successful result with output and metadata.
func SuccessWithMetadata(output string, metadata map[string]any) Result {
	return Result{Output: output, Metadata: metadata}
}

// Failure creates a failed result with the given error message.
func Failure(errMsg string) Result {
	return Result{Error: errMsg}
}

// IsSuccess returns true if the result represents a successful execution.
func (r Result) IsSuccess() bool {
	return r.Error == ""
}

// String returns the result as a string for the LLM.
// If there's an error, it returns the error message.
func (r Result) String() string {
	if r.Error != "" {
		return "Error: " + r.Error
	}
	return r.Output
}

// Definition represents a tool definition for the OpenAI API.
// This is the format expected by the tools parameter in chat completion.
type Definition struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition represents a function definition for OpenAI.
type FunctionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  ParameterSchema `json:"parameters"`
}

// ToDefinition converts a Tool to an OpenAI-compatible definition.
func ToDefinition(t Tool) Definition {
	return Definition{
		Type: "function",
		Function: FunctionDefinition{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Parameters(),
		},
	}
}

// ParseArguments parses the JSON arguments string into the given struct.
// This is a helper function for tool implementations.
func ParseArguments[T any](arguments string) (T, error) {
	var args T
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return args, err
	}
	return args, nil
}
