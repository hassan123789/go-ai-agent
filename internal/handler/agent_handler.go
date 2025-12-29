package handler

import (
	"net/http"

	"github.com/labstack/echo/v4"

	"github.com/hassan123789/go-ai-agent/internal/agent"
)

// AgentHandler handles agent-related HTTP requests.
type AgentHandler struct {
	agent agent.Agent
}

// NewAgentHandler creates a new AgentHandler.
func NewAgentHandler(a agent.Agent) *AgentHandler {
	return &AgentHandler{
		agent: a,
	}
}

// AgentRequest represents the request body for agent endpoint.
type AgentRequest struct {
	Query   string         `json:"query" validate:"required"`
	History []AgentMessage `json:"history,omitempty"`
	Verbose bool           `json:"verbose,omitempty"`
}

// AgentMessage represents a message in the conversation history.
type AgentMessage struct {
	Role    string `json:"role" validate:"required,oneof=system user assistant"`
	Content string `json:"content" validate:"required"`
}

// AgentResponse represents the response from the agent.
type AgentResponse struct {
	Output string     `json:"output"`
	Steps  []StepInfo `json:"steps,omitempty"`
	Usage  UsageInfo  `json:"usage"`
}

// StepInfo represents a single step in the agent's reasoning.
type StepInfo struct {
	Type       string `json:"type"`
	Content    string `json:"content,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
	ToolInput  string `json:"tool_input,omitempty"`
	ToolOutput string `json:"tool_output,omitempty"`
}

// Run handles POST /api/agent requests.
func (h *AgentHandler) Run(c echo.Context) error {
	var req AgentRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "invalid_request",
			Message: "Failed to parse request body",
		})
	}

	if req.Query == "" {
		return c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "validation_error",
			Message: "Query is required",
		})
	}

	// Convert history to agent messages
	history := make([]agent.Message, 0, len(req.History))
	for _, msg := range req.History {
		history = append(history, agent.Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Run the agent
	resp, err := h.agent.RunWithHistory(c.Request().Context(), history, req.Query)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "agent_error",
			Message: err.Error(),
		})
	}

	// Convert steps
	var steps []StepInfo
	if req.Verbose {
		for _, step := range resp.Steps {
			steps = append(steps, StepInfo{
				Type:       step.Type,
				Content:    step.Content,
				ToolName:   step.ToolName,
				ToolInput:  step.ToolInput,
				ToolOutput: step.ToolOutput,
			})
		}
	}

	return c.JSON(http.StatusOK, AgentResponse{
		Output: resp.Output,
		Steps:  steps,
		Usage: UsageInfo{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	})
}
