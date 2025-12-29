package handler

import (
	"net/http"

	"github.com/labstack/echo/v4"

	"github.com/hassan123789/go-ai-agent/internal/agent"
)

// ReflexionHandler handles Reflexion agent HTTP requests.
type ReflexionHandler struct {
	agent *agent.ReflexionAgent
}

// NewReflexionHandler creates a new ReflexionHandler.
func NewReflexionHandler(a *agent.ReflexionAgent) *ReflexionHandler {
	return &ReflexionHandler{
		agent: a,
	}
}

// ReflexionRequest represents the request body for reflexion endpoint.
type ReflexionRequest struct {
	Query   string         `json:"query" validate:"required"`
	History []AgentMessage `json:"history,omitempty"`
	Verbose bool           `json:"verbose,omitempty"`
}

// ReflexionResponse represents the response from the reflexion agent.
type ReflexionResponse struct {
	Output          string           `json:"output"`
	Steps           []StepInfo       `json:"steps,omitempty"`
	Reflections     []ReflectionInfo `json:"reflections,omitempty"`
	FinalEvaluation *EvaluationInfo  `json:"final_evaluation,omitempty"`
	TotalAttempts   int              `json:"total_attempts"`
	Usage           UsageInfo        `json:"usage"`
}

// ReflectionInfo represents a single reflection iteration.
type ReflectionInfo struct {
	Attempt    int             `json:"attempt"`
	Score      float64         `json:"score"`
	Evaluation *EvaluationInfo `json:"evaluation,omitempty"`
	Feedback   string          `json:"feedback,omitempty"`
}

// EvaluationInfo represents the evaluation of a response.
type EvaluationInfo struct {
	Score      float64  `json:"score"`
	Reasoning  string   `json:"reasoning"`
	Strengths  []string `json:"strengths,omitempty"`
	Weaknesses []string `json:"weaknesses,omitempty"`
}

// Run handles POST /api/reflexion requests.
func (h *ReflexionHandler) Run(c echo.Context) error {
	var req ReflexionRequest
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

	// Run the reflexion agent
	resp, err := h.agent.RunWithHistory(c.Request().Context(), history, req.Query)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "reflexion_error",
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

	// Extract reflections from metadata if available
	var reflections []ReflectionInfo
	var finalEval *EvaluationInfo
	totalAttempts := 1

	if metadata, ok := resp.Metadata["reflexion"]; ok {
		if refData, ok := metadata.(map[string]any); ok {
			if attempts, ok := refData["attempts"].(int); ok {
				totalAttempts = attempts
			}
			if refList, ok := refData["reflections"].([]agent.Reflection); ok {
				for i, ref := range refList {
					info := ReflectionInfo{
						Attempt:  i + 1,
						Score:    ref.Evaluation.Score,
						Feedback: ref.Feedback,
					}
					if req.Verbose {
						info.Evaluation = &EvaluationInfo{
							Score:      ref.Evaluation.Score,
							Reasoning:  ref.Evaluation.Reasoning,
							Strengths:  ref.Evaluation.Strengths,
							Weaknesses: ref.Evaluation.Weaknesses,
						}
					}
					reflections = append(reflections, info)
				}
			}
			if eval, ok := refData["final_evaluation"].(*agent.Evaluation); ok && eval != nil {
				finalEval = &EvaluationInfo{
					Score:      eval.Score,
					Reasoning:  eval.Reasoning,
					Strengths:  eval.Strengths,
					Weaknesses: eval.Weaknesses,
				}
			}
		}
	}

	return c.JSON(http.StatusOK, ReflexionResponse{
		Output:          resp.Output,
		Steps:           steps,
		Reflections:     reflections,
		FinalEvaluation: finalEval,
		TotalAttempts:   totalAttempts,
		Usage: UsageInfo{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	})
}
