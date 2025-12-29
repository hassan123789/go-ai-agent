package handler

import (
	"net/http"

	"github.com/labstack/echo/v4"

	"github.com/hassan123789/go-ai-agent/internal/agent"
)

// OrchestratorHandler handles Orchestrator agent HTTP requests.
type OrchestratorHandler struct {
	agent *agent.OrchestratorAgent
}

// NewOrchestratorHandler creates a new OrchestratorHandler.
func NewOrchestratorHandler(a *agent.OrchestratorAgent) *OrchestratorHandler {
	return &OrchestratorHandler{
		agent: a,
	}
}

// OrchestratorRequest represents the request body for orchestrator endpoint.
type OrchestratorRequest struct {
	Query   string         `json:"query" validate:"required"`
	History []AgentMessage `json:"history,omitempty"`
	Verbose bool           `json:"verbose,omitempty"`
}

// OrchestratorResponse represents the response from the orchestrator agent.
type OrchestratorResponse struct {
	Output   string        `json:"output"`
	Plan     *TaskPlanInfo `json:"plan,omitempty"`
	Workers  []WorkerInfo  `json:"workers,omitempty"`
	Subtasks []SubtaskInfo `json:"subtasks,omitempty"`
	Usage    UsageInfo     `json:"usage"`
}

// TaskPlanInfo represents the task decomposition plan.
type TaskPlanInfo struct {
	Analysis     string `json:"analysis"`
	SubtaskCount int    `json:"subtask_count"`
}

// WorkerInfo represents information about a worker agent.
type WorkerInfo struct {
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	Capabilities []string `json:"capabilities,omitempty"`
}

// SubtaskInfo represents a subtask and its result.
type SubtaskInfo struct {
	ID          string `json:"id"`
	Description string `json:"description"`
	WorkerType  string `json:"worker_type"`
	Success     bool   `json:"success"`
	Output      string `json:"output,omitempty"`
	Error       string `json:"error,omitempty"`
	DurationMs  int64  `json:"duration_ms"`
}

// Run handles POST /api/orchestrator requests.
func (h *OrchestratorHandler) Run(c echo.Context) error {
	var req OrchestratorRequest
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

	// Run the orchestrator agent
	resp, err := h.agent.RunWithHistory(c.Request().Context(), history, req.Query)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "orchestrator_error",
			Message: err.Error(),
		})
	}

	// Build response
	result := OrchestratorResponse{
		Output: resp.Output,
		Usage: UsageInfo{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}

	// Extract detailed info if verbose mode
	if req.Verbose {
		// Extract plan from metadata
		if metadata, ok := resp.Metadata["orchestrator"]; ok {
			if orchData, ok := metadata.(map[string]any); ok {
				// Plan info
				if plan, ok := orchData["plan"].(*agent.TaskPlan); ok && plan != nil {
					result.Plan = &TaskPlanInfo{
						Analysis:     plan.Analysis,
						SubtaskCount: len(plan.Subtasks),
					}

					// Subtask results
					for _, subtask := range plan.Subtasks {
						result.Subtasks = append(result.Subtasks, SubtaskInfo{
							ID:          subtask.ID,
							Description: subtask.Description,
							WorkerType:  subtask.WorkerType,
						})
					}
				}

				// Results
				if results, ok := orchData["results"].([]agent.SubtaskResult); ok {
					resultMap := make(map[string]agent.SubtaskResult)
					for _, r := range results {
						resultMap[r.ID] = r
					}

					// Update subtasks with results
					for i := range result.Subtasks {
						if r, ok := resultMap[result.Subtasks[i].ID]; ok {
							result.Subtasks[i].Success = r.Success
							result.Subtasks[i].Output = r.Output
							result.Subtasks[i].Error = r.Error
							result.Subtasks[i].DurationMs = r.Duration
						}
					}
				}

				// Workers info
				if workers, ok := orchData["workers"].([]*agent.WorkerAgent); ok {
					for _, w := range workers {
						result.Workers = append(result.Workers, WorkerInfo{
							Name:         w.Name,
							Description:  w.Description,
							Capabilities: w.Tools,
						})
					}
				}
			}
		}
	}

	return c.JSON(http.StatusOK, result)
}

// ListWorkers handles GET /api/orchestrator/workers requests.
func (h *OrchestratorHandler) ListWorkers(c echo.Context) error {
	workers := h.agent.ListWorkers()

	workerInfos := make([]WorkerInfo, 0, len(workers))
	for _, w := range workers {
		workerInfos = append(workerInfos, WorkerInfo{
			Name:         w.Name,
			Description:  w.Description,
			Capabilities: w.Tools,
		})
	}

	return c.JSON(http.StatusOK, map[string]any{
		"workers": workerInfos,
		"count":   len(workerInfos),
	})
}
