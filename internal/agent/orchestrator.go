package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"

	"github.com/hassan123789/go-ai-agent/internal/llm"
	"github.com/hassan123789/go-ai-agent/internal/tools"
)

// OrchestratorAgent implements the Orchestrator-Workers pattern.
// It dynamically decomposes tasks and delegates to specialized worker agents.
//
// Reference: Anthropic - "Building Effective Agents"
// https://www.anthropic.com/research/building-effective-agents
//
// Flow:
//  1. Orchestrator analyzes the task and creates a plan
//  2. Tasks are delegated to specialized workers
//  3. Workers execute in parallel where possible
//  4. Results are synthesized by the orchestrator
type OrchestratorAgent struct {
	llm     *llm.OpenAIClient
	tools   *tools.Registry
	workers map[string]*WorkerAgent
	config  OrchestratorConfig
}

// OrchestratorConfig contains configuration for the orchestrator.
type OrchestratorConfig struct {
	Config

	// MaxWorkers is the maximum number of parallel workers.
	MaxWorkers int

	// PlanningPrompt is the prompt for task decomposition.
	PlanningPrompt string

	// SynthesisPrompt is the prompt for result synthesis.
	SynthesisPrompt string
}

// WorkerAgent is a specialized agent for specific tasks.
type WorkerAgent struct {
	Name         string
	Description  string
	SystemPrompt string
	Tools        []string // Tool names this worker can use
	llm          *llm.OpenAIClient
	registry     *tools.Registry
}

// TaskPlan represents the orchestrator's plan.
type TaskPlan struct {
	// Analysis is the orchestrator's analysis of the task.
	Analysis string `json:"analysis"`

	// Subtasks are the decomposed tasks.
	Subtasks []Subtask `json:"subtasks"`

	// Dependencies maps subtask IDs to their dependencies.
	Dependencies map[string][]string `json:"dependencies,omitempty"`
}

// Subtask represents a single unit of work.
type Subtask struct {
	ID          string `json:"id"`
	Description string `json:"description"`
	WorkerType  string `json:"worker_type"`
	Input       string `json:"input"`
}

// SubtaskResult contains the result of a subtask execution.
type SubtaskResult struct {
	ID       string `json:"id"`
	Success  bool   `json:"success"`
	Output   string `json:"output"`
	Error    string `json:"error,omitempty"`
	Duration int64  `json:"duration_ms"`
}

// NewOrchestratorAgent creates a new orchestrator agent.
func NewOrchestratorAgent(llmClient *llm.OpenAIClient, toolRegistry *tools.Registry, config OrchestratorConfig) *OrchestratorAgent {
	if config.MaxIterations <= 0 {
		config.MaxIterations = 10
	}
	if config.MaxWorkers <= 0 {
		config.MaxWorkers = 5
	}
	if config.SystemPrompt == "" {
		config.SystemPrompt = defaultOrchestratorPrompt
	}
	if config.PlanningPrompt == "" {
		config.PlanningPrompt = defaultPlanningPrompt
	}
	if config.SynthesisPrompt == "" {
		config.SynthesisPrompt = defaultSynthesisPrompt
	}

	return &OrchestratorAgent{
		llm:     llmClient,
		tools:   toolRegistry,
		workers: make(map[string]*WorkerAgent),
		config:  config,
	}
}

const defaultOrchestratorPrompt = `You are an orchestrator agent that breaks down complex tasks into smaller subtasks.

Your role is to:
1. Analyze the user's request
2. Decompose it into manageable subtasks
3. Assign each subtask to the appropriate worker type
4. Synthesize the results into a coherent response

Available worker types:
- general: For general reasoning and simple tasks
- calculator: For mathematical computations
- researcher: For information gathering and analysis
- writer: For content creation and editing

Be strategic in your task decomposition to maximize efficiency.`

const defaultPlanningPrompt = `Analyze this task and create a plan:

Task: %s

Create a JSON plan with this structure:
{
  "analysis": "Your analysis of what needs to be done",
  "subtasks": [
    {
      "id": "task_1",
      "description": "What this subtask accomplishes",
      "worker_type": "general|calculator|researcher|writer",
      "input": "The specific input for this subtask"
    }
  ],
  "dependencies": {
    "task_2": ["task_1"]  // task_2 depends on task_1
  }
}

Return only valid JSON.`

const defaultSynthesisPrompt = `Synthesize these subtask results into a final response:

Original Task: %s

Subtask Results:
%s

Create a coherent, comprehensive response that addresses the original task.
Integrate all relevant information from the subtask results.`

// RegisterWorker adds a specialized worker agent.
func (o *OrchestratorAgent) RegisterWorker(worker *WorkerAgent) {
	worker.llm = o.llm
	worker.registry = o.tools
	o.workers[worker.Name] = worker
}

// ListWorkers returns all registered worker agents.
func (o *OrchestratorAgent) ListWorkers() []*WorkerAgent {
	workers := make([]*WorkerAgent, 0, len(o.workers))
	for _, w := range o.workers {
		workers = append(workers, w)
	}
	return workers
}

// Run processes a query using the orchestrator-workers pattern.
func (o *OrchestratorAgent) Run(ctx context.Context, query string) (*Response, error) {
	return o.RunWithHistory(ctx, nil, query)
}

// RunWithHistory processes a query with conversation history.
func (o *OrchestratorAgent) RunWithHistory(ctx context.Context, history []Message, query string) (*Response, error) {
	var allSteps []Step
	var totalUsage Usage

	// Step 1: Create execution plan
	if o.config.Verbose {
		log.Printf("[Orchestrator] Creating plan for: %s", truncate(query, 50))
	}

	plan, planUsage, err := o.createPlan(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("planning failed: %w", err)
	}
	totalUsage = addUsage(totalUsage, planUsage)

	allSteps = append(allSteps, Step{
		Type:    "planning",
		Content: plan.Analysis,
	})

	if o.config.Verbose {
		log.Printf("[Orchestrator] Plan created with %d subtasks", len(plan.Subtasks))
	}

	// Step 2: Execute subtasks
	results, execUsage := o.executeSubtasks(ctx, plan)
	totalUsage = addUsage(totalUsage, execUsage)

	for _, result := range results {
		stepType := StepTypeObservation
		if !result.Success {
			stepType = "error"
		}
		allSteps = append(allSteps, Step{
			Type:       stepType,
			ToolName:   result.ID,
			ToolOutput: result.Output,
		})
	}

	// Step 3: Synthesize results
	if o.config.Verbose {
		log.Printf("[Orchestrator] Synthesizing %d results", len(results))
	}

	synthesis, synthUsage, err := o.synthesize(ctx, query, results)
	if err != nil {
		return nil, fmt.Errorf("synthesis failed: %w", err)
	}
	totalUsage = addUsage(totalUsage, synthUsage)

	allSteps = append(allSteps, Step{
		Type:    "synthesis",
		Content: synthesis,
	})

	return &Response{
		Output: synthesis,
		Steps:  allSteps,
		Usage:  totalUsage,
	}, nil
}

// createPlan generates an execution plan.
func (o *OrchestratorAgent) createPlan(ctx context.Context, query string) (*TaskPlan, Usage, error) {
	prompt := fmt.Sprintf(o.config.PlanningPrompt, query)

	resp, err := o.llm.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: o.config.SystemPrompt},
			{Role: llm.RoleUser, Content: prompt},
		},
		Temperature: 0.3,
	})
	if err != nil {
		return nil, Usage{}, err
	}

	var plan TaskPlan
	if err := json.Unmarshal([]byte(extractJSON(resp.Content)), &plan); err != nil {
		// Create a simple single-task plan if parsing fails
		plan = TaskPlan{
			Analysis: "Direct execution",
			Subtasks: []Subtask{{
				ID:          "task_1",
				Description: query,
				WorkerType:  "general",
				Input:       query,
			}},
		}
	}

	return &plan, Usage{
		PromptTokens:     resp.Usage.PromptTokens,
		CompletionTokens: resp.Usage.CompletionTokens,
		TotalTokens:      resp.Usage.TotalTokens,
	}, nil
}

// executeSubtasks runs all subtasks, respecting dependencies.
func (o *OrchestratorAgent) executeSubtasks(ctx context.Context, plan *TaskPlan) ([]SubtaskResult, Usage) {
	var totalUsage Usage
	results := make([]SubtaskResult, 0, len(plan.Subtasks))
	resultMap := make(map[string]*SubtaskResult)
	var mu sync.Mutex

	// Group tasks by dependency level
	levels := o.groupByLevel(plan)

	for _, level := range levels {
		// Execute tasks at this level in parallel
		var wg sync.WaitGroup
		semaphore := make(chan struct{}, o.config.MaxWorkers)

		for _, task := range level {
			wg.Add(1)
			go func(t Subtask) {
				defer wg.Done()
				semaphore <- struct{}{}
				defer func() { <-semaphore }()

				// Build input with dependency results
				input := o.buildTaskInput(t, plan.Dependencies, resultMap)

				result, usage := o.executeSubtask(ctx, t, input)

				mu.Lock()
				results = append(results, result)
				resultMap[t.ID] = &result
				totalUsage = addUsage(totalUsage, usage)
				mu.Unlock()
			}(task)
		}

		wg.Wait()
	}

	return results, totalUsage
}

// groupByLevel groups subtasks by dependency level.
func (o *OrchestratorAgent) groupByLevel(plan *TaskPlan) [][]Subtask {
	if len(plan.Subtasks) == 0 {
		return nil
	}

	// Simple case: no dependencies
	if len(plan.Dependencies) == 0 {
		return [][]Subtask{plan.Subtasks}
	}

	// Build dependency graph
	taskMap := make(map[string]Subtask)
	for _, t := range plan.Subtasks {
		taskMap[t.ID] = t
	}

	// Calculate levels using topological sort
	levels := make([][]Subtask, 0)
	completed := make(map[string]bool)

	for len(completed) < len(plan.Subtasks) {
		var currentLevel []Subtask

		for _, task := range plan.Subtasks {
			if completed[task.ID] {
				continue
			}

			// Check if all dependencies are completed
			deps := plan.Dependencies[task.ID]
			allDepsCompleted := true
			for _, dep := range deps {
				if !completed[dep] {
					allDepsCompleted = false
					break
				}
			}

			if allDepsCompleted {
				currentLevel = append(currentLevel, task)
			}
		}

		if len(currentLevel) == 0 {
			// Circular dependency or missing task - add remaining
			for _, task := range plan.Subtasks {
				if !completed[task.ID] {
					currentLevel = append(currentLevel, task)
				}
			}
		}

		for _, task := range currentLevel {
			completed[task.ID] = true
		}

		levels = append(levels, currentLevel)
	}

	return levels
}

// buildTaskInput creates input with dependency results.
func (o *OrchestratorAgent) buildTaskInput(task Subtask, deps map[string][]string, results map[string]*SubtaskResult) string {
	taskDeps := deps[task.ID]
	if len(taskDeps) == 0 {
		return task.Input
	}

	var sb strings.Builder
	sb.WriteString(task.Input)
	sb.WriteString("\n\nContext from previous tasks:\n")

	for _, depID := range taskDeps {
		if result, ok := results[depID]; ok && result.Success {
			sb.WriteString(fmt.Sprintf("- %s: %s\n", depID, truncate(result.Output, 200)))
		}
	}

	return sb.String()
}

// executeSubtask runs a single subtask.
func (o *OrchestratorAgent) executeSubtask(ctx context.Context, task Subtask, input string) (SubtaskResult, Usage) {
	worker := o.workers[task.WorkerType]
	if worker == nil {
		worker = o.workers["general"]
	}

	if worker == nil {
		// Use default execution
		return o.executeDefault(ctx, task, input)
	}

	return worker.Execute(ctx, input)
}

// executeDefault runs a task without a specialized worker.
func (o *OrchestratorAgent) executeDefault(ctx context.Context, task Subtask, input string) (SubtaskResult, Usage) {
	resp, err := o.llm.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a helpful assistant. Complete the given task."},
			{Role: llm.RoleUser, Content: input},
		},
	})
	if err != nil {
		return SubtaskResult{
			ID:      task.ID,
			Success: false,
			Error:   err.Error(),
		}, Usage{}
	}

	return SubtaskResult{
			ID:      task.ID,
			Success: true,
			Output:  resp.Content,
		}, Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
}

// synthesize combines subtask results.
func (o *OrchestratorAgent) synthesize(ctx context.Context, query string, results []SubtaskResult) (string, Usage, error) {
	var sb strings.Builder
	for _, r := range results {
		status := "✓"
		if !r.Success {
			status = "✗"
		}
		sb.WriteString(fmt.Sprintf("[%s] %s: %s\n\n", status, r.ID, r.Output))
	}

	prompt := fmt.Sprintf(o.config.SynthesisPrompt, query, sb.String())

	resp, err := o.llm.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a skilled synthesizer. Create coherent responses from multiple inputs."},
			{Role: llm.RoleUser, Content: prompt},
		},
	})
	if err != nil {
		return "", Usage{}, err
	}

	return resp.Content, Usage{
		PromptTokens:     resp.Usage.PromptTokens,
		CompletionTokens: resp.Usage.CompletionTokens,
		TotalTokens:      resp.Usage.TotalTokens,
	}, nil
}

// Execute runs the worker on an input.
func (w *WorkerAgent) Execute(ctx context.Context, input string) (SubtaskResult, Usage) {
	// Nil check for LLM
	if w.llm == nil {
		return SubtaskResult{
			ID:      "no-llm",
			Success: false,
			Error:   "LLM not configured for worker",
		}, Usage{}
	}

	prompt := w.SystemPrompt
	if prompt == "" {
		prompt = fmt.Sprintf("You are a %s agent. %s", w.Name, w.Description)
	}

	// If worker has specific tools, use them
	if len(w.Tools) > 0 && w.registry != nil {
		// Create filtered tool registry
		filteredTools := make([]llm.ToolDefinition, 0)
		for _, toolName := range w.Tools {
			if tool := w.registry.Get(toolName); tool != nil {
				params := tool.Parameters()
				filteredTools = append(filteredTools, llm.ToolDefinition{
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
				})
			}
		}

		if len(filteredTools) > 0 {
			return w.executeWithTools(ctx, input, prompt, filteredTools)
		}
	}

	// Simple execution without tools
	resp, err := w.llm.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: prompt},
			{Role: llm.RoleUser, Content: input},
		},
	})
	if err != nil {
		return SubtaskResult{
			Success: false,
			Error:   err.Error(),
		}, Usage{}
	}

	return SubtaskResult{
			Success: true,
			Output:  resp.Content,
		}, Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
}

// executeWithTools runs the worker with tool calling.
func (w *WorkerAgent) executeWithTools(ctx context.Context, input, prompt string, tools []llm.ToolDefinition) (SubtaskResult, Usage) {
	resp, err := w.llm.ChatWithTools(ctx, &llm.ChatWithToolsRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: prompt},
			{Role: llm.RoleUser, Content: input},
		},
		Tools: tools,
	})
	if err != nil {
		return SubtaskResult{
			Success: false,
			Error:   err.Error(),
		}, Usage{}
	}

	// Handle tool calls
	if resp.HasToolCalls() {
		var results []string
		for _, tc := range resp.ToolCalls {
			if tool := w.registry.Get(tc.Name); tool != nil {
				result, err := tool.Execute(ctx, tc.Arguments)
				if err != nil {
					results = append(results, fmt.Sprintf("%s error: %v", tc.Name, err))
				} else {
					results = append(results, fmt.Sprintf("%s: %s", tc.Name, result.String()))
				}
			}
		}
		return SubtaskResult{
				Success: true,
				Output:  strings.Join(results, "\n"),
			}, Usage{
				PromptTokens:     resp.Usage.PromptTokens,
				CompletionTokens: resp.Usage.CompletionTokens,
				TotalTokens:      resp.Usage.TotalTokens,
			}
	}

	return SubtaskResult{
			Success: true,
			Output:  resp.Content,
		}, Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
}

// addUsage combines two usage stats.
func addUsage(a, b Usage) Usage {
	return Usage{
		PromptTokens:     a.PromptTokens + b.PromptTokens,
		CompletionTokens: a.CompletionTokens + b.CompletionTokens,
		TotalTokens:      a.TotalTokens + b.TotalTokens,
	}
}

// NewGeneralWorker creates a general-purpose worker.
func NewGeneralWorker() *WorkerAgent {
	return &WorkerAgent{
		Name:        "general",
		Description: "A general-purpose assistant for reasoning and simple tasks",
		SystemPrompt: `You are a general-purpose AI assistant.
Handle the given task thoughtfully and provide a clear, helpful response.`,
	}
}

// NewCalculatorWorker creates a calculator worker.
func NewCalculatorWorker() *WorkerAgent {
	return &WorkerAgent{
		Name:        "calculator",
		Description: "Specialized in mathematical computations",
		SystemPrompt: `You are a mathematical computation specialist.
Solve mathematical problems step by step, showing your work.
Use the calculator tool for accurate computations.`,
		Tools: []string{"calculator"},
	}
}

// NewResearcherWorker creates a research worker.
func NewResearcherWorker() *WorkerAgent {
	return &WorkerAgent{
		Name:        "researcher",
		Description: "Specialized in information gathering and analysis",
		SystemPrompt: `You are a research specialist.
Analyze information thoroughly and provide well-reasoned conclusions.
Cite sources and explain your reasoning.`,
	}
}

// NewWriterWorker creates a content writer worker.
func NewWriterWorker() *WorkerAgent {
	return &WorkerAgent{
		Name:        "writer",
		Description: "Specialized in content creation and editing",
		SystemPrompt: `You are a skilled content writer.
Create clear, engaging, and well-structured content.
Adapt your style to the task requirements.`,
	}
}
