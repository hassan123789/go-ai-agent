package agent

import (
	"context"
	"testing"
)

func TestOrchestratorAgent_TaskPlan(t *testing.T) {
	plan := TaskPlan{
		Analysis: "Test analysis",
		Subtasks: []Subtask{
			{ID: "task_1", Description: "First task", WorkerType: "general", Input: "input1"},
			{ID: "task_2", Description: "Second task", WorkerType: "calculator", Input: "input2"},
		},
		Dependencies: map[string][]string{
			"task_2": {"task_1"},
		},
	}

	if plan.Analysis != "Test analysis" {
		t.Errorf("expected Analysis 'Test analysis', got %s", plan.Analysis)
	}

	if len(plan.Subtasks) != 2 {
		t.Errorf("expected 2 subtasks, got %d", len(plan.Subtasks))
	}

	if plan.Subtasks[0].WorkerType != "general" {
		t.Errorf("expected WorkerType 'general', got %s", plan.Subtasks[0].WorkerType)
	}

	deps := plan.Dependencies["task_2"]
	if len(deps) != 1 || deps[0] != "task_1" {
		t.Errorf("expected task_2 to depend on task_1, got %v", deps)
	}
}

func TestOrchestratorAgent_SubtaskResult(t *testing.T) {
	result := SubtaskResult{
		ID:       "task_1",
		Success:  true,
		Output:   "result output",
		Duration: 100,
	}

	if result.ID != "task_1" {
		t.Errorf("expected ID 'task_1', got %s", result.ID)
	}

	if !result.Success {
		t.Error("expected Success to be true")
	}

	if result.Duration != 100 {
		t.Errorf("expected Duration 100, got %d", result.Duration)
	}
}

func TestOrchestratorAgent_SubtaskResultError(t *testing.T) {
	result := SubtaskResult{
		ID:      "task_1",
		Success: false,
		Error:   "something went wrong",
	}

	if result.Success {
		t.Error("expected Success to be false")
	}

	if result.Error != "something went wrong" {
		t.Errorf("expected Error message, got %s", result.Error)
	}
}

func TestOrchestratorConfig_Defaults(t *testing.T) {
	config := OrchestratorConfig{}

	if config.MaxWorkers != 0 {
		t.Errorf("expected 0 default MaxWorkers, got %d", config.MaxWorkers)
	}

	if config.MaxIterations != 0 {
		t.Errorf("expected 0 default MaxIterations, got %d", config.MaxIterations)
	}
}

func TestWorkerAgent_Structure(t *testing.T) {
	worker := &WorkerAgent{
		Name:         "test_worker",
		Description:  "A test worker",
		SystemPrompt: "You are a test assistant",
		Tools:        []string{"calculator"},
	}

	if worker.Name != "test_worker" {
		t.Errorf("expected Name 'test_worker', got %s", worker.Name)
	}

	if len(worker.Tools) != 1 || worker.Tools[0] != "calculator" {
		t.Errorf("expected Tools ['calculator'], got %v", worker.Tools)
	}
}

func TestNewGeneralWorker(t *testing.T) {
	worker := NewGeneralWorker()

	if worker.Name != "general" {
		t.Errorf("expected Name 'general', got %s", worker.Name)
	}

	if worker.SystemPrompt == "" {
		t.Error("expected SystemPrompt to be set")
	}
}

func TestNewCalculatorWorker(t *testing.T) {
	worker := NewCalculatorWorker()

	if worker.Name != "calculator" {
		t.Errorf("expected Name 'calculator', got %s", worker.Name)
	}

	if len(worker.Tools) != 1 || worker.Tools[0] != "calculator" {
		t.Errorf("expected Tools ['calculator'], got %v", worker.Tools)
	}
}

func TestNewResearcherWorker(t *testing.T) {
	worker := NewResearcherWorker()

	if worker.Name != "researcher" {
		t.Errorf("expected Name 'researcher', got %s", worker.Name)
	}

	if worker.Description == "" {
		t.Error("expected Description to be set")
	}
}

func TestNewWriterWorker(t *testing.T) {
	worker := NewWriterWorker()

	if worker.Name != "writer" {
		t.Errorf("expected Name 'writer', got %s", worker.Name)
	}

	if worker.SystemPrompt == "" {
		t.Error("expected SystemPrompt to be set")
	}
}

func TestAddUsage(t *testing.T) {
	a := Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150}
	b := Usage{PromptTokens: 200, CompletionTokens: 100, TotalTokens: 300}

	result := addUsage(a, b)

	if result.PromptTokens != 300 {
		t.Errorf("expected PromptTokens 300, got %d", result.PromptTokens)
	}

	if result.CompletionTokens != 150 {
		t.Errorf("expected CompletionTokens 150, got %d", result.CompletionTokens)
	}

	if result.TotalTokens != 450 {
		t.Errorf("expected TotalTokens 450, got %d", result.TotalTokens)
	}
}

func TestOrchestratorAgent_GroupByLevel_NoDependencies(t *testing.T) {
	orch := &OrchestratorAgent{
		config: OrchestratorConfig{},
	}

	plan := &TaskPlan{
		Subtasks: []Subtask{
			{ID: "task_1"},
			{ID: "task_2"},
			{ID: "task_3"},
		},
		Dependencies: nil,
	}

	levels := orch.groupByLevel(plan)

	if len(levels) != 1 {
		t.Errorf("expected 1 level for independent tasks, got %d", len(levels))
	}

	if len(levels[0]) != 3 {
		t.Errorf("expected 3 tasks in first level, got %d", len(levels[0]))
	}
}

func TestOrchestratorAgent_GroupByLevel_WithDependencies(t *testing.T) {
	orch := &OrchestratorAgent{
		config: OrchestratorConfig{},
	}

	plan := &TaskPlan{
		Subtasks: []Subtask{
			{ID: "task_1"},
			{ID: "task_2"},
			{ID: "task_3"},
		},
		Dependencies: map[string][]string{
			"task_2": {"task_1"},
			"task_3": {"task_2"},
		},
	}

	levels := orch.groupByLevel(plan)

	if len(levels) != 3 {
		t.Errorf("expected 3 levels for sequential tasks, got %d", len(levels))
	}

	// First level should have task_1
	if len(levels[0]) != 1 || levels[0][0].ID != "task_1" {
		t.Errorf("expected first level to have task_1")
	}

	// Second level should have task_2
	if len(levels[1]) != 1 || levels[1][0].ID != "task_2" {
		t.Errorf("expected second level to have task_2")
	}

	// Third level should have task_3
	if len(levels[2]) != 1 || levels[2][0].ID != "task_3" {
		t.Errorf("expected third level to have task_3")
	}
}

func TestOrchestratorAgent_GroupByLevel_Empty(t *testing.T) {
	orch := &OrchestratorAgent{
		config: OrchestratorConfig{},
	}

	plan := &TaskPlan{
		Subtasks: []Subtask{},
	}

	levels := orch.groupByLevel(plan)

	if levels != nil {
		t.Errorf("expected nil for empty subtasks, got %v", levels)
	}
}

func TestOrchestratorAgent_BuildTaskInput_NoDependencies(t *testing.T) {
	orch := &OrchestratorAgent{}

	task := Subtask{ID: "task_1", Input: "original input"}
	deps := map[string][]string{}
	results := map[string]*SubtaskResult{}

	input := orch.buildTaskInput(task, deps, results)

	if input != "original input" {
		t.Errorf("expected 'original input', got %s", input)
	}
}

func TestOrchestratorAgent_BuildTaskInput_WithDependencies(t *testing.T) {
	orch := &OrchestratorAgent{}

	task := Subtask{ID: "task_2", Input: "second task"}
	deps := map[string][]string{
		"task_2": {"task_1"},
	}
	results := map[string]*SubtaskResult{
		"task_1": {ID: "task_1", Success: true, Output: "first result"},
	}

	input := orch.buildTaskInput(task, deps, results)

	if input == "second task" {
		t.Error("expected input to include dependency results")
	}

	if len(input) < len("second task") {
		t.Error("expected input to be longer with context")
	}
}

func TestSubtask_Structure(t *testing.T) {
	subtask := Subtask{
		ID:          "test_task",
		Description: "A test subtask",
		WorkerType:  "general",
		Input:       "test input",
	}

	if subtask.ID != "test_task" {
		t.Errorf("expected ID 'test_task', got %s", subtask.ID)
	}

	if subtask.WorkerType != "general" {
		t.Errorf("expected WorkerType 'general', got %s", subtask.WorkerType)
	}
}

func TestWorkerAgent_ExecuteWithoutLLM(t *testing.T) {
	worker := &WorkerAgent{
		Name:        "test",
		Description: "Test worker",
	}

	// Without LLM set, Execute should handle gracefully
	result, usage := worker.Execute(context.Background(), "test input")

	// Should fail gracefully without LLM
	if result.Success {
		t.Error("Worker without LLM should not succeed")
	}

	if result.Error != "LLM not configured for worker" {
		t.Errorf("expected error 'LLM not configured for worker', got %s", result.Error)
	}

	// Usage should be zero
	if usage.TotalTokens != 0 {
		t.Errorf("expected 0 tokens, got %d", usage.TotalTokens)
	}
}
