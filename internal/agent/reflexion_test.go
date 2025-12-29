package agent

import (
	"context"
	"testing"

	"github.com/hassan123789/go-ai-agent/internal/llm"
	"github.com/hassan123789/go-ai-agent/internal/tools"
)

// MockLLMClient is a mock LLM client for testing.
type MockLLMClient struct {
	ChatFunc          func(ctx context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error)
	ChatWithToolsFunc func(ctx context.Context, req *llm.ChatWithToolsRequest) (*llm.ChatWithToolsResponse, error)
}

func (m *MockLLMClient) Chat(ctx context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
	if m.ChatFunc != nil {
		return m.ChatFunc(ctx, req)
	}
	return &llm.ChatResponse{Content: "mock response"}, nil
}

func (m *MockLLMClient) ChatWithTools(ctx context.Context, req *llm.ChatWithToolsRequest) (*llm.ChatWithToolsResponse, error) {
	if m.ChatWithToolsFunc != nil {
		return m.ChatWithToolsFunc(ctx, req)
	}
	return &llm.ChatWithToolsResponse{Content: "mock response"}, nil
}

func TestReflexionAgent_NewReflexionAgent(t *testing.T) {
	registry := tools.NewRegistry()

	config := ReflexionConfig{
		Config: Config{
			MaxIterations: 5,
		},
		MaxReflections:   3,
		QualityThreshold: 7.0,
	}

	// Note: ReflexionAgent requires actual OpenAIClient, so we test creation only
	t.Run("config defaults", func(t *testing.T) {
		if config.MaxReflections != 3 {
			t.Errorf("expected MaxReflections 3, got %d", config.MaxReflections)
		}
		if config.QualityThreshold != 7.0 {
			t.Errorf("expected QualityThreshold 7.0, got %f", config.QualityThreshold)
		}
	})

	t.Run("registry is usable", func(t *testing.T) {
		if registry == nil {
			t.Error("registry should not be nil")
		}
	})
}

func TestReflexionAgent_ExtractJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "raw JSON",
			input:    `{"score": 8, "reasoning": "good"}`,
			expected: `{"score": 8, "reasoning": "good"}`,
		},
		{
			name:     "JSON in code block",
			input:    "```json\n{\"score\": 8}\n```",
			expected: `{"score": 8}`,
		},
		{
			name:     "JSON in plain code block",
			input:    "```\n{\"score\": 8}\n```",
			expected: `{"score": 8}`,
		},
		{
			name:     "JSON with surrounding text",
			input:    "Here is the result: {\"score\": 8} as requested",
			expected: `{"score": 8}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractJSON(tt.input)
			if result != tt.expected {
				t.Errorf("extractJSON(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestReflexionAgent_Truncate(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxLen   int
		expected string
	}{
		{
			name:     "short string",
			input:    "hello",
			maxLen:   10,
			expected: "hello",
		},
		{
			name:     "exact length",
			input:    "hello",
			maxLen:   5,
			expected: "hello",
		},
		{
			name:     "truncated",
			input:    "hello world",
			maxLen:   8,
			expected: "hello...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := truncate(tt.input, tt.maxLen)
			if result != tt.expected {
				t.Errorf("truncate(%q, %d) = %q, want %q", tt.input, tt.maxLen, result, tt.expected)
			}
		})
	}
}

func TestReflexionConfig_Defaults(t *testing.T) {
	config := ReflexionConfig{}

	if config.MaxIterations != 0 {
		t.Errorf("expected 0 default MaxIterations, got %d", config.MaxIterations)
	}

	if config.MaxReflections != 0 {
		t.Errorf("expected 0 default MaxReflections, got %d", config.MaxReflections)
	}

	if config.QualityThreshold != 0 {
		t.Errorf("expected 0 default QualityThreshold, got %f", config.QualityThreshold)
	}
}

func TestEvaluation_Structure(t *testing.T) {
	eval := Evaluation{
		Score:      8.5,
		Strengths:  []string{"accurate", "clear"},
		Weaknesses: []string{"could be more detailed"},
		Reasoning:  "Good response overall",
	}

	if eval.Score != 8.5 {
		t.Errorf("expected Score 8.5, got %f", eval.Score)
	}

	if len(eval.Strengths) != 2 {
		t.Errorf("expected 2 strengths, got %d", len(eval.Strengths))
	}

	if len(eval.Weaknesses) != 1 {
		t.Errorf("expected 1 weakness, got %d", len(eval.Weaknesses))
	}
}

func TestReflection_Structure(t *testing.T) {
	reflection := Reflection{
		Query:    "test query",
		Attempt:  1,
		Response: "test response",
		Evaluation: Evaluation{
			Score: 7.0,
		},
		Feedback: "improve accuracy",
	}

	if reflection.Query != "test query" {
		t.Errorf("expected Query 'test query', got %s", reflection.Query)
	}

	if reflection.Attempt != 1 {
		t.Errorf("expected Attempt 1, got %d", reflection.Attempt)
	}

	if reflection.Evaluation.Score != 7.0 {
		t.Errorf("expected Score 7.0, got %f", reflection.Evaluation.Score)
	}
}
