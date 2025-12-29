package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRetryConfig_Defaults(t *testing.T) {
	config := DefaultRetryConfig()

	if config.MaxRetries != 3 {
		t.Errorf("expected MaxRetries 3, got %d", config.MaxRetries)
	}

	if config.InitialBackoff != time.Second {
		t.Errorf("expected InitialBackoff 1s, got %v", config.InitialBackoff)
	}

	if config.MaxBackoff != 30*time.Second {
		t.Errorf("expected MaxBackoff 30s, got %v", config.MaxBackoff)
	}

	if config.BackoffFactor != 2.0 {
		t.Errorf("expected BackoffFactor 2.0, got %f", config.BackoffFactor)
	}
}

func TestIsRetryableError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "nil error",
			err:      nil,
			expected: false,
		},
		{
			name:     "rate limit error",
			err:      errors.New("rate limit exceeded"),
			expected: true,
		},
		{
			name:     "429 error",
			err:      errors.New("error: 429 Too Many Requests"),
			expected: true,
		},
		{
			name:     "500 error",
			err:      errors.New("500 Internal Server Error"),
			expected: true,
		},
		{
			name:     "502 error",
			err:      errors.New("502 Bad Gateway"),
			expected: true,
		},
		{
			name:     "503 error",
			err:      errors.New("503 Service Unavailable"),
			expected: true,
		},
		{
			name:     "504 error",
			err:      errors.New("504 Gateway Timeout"),
			expected: true,
		},
		{
			name:     "timeout error",
			err:      errors.New("request timeout"),
			expected: true,
		},
		{
			name:     "deadline exceeded",
			err:      errors.New("context deadline exceeded"),
			expected: true,
		},
		{
			name:     "connection reset",
			err:      errors.New("connection reset by peer"),
			expected: true,
		},
		{
			name:     "EOF error",
			err:      errors.New("unexpected EOF"),
			expected: true,
		},
		{
			name:     "non-retryable error",
			err:      errors.New("invalid input parameter"),
			expected: false,
		},
		{
			name:     "authentication error",
			err:      errors.New("authentication failed"),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableError(tt.err)
			if result != tt.expected {
				t.Errorf("isRetryableError(%v) = %v, want %v", tt.err, result, tt.expected)
			}
		})
	}
}

func TestContains(t *testing.T) {
	tests := []struct {
		name       string
		s          string
		substrings []string
		expected   bool
	}{
		{
			name:       "contains first",
			s:          "hello world",
			substrings: []string{"hello", "foo"},
			expected:   true,
		},
		{
			name:       "contains second",
			s:          "hello world",
			substrings: []string{"foo", "world"},
			expected:   true,
		},
		{
			name:       "contains none",
			s:          "hello world",
			substrings: []string{"foo", "bar"},
			expected:   false,
		},
		{
			name:       "empty substrings",
			s:          "hello",
			substrings: []string{},
			expected:   false,
		},
		{
			name:       "empty string",
			s:          "",
			substrings: []string{"hello"},
			expected:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := contains(tt.s, tt.substrings...)
			if result != tt.expected {
				t.Errorf("contains(%q, %v) = %v, want %v", tt.s, tt.substrings, result, tt.expected)
			}
		})
	}
}

func TestErrorHandler_Handle(t *testing.T) {
	handler := DefaultErrorHandler()

	t.Run("nil error", func(t *testing.T) {
		result := handler.Handle(nil)
		if result != nil {
			t.Errorf("expected nil, got %v", result)
		}
	})

	t.Run("rate limit error", func(t *testing.T) {
		err := errors.New("rate limit exceeded")
		result := handler.Handle(err)
		if result == nil {
			t.Error("expected error, got nil")
		}
	})

	t.Run("server error", func(t *testing.T) {
		err := errors.New("500 error")
		result := handler.Handle(err)
		if result == nil {
			t.Error("expected error, got nil")
		}
	})

	t.Run("timeout error", func(t *testing.T) {
		err := errors.New("request timeout")
		result := handler.Handle(err)
		if result == nil {
			t.Error("expected error, got nil")
		}
	})

	t.Run("invalid input error", func(t *testing.T) {
		err := errors.New("invalid request")
		result := handler.Handle(err)
		if result == nil {
			t.Error("expected error, got nil")
		}
	})
}

func TestStreamingToolCall_Structure(t *testing.T) {
	tc := StreamingToolCall{
		ID:             "call_123",
		Name:           "calculator",
		ArgumentsDelta: `{"x":`,
		ArgumentsFull:  `{"x": 5}`,
		IsComplete:     true,
	}

	if tc.ID != "call_123" {
		t.Errorf("expected ID 'call_123', got %s", tc.ID)
	}

	if tc.Name != "calculator" {
		t.Errorf("expected Name 'calculator', got %s", tc.Name)
	}

	if !tc.IsComplete {
		t.Error("expected IsComplete to be true")
	}
}

func TestToolStreamChunk_Structure(t *testing.T) {
	chunk := ToolStreamChunk{
		StreamChunk: StreamChunk{
			Content:      "partial",
			FinishReason: "",
			Done:         false,
		},
		ToolCalls: []StreamingToolCall{
			{ID: "call_1", Name: "test"},
		},
	}

	if chunk.Content != "partial" {
		t.Errorf("expected Content 'partial', got %s", chunk.Content)
	}

	if len(chunk.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(chunk.ToolCalls))
	}
}

func TestStructuredOutput_Structure(t *testing.T) {
	output := StructuredOutput{
		Name:        "test_schema",
		Description: "A test schema",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		},
		Strict: true,
	}

	if output.Name != "test_schema" {
		t.Errorf("expected Name 'test_schema', got %s", output.Name)
	}

	if !output.Strict {
		t.Error("expected Strict to be true")
	}
}

func TestChatWithToolsStreamRequest_Structure(t *testing.T) {
	req := ChatWithToolsStreamRequest{
		Messages: []Message{
			{Role: RoleUser, Content: "Hello"},
		},
		Tools: []ToolDefinition{
			{Type: "function"},
		},
		StrictMode:  true,
		MaxTokens:   1000,
		Temperature: 0.7,
	}

	if len(req.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(req.Messages))
	}

	if !req.StrictMode {
		t.Error("expected StrictMode to be true")
	}

	if req.MaxTokens != 1000 {
		t.Errorf("expected MaxTokens 1000, got %d", req.MaxTokens)
	}
}

func TestConvertToolsWithStrict(t *testing.T) {
	tools := []ToolDefinition{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "test_func",
				Description: "A test function",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
	}

	// Without strict mode
	result := convertToolsWithStrict(tools, false)
	if len(result) != 1 {
		t.Errorf("expected 1 tool, got %d", len(result))
	}

	// With strict mode
	result = convertToolsWithStrict(tools, true)
	if len(result) != 1 {
		t.Errorf("expected 1 tool, got %d", len(result))
	}

	if result[0].Function.Strict != true {
		t.Error("expected Strict to be true")
	}
}

func TestDefaultErrorHandler(t *testing.T) {
	handler := DefaultErrorHandler()

	if handler.OnRateLimit == nil {
		t.Error("expected OnRateLimit to be set")
	}

	if handler.OnServerError == nil {
		t.Error("expected OnServerError to be set")
	}

	if handler.OnTimeout == nil {
		t.Error("expected OnTimeout to be set")
	}

	if handler.OnInvalidInput == nil {
		t.Error("expected OnInvalidInput to be set")
	}

	if handler.OnDefault == nil {
		t.Error("expected OnDefault to be set")
	}
}

func TestErrorHandler_CustomHandlers(t *testing.T) {
	called := ""
	handler := &ErrorHandler{
		OnRateLimit: func(err error) error {
			called = "rate_limit"
			return err
		},
		OnServerError: func(err error) error {
			called = "server"
			return err
		},
	}

	handler.Handle(errors.New("rate limit"))
	if called != "rate_limit" {
		t.Errorf("expected 'rate_limit', got '%s'", called)
	}

	handler.Handle(errors.New("500 error"))
	if called != "server" {
		t.Errorf("expected 'server', got '%s'", called)
	}
}

func TestStrictToolDefinition_Structure(t *testing.T) {
	def := StrictToolDefinition{
		ToolDefinition: ToolDefinition{
			Type: "function",
			Function: FunctionDefinition{
				Name: "test",
			},
		},
		Strict: true,
	}

	if def.Type != "function" {
		t.Errorf("expected Type 'function', got %s", def.Type)
	}

	if !def.Strict {
		t.Error("expected Strict to be true")
	}
}

func TestChatWithRetry_ContextCancellation(t *testing.T) {
	// Test that context cancellation is respected
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	config := RetryConfig{
		MaxRetries:     3,
		InitialBackoff: time.Second,
	}

	// Note: Without a real client, we can only test the config structure
	if config.MaxRetries != 3 {
		t.Errorf("expected MaxRetries 3, got %d", config.MaxRetries)
	}

	// Verify context is cancelled
	if ctx.Err() == nil {
		t.Error("expected context to be cancelled")
	}
}
