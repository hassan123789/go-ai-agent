package tools

import (
	"context"
	"testing"
)

func TestCalculator_Name(t *testing.T) {
	calc := NewCalculator()
	if calc.Name() != "calculator" {
		t.Errorf("expected name 'calculator', got %q", calc.Name())
	}
}

func TestCalculator_Description(t *testing.T) {
	calc := NewCalculator()
	if calc.Description() == "" {
		t.Error("description should not be empty")
	}
}

func TestCalculator_Parameters(t *testing.T) {
	calc := NewCalculator()
	params := calc.Parameters()

	if params.Type != "object" {
		t.Errorf("expected type 'object', got %q", params.Type)
	}

	if _, exists := params.Properties["expression"]; !exists {
		t.Error("expected 'expression' property")
	}

	if len(params.Required) != 1 || params.Required[0] != "expression" {
		t.Errorf("expected required ['expression'], got %v", params.Required)
	}
}

func TestCalculator_Execute(t *testing.T) {
	calc := NewCalculator()
	ctx := context.Background()

	tests := []struct {
		name       string
		args       string
		wantOutput string
		wantError  bool
	}{
		{
			name:       "simple addition",
			args:       `{"expression": "2 + 3"}`,
			wantOutput: "5",
		},
		{
			name:       "simple subtraction",
			args:       `{"expression": "10 - 4"}`,
			wantOutput: "6",
		},
		{
			name:       "simple multiplication",
			args:       `{"expression": "6 * 7"}`,
			wantOutput: "42",
		},
		{
			name:       "simple division",
			args:       `{"expression": "20 / 4"}`,
			wantOutput: "5",
		},
		{
			name:       "parentheses",
			args:       `{"expression": "(2 + 3) * 4"}`,
			wantOutput: "20",
		},
		{
			name:       "complex expression",
			args:       `{"expression": "2 + 3 * 4 - 1"}`,
			wantOutput: "13",
		},
		{
			name:       "negative number",
			args:       `{"expression": "-5 + 3"}`,
			wantOutput: "-2",
		},
		{
			name:       "decimal numbers",
			args:       `{"expression": "3.5 + 2.5"}`,
			wantOutput: "6",
		},
		{
			name:       "decimal result",
			args:       `{"expression": "7 / 2"}`,
			wantOutput: "3.5",
		},
		{
			name:      "division by zero",
			args:      `{"expression": "5 / 0"}`,
			wantError: true,
		},
		{
			name:      "empty expression",
			args:      `{"expression": ""}`,
			wantError: true,
		},
		{
			name:      "invalid expression - letters",
			args:      `{"expression": "abc + def"}`,
			wantError: true,
		},
		{
			name:      "invalid JSON",
			args:      `{invalid}`,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := calc.Execute(ctx, tt.args)
			if err != nil {
				t.Fatalf("Execute returned error: %v", err)
			}

			if tt.wantError {
				if result.IsSuccess() {
					t.Errorf("expected error, got success with output %q", result.Output)
				}
			} else {
				if !result.IsSuccess() {
					t.Errorf("expected success, got error %q", result.Error)
				}
				if result.Output != tt.wantOutput {
					t.Errorf("expected output %q, got %q", tt.wantOutput, result.Output)
				}
			}
		})
	}
}
