package tools

import (
	"testing"
)

func TestResult_Success(t *testing.T) {
	result := Success("test output")

	if !result.IsSuccess() {
		t.Error("Success result should be successful")
	}

	if result.Output != "test output" {
		t.Errorf("expected 'test output', got %q", result.Output)
	}

	if result.String() != "test output" {
		t.Errorf("String() expected 'test output', got %q", result.String())
	}
}

func TestResult_SuccessWithMetadata(t *testing.T) {
	metadata := map[string]any{"key": "value"}
	result := SuccessWithMetadata("test output", metadata)

	if !result.IsSuccess() {
		t.Error("SuccessWithMetadata result should be successful")
	}

	if result.Metadata["key"] != "value" {
		t.Error("metadata not set correctly")
	}
}

func TestResult_Failure(t *testing.T) {
	result := Failure("error message")

	if result.IsSuccess() {
		t.Error("Failure result should not be successful")
	}

	if result.Error != "error message" {
		t.Errorf("expected 'error message', got %q", result.Error)
	}

	if result.String() != "Error: error message" {
		t.Errorf("String() expected 'Error: error message', got %q", result.String())
	}
}

func TestToDefinition(t *testing.T) {
	calc := NewCalculator()
	def := ToDefinition(calc)

	if def.Type != "function" {
		t.Errorf("expected type 'function', got %q", def.Type)
	}

	if def.Function.Name != "calculator" {
		t.Errorf("expected name 'calculator', got %q", def.Function.Name)
	}

	if def.Function.Description == "" {
		t.Error("description should not be empty")
	}

	if def.Function.Parameters.Type != "object" {
		t.Errorf("expected parameters type 'object', got %q", def.Function.Parameters.Type)
	}
}

func TestParseArguments(t *testing.T) {
	type testArgs struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	t.Run("valid JSON", func(t *testing.T) {
		args, err := ParseArguments[testArgs](`{"name": "test", "value": 42}`)
		if err != nil {
			t.Fatalf("ParseArguments failed: %v", err)
		}

		if args.Name != "test" {
			t.Errorf("expected name 'test', got %q", args.Name)
		}
		if args.Value != 42 {
			t.Errorf("expected value 42, got %d", args.Value)
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		_, err := ParseArguments[testArgs](`{invalid}`)
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})
}
