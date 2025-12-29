package tools

import (
	"testing"
)

func TestRegistry_New(t *testing.T) {
	r := NewRegistry()
	if r == nil {
		t.Fatal("NewRegistry returned nil")
	}
	if r.Count() != 0 {
		t.Errorf("expected empty registry, got %d tools", r.Count())
	}
}

func TestRegistry_Register(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()

	err := r.Register(calc)
	if err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	if r.Count() != 1 {
		t.Errorf("expected 1 tool, got %d", r.Count())
	}

	// Registering the same tool again should fail
	err = r.Register(calc)
	if err == nil {
		t.Error("expected error when registering duplicate tool")
	}
}

func TestRegistry_MustRegister(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()

	// Should not panic
	r.MustRegister(calc)

	if r.Count() != 1 {
		t.Errorf("expected 1 tool, got %d", r.Count())
	}

	// Should panic on duplicate
	defer func() {
		if recover() == nil {
			t.Error("expected panic on duplicate registration")
		}
	}()
	r.MustRegister(calc)
}

func TestRegistry_Get(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()
	r.MustRegister(calc)

	// Get existing tool
	tool := r.Get("calculator")
	if tool == nil {
		t.Fatal("Get returned nil for existing tool")
	}
	if tool.Name() != "calculator" {
		t.Errorf("expected 'calculator', got %q", tool.Name())
	}

	// Get non-existing tool
	tool = r.Get("nonexistent")
	if tool != nil {
		t.Error("Get should return nil for non-existing tool")
	}
}

func TestRegistry_Has(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()
	r.MustRegister(calc)

	if !r.Has("calculator") {
		t.Error("Has should return true for existing tool")
	}

	if r.Has("nonexistent") {
		t.Error("Has should return false for non-existing tool")
	}
}

func TestRegistry_List(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()
	r.MustRegister(calc)

	tools := r.List()
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	if tools[0].Name() != "calculator" {
		t.Errorf("expected 'calculator', got %q", tools[0].Name())
	}
}

func TestRegistry_Names(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()
	r.MustRegister(calc)

	names := r.Names()
	if len(names) != 1 {
		t.Fatalf("expected 1 name, got %d", len(names))
	}

	if names[0] != "calculator" {
		t.Errorf("expected 'calculator', got %q", names[0])
	}
}

func TestRegistry_Definitions(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()
	r.MustRegister(calc)

	defs := r.Definitions()
	if len(defs) != 1 {
		t.Fatalf("expected 1 definition, got %d", len(defs))
	}

	def := defs[0]
	if def.Type != "function" {
		t.Errorf("expected type 'function', got %q", def.Type)
	}
	if def.Function.Name != "calculator" {
		t.Errorf("expected name 'calculator', got %q", def.Function.Name)
	}
}

func TestRegistry_Unregister(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()
	r.MustRegister(calc)

	err := r.Unregister("calculator")
	if err != nil {
		t.Fatalf("Unregister failed: %v", err)
	}

	if r.Count() != 0 {
		t.Errorf("expected 0 tools, got %d", r.Count())
	}

	// Unregistering again should fail
	err = r.Unregister("calculator")
	if err == nil {
		t.Error("expected error when unregistering non-existing tool")
	}
}

func TestRegistry_Clear(t *testing.T) {
	r := NewRegistry()
	calc := NewCalculator()
	r.MustRegister(calc)

	r.Clear()

	if r.Count() != 0 {
		t.Errorf("expected 0 tools after Clear, got %d", r.Count())
	}
}
