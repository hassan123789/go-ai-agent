package embedding

import (
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        Vector
		b        Vector
		expected float32
	}{
		{
			name:     "identical vectors",
			a:        Vector{1, 0, 0},
			b:        Vector{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "orthogonal vectors",
			a:        Vector{1, 0, 0},
			b:        Vector{0, 1, 0},
			expected: 0.0,
		},
		{
			name:     "opposite vectors",
			a:        Vector{1, 0, 0},
			b:        Vector{-1, 0, 0},
			expected: -1.0,
		},
		{
			name:     "empty vectors",
			a:        Vector{},
			b:        Vector{},
			expected: 0.0,
		},
		{
			name:     "different lengths",
			a:        Vector{1, 0},
			b:        Vector{1, 0, 0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CosineSimilarity(tt.a, tt.b)
			if abs(result-tt.expected) > 0.001 {
				t.Errorf("expected %f, got %f", tt.expected, result)
			}
		})
	}
}

func TestDotProduct(t *testing.T) {
	a := Vector{1, 2, 3}
	b := Vector{4, 5, 6}

	result := DotProduct(a, b)
	expected := float32(32) // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

	if abs(result-expected) > 0.001 {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

func TestNormalize(t *testing.T) {
	v := Vector{3, 4}
	normalized := Normalize(v)

	// Length should be 1
	var length float32
	for _, val := range normalized {
		length += val * val
	}
	length = sqrt(length)

	if abs(length-1.0) > 0.001 {
		t.Errorf("expected length 1, got %f", length)
	}
}

func TestEuclideanDistance(t *testing.T) {
	a := Vector{0, 0}
	b := Vector{3, 4}

	result := EuclideanDistance(a, b)
	expected := float32(5) // sqrt(9 + 16) = 5

	if abs(result-expected) > 0.001 {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

func TestNewDocument(t *testing.T) {
	doc := NewDocument("id1", "Hello, World!")

	if doc.ID != "id1" {
		t.Errorf("expected ID 'id1', got '%s'", doc.ID)
	}

	if doc.Content != "Hello, World!" {
		t.Errorf("expected content 'Hello, World!', got '%s'", doc.Content)
	}

	if doc.Embedding != nil {
		t.Error("embedding should be nil")
	}
}

func TestNewDocumentWithMetadata(t *testing.T) {
	meta := map[string]any{"source": "test", "page": 1}
	doc := NewDocumentWithMetadata("id1", "content", meta)

	if doc.Metadata["source"] != "test" {
		t.Error("metadata source should be 'test'")
	}

	if doc.Metadata["page"] != 1 {
		t.Error("metadata page should be 1")
	}
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
