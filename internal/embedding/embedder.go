// Package embedding provides text embedding capabilities for vector search.
package embedding

import (
	"context"
)

// Vector represents an embedding vector.
type Vector []float32

// Embedder is the interface for text embedding providers.
type Embedder interface {
	// Embed converts a single text into a vector.
	Embed(ctx context.Context, text string) (Vector, error)

	// EmbedBatch converts multiple texts into vectors.
	// More efficient than calling Embed multiple times.
	EmbedBatch(ctx context.Context, texts []string) ([]Vector, error)

	// Dimension returns the dimension of the embedding vectors.
	Dimension() int

	// Model returns the name of the embedding model.
	Model() string
}

// Document represents a text document with its embedding.
type Document struct {
	// ID is the unique identifier for the document.
	ID string `json:"id"`

	// Content is the text content.
	Content string `json:"content"`

	// Embedding is the vector representation.
	Embedding Vector `json:"embedding,omitempty"`

	// Metadata contains additional information.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewDocument creates a new document without embedding.
func NewDocument(id, content string) Document {
	return Document{
		ID:      id,
		Content: content,
	}
}

// NewDocumentWithMetadata creates a new document with metadata.
func NewDocumentWithMetadata(id, content string, metadata map[string]any) Document {
	return Document{
		ID:       id,
		Content:  content,
		Metadata: metadata,
	}
}

// CosineSimilarity calculates the cosine similarity between two vectors.
// Returns a value between -1 and 1, where 1 means identical.
func CosineSimilarity(a, b Vector) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

// sqrt is a simple square root implementation for float32.
func sqrt(x float32) float32 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	z := x / 2
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

// EuclideanDistance calculates the Euclidean distance between two vectors.
func EuclideanDistance(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}

	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sqrt(sum)
}

// DotProduct calculates the dot product of two vectors.
func DotProduct(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}

	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

// Normalize returns a normalized (unit) vector.
func Normalize(v Vector) Vector {
	var sum float32
	for _, val := range v {
		sum += val * val
	}

	if sum == 0 {
		return v
	}

	norm := sqrt(sum)
	result := make(Vector, len(v))
	for i, val := range v {
		result[i] = val / norm
	}

	return result
}
