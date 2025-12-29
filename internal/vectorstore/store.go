// Package vectorstore provides vector storage and similarity search.
package vectorstore

import (
	"context"

	"github.com/hassan123789/go-ai-agent/internal/embedding"
)

// SearchResult represents a document with its similarity score.
type SearchResult struct {
	// Document is the matched document.
	Document embedding.Document

	// Score is the similarity score (higher is more similar).
	Score float32
}

// VectorStore is the interface for vector storage backends.
type VectorStore interface {
	// Add stores documents with their embeddings.
	Add(ctx context.Context, docs []embedding.Document) error

	// Search finds documents similar to the query vector.
	// Returns up to limit results, sorted by similarity (highest first).
	Search(ctx context.Context, query embedding.Vector, limit int) ([]SearchResult, error)

	// Delete removes documents by their IDs.
	Delete(ctx context.Context, ids []string) error

	// Get retrieves a document by its ID.
	Get(ctx context.Context, id string) (*embedding.Document, error)

	// Count returns the number of documents in the store.
	Count(ctx context.Context) (int, error)

	// Clear removes all documents from the store.
	Clear(ctx context.Context) error
}

// SemanticStore combines VectorStore with an Embedder for text-based search.
type SemanticStore interface {
	VectorStore

	// AddTexts stores documents and automatically computes their embeddings.
	AddTexts(ctx context.Context, docs []embedding.Document) error

	// SearchText finds documents similar to the query text.
	SearchText(ctx context.Context, query string, limit int) ([]SearchResult, error)

	// Embedder returns the underlying embedder.
	Embedder() embedding.Embedder
}

// FilterFunc is a function that filters search results.
type FilterFunc func(doc embedding.Document) bool

// FilteredVectorStore extends VectorStore with filtering capabilities.
type FilteredVectorStore interface {
	VectorStore

	// SearchWithFilter finds documents matching both similarity and filter criteria.
	SearchWithFilter(ctx context.Context, query embedding.Vector, limit int, filter FilterFunc) ([]SearchResult, error)
}

// Metadata filter helpers.

// HasMetadata returns a filter that matches documents with the specified metadata key.
func HasMetadata(key string) FilterFunc {
	return func(doc embedding.Document) bool {
		if doc.Metadata == nil {
			return false
		}
		_, exists := doc.Metadata[key]
		return exists
	}
}

// MetadataEquals returns a filter that matches documents where metadata[key] equals value.
func MetadataEquals(key string, value any) FilterFunc {
	return func(doc embedding.Document) bool {
		if doc.Metadata == nil {
			return false
		}
		v, exists := doc.Metadata[key]
		return exists && v == value
	}
}

// And combines multiple filters with AND logic.
func And(filters ...FilterFunc) FilterFunc {
	return func(doc embedding.Document) bool {
		for _, f := range filters {
			if !f(doc) {
				return false
			}
		}
		return true
	}
}

// Or combines multiple filters with OR logic.
func Or(filters ...FilterFunc) FilterFunc {
	return func(doc embedding.Document) bool {
		for _, f := range filters {
			if f(doc) {
				return true
			}
		}
		return false
	}
}

// Not negates a filter.
func Not(filter FilterFunc) FilterFunc {
	return func(doc embedding.Document) bool {
		return !filter(doc)
	}
}
