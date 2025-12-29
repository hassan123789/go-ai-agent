// Package vectorstore provides in-memory vector storage implementation.
package vectorstore

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/hassan123789/go-ai-agent/internal/embedding"
)

// MemoryStore is an in-memory implementation of VectorStore.
// Suitable for testing and small datasets.
type MemoryStore struct {
	embedder  embedding.Embedder
	documents map[string]embedding.Document
	mu        sync.RWMutex
}

// NewMemoryStore creates a new in-memory vector store.
// If embedder is nil, AddTexts and SearchText will not be available.
func NewMemoryStore(embedder embedding.Embedder) *MemoryStore {
	return &MemoryStore{
		embedder:  embedder,
		documents: make(map[string]embedding.Document),
	}
}

// Add stores documents with their embeddings.
func (m *MemoryStore) Add(_ context.Context, docs []embedding.Document) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, doc := range docs {
		if doc.ID == "" {
			return fmt.Errorf("document ID is required")
		}
		m.documents[doc.ID] = doc
	}

	return nil
}

// Search finds documents similar to the query vector.
func (m *MemoryStore) Search(_ context.Context, query embedding.Vector, limit int) ([]SearchResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if limit <= 0 {
		limit = 10
	}

	results := make([]SearchResult, 0, len(m.documents))

	for _, doc := range m.documents {
		if len(doc.Embedding) == 0 {
			continue
		}

		score := embedding.CosineSimilarity(query, doc.Embedding)
		results = append(results, SearchResult{
			Document: doc,
			Score:    score,
		})
	}

	// Sort by score (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// SearchWithFilter finds documents matching both similarity and filter criteria.
func (m *MemoryStore) SearchWithFilter(_ context.Context, query embedding.Vector, limit int, filter FilterFunc) ([]SearchResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if limit <= 0 {
		limit = 10
	}

	results := make([]SearchResult, 0, len(m.documents))

	for _, doc := range m.documents {
		if len(doc.Embedding) == 0 {
			continue
		}

		if filter != nil && !filter(doc) {
			continue
		}

		score := embedding.CosineSimilarity(query, doc.Embedding)
		results = append(results, SearchResult{
			Document: doc,
			Score:    score,
		})
	}

	// Sort by score (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// Delete removes documents by their IDs.
func (m *MemoryStore) Delete(_ context.Context, ids []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, id := range ids {
		delete(m.documents, id)
	}

	return nil
}

// Get retrieves a document by its ID.
func (m *MemoryStore) Get(_ context.Context, id string) (*embedding.Document, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	doc, exists := m.documents[id]
	if !exists {
		return nil, nil
	}

	return &doc, nil
}

// Count returns the number of documents in the store.
func (m *MemoryStore) Count(_ context.Context) (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.documents), nil
}

// Clear removes all documents from the store.
func (m *MemoryStore) Clear(_ context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.documents = make(map[string]embedding.Document)
	return nil
}

// AddTexts stores documents and automatically computes their embeddings.
func (m *MemoryStore) AddTexts(ctx context.Context, docs []embedding.Document) error {
	if m.embedder == nil {
		return fmt.Errorf("embedder is required for AddTexts")
	}

	// Extract texts for batch embedding
	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.Content
	}

	// Get embeddings
	vectors, err := m.embedder.EmbedBatch(ctx, texts)
	if err != nil {
		return fmt.Errorf("failed to embed texts: %w", err)
	}

	// Assign embeddings to documents
	for i := range docs {
		docs[i].Embedding = vectors[i]
	}

	return m.Add(ctx, docs)
}

// SearchText finds documents similar to the query text.
func (m *MemoryStore) SearchText(ctx context.Context, query string, limit int) ([]SearchResult, error) {
	if m.embedder == nil {
		return nil, fmt.Errorf("embedder is required for SearchText")
	}

	// Embed the query
	vector, err := m.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	return m.Search(ctx, vector, limit)
}

// Embedder returns the underlying embedder.
func (m *MemoryStore) Embedder() embedding.Embedder {
	return m.embedder
}

// Ensure MemoryStore implements the interfaces.
var (
	_ VectorStore         = (*MemoryStore)(nil)
	_ SemanticStore       = (*MemoryStore)(nil)
	_ FilteredVectorStore = (*MemoryStore)(nil)
)
