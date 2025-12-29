package vectorstore

import (
	"context"
	"testing"

	"github.com/hassan123789/go-ai-agent/internal/embedding"
)

// MockEmbedder for testing.
type MockEmbedder struct{}

func (m *MockEmbedder) Embed(_ context.Context, text string) (embedding.Vector, error) {
	// Return a simple hash-based embedding for testing
	vec := make(embedding.Vector, 8)
	for i, c := range text {
		if i >= len(vec) {
			break
		}
		vec[i] = float32(c) / 255.0
	}
	return vec, nil
}

func (m *MockEmbedder) EmbedBatch(_ context.Context, texts []string) ([]embedding.Vector, error) {
	vecs := make([]embedding.Vector, len(texts))
	for i, text := range texts {
		vec, _ := m.Embed(context.Background(), text)
		vecs[i] = vec
	}
	return vecs, nil
}

func (m *MockEmbedder) Dimension() int {
	return 8
}

func (m *MockEmbedder) Model() string {
	return "mock-embedding-model"
}

func TestRAPTORStore_Add(t *testing.T) {
	embedder := &MockEmbedder{}
	base := NewMemoryStore(embedder)
	summarizer := NewSimpleSummarizer()

	store := NewRAPTORStore(base, embedder, summarizer, RAPTORConfig{
		MaxLevels:           2,
		ClusterSize:         3,
		SimilarityThreshold: 0.5,
	})

	docs := []embedding.Document{
		{ID: "doc1", Content: "Hello world", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}},
		{ID: "doc2", Content: "Goodbye world", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9}},
	}

	err := store.Add(context.Background(), docs)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	count, _ := store.Count(context.Background())
	if count != 2 {
		t.Errorf("expected count 2, got %d", count)
	}
}

func TestRAPTORStore_Search(t *testing.T) {
	embedder := &MockEmbedder{}
	base := NewMemoryStore(embedder)
	summarizer := NewSimpleSummarizer()

	store := NewRAPTORStore(base, embedder, summarizer, RAPTORConfig{
		MaxLevels:           2,
		ClusterSize:         3,
		SimilarityThreshold: 0.3,
	})

	docs := []embedding.Document{
		{ID: "doc1", Content: "Hello world", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}},
		{ID: "doc2", Content: "Goodbye world", Embedding: []float32{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}},
		{ID: "doc3", Content: "Test document", Embedding: []float32{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}},
	}

	err := store.Add(context.Background(), docs)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search with similar vector
	query := embedding.Vector{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	results, err := store.Search(context.Background(), query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected at least one result")
	}
}

func TestRAPTORStore_Get(t *testing.T) {
	embedder := &MockEmbedder{}
	base := NewMemoryStore(embedder)
	summarizer := NewSimpleSummarizer()

	store := NewRAPTORStore(base, embedder, summarizer, RAPTORConfig{})

	docs := []embedding.Document{
		{ID: "doc1", Content: "Hello world", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}},
	}

	_ = store.Add(context.Background(), docs)

	doc, err := store.Get(context.Background(), "doc1")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	if doc.Content != "Hello world" {
		t.Errorf("expected content 'Hello world', got %s", doc.Content)
	}
}

func TestRAPTORStore_Delete(t *testing.T) {
	embedder := &MockEmbedder{}
	base := NewMemoryStore(embedder)
	summarizer := NewSimpleSummarizer()

	store := NewRAPTORStore(base, embedder, summarizer, RAPTORConfig{})

	docs := []embedding.Document{
		{ID: "doc1", Content: "Hello", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}},
		{ID: "doc2", Content: "World", Embedding: []float32{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}},
	}

	_ = store.Add(context.Background(), docs)

	err := store.Delete(context.Background(), []string{"doc1"})
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	count, _ := store.Count(context.Background())
	if count != 1 {
		t.Errorf("expected count 1, got %d", count)
	}
}

func TestRAPTORStore_Clear(t *testing.T) {
	embedder := &MockEmbedder{}
	base := NewMemoryStore(embedder)
	summarizer := NewSimpleSummarizer()

	store := NewRAPTORStore(base, embedder, summarizer, RAPTORConfig{})

	docs := []embedding.Document{
		{ID: "doc1", Content: "Hello", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}},
	}

	_ = store.Add(context.Background(), docs)

	err := store.Clear(context.Background())
	if err != nil {
		t.Fatalf("Clear failed: %v", err)
	}

	count, _ := store.Count(context.Background())
	if count != 0 {
		t.Errorf("expected count 0, got %d", count)
	}

	// Tree should also be cleared
	tree := store.GetTree()
	if len(tree.Levels[0]) != 0 {
		t.Error("expected tree levels to be cleared")
	}
}

func TestRAPTORStore_GetTree(t *testing.T) {
	embedder := &MockEmbedder{}
	base := NewMemoryStore(embedder)
	summarizer := NewSimpleSummarizer()

	store := NewRAPTORStore(base, embedder, summarizer, RAPTORConfig{
		MaxLevels:   2,
		ClusterSize: 2,
	})

	docs := []embedding.Document{
		{ID: "doc1", Content: "Hello", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}},
		{ID: "doc2", Content: "World", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9}},
	}

	_ = store.Add(context.Background(), docs)

	tree := store.GetTree()
	if tree == nil {
		t.Fatal("expected tree to be non-nil")
	}

	// Level 0 should have leaf nodes
	if len(tree.Levels[0]) != 2 {
		t.Errorf("expected 2 leaf nodes, got %d", len(tree.Levels[0]))
	}
}

func TestRAPTORConfig_Defaults(t *testing.T) {
	embedder := &MockEmbedder{}
	base := NewMemoryStore(embedder)
	summarizer := NewSimpleSummarizer()

	store := NewRAPTORStore(base, embedder, summarizer, RAPTORConfig{})

	if store.config.MaxLevels != 3 {
		t.Errorf("expected default MaxLevels 3, got %d", store.config.MaxLevels)
	}

	if store.config.ClusterSize != 5 {
		t.Errorf("expected default ClusterSize 5, got %d", store.config.ClusterSize)
	}

	if store.config.SimilarityThreshold != 0.7 {
		t.Errorf("expected default SimilarityThreshold 0.7, got %f", store.config.SimilarityThreshold)
	}

	if store.config.MaxChildrenPerNode != 10 {
		t.Errorf("expected default MaxChildrenPerNode 10, got %d", store.config.MaxChildrenPerNode)
	}
}

func TestTreeNode_Structure(t *testing.T) {
	node := &TreeNode{
		ID:        "node1",
		Level:     1,
		Content:   "content",
		Summary:   "summary",
		Documents: []string{"doc1", "doc2"},
		Metadata: map[string]any{
			"key": "value",
		},
	}

	if node.ID != "node1" {
		t.Errorf("expected ID 'node1', got %s", node.ID)
	}

	if node.Level != 1 {
		t.Errorf("expected Level 1, got %d", node.Level)
	}

	if len(node.Documents) != 2 {
		t.Errorf("expected 2 documents, got %d", len(node.Documents))
	}
}

func TestSimpleSummarizer_Summarize(t *testing.T) {
	summarizer := NewSimpleSummarizer()

	docs := []string{"Hello world", "Goodbye world", "Test document"}
	summary, err := summarizer.Summarize(context.Background(), docs)

	if err != nil {
		t.Fatalf("Summarize failed: %v", err)
	}

	if summary == "" {
		t.Error("expected non-empty summary")
	}
}

func TestSimpleSummarizer_EmptyInput(t *testing.T) {
	summarizer := NewSimpleSummarizer()

	summary, err := summarizer.Summarize(context.Background(), []string{})
	if err != nil {
		t.Fatalf("Summarize failed: %v", err)
	}

	if summary != "" {
		t.Errorf("expected empty summary for empty input, got %s", summary)
	}
}

func TestRAPTORStore_ClusterNodes(t *testing.T) {
	store := &RAPTORStore{
		config: RAPTORConfig{
			ClusterSize:         3,
			SimilarityThreshold: 0.5,
		},
	}

	nodes := []*TreeNode{
		{ID: "1", Embedding: []float32{0.1, 0.2, 0.3, 0.4}},
		{ID: "2", Embedding: []float32{0.1, 0.2, 0.3, 0.5}},
		{ID: "3", Embedding: []float32{0.9, 0.8, 0.7, 0.6}},
	}

	clusters := store.clusterNodes(nodes)

	if len(clusters) == 0 {
		t.Error("expected at least one cluster")
	}
}

func TestRAPTORStore_SearchWithContext(t *testing.T) {
	embedder := &MockEmbedder{}
	base := NewMemoryStore(embedder)
	summarizer := NewSimpleSummarizer()

	store := NewRAPTORStore(base, embedder, summarizer, RAPTORConfig{
		MaxLevels:           2,
		ClusterSize:         2,
		SimilarityThreshold: 0.3,
	})

	docs := []embedding.Document{
		{ID: "doc1", Content: "Hello world", Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}},
		{ID: "doc2", Content: "Test content", Embedding: []float32{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}},
	}

	_ = store.Add(context.Background(), docs)

	results, err := store.SearchWithContext(context.Background(), "Hello", 5)
	if err != nil {
		t.Fatalf("SearchWithContext failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected at least one result")
	}
}
