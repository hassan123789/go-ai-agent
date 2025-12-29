package vectorstore

import (
	"context"
	"testing"

	"github.com/hassan123789/go-ai-agent/internal/embedding"
)

func TestMemoryStore_Add(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore(nil)

	docs := []embedding.Document{
		{ID: "1", Content: "Hello", Embedding: embedding.Vector{1, 0, 0}},
		{ID: "2", Content: "World", Embedding: embedding.Vector{0, 1, 0}},
	}

	err := store.Add(ctx, docs)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	count, _ := store.Count(ctx)
	if count != 2 {
		t.Errorf("expected count 2, got %d", count)
	}
}

func TestMemoryStore_AddEmptyID(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore(nil)

	docs := []embedding.Document{
		{ID: "", Content: "No ID"},
	}

	err := store.Add(ctx, docs)
	if err == nil {
		t.Error("Add should fail for empty ID")
	}
}

func TestMemoryStore_Search(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore(nil)

	// Add documents with embeddings
	docs := []embedding.Document{
		{ID: "1", Content: "Hello", Embedding: embedding.Vector{1, 0, 0}},
		{ID: "2", Content: "World", Embedding: embedding.Vector{0, 1, 0}},
		{ID: "3", Content: "Test", Embedding: embedding.Vector{0.9, 0.1, 0}},
	}
	_ = store.Add(ctx, docs)

	// Search with query similar to doc 1 and 3
	query := embedding.Vector{1, 0, 0}
	results, err := store.Search(ctx, query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// First result should be doc 1 (exact match)
	if results[0].Document.ID != "1" {
		t.Errorf("expected first result to be doc 1, got %s", results[0].Document.ID)
	}

	// Score should be 1.0 for exact match
	if results[0].Score < 0.99 {
		t.Errorf("expected score ~1.0, got %f", results[0].Score)
	}
}

func TestMemoryStore_SearchWithFilter(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore(nil)

	docs := []embedding.Document{
		{ID: "1", Content: "A", Embedding: embedding.Vector{1, 0, 0}, Metadata: map[string]any{"type": "a"}},
		{ID: "2", Content: "B", Embedding: embedding.Vector{0.9, 0.1, 0}, Metadata: map[string]any{"type": "b"}},
		{ID: "3", Content: "C", Embedding: embedding.Vector{0.8, 0.2, 0}, Metadata: map[string]any{"type": "a"}},
	}
	_ = store.Add(ctx, docs)

	query := embedding.Vector{1, 0, 0}
	filter := MetadataEquals("type", "a")

	results, err := store.SearchWithFilter(ctx, query, 10, filter)
	if err != nil {
		t.Fatalf("SearchWithFilter failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	for _, r := range results {
		if r.Document.Metadata["type"] != "a" {
			t.Errorf("result should have type 'a', got '%v'", r.Document.Metadata["type"])
		}
	}
}

func TestMemoryStore_Delete(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore(nil)

	docs := []embedding.Document{
		{ID: "1", Content: "A", Embedding: embedding.Vector{1, 0, 0}},
		{ID: "2", Content: "B", Embedding: embedding.Vector{0, 1, 0}},
	}
	_ = store.Add(ctx, docs)

	err := store.Delete(ctx, []string{"1"})
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	count, _ := store.Count(ctx)
	if count != 1 {
		t.Errorf("expected count 1, got %d", count)
	}

	doc, _ := store.Get(ctx, "1")
	if doc != nil {
		t.Error("doc 1 should be deleted")
	}
}

func TestMemoryStore_Get(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore(nil)

	docs := []embedding.Document{
		{ID: "1", Content: "Hello", Embedding: embedding.Vector{1, 0, 0}},
	}
	_ = store.Add(ctx, docs)

	// Get existing document
	doc, err := store.Get(ctx, "1")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if doc == nil {
		t.Fatal("doc should not be nil")
	}
	if doc.Content != "Hello" {
		t.Errorf("expected content 'Hello', got '%s'", doc.Content)
	}

	// Get non-existing document
	doc, err = store.Get(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if doc != nil {
		t.Error("doc should be nil for non-existing ID")
	}
}

func TestMemoryStore_Clear(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore(nil)

	docs := []embedding.Document{
		{ID: "1", Content: "A", Embedding: embedding.Vector{1, 0, 0}},
		{ID: "2", Content: "B", Embedding: embedding.Vector{0, 1, 0}},
	}
	_ = store.Add(ctx, docs)

	err := store.Clear(ctx)
	if err != nil {
		t.Fatalf("Clear failed: %v", err)
	}

	count, _ := store.Count(ctx)
	if count != 0 {
		t.Errorf("expected count 0, got %d", count)
	}
}

func TestFilterFunctions(t *testing.T) {
	doc := embedding.Document{
		ID:       "1",
		Content:  "Test",
		Metadata: map[string]any{"type": "a", "count": 5},
	}

	// Test HasMetadata
	if !HasMetadata("type")(doc) {
		t.Error("HasMetadata should return true for existing key")
	}
	if HasMetadata("nonexistent")(doc) {
		t.Error("HasMetadata should return false for non-existing key")
	}

	// Test MetadataEquals
	if !MetadataEquals("type", "a")(doc) {
		t.Error("MetadataEquals should return true for matching value")
	}
	if MetadataEquals("type", "b")(doc) {
		t.Error("MetadataEquals should return false for non-matching value")
	}

	// Test And
	filter := And(HasMetadata("type"), MetadataEquals("count", 5))
	if !filter(doc) {
		t.Error("And filter should return true when all conditions match")
	}

	// Test Or
	filter = Or(MetadataEquals("type", "b"), MetadataEquals("count", 5))
	if !filter(doc) {
		t.Error("Or filter should return true when any condition matches")
	}

	// Test Not
	filter = Not(MetadataEquals("type", "b"))
	if !filter(doc) {
		t.Error("Not filter should negate the result")
	}
}
