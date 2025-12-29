// Package embedding provides OpenAI embedding implementation.
package embedding

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// OpenAIEmbedder implements Embedder using OpenAI's embedding API.
type OpenAIEmbedder struct {
	client    *openai.Client
	model     openai.EmbeddingModel
	dimension int
}

// OpenAIConfig contains configuration for OpenAI embedder.
type OpenAIConfig struct {
	// APIKey is the OpenAI API key.
	APIKey string

	// Model is the embedding model to use.
	// Default is text-embedding-3-small.
	Model openai.EmbeddingModel

	// Dimension is the embedding dimension.
	// Default is 1536 for text-embedding-3-small.
	Dimension int
}

// DefaultOpenAIConfig returns the default OpenAI embedding configuration.
func DefaultOpenAIConfig(apiKey string) OpenAIConfig {
	return OpenAIConfig{
		APIKey:    apiKey,
		Model:     openai.SmallEmbedding3,
		Dimension: 1536,
	}
}

// NewOpenAIEmbedder creates a new OpenAI embedder.
func NewOpenAIEmbedder(cfg OpenAIConfig) (*OpenAIEmbedder, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	if cfg.Model == "" {
		cfg.Model = openai.SmallEmbedding3
	}

	if cfg.Dimension <= 0 {
		cfg.Dimension = 1536
	}

	client := openai.NewClient(cfg.APIKey)

	return &OpenAIEmbedder{
		client:    client,
		model:     cfg.Model,
		dimension: cfg.Dimension,
	}, nil
}

// Embed converts a single text into a vector.
func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) (Vector, error) {
	resp, err := e.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{text},
		Model: e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	return resp.Data[0].Embedding, nil
}

// EmbedBatch converts multiple texts into vectors.
func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([]Vector, error) {
	if len(texts) == 0 {
		return []Vector{}, nil
	}

	resp, err := e.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: texts,
		Model: e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create embeddings: %w", err)
	}

	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(resp.Data))
	}

	vectors := make([]Vector, len(resp.Data))
	for i, data := range resp.Data {
		vectors[i] = data.Embedding
	}

	return vectors, nil
}

// Dimension returns the dimension of the embedding vectors.
func (e *OpenAIEmbedder) Dimension() int {
	return e.dimension
}

// Model returns the name of the embedding model.
func (e *OpenAIEmbedder) Model() string {
	return string(e.model)
}

// Ensure OpenAIEmbedder implements Embedder interface.
var _ Embedder = (*OpenAIEmbedder)(nil) // Ensure OpenAIEmbedder implements Embedder interface.
var _ Embedder = (*OpenAIEmbedder)(nil)
