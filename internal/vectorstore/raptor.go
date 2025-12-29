// Package vectorstore provides RAPTOR-style hierarchical retrieval.
package vectorstore

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/hassan123789/go-ai-agent/internal/embedding"
)

// RAPTORStore implements hierarchical tree-structured retrieval.
// Reference: Sarthi et al., 2024 - "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
// https://arxiv.org/abs/2401.18059
//
// Key features:
//   - Recursive embedding and clustering of documents
//   - Tree-structured organization with summaries at each level
//   - Multi-hop reasoning through hierarchical search
type RAPTORStore struct {
	base       VectorStore
	embedder   embedding.Embedder
	summarizer Summarizer
	config     RAPTORConfig
	tree       *RAPTORTree
	mu         sync.RWMutex
}

// RAPTORConfig configures the RAPTOR store.
type RAPTORConfig struct {
	// MaxLevels is the maximum tree depth.
	MaxLevels int

	// ClusterSize is the target number of documents per cluster.
	ClusterSize int

	// SimilarityThreshold for clustering.
	SimilarityThreshold float32

	// MaxChildrenPerNode limits branching factor.
	MaxChildrenPerNode int
}

// Summarizer interface for creating document summaries.
type Summarizer interface {
	Summarize(ctx context.Context, documents []string) (string, error)
}

// RAPTORTree represents the hierarchical document structure.
type RAPTORTree struct {
	Root   *TreeNode           `json:"root"`
	Levels map[int][]*TreeNode `json:"levels"`
}

// TreeNode represents a node in the RAPTOR tree.
type TreeNode struct {
	ID        string           `json:"id"`
	Level     int              `json:"level"`
	Content   string           `json:"content"`
	Summary   string           `json:"summary"`
	Embedding embedding.Vector `json:"embedding"`
	Children  []*TreeNode      `json:"children,omitempty"`
	Parent    *TreeNode        `json:"-"`
	Documents []string         `json:"documents,omitempty"`
	Metadata  map[string]any   `json:"metadata,omitempty"`
}

// NewRAPTORStore creates a new RAPTOR-style hierarchical store.
func NewRAPTORStore(base VectorStore, embedder embedding.Embedder, summarizer Summarizer, config RAPTORConfig) *RAPTORStore {
	if config.MaxLevels <= 0 {
		config.MaxLevels = 3
	}
	if config.ClusterSize <= 0 {
		config.ClusterSize = 5
	}
	if config.SimilarityThreshold <= 0 {
		config.SimilarityThreshold = 0.7
	}
	if config.MaxChildrenPerNode <= 0 {
		config.MaxChildrenPerNode = 10
	}

	return &RAPTORStore{
		base:       base,
		embedder:   embedder,
		summarizer: summarizer,
		config:     config,
		tree: &RAPTORTree{
			Levels: make(map[int][]*TreeNode),
		},
	}
}

// Add stores documents and builds the hierarchical structure.
func (r *RAPTORStore) Add(ctx context.Context, docs []embedding.Document) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Add to base store
	if err := r.base.Add(ctx, docs); err != nil {
		return fmt.Errorf("base add failed: %w", err)
	}

	// Create leaf nodes
	leafNodes := make([]*TreeNode, len(docs))
	for i, doc := range docs {
		leafNodes[i] = &TreeNode{
			ID:        doc.ID,
			Level:     0,
			Content:   doc.Content,
			Embedding: doc.Embedding,
			Metadata:  doc.Metadata,
			Documents: []string{doc.ID},
		}
	}
	r.tree.Levels[0] = append(r.tree.Levels[0], leafNodes...)

	// Build tree levels
	if err := r.buildTree(ctx, leafNodes); err != nil {
		return fmt.Errorf("tree building failed: %w", err)
	}

	return nil
}

// buildTree recursively clusters and summarizes documents.
func (r *RAPTORStore) buildTree(ctx context.Context, nodes []*TreeNode) error {
	currentLevel := nodes
	level := 0

	for level < r.config.MaxLevels && len(currentLevel) > 1 {
		level++

		// Cluster nodes
		clusters := r.clusterNodes(currentLevel)

		if len(clusters) == len(currentLevel) {
			// No more clustering possible
			break
		}

		// Create parent nodes for each cluster
		parentNodes := make([]*TreeNode, 0, len(clusters))

		for i, cluster := range clusters {
			if len(cluster) == 0 {
				continue
			}

			// Collect content for summarization
			var contents []string
			var allDocs []string
			for _, node := range cluster {
				if node.Summary != "" {
					contents = append(contents, node.Summary)
				} else {
					contents = append(contents, node.Content)
				}
				allDocs = append(allDocs, node.Documents...)
			}

			// Create summary
			summary, err := r.summarizer.Summarize(ctx, contents)
			if err != nil {
				// Use concatenation as fallback
				summary = strings.Join(contents, "\n\n")
				if len(summary) > 500 {
					summary = summary[:500] + "..."
				}
			}

			// Get embedding for summary
			emb, err := r.embedder.Embed(ctx, summary)
			if err != nil {
				continue
			}

			parentNode := &TreeNode{
				ID:        fmt.Sprintf("cluster_l%d_%d", level, i),
				Level:     level,
				Summary:   summary,
				Embedding: emb,
				Children:  cluster,
				Documents: allDocs,
				Metadata: map[string]any{
					"level":       level,
					"child_count": len(cluster),
				},
			}

			// Set parent references
			for _, child := range cluster {
				child.Parent = parentNode
			}

			parentNodes = append(parentNodes, parentNode)
		}

		r.tree.Levels[level] = parentNodes
		currentLevel = parentNodes
	}

	// Set root
	if len(currentLevel) == 1 {
		r.tree.Root = currentLevel[0]
	} else if len(currentLevel) > 1 {
		// Create root node
		r.tree.Root = &TreeNode{
			ID:       "root",
			Level:    level + 1,
			Children: currentLevel,
			Summary:  "Root node",
		}
	}

	return nil
}

// clusterNodes groups similar nodes together.
func (r *RAPTORStore) clusterNodes(nodes []*TreeNode) [][]*TreeNode {
	if len(nodes) <= r.config.ClusterSize {
		return [][]*TreeNode{nodes}
	}

	// Simple greedy clustering based on similarity
	used := make([]bool, len(nodes))
	var clusters [][]*TreeNode

	for i := 0; i < len(nodes); i++ {
		if used[i] {
			continue
		}

		cluster := []*TreeNode{nodes[i]}
		used[i] = true

		for j := i + 1; j < len(nodes) && len(cluster) < r.config.ClusterSize; j++ {
			if used[j] {
				continue
			}

			// Check similarity with cluster centroid
			sim := embedding.CosineSimilarity(nodes[i].Embedding, nodes[j].Embedding)
			if sim >= r.config.SimilarityThreshold {
				cluster = append(cluster, nodes[j])
				used[j] = true
			}
		}

		clusters = append(clusters, cluster)
	}

	return clusters
}

// Search performs hierarchical search through the tree.
func (r *RAPTORStore) Search(ctx context.Context, query embedding.Vector, limit int) ([]SearchResult, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Search at multiple levels and combine results
	var allResults []SearchResult

	// Search each level
	for level := r.config.MaxLevels; level >= 0; level-- {
		nodes := r.tree.Levels[level]
		if len(nodes) == 0 {
			continue
		}

		for _, node := range nodes {
			score := embedding.CosineSimilarity(query, node.Embedding)

			if level == 0 {
				// Leaf node - add directly
				allResults = append(allResults, SearchResult{
					Document: embedding.Document{
						ID:        node.ID,
						Content:   node.Content,
						Embedding: node.Embedding,
						Metadata:  node.Metadata,
					},
					Score: score,
				})
			} else if score > r.config.SimilarityThreshold {
				// Non-leaf node with good score - traverse children
				childResults := r.searchChildren(query, node, limit)
				allResults = append(allResults, childResults...)
			}
		}
	}

	// Sort by score
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})

	// Deduplicate and limit
	seen := make(map[string]bool)
	var unique []SearchResult
	for _, r := range allResults {
		if !seen[r.Document.ID] {
			seen[r.Document.ID] = true
			unique = append(unique, r)
			if len(unique) >= limit {
				break
			}
		}
	}

	return unique, nil
}

// searchChildren recursively searches child nodes.
func (r *RAPTORStore) searchChildren(query embedding.Vector, node *TreeNode, limit int) []SearchResult {
	var results []SearchResult

	for _, child := range node.Children {
		score := embedding.CosineSimilarity(query, child.Embedding)

		if child.Level == 0 {
			// Leaf node
			results = append(results, SearchResult{
				Document: embedding.Document{
					ID:        child.ID,
					Content:   child.Content,
					Embedding: child.Embedding,
					Metadata:  child.Metadata,
				},
				Score: score,
			})
		} else if len(child.Children) > 0 {
			// Recurse
			childResults := r.searchChildren(query, child, limit)
			results = append(results, childResults...)
		}
	}

	return results
}

// SearchWithContext performs multi-hop retrieval.
func (r *RAPTORStore) SearchWithContext(ctx context.Context, query string, limit int) ([]SearchResult, error) {
	// Get query embedding
	queryEmb, err := r.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	// First hop: find relevant clusters
	results, err := r.Search(ctx, queryEmb, limit*2)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return nil, nil
	}

	// Second hop: expand context from parent summaries
	contextResults := make([]SearchResult, 0, len(results))
	for _, result := range results {
		// Add the result
		contextResults = append(contextResults, result)

		// Add parent context if available
		if node := r.findNode(result.Document.ID); node != nil && node.Parent != nil {
			contextResults = append(contextResults, SearchResult{
				Document: embedding.Document{
					ID:      node.Parent.ID + "_context",
					Content: fmt.Sprintf("[Context] %s", node.Parent.Summary),
					Metadata: map[string]any{
						"type":  "context",
						"level": node.Parent.Level,
					},
				},
				Score: result.Score * 0.8, // Slightly lower score for context
			})
		}
	}

	// Sort and limit
	sort.Slice(contextResults, func(i, j int) bool {
		return contextResults[i].Score > contextResults[j].Score
	})

	if len(contextResults) > limit {
		contextResults = contextResults[:limit]
	}

	return contextResults, nil
}

// findNode finds a node by ID.
func (r *RAPTORStore) findNode(id string) *TreeNode {
	for _, nodes := range r.tree.Levels {
		for _, node := range nodes {
			if node.ID == id {
				return node
			}
		}
	}
	return nil
}

// Delete removes documents from the store.
func (r *RAPTORStore) Delete(ctx context.Context, ids []string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Delete from base
	if err := r.base.Delete(ctx, ids); err != nil {
		return err
	}

	// Remove from tree (simplified - full rebuild would be better)
	idSet := make(map[string]bool)
	for _, id := range ids {
		idSet[id] = true
	}

	for level := range r.tree.Levels {
		var remaining []*TreeNode
		for _, node := range r.tree.Levels[level] {
			if !idSet[node.ID] {
				remaining = append(remaining, node)
			}
		}
		r.tree.Levels[level] = remaining
	}

	return nil
}

// Get retrieves a document by ID.
func (r *RAPTORStore) Get(ctx context.Context, id string) (*embedding.Document, error) {
	return r.base.Get(ctx, id)
}

// Count returns the number of documents.
func (r *RAPTORStore) Count(ctx context.Context) (int, error) {
	return r.base.Count(ctx)
}

// Clear removes all documents.
func (r *RAPTORStore) Clear(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.tree = &RAPTORTree{
		Levels: make(map[int][]*TreeNode),
	}

	return r.base.Clear(ctx)
}

// GetTree returns the tree structure for debugging.
func (r *RAPTORStore) GetTree() *RAPTORTree {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.tree
}

// SimpleSummarizer provides a basic summarization implementation.
type SimpleSummarizer struct{}

// NewSimpleSummarizer creates a simple summarizer.
func NewSimpleSummarizer() *SimpleSummarizer {
	return &SimpleSummarizer{}
}

// Summarize creates a simple concatenated summary.
func (s *SimpleSummarizer) Summarize(_ context.Context, documents []string) (string, error) {
	if len(documents) == 0 {
		return "", nil
	}

	// Simple: concatenate with truncation
	var sb strings.Builder
	maxPerDoc := 200 / len(documents)
	if maxPerDoc < 50 {
		maxPerDoc = 50
	}

	for i, doc := range documents {
		if i > 0 {
			sb.WriteString(" | ")
		}
		if len(doc) > maxPerDoc {
			sb.WriteString(doc[:maxPerDoc])
			sb.WriteString("...")
		} else {
			sb.WriteString(doc)
		}
	}

	return sb.String(), nil
}

// LLMSummarizer uses an LLM for summarization.
type LLMSummarizer struct {
	llm interface {
		Summarize(ctx context.Context, text string) (string, error)
	}
}

// Summarize creates an LLM-generated summary.
func (s *LLMSummarizer) Summarize(ctx context.Context, documents []string) (string, error) {
	combined := strings.Join(documents, "\n\n---\n\n")
	return s.llm.Summarize(ctx, combined)
}
