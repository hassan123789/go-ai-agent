// Package memory provides hierarchical memory architecture for AI agents.
// Implements Working, Episodic, and Semantic memory layers.
package memory

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// HierarchicalMemory implements a 3-layer memory architecture:
//   - Working Memory: Current conversation context (short-term)
//   - Episodic Memory: Conversation episodes with temporal ordering
//   - Semantic Memory: Long-term knowledge with semantic search
//
// Reference: Lilian Weng - "LLM Powered Autonomous Agents"
// https://lilianweng.github.io/posts/2023-06-23-agent/
type HierarchicalMemory struct {
	working   *WorkingMemory
	episodic  *EpisodicMemory
	semantic  SemanticProvider
	config    HierarchicalConfig
	mu        sync.RWMutex
}

// HierarchicalConfig configures the hierarchical memory.
type HierarchicalConfig struct {
	// WorkingCapacity is the max messages in working memory.
	WorkingCapacity int

	// EpisodeThreshold is the inactivity time before starting a new episode.
	EpisodeThreshold time.Duration

	// ConsolidationThreshold triggers semantic consolidation.
	ConsolidationThreshold int
}

// SemanticProvider interface for semantic memory operations.
type SemanticProvider interface {
	// Store adds content to semantic memory.
	Store(ctx context.Context, content string, metadata map[string]any) error

	// Search finds semantically similar content.
	Search(ctx context.Context, query string, limit int) ([]SemanticResult, error)
}

// SemanticResult represents a semantic search result.
type SemanticResult struct {
	Content  string         `json:"content"`
	Score    float32        `json:"score"`
	Metadata map[string]any `json:"metadata"`
}

// NewHierarchicalMemory creates a new hierarchical memory system.
func NewHierarchicalMemory(semantic SemanticProvider, config HierarchicalConfig) *HierarchicalMemory {
	if config.WorkingCapacity <= 0 {
		config.WorkingCapacity = 20
	}
	if config.EpisodeThreshold <= 0 {
		config.EpisodeThreshold = 30 * time.Minute
	}
	if config.ConsolidationThreshold <= 0 {
		config.ConsolidationThreshold = 50
	}

	return &HierarchicalMemory{
		working:  NewWorkingMemory(config.WorkingCapacity),
		episodic: NewEpisodicMemory(),
		semantic: semantic,
		config:   config,
	}
}

// Add stores a message in the appropriate memory layer.
func (h *HierarchicalMemory) Add(ctx context.Context, msg Message) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Add to working memory
	if err := h.working.Add(ctx, msg); err != nil {
		return fmt.Errorf("working memory add failed: %w", err)
	}

	// Check if we need to start a new episode
	if h.episodic.ShouldStartNewEpisode(h.config.EpisodeThreshold) {
		h.episodic.StartNewEpisode()
	}

	// Add to current episode
	h.episodic.AddToCurrentEpisode(msg)

	// Check if consolidation is needed
	if h.working.Len() >= h.config.ConsolidationThreshold {
		if err := h.consolidate(ctx); err != nil {
			// Log but don't fail
			fmt.Printf("consolidation warning: %v\n", err)
		}
	}

	return nil
}

// Get retrieves messages from working memory.
func (h *HierarchicalMemory) Get(ctx context.Context, limit int) ([]Message, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.working.Get(ctx, limit)
}

// Clear removes all messages from all memory layers.
func (h *HierarchicalMemory) Clear(ctx context.Context) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if err := h.working.Clear(ctx); err != nil {
		return err
	}
	h.episodic.Clear()
	return nil
}

// Count returns the number of messages in working memory.
func (h *HierarchicalMemory) Count(ctx context.Context) (int, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.working.Count(ctx)
}

// Recall retrieves relevant context from all memory layers.
func (h *HierarchicalMemory) Recall(ctx context.Context, query string, limit int) ([]Message, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	var results []Message

	// Get recent from working memory
	working, err := h.working.Get(ctx, limit/2)
	if err != nil {
		return nil, err
	}
	results = append(results, working...)

	// Search episodic memory
	episodic := h.episodic.SearchEpisodes(query, limit/4)
	results = append(results, episodic...)

	// Search semantic memory if available
	if h.semantic != nil {
		semantic, err := h.semantic.Search(ctx, query, limit/4)
		if err == nil {
			for _, s := range semantic {
				results = append(results, Message{
					Role:     "system",
					Content:  fmt.Sprintf("[Semantic Memory] %s", s.Content),
					Metadata: s.Metadata,
				})
			}
		}
	}

	return results, nil
}

// consolidate moves old working memory to semantic storage.
func (h *HierarchicalMemory) consolidate(ctx context.Context) error {
	if h.semantic == nil {
		// Just trim working memory
		h.working.Trim(h.config.WorkingCapacity / 2)
		return nil
	}

	// Get messages to consolidate
	messages, err := h.working.Get(ctx, 0)
	if err != nil {
		return err
	}

	if len(messages) < h.config.ConsolidationThreshold {
		return nil
	}

	// Consolidate older half to semantic memory
	toConsolidate := messages[:len(messages)/2]
	summary := h.summarizeMessages(toConsolidate)

	if err := h.semantic.Store(ctx, summary, map[string]any{
		"type":      "consolidated",
		"count":     len(toConsolidate),
		"timestamp": time.Now().Unix(),
	}); err != nil {
		return err
	}

	// Trim working memory
	h.working.Trim(len(messages) / 2)

	return nil
}

// summarizeMessages creates a summary of messages.
func (h *HierarchicalMemory) summarizeMessages(messages []Message) string {
	var sb strings.Builder
	sb.WriteString("Conversation summary:\n")
	for _, msg := range messages {
		if msg.Role == "user" || msg.Role == "assistant" {
			content := msg.Content
			if len(content) > 200 {
				content = content[:200] + "..."
			}
			sb.WriteString(fmt.Sprintf("- %s: %s\n", msg.Role, content))
		}
	}
	return sb.String()
}

// GetEpisodes returns all episodes from episodic memory.
func (h *HierarchicalMemory) GetEpisodes() []Episode {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.episodic.GetEpisodes()
}

// WorkingMemory implements short-term conversation memory.
type WorkingMemory struct {
	messages []Message
	capacity int
	mu       sync.RWMutex
}

// NewWorkingMemory creates a new working memory.
func NewWorkingMemory(capacity int) *WorkingMemory {
	return &WorkingMemory{
		messages: make([]Message, 0, capacity),
		capacity: capacity,
	}
}

// Add stores a message in working memory.
func (w *WorkingMemory) Add(_ context.Context, msg Message) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.messages = append(w.messages, msg)

	// Trim if over capacity
	if len(w.messages) > w.capacity {
		w.messages = w.messages[len(w.messages)-w.capacity:]
	}

	return nil
}

// Get retrieves messages from working memory.
func (w *WorkingMemory) Get(_ context.Context, limit int) ([]Message, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if limit <= 0 || limit > len(w.messages) {
		result := make([]Message, len(w.messages))
		copy(result, w.messages)
		return result, nil
	}

	start := len(w.messages) - limit
	result := make([]Message, limit)
	copy(result, w.messages[start:])
	return result, nil
}

// Clear removes all messages.
func (w *WorkingMemory) Clear(_ context.Context) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.messages = make([]Message, 0, w.capacity)
	return nil
}

// Count returns the number of messages.
func (w *WorkingMemory) Count(_ context.Context) (int, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return len(w.messages), nil
}

// Len returns the current length.
func (w *WorkingMemory) Len() int {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return len(w.messages)
}

// Trim removes oldest messages.
func (w *WorkingMemory) Trim(keep int) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if len(w.messages) > keep {
		w.messages = w.messages[len(w.messages)-keep:]
	}
}

// Episode represents a conversation episode.
type Episode struct {
	ID        string    `json:"id"`
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	Messages  []Message `json:"messages"`
	Summary   string    `json:"summary,omitempty"`
}

// EpisodicMemory stores conversation episodes.
type EpisodicMemory struct {
	episodes       []Episode
	currentEpisode *Episode
	mu             sync.RWMutex
}

// NewEpisodicMemory creates a new episodic memory.
func NewEpisodicMemory() *EpisodicMemory {
	em := &EpisodicMemory{
		episodes: make([]Episode, 0),
	}
	em.StartNewEpisode()
	return em
}

// ShouldStartNewEpisode checks if enough time has passed.
func (e *EpisodicMemory) ShouldStartNewEpisode(threshold time.Duration) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.currentEpisode == nil || len(e.currentEpisode.Messages) == 0 {
		return false
	}

	lastMsg := e.currentEpisode.Messages[len(e.currentEpisode.Messages)-1]
	return time.Since(lastMsg.Timestamp) > threshold
}

// StartNewEpisode starts a new episode.
func (e *EpisodicMemory) StartNewEpisode() {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Close current episode if exists
	if e.currentEpisode != nil && len(e.currentEpisode.Messages) > 0 {
		e.currentEpisode.EndTime = time.Now()
		e.episodes = append(e.episodes, *e.currentEpisode)
	}

	e.currentEpisode = &Episode{
		ID:        fmt.Sprintf("ep_%d", time.Now().UnixNano()),
		StartTime: time.Now(),
		Messages:  make([]Message, 0),
	}
}

// AddToCurrentEpisode adds a message to the current episode.
func (e *EpisodicMemory) AddToCurrentEpisode(msg Message) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.currentEpisode == nil {
		e.currentEpisode = &Episode{
			ID:        fmt.Sprintf("ep_%d", time.Now().UnixNano()),
			StartTime: time.Now(),
			Messages:  make([]Message, 0),
		}
	}

	e.currentEpisode.Messages = append(e.currentEpisode.Messages, msg)
}

// SearchEpisodes searches for messages containing the query.
func (e *EpisodicMemory) SearchEpisodes(query string, limit int) []Message {
	e.mu.RLock()
	defer e.mu.RUnlock()

	query = strings.ToLower(query)
	var results []Message

	// Search all episodes
	allEpisodes := append(e.episodes, *e.currentEpisode)
	for i := len(allEpisodes) - 1; i >= 0 && len(results) < limit; i-- {
		ep := allEpisodes[i]
		for _, msg := range ep.Messages {
			if strings.Contains(strings.ToLower(msg.Content), query) {
				results = append(results, msg)
				if len(results) >= limit {
					break
				}
			}
		}
	}

	return results
}

// GetEpisodes returns all episodes.
func (e *EpisodicMemory) GetEpisodes() []Episode {
	e.mu.RLock()
	defer e.mu.RUnlock()

	result := make([]Episode, len(e.episodes))
	copy(result, e.episodes)
	if e.currentEpisode != nil {
		result = append(result, *e.currentEpisode)
	}
	return result
}

// Clear removes all episodes.
func (e *EpisodicMemory) Clear() {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.episodes = make([]Episode, 0)
	e.currentEpisode = &Episode{
		ID:        fmt.Sprintf("ep_%d", time.Now().UnixNano()),
		StartTime: time.Now(),
		Messages:  make([]Message, 0),
	}
}

// RecentEpisodes returns the N most recent episodes.
func (e *EpisodicMemory) RecentEpisodes(n int) []Episode {
	e.mu.RLock()
	defer e.mu.RUnlock()

	all := e.GetEpisodes()
	if n >= len(all) {
		return all
	}

	// Sort by start time descending
	sort.Slice(all, func(i, j int) bool {
		return all[i].StartTime.After(all[j].StartTime)
	})

	return all[:n]
}
