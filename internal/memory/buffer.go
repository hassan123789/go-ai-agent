// Package memory provides conversation memory management for AI agents.
// It supports short-term buffer memory and long-term persistence.
package memory

import (
	"context"
	"sync"
)

// BufferMemory implements Memory with an in-memory ring buffer.
// It stores the most recent N messages, discarding older ones.
type BufferMemory struct {
	messages  []Message
	maxSize   int
	sessionID string
	mu        sync.RWMutex
}

// BufferConfig contains configuration for BufferMemory.
type BufferConfig struct {
	// MaxSize is the maximum number of messages to store.
	// Default is 100.
	MaxSize int

	// SessionID is the initial session identifier.
	SessionID string
}

// DefaultBufferConfig returns the default buffer configuration.
func DefaultBufferConfig() BufferConfig {
	return BufferConfig{
		MaxSize:   100,
		SessionID: "default",
	}
}

// NewBufferMemory creates a new in-memory buffer.
func NewBufferMemory(cfg BufferConfig) *BufferMemory {
	if cfg.MaxSize <= 0 {
		cfg.MaxSize = 100
	}
	if cfg.SessionID == "" {
		cfg.SessionID = "default"
	}
	return &BufferMemory{
		messages:  make([]Message, 0, cfg.MaxSize),
		maxSize:   cfg.MaxSize,
		sessionID: cfg.SessionID,
	}
}

// Add stores a new message in the buffer.
// If the buffer is full, the oldest message is discarded.
func (b *BufferMemory) Add(_ context.Context, msg Message) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// If at capacity, remove the oldest message
	if len(b.messages) >= b.maxSize {
		b.messages = b.messages[1:]
	}

	b.messages = append(b.messages, msg)
	return nil
}

// Get retrieves messages from the buffer.
// If limit is 0 or greater than buffer size, all messages are returned.
func (b *BufferMemory) Get(_ context.Context, limit int) ([]Message, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if limit <= 0 || limit > len(b.messages) {
		limit = len(b.messages)
	}

	// Return the most recent messages
	start := len(b.messages) - limit
	result := make([]Message, limit)
	copy(result, b.messages[start:])
	return result, nil
}

// Clear removes all messages from the buffer.
func (b *BufferMemory) Clear(_ context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.messages = make([]Message, 0, b.maxSize)
	return nil
}

// Count returns the number of messages in the buffer.
func (b *BufferMemory) Count(_ context.Context) (int, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return len(b.messages), nil
}

// SessionID returns the current session identifier.
func (b *BufferMemory) SessionID() string {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return b.sessionID
}

// SetSessionID changes the current session and clears the buffer.
func (b *BufferMemory) SetSessionID(id string) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.sessionID = id
	b.messages = make([]Message, 0, b.maxSize)
}

// GetAll returns all messages in the buffer (alias for Get with limit 0).
func (b *BufferMemory) GetAll(ctx context.Context) ([]Message, error) {
	return b.Get(ctx, 0)
}

// AddUserMessage is a convenience method to add a user message.
func (b *BufferMemory) AddUserMessage(ctx context.Context, content string) error {
	return b.Add(ctx, NewMessage(RoleUser, content))
}

// AddAssistantMessage is a convenience method to add an assistant message.
func (b *BufferMemory) AddAssistantMessage(ctx context.Context, content string) error {
	return b.Add(ctx, NewMessage(RoleAssistant, content))
}

// AddSystemMessage is a convenience method to add a system message.
func (b *BufferMemory) AddSystemMessage(ctx context.Context, content string) error {
	return b.Add(ctx, NewMessage(RoleSystem, content))
}

// Ensure BufferMemory implements SessionMemory interface.
var _ SessionMemory = (*BufferMemory)(nil)
