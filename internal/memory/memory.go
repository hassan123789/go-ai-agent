// Package memory provides conversation memory management for AI agents.
// It supports short-term buffer memory and long-term persistence.
package memory

import (
	"context"
	"time"
)

// Message represents a single message in conversation history.
type Message struct {
	// ID is the unique identifier for the message.
	ID string `json:"id"`

	// Role is the sender role (user, assistant, system, tool).
	Role string `json:"role"`

	// Content is the message text.
	Content string `json:"content"`

	// Timestamp is when the message was created.
	Timestamp time.Time `json:"timestamp"`

	// Metadata contains additional information.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Memory is the interface for conversation memory stores.
type Memory interface {
	// Add stores a new message in memory.
	Add(ctx context.Context, msg Message) error

	// Get retrieves messages from memory.
	// The limit parameter controls how many recent messages to return.
	// If limit is 0, all messages are returned.
	Get(ctx context.Context, limit int) ([]Message, error)

	// Clear removes all messages from memory.
	Clear(ctx context.Context) error

	// Count returns the number of messages in memory.
	Count(ctx context.Context) (int, error)
}

// SessionMemory extends Memory with session management.
type SessionMemory interface {
	Memory

	// SessionID returns the current session identifier.
	SessionID() string

	// SetSessionID changes the current session.
	SetSessionID(id string)
}

// SummaryMemory provides summarization capabilities for long-term memory.
type SummaryMemory interface {
	Memory

	// Summarize creates a summary of the conversation history.
	Summarize(ctx context.Context) (string, error)

	// GetSummary retrieves the current summary.
	GetSummary(ctx context.Context) (string, error)

	// SetSummary stores a summary.
	SetSummary(ctx context.Context, summary string) error
}

// SearchableMemory provides semantic search over conversation history.
type SearchableMemory interface {
	Memory

	// Search finds messages similar to the query.
	Search(ctx context.Context, query string, limit int) ([]Message, error)
}

// NewMessage creates a new message with the current timestamp.
func NewMessage(role, content string) Message {
	return Message{
		ID:        generateID(),
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
	}
}

// NewMessageWithMetadata creates a new message with metadata.
func NewMessageWithMetadata(role, content string, metadata map[string]any) Message {
	msg := NewMessage(role, content)
	msg.Metadata = metadata
	return msg
}

// generateID creates a simple unique identifier.
func generateID() string {
	return time.Now().Format("20060102150405.000000000")
}

// Role constants for messages.
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleSystem    = "system"
	RoleTool      = "tool"
)
