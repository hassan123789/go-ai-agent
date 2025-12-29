// Package memory provides conversation memory management for AI agents.
package memory

import (
	"context"
	"sync"
)

// ConversationManager manages multiple conversation sessions.
// Each session has its own memory buffer.
type ConversationManager struct {
	sessions   map[string]*BufferMemory
	activeID   string
	defaultCfg BufferConfig
	mu         sync.RWMutex
}

// NewConversationManager creates a new conversation manager.
func NewConversationManager(defaultCfg BufferConfig) *ConversationManager {
	if defaultCfg.MaxSize <= 0 {
		defaultCfg.MaxSize = 100
	}
	return &ConversationManager{
		sessions:   make(map[string]*BufferMemory),
		activeID:   "",
		defaultCfg: defaultCfg,
	}
}

// CreateSession creates a new conversation session.
// If a session with the given ID already exists, it returns the existing one.
func (cm *ConversationManager) CreateSession(id string) *BufferMemory {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if session, exists := cm.sessions[id]; exists {
		return session
	}

	cfg := cm.defaultCfg
	cfg.SessionID = id
	session := NewBufferMemory(cfg)
	cm.sessions[id] = session

	if cm.activeID == "" {
		cm.activeID = id
	}

	return session
}

// GetSession retrieves a session by ID.
// Returns nil if the session doesn't exist.
func (cm *ConversationManager) GetSession(id string) *BufferMemory {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return cm.sessions[id]
}

// GetOrCreateSession retrieves a session or creates it if it doesn't exist.
func (cm *ConversationManager) GetOrCreateSession(id string) *BufferMemory {
	cm.mu.RLock()
	session, exists := cm.sessions[id]
	cm.mu.RUnlock()

	if exists {
		return session
	}

	return cm.CreateSession(id)
}

// DeleteSession removes a session from the manager.
func (cm *ConversationManager) DeleteSession(id string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if _, exists := cm.sessions[id]; !exists {
		return nil
	}

	delete(cm.sessions, id)

	// If we deleted the active session, clear active ID
	if cm.activeID == id {
		cm.activeID = ""
	}

	return nil
}

// ListSessions returns all session IDs.
func (cm *ConversationManager) ListSessions() []string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	ids := make([]string, 0, len(cm.sessions))
	for id := range cm.sessions {
		ids = append(ids, id)
	}
	return ids
}

// ActiveSession returns the currently active session.
// Returns nil if no session is active.
func (cm *ConversationManager) ActiveSession() *BufferMemory {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	if cm.activeID == "" {
		return nil
	}
	return cm.sessions[cm.activeID]
}

// SetActiveSession sets the active session by ID.
// Creates the session if it doesn't exist.
func (cm *ConversationManager) SetActiveSession(id string) *BufferMemory {
	cm.mu.Lock()
	session, exists := cm.sessions[id]
	cm.mu.Unlock()

	if !exists {
		session = cm.CreateSession(id)
	}

	cm.mu.Lock()
	cm.activeID = id
	cm.mu.Unlock()

	return session
}

// ActiveSessionID returns the ID of the currently active session.
func (cm *ConversationManager) ActiveSessionID() string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return cm.activeID
}

// ClearAll removes all sessions.
func (cm *ConversationManager) ClearAll() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.sessions = make(map[string]*BufferMemory)
	cm.activeID = ""
}

// SessionCount returns the number of sessions.
func (cm *ConversationManager) SessionCount() int {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return len(cm.sessions)
}

// AddToSession adds a message to a specific session.
// Creates the session if it doesn't exist.
func (cm *ConversationManager) AddToSession(ctx context.Context, sessionID string, msg Message) error {
	session := cm.GetOrCreateSession(sessionID)
	return session.Add(ctx, msg)
}

// GetFromSession retrieves messages from a specific session.
// Returns empty slice if the session doesn't exist.
func (cm *ConversationManager) GetFromSession(ctx context.Context, sessionID string, limit int) ([]Message, error) {
	session := cm.GetSession(sessionID)
	if session == nil {
		return []Message{}, nil
	}
	return session.Get(ctx, limit)
}
