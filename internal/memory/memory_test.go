package memory

import (
	"context"
	"testing"
)

func TestBufferMemory_Add(t *testing.T) {
	ctx := context.Background()
	buf := NewBufferMemory(DefaultBufferConfig())

	err := buf.Add(ctx, NewMessage(RoleUser, "Hello"))
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	count, _ := buf.Count(ctx)
	if count != 1 {
		t.Errorf("expected count 1, got %d", count)
	}
}

func TestBufferMemory_Get(t *testing.T) {
	ctx := context.Background()
	buf := NewBufferMemory(DefaultBufferConfig())

	_ = buf.Add(ctx, NewMessage(RoleUser, "First"))
	_ = buf.Add(ctx, NewMessage(RoleAssistant, "Second"))
	_ = buf.Add(ctx, NewMessage(RoleUser, "Third"))

	// Get all messages
	msgs, err := buf.Get(ctx, 0)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if len(msgs) != 3 {
		t.Errorf("expected 3 messages, got %d", len(msgs))
	}

	// Get last 2 messages
	msgs, err = buf.Get(ctx, 2)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Content != "Second" {
		t.Errorf("expected 'Second', got '%s'", msgs[0].Content)
	}
}

func TestBufferMemory_MaxSize(t *testing.T) {
	ctx := context.Background()
	cfg := BufferConfig{MaxSize: 3, SessionID: "test"}
	buf := NewBufferMemory(cfg)

	// Add 5 messages
	for i := 0; i < 5; i++ {
		_ = buf.Add(ctx, NewMessage(RoleUser, string(rune('A'+i))))
	}

	// Should only have 3 messages
	count, _ := buf.Count(ctx)
	if count != 3 {
		t.Errorf("expected count 3, got %d", count)
	}

	// Should have the last 3 messages
	msgs, _ := buf.Get(ctx, 0)
	if msgs[0].Content != "C" {
		t.Errorf("expected first message 'C', got '%s'", msgs[0].Content)
	}
	if msgs[2].Content != "E" {
		t.Errorf("expected last message 'E', got '%s'", msgs[2].Content)
	}
}

func TestBufferMemory_Clear(t *testing.T) {
	ctx := context.Background()
	buf := NewBufferMemory(DefaultBufferConfig())

	_ = buf.Add(ctx, NewMessage(RoleUser, "Hello"))
	_ = buf.Add(ctx, NewMessage(RoleAssistant, "Hi"))

	err := buf.Clear(ctx)
	if err != nil {
		t.Fatalf("Clear failed: %v", err)
	}

	count, _ := buf.Count(ctx)
	if count != 0 {
		t.Errorf("expected count 0 after clear, got %d", count)
	}
}

func TestBufferMemory_SessionID(t *testing.T) {
	cfg := BufferConfig{MaxSize: 100, SessionID: "session-1"}
	buf := NewBufferMemory(cfg)

	if buf.SessionID() != "session-1" {
		t.Errorf("expected session ID 'session-1', got '%s'", buf.SessionID())
	}

	buf.SetSessionID("session-2")
	if buf.SessionID() != "session-2" {
		t.Errorf("expected session ID 'session-2', got '%s'", buf.SessionID())
	}

	// Buffer should be cleared after session change
	count, _ := buf.Count(context.Background())
	if count != 0 {
		t.Errorf("expected count 0 after session change, got %d", count)
	}
}

func TestConversationManager_CreateSession(t *testing.T) {
	cm := NewConversationManager(DefaultBufferConfig())

	session1 := cm.CreateSession("s1")
	if session1 == nil {
		t.Fatal("CreateSession returned nil")
	}

	if session1.SessionID() != "s1" {
		t.Errorf("expected session ID 's1', got '%s'", session1.SessionID())
	}

	// Creating same session should return existing
	session1Again := cm.CreateSession("s1")
	if session1Again != session1 {
		t.Error("CreateSession should return existing session")
	}
}

func TestConversationManager_GetSession(t *testing.T) {
	cm := NewConversationManager(DefaultBufferConfig())

	// Non-existent session
	session := cm.GetSession("nonexistent")
	if session != nil {
		t.Error("GetSession should return nil for non-existent session")
	}

	// Create and get
	cm.CreateSession("test")
	session = cm.GetSession("test")
	if session == nil {
		t.Error("GetSession should return existing session")
	}
}

func TestConversationManager_ListSessions(t *testing.T) {
	cm := NewConversationManager(DefaultBufferConfig())

	cm.CreateSession("s1")
	cm.CreateSession("s2")
	cm.CreateSession("s3")

	sessions := cm.ListSessions()
	if len(sessions) != 3 {
		t.Errorf("expected 3 sessions, got %d", len(sessions))
	}
}

func TestConversationManager_DeleteSession(t *testing.T) {
	cm := NewConversationManager(DefaultBufferConfig())

	cm.CreateSession("s1")
	cm.CreateSession("s2")

	_ = cm.DeleteSession("s1")

	if cm.GetSession("s1") != nil {
		t.Error("session s1 should be deleted")
	}

	if cm.SessionCount() != 1 {
		t.Errorf("expected 1 session, got %d", cm.SessionCount())
	}
}

func TestConversationManager_ActiveSession(t *testing.T) {
	cm := NewConversationManager(DefaultBufferConfig())

	// No active session initially
	if cm.ActiveSession() != nil {
		t.Error("ActiveSession should be nil initially")
	}

	// First created session becomes active
	cm.CreateSession("s1")
	if cm.ActiveSessionID() != "s1" {
		t.Errorf("expected active session 's1', got '%s'", cm.ActiveSessionID())
	}

	// Set different active session
	cm.SetActiveSession("s2")
	if cm.ActiveSessionID() != "s2" {
		t.Errorf("expected active session 's2', got '%s'", cm.ActiveSessionID())
	}
}

func TestNewMessage(t *testing.T) {
	msg := NewMessage(RoleUser, "Hello")

	if msg.Role != RoleUser {
		t.Errorf("expected role '%s', got '%s'", RoleUser, msg.Role)
	}

	if msg.Content != "Hello" {
		t.Errorf("expected content 'Hello', got '%s'", msg.Content)
	}

	if msg.ID == "" {
		t.Error("message ID should not be empty")
	}

	if msg.Timestamp.IsZero() {
		t.Error("message timestamp should not be zero")
	}
}
