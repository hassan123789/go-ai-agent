package memory

import (
	"context"
	"testing"
	"time"
)

func TestWorkingMemory_Add(t *testing.T) {
	wm := NewWorkingMemory(5)

	msg := Message{
		ID:        "1",
		Role:      RoleUser,
		Content:   "Hello",
		Timestamp: time.Now(),
	}

	err := wm.Add(context.Background(), msg)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	count, _ := wm.Count(context.Background())
	if count != 1 {
		t.Errorf("expected count 1, got %d", count)
	}
}

func TestWorkingMemory_Capacity(t *testing.T) {
	wm := NewWorkingMemory(3)

	// Add more than capacity
	for i := 0; i < 5; i++ {
		msg := Message{
			ID:        string(rune('1' + i)),
			Role:      RoleUser,
			Content:   "Message",
			Timestamp: time.Now(),
		}
		_ = wm.Add(context.Background(), msg)
	}

	count, _ := wm.Count(context.Background())
	if count != 3 {
		t.Errorf("expected count 3 (capacity), got %d", count)
	}
}

func TestWorkingMemory_Get(t *testing.T) {
	wm := NewWorkingMemory(10)

	for i := 0; i < 5; i++ {
		msg := Message{
			ID:        string(rune('1' + i)),
			Role:      RoleUser,
			Content:   "Message",
			Timestamp: time.Now(),
		}
		_ = wm.Add(context.Background(), msg)
	}

	// Get all
	msgs, err := wm.Get(context.Background(), 0)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if len(msgs) != 5 {
		t.Errorf("expected 5 messages, got %d", len(msgs))
	}

	// Get limited
	msgs, err = wm.Get(context.Background(), 3)
	if err != nil {
		t.Fatalf("Get limited failed: %v", err)
	}
	if len(msgs) != 3 {
		t.Errorf("expected 3 messages, got %d", len(msgs))
	}
}

func TestWorkingMemory_Clear(t *testing.T) {
	wm := NewWorkingMemory(10)

	msg := Message{ID: "1", Role: RoleUser, Content: "Hello"}
	_ = wm.Add(context.Background(), msg)

	err := wm.Clear(context.Background())
	if err != nil {
		t.Fatalf("Clear failed: %v", err)
	}

	count, _ := wm.Count(context.Background())
	if count != 0 {
		t.Errorf("expected count 0 after clear, got %d", count)
	}
}

func TestWorkingMemory_Trim(t *testing.T) {
	wm := NewWorkingMemory(10)

	for i := 0; i < 5; i++ {
		msg := Message{ID: string(rune('1' + i)), Role: RoleUser, Content: "Message"}
		_ = wm.Add(context.Background(), msg)
	}

	wm.Trim(3)

	count, _ := wm.Count(context.Background())
	if count != 3 {
		t.Errorf("expected count 3 after trim, got %d", count)
	}
}

func TestEpisodicMemory_StartNewEpisode(t *testing.T) {
	em := NewEpisodicMemory()

	em.AddToCurrentEpisode(Message{ID: "1", Role: RoleUser, Content: "Hello"})
	em.StartNewEpisode()
	em.AddToCurrentEpisode(Message{ID: "2", Role: RoleUser, Content: "Hi again"})

	episodes := em.GetEpisodes()
	if len(episodes) != 2 {
		t.Errorf("expected 2 episodes, got %d", len(episodes))
	}
}

func TestEpisodicMemory_SearchEpisodes(t *testing.T) {
	em := NewEpisodicMemory()

	em.AddToCurrentEpisode(Message{ID: "1", Role: RoleUser, Content: "Hello world"})
	em.AddToCurrentEpisode(Message{ID: "2", Role: RoleUser, Content: "Goodbye world"})
	em.AddToCurrentEpisode(Message{ID: "3", Role: RoleUser, Content: "Testing"})

	results := em.SearchEpisodes("world", 10)
	if len(results) != 2 {
		t.Errorf("expected 2 results for 'world', got %d", len(results))
	}
}

func TestEpisodicMemory_ShouldStartNewEpisode(t *testing.T) {
	em := NewEpisodicMemory()

	// No messages yet
	if em.ShouldStartNewEpisode(time.Minute) {
		t.Error("should not start new episode with no messages")
	}

	// Add a message
	em.AddToCurrentEpisode(Message{
		ID:        "1",
		Role:      RoleUser,
		Content:   "Hello",
		Timestamp: time.Now().Add(-2 * time.Minute),
	})

	// Should start new episode after threshold
	if !em.ShouldStartNewEpisode(time.Minute) {
		t.Error("should start new episode after threshold")
	}
}

func TestEpisodicMemory_Clear(t *testing.T) {
	em := NewEpisodicMemory()

	em.AddToCurrentEpisode(Message{ID: "1", Role: RoleUser, Content: "Hello"})
	em.StartNewEpisode()
	em.AddToCurrentEpisode(Message{ID: "2", Role: RoleUser, Content: "Hi"})

	em.Clear()

	episodes := em.GetEpisodes()
	// Should have 1 empty episode after clear
	if len(episodes) != 1 {
		t.Errorf("expected 1 episode after clear, got %d", len(episodes))
	}

	if len(episodes[0].Messages) != 0 {
		t.Errorf("expected 0 messages in episode after clear, got %d", len(episodes[0].Messages))
	}
}

func TestEpisodicMemory_RecentEpisodes(t *testing.T) {
	em := NewEpisodicMemory()

	// Create multiple episodes
	for i := 0; i < 5; i++ {
		em.AddToCurrentEpisode(Message{
			ID:        string(rune('1' + i)),
			Role:      RoleUser,
			Content:   "Message",
			Timestamp: time.Now(),
		})
		if i < 4 {
			em.StartNewEpisode()
		}
	}

	recent := em.RecentEpisodes(3)
	if len(recent) != 3 {
		t.Errorf("expected 3 recent episodes, got %d", len(recent))
	}
}

func TestHierarchicalMemory_Add(t *testing.T) {
	hm := NewHierarchicalMemory(nil, HierarchicalConfig{
		WorkingCapacity: 10,
	})

	msg := NewMessage(RoleUser, "Hello")
	err := hm.Add(context.Background(), msg)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	count, _ := hm.Count(context.Background())
	if count != 1 {
		t.Errorf("expected count 1, got %d", count)
	}
}

func TestHierarchicalMemory_Get(t *testing.T) {
	hm := NewHierarchicalMemory(nil, HierarchicalConfig{
		WorkingCapacity: 10,
	})

	for i := 0; i < 5; i++ {
		msg := NewMessage(RoleUser, "Message")
		_ = hm.Add(context.Background(), msg)
	}

	msgs, err := hm.Get(context.Background(), 3)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	if len(msgs) != 3 {
		t.Errorf("expected 3 messages, got %d", len(msgs))
	}
}

func TestHierarchicalMemory_Clear(t *testing.T) {
	hm := NewHierarchicalMemory(nil, HierarchicalConfig{
		WorkingCapacity: 10,
	})

	msg := NewMessage(RoleUser, "Hello")
	_ = hm.Add(context.Background(), msg)

	err := hm.Clear(context.Background())
	if err != nil {
		t.Fatalf("Clear failed: %v", err)
	}

	count, _ := hm.Count(context.Background())
	if count != 0 {
		t.Errorf("expected count 0, got %d", count)
	}
}

func TestHierarchicalMemory_GetEpisodes(t *testing.T) {
	hm := NewHierarchicalMemory(nil, HierarchicalConfig{
		WorkingCapacity:  10,
		EpisodeThreshold: time.Millisecond,
	})

	msg := NewMessage(RoleUser, "Hello")
	_ = hm.Add(context.Background(), msg)

	episodes := hm.GetEpisodes()
	if len(episodes) < 1 {
		t.Error("expected at least 1 episode")
	}
}

func TestHierarchicalConfig_Defaults(t *testing.T) {
	hm := NewHierarchicalMemory(nil, HierarchicalConfig{})

	// Check that defaults are applied
	if hm.config.WorkingCapacity != 20 {
		t.Errorf("expected default WorkingCapacity 20, got %d", hm.config.WorkingCapacity)
	}

	if hm.config.EpisodeThreshold != 30*time.Minute {
		t.Errorf("expected default EpisodeThreshold 30m, got %v", hm.config.EpisodeThreshold)
	}

	if hm.config.ConsolidationThreshold != 50 {
		t.Errorf("expected default ConsolidationThreshold 50, got %d", hm.config.ConsolidationThreshold)
	}
}

func TestEpisode_Structure(t *testing.T) {
	now := time.Now()
	episode := Episode{
		ID:        "ep_1",
		StartTime: now,
		EndTime:   now.Add(time.Hour),
		Messages:  []Message{{ID: "1", Role: RoleUser, Content: "Hello"}},
		Summary:   "Test episode",
	}

	if episode.ID != "ep_1" {
		t.Errorf("expected ID 'ep_1', got %s", episode.ID)
	}

	if len(episode.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(episode.Messages))
	}

	if episode.Summary != "Test episode" {
		t.Errorf("expected Summary 'Test episode', got %s", episode.Summary)
	}
}

func TestSemanticResult_Structure(t *testing.T) {
	result := SemanticResult{
		Content: "test content",
		Score:   0.95,
		Metadata: map[string]any{
			"source": "test",
		},
	}

	if result.Content != "test content" {
		t.Errorf("expected Content 'test content', got %s", result.Content)
	}

	if result.Score != 0.95 {
		t.Errorf("expected Score 0.95, got %f", result.Score)
	}

	if result.Metadata["source"] != "test" {
		t.Errorf("expected Metadata source 'test', got %v", result.Metadata["source"])
	}
}
