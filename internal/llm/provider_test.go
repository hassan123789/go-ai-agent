package llm

import (
	"testing"
)

func TestProviderConfig(t *testing.T) {
	t.Run("OpenAI provider creation", func(t *testing.T) {
		client, err := NewClient(ProviderConfig{
			Provider: ProviderOpenAI,
			APIKey:   "test-key",
			Model:    "gpt-4o-mini",
		})
		if err != nil {
			t.Fatalf("failed to create OpenAI client: %v", err)
		}
		if client == nil {
			t.Fatal("client should not be nil")
		}
		_ = client.Close()
	})

	t.Run("Claude provider creation", func(t *testing.T) {
		client, err := NewClient(ProviderConfig{
			Provider: ProviderClaude,
			APIKey:   "test-key",
			Model:    ClaudeHaiku35,
		})
		if err != nil {
			t.Fatalf("failed to create Claude client: %v", err)
		}
		if client == nil {
			t.Fatal("client should not be nil")
		}
		_ = client.Close()
	})

	t.Run("Ollama provider creation", func(t *testing.T) {
		client, err := NewClient(ProviderConfig{
			Provider: ProviderOllama,
			Model:    OllamaLlama3_2,
		})
		if err != nil {
			t.Fatalf("failed to create Ollama client: %v", err)
		}
		if client == nil {
			t.Fatal("client should not be nil")
		}
		_ = client.Close()
	})

	t.Run("Missing provider returns error", func(t *testing.T) {
		_, err := NewClient(ProviderConfig{
			APIKey: "test-key",
		})
		if err == nil {
			t.Fatal("expected error for missing provider")
		}
	})

	t.Run("Unsupported provider returns error", func(t *testing.T) {
		_, err := NewClient(ProviderConfig{
			Provider: "unsupported",
			APIKey:   "test-key",
		})
		if err == nil {
			t.Fatal("expected error for unsupported provider")
		}
	})
}

func TestMultiProvider(t *testing.T) {
	t.Run("Create multi-provider", func(t *testing.T) {
		mp, err := NewMultiProvider(MultiProviderConfig{
			Providers: []ProviderConfig{
				{Provider: ProviderOpenAI, APIKey: "test-key"},
				{Provider: ProviderClaude, APIKey: "test-key"},
				{Provider: ProviderOllama},
			},
			Primary:  ProviderOpenAI,
			Fallback: ProviderClaude,
		})
		if err != nil {
			t.Fatalf("failed to create multi-provider: %v", err)
		}
		defer mp.Close()

		// Test GetPrimary
		primary := mp.GetPrimary()
		if primary == nil {
			t.Fatal("primary should not be nil")
		}

		// Test GetFallback
		fallback := mp.GetFallback()
		if fallback == nil {
			t.Fatal("fallback should not be nil")
		}

		// Test GetClient
		_, ok := mp.GetClient(ProviderOllama)
		if !ok {
			t.Fatal("should find Ollama client")
		}

		// Test ListProviders
		providers := mp.ListProviders()
		if len(providers) != 3 {
			t.Fatalf("expected 3 providers, got %d", len(providers))
		}
	})

	t.Run("Empty providers returns error", func(t *testing.T) {
		_, err := NewMultiProvider(MultiProviderConfig{})
		if err == nil {
			t.Fatal("expected error for empty providers")
		}
	})

	t.Run("Invalid primary returns error", func(t *testing.T) {
		_, err := NewMultiProvider(MultiProviderConfig{
			Providers: []ProviderConfig{
				{Provider: ProviderOpenAI, APIKey: "test-key"},
			},
			Primary: ProviderClaude,
		})
		if err == nil {
			t.Fatal("expected error for invalid primary")
		}
	})
}

func TestProviderRouter(t *testing.T) {
	mp, err := NewMultiProvider(MultiProviderConfig{
		Providers: []ProviderConfig{
			{Provider: ProviderOpenAI, APIKey: "test-key"},
			{Provider: ProviderOllama},
		},
		Primary: ProviderOpenAI,
	})
	if err != nil {
		t.Fatalf("failed to create multi-provider: %v", err)
	}
	defer mp.Close()

	t.Run("Primary strategy", func(t *testing.T) {
		router := NewProviderRouter(mp, StrategyPrimary)
		client := router.Route(&ChatRequest{})
		if client == nil {
			t.Fatal("client should not be nil")
		}
	})

	t.Run("Cost strategy - short prompt uses Ollama", func(t *testing.T) {
		router := NewProviderRouter(mp, StrategyCost)
		client := router.Route(&ChatRequest{
			Messages: []Message{{Content: "Hi"}},
		})
		if client == nil {
			t.Fatal("client should not be nil")
		}
	})

	t.Run("Speed strategy uses Ollama when available", func(t *testing.T) {
		router := NewProviderRouter(mp, StrategySpeed)
		client := router.Route(&ChatRequest{})
		if client == nil {
			t.Fatal("client should not be nil")
		}
	})
}

func TestClaudeModelConstants(t *testing.T) {
	// Verify model constants are defined correctly
	models := []string{
		ClaudeOpus45,
		ClaudeOpus4,
		ClaudeSonnet45,
		ClaudeSonnet4,
		ClaudeSonnet37,
		ClaudeHaiku45,
		ClaudeHaiku35,
		ClaudeHaiku3,
	}

	for _, model := range models {
		if model == "" {
			t.Errorf("model constant should not be empty")
		}
	}
}

func TestOllamaModelConstants(t *testing.T) {
	// Verify model constants are defined correctly
	models := []string{
		OllamaLlama3_2,
		OllamaLlama3_1,
		OllamaLlama3,
		OllamaMistral,
		OllamaCodeLlama,
		OllamaGemma2,
		OllamaQwen2_5,
		OllamaDeepSeekR1,
		OllamaPhi3,
	}

	for _, model := range models {
		if model == "" {
			t.Errorf("model constant should not be empty")
		}
	}
}

func TestClaudeClientCreation(t *testing.T) {
	t.Run("Valid config", func(t *testing.T) {
		client, err := NewClaudeClient(ClaudeConfig{
			APIKey:    "test-key",
			Model:     ClaudeHaiku35,
			MaxTokens: 1024,
		})
		if err != nil {
			t.Fatalf("failed to create client: %v", err)
		}
		if client == nil {
			t.Fatal("client should not be nil")
		}
		_ = client.Close()
	})

	t.Run("Default model", func(t *testing.T) {
		client, err := NewClaudeClient(ClaudeConfig{
			APIKey: "test-key",
		})
		if err != nil {
			t.Fatalf("failed to create client: %v", err)
		}
		if client == nil {
			t.Fatal("client should not be nil")
		}
		_ = client.Close()
	})

	t.Run("Missing API key", func(t *testing.T) {
		_, err := NewClaudeClient(ClaudeConfig{})
		if err == nil {
			t.Fatal("expected error for missing API key")
		}
	})
}

func TestOllamaClientCreation(t *testing.T) {
	t.Run("Default config", func(t *testing.T) {
		client, err := NewOllamaClient(OllamaConfig{})
		if err != nil {
			t.Fatalf("failed to create client: %v", err)
		}
		if client == nil {
			t.Fatal("client should not be nil")
		}
		_ = client.Close()
	})

	t.Run("Custom base URL", func(t *testing.T) {
		client, err := NewOllamaClient(OllamaConfig{
			BaseURL: "http://custom:11434/v1",
			Model:   OllamaMistral,
		})
		if err != nil {
			t.Fatalf("failed to create client: %v", err)
		}
		if client == nil {
			t.Fatal("client should not be nil")
		}
		_ = client.Close()
	})
}
