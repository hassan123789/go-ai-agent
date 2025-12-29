package llm

import (
	"errors"
	"fmt"
)

// Provider represents the type of LLM provider.
type Provider string

const (
	ProviderOpenAI Provider = "openai"
	ProviderClaude Provider = "claude"
	ProviderOllama Provider = "ollama"
)

// ProviderConfig contains configuration for creating an LLM client.
type ProviderConfig struct {
	// Provider specifies which LLM provider to use
	Provider Provider

	// APIKey is the API key for cloud providers (OpenAI, Claude)
	APIKey string

	// Model is the model name to use
	Model string

	// MaxTokens is the default max tokens for completions
	MaxTokens int

	// BaseURL is the custom base URL (useful for Ollama or proxies)
	BaseURL string
}

// NewClient creates a new LLM client based on the provider configuration.
// This is the main factory function for creating LLM clients.
func NewClient(cfg ProviderConfig) (Client, error) {
	switch cfg.Provider {
	case ProviderOpenAI:
		return NewOpenAIClient(OpenAIConfig{
			APIKey:    cfg.APIKey,
			Model:     cfg.Model,
			MaxTokens: cfg.MaxTokens,
		})

	case ProviderClaude:
		return NewClaudeClient(ClaudeConfig{
			APIKey:    cfg.APIKey,
			Model:     cfg.Model,
			MaxTokens: cfg.MaxTokens,
		})

	case ProviderOllama:
		return NewOllamaClient(OllamaConfig{
			BaseURL:   cfg.BaseURL,
			Model:     cfg.Model,
			MaxTokens: cfg.MaxTokens,
		})

	case "":
		return nil, errors.New("provider is required")

	default:
		return nil, fmt.Errorf("unsupported provider: %s", cfg.Provider)
	}
}

// MultiProvider manages multiple LLM clients and enables routing between them.
type MultiProvider struct {
	clients  map[Provider]Client
	primary  Provider
	fallback Provider
}

// MultiProviderConfig contains configuration for the multi-provider.
type MultiProviderConfig struct {
	Providers []ProviderConfig
	Primary   Provider
	Fallback  Provider
}

// NewMultiProvider creates a new multi-provider with multiple LLM clients.
func NewMultiProvider(cfg MultiProviderConfig) (*MultiProvider, error) {
	if len(cfg.Providers) == 0 {
		return nil, errors.New("at least one provider is required")
	}

	clients := make(map[Provider]Client)
	for _, pc := range cfg.Providers {
		client, err := NewClient(pc)
		if err != nil {
			return nil, fmt.Errorf("failed to create %s client: %w", pc.Provider, err)
		}
		clients[pc.Provider] = client
	}

	primary := cfg.Primary
	if primary == "" {
		primary = cfg.Providers[0].Provider
	}

	if _, ok := clients[primary]; !ok {
		return nil, fmt.Errorf("primary provider %s not found", primary)
	}

	fallback := cfg.Fallback
	if fallback != "" {
		if _, ok := clients[fallback]; !ok {
			return nil, fmt.Errorf("fallback provider %s not found", fallback)
		}
	}

	return &MultiProvider{
		clients:  clients,
		primary:  primary,
		fallback: fallback,
	}, nil
}

// GetClient returns the client for the specified provider.
func (mp *MultiProvider) GetClient(provider Provider) (Client, bool) {
	client, ok := mp.clients[provider]
	return client, ok
}

// GetPrimary returns the primary client.
func (mp *MultiProvider) GetPrimary() Client {
	return mp.clients[mp.primary]
}

// GetFallback returns the fallback client, or nil if not configured.
func (mp *MultiProvider) GetFallback() Client {
	if mp.fallback == "" {
		return nil
	}
	return mp.clients[mp.fallback]
}

// ListProviders returns all available provider names.
func (mp *MultiProvider) ListProviders() []Provider {
	providers := make([]Provider, 0, len(mp.clients))
	for p := range mp.clients {
		providers = append(providers, p)
	}
	return providers
}

// Close closes all clients.
func (mp *MultiProvider) Close() error {
	var errs []error
	for _, client := range mp.clients {
		if err := client.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("errors closing clients: %v", errs)
	}
	return nil
}

// ProviderRouter provides intelligent routing between providers.
type ProviderRouter struct {
	mp       *MultiProvider
	strategy RoutingStrategy
}

// RoutingStrategy defines how requests should be routed.
type RoutingStrategy string

const (
	// StrategyPrimary always uses the primary provider
	StrategyPrimary RoutingStrategy = "primary"
	// StrategyFallback uses fallback on primary failure
	StrategyFallback RoutingStrategy = "fallback"
	// StrategyCost routes based on cost optimization
	StrategyCost RoutingStrategy = "cost"
	// StrategySpeed routes based on speed optimization
	StrategySpeed RoutingStrategy = "speed"
)

// NewProviderRouter creates a router with the given strategy.
func NewProviderRouter(mp *MultiProvider, strategy RoutingStrategy) *ProviderRouter {
	if strategy == "" {
		strategy = StrategyPrimary
	}
	return &ProviderRouter{
		mp:       mp,
		strategy: strategy,
	}
}

// Route returns the appropriate client based on the routing strategy and request characteristics.
func (r *ProviderRouter) Route(req *ChatRequest) Client {
	switch r.strategy {
	case StrategyCost:
		return r.routeByCost(req)
	case StrategySpeed:
		return r.routeBySpeed(req)
	default:
		return r.mp.GetPrimary()
	}
}

// routeByCost selects cheaper providers for simpler tasks.
func (r *ProviderRouter) routeByCost(req *ChatRequest) Client {
	// Simple heuristic: use cheaper models for short prompts
	totalLen := 0
	for _, msg := range req.Messages {
		totalLen += len(msg.Content)
	}

	// For short prompts, prefer Ollama (free) or Haiku (cheap)
	if totalLen < 500 {
		if client, ok := r.mp.GetClient(ProviderOllama); ok {
			return client
		}
	}

	// For medium prompts, prefer Claude Haiku
	if totalLen < 2000 {
		if client, ok := r.mp.GetClient(ProviderClaude); ok {
			return client
		}
	}

	// Default to primary
	return r.mp.GetPrimary()
}

// routeBySpeed selects faster providers.
func (r *ProviderRouter) routeBySpeed(req *ChatRequest) Client {
	// Ollama is typically fastest for local inference
	if client, ok := r.mp.GetClient(ProviderOllama); ok {
		return client
	}
	return r.mp.GetPrimary()
}
