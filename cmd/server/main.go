package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/hassan123789/go-ai-agent/internal/config"
	"github.com/hassan123789/go-ai-agent/internal/handler"
	"github.com/hassan123789/go-ai-agent/internal/llm"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize LLM client
	llmClient, err := llm.NewOpenAIClient(llm.OpenAIConfig{
		APIKey:    cfg.OpenAIAPIKey,
		Model:     cfg.OpenAIModel,
		MaxTokens: cfg.OpenAIMaxToken,
	})
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}
	defer func() {
		if err := llmClient.Close(); err != nil {
			log.Printf("Failed to close LLM client: %v", err)
		}
	}()

	// Create Echo instance
	e := echo.New()
	e.HideBanner = true

	// Middleware
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())
	e.Use(middleware.CORS())
	e.Use(middleware.RequestID())

	// Initialize handlers
	chatHandler := handler.NewChatHandler(llmClient)

	// Routes
	e.GET("/health", chatHandler.Health)

	api := e.Group("/api")
	api.POST("/chat", chatHandler.Chat)

	// Start server with graceful shutdown
	go func() {
		addr := cfg.Address()
		log.Printf("Starting server on %s", addr)
		if err := e.Start(addr); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := e.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited")
}
