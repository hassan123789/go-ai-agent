package handler

import (
	"net/http"

	"github.com/labstack/echo/v4"

	"github.com/hassan123789/go-ai-agent/internal/llm"
)

// ChatHandler handles chat-related HTTP requests.
type ChatHandler struct {
	llmClient llm.Client
}

// NewChatHandler creates a new ChatHandler.
func NewChatHandler(client llm.Client) *ChatHandler {
	return &ChatHandler{
		llmClient: client,
	}
}

// ChatRequest represents the request body for chat endpoint.
type ChatRequest struct {
	Messages    []MessageRequest `json:"messages" validate:"required,min=1"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Temperature float32          `json:"temperature,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
}

// MessageRequest represents a single message in the request.
type MessageRequest struct {
	Role    string `json:"role" validate:"required,oneof=system user assistant"`
	Content string `json:"content" validate:"required"`
}

// ChatResponse represents the response body for chat endpoint.
type ChatResponse struct {
	Content      string    `json:"content"`
	FinishReason string    `json:"finish_reason"`
	Usage        UsageInfo `json:"usage"`
}

// UsageInfo contains token usage information.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ErrorResponse represents an error response.
type ErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// Chat handles POST /api/chat requests.
func (h *ChatHandler) Chat(c echo.Context) error {
	var req ChatRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "invalid_request",
			Message: "Failed to parse request body",
		})
	}

	if len(req.Messages) == 0 {
		return c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "validation_error",
			Message: "At least one message is required",
		})
	}

	// Convert request messages to LLM messages
	messages := make([]llm.Message, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = llm.Message{
			Role:    llm.Role(msg.Role),
			Content: msg.Content,
		}
	}

	// Handle streaming response
	if req.Stream {
		return h.handleStreamingChat(c, messages, req.MaxTokens, req.Temperature)
	}

	// Non-streaming response
	resp, err := h.llmClient.Chat(c.Request().Context(), &llm.ChatRequest{
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	})
	if err != nil {
		return c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "llm_error",
			Message: err.Error(),
		})
	}

	return c.JSON(http.StatusOK, ChatResponse{
		Content:      resp.Content,
		FinishReason: resp.FinishReason,
		Usage: UsageInfo{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	})
}

// handleStreamingChat handles streaming chat responses using SSE.
func (h *ChatHandler) handleStreamingChat(c echo.Context, messages []llm.Message, maxTokens int, temperature float32) error {
	c.Response().Header().Set(echo.HeaderContentType, "text/event-stream")
	c.Response().Header().Set("Cache-Control", "no-cache")
	c.Response().Header().Set("Connection", "keep-alive")
	c.Response().WriteHeader(http.StatusOK)

	stream, err := h.llmClient.ChatStream(c.Request().Context(), &llm.ChatRequest{
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Stream:      true,
	})
	if err != nil {
		return err
	}

	flusher, ok := c.Response().Writer.(http.Flusher)
	if !ok {
		return echo.NewHTTPError(http.StatusInternalServerError, "Streaming not supported")
	}

	for chunk := range stream {
		if chunk.Error != nil {
			_, _ = c.Response().Write([]byte("event: error\ndata: " + chunk.Error.Error() + "\n\n"))
			flusher.Flush()
			break
		}

		if chunk.Content != "" {
			_, _ = c.Response().Write([]byte("data: " + chunk.Content + "\n\n"))
			flusher.Flush()
		}

		if chunk.Done {
			_, _ = c.Response().Write([]byte("event: done\ndata: [DONE]\n\n"))
			flusher.Flush()
			break
		}
	}

	return nil
}

// Health handles GET /health requests.
func (h *ChatHandler) Health(c echo.Context) error {
	return c.JSON(http.StatusOK, map[string]string{
		"status": "healthy",
	})
}
