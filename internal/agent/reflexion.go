package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/hassan123789/go-ai-agent/internal/llm"
	"github.com/hassan123789/go-ai-agent/internal/tools"
)

// ReflexionAgent implements self-reflection and self-improvement.
// It extends ReAct with an evaluation loop that critiques and retries.
//
// Reference: Shinn & Labash, 2023 - "Reflexion: Language Agents with Verbal Reinforcement Learning"
// https://arxiv.org/abs/2303.11366
//
// Flow:
//  1. Execute task using ReAct pattern
//  2. Evaluate the result with a critic
//  3. If unsatisfactory, reflect and retry with feedback
//  4. Store successful strategies in episodic memory
type ReflexionAgent struct {
	llm             *llm.OpenAIClient
	tools           *tools.Registry
	config          ReflexionConfig
	episodicMemory  []Reflection
	maxReflections  int
}

// ReflexionConfig contains configuration for the Reflexion agent.
type ReflexionConfig struct {
	Config

	// MaxReflections is the maximum number of self-reflection attempts.
	MaxReflections int

	// EvaluationPrompt is the prompt used for self-evaluation.
	EvaluationPrompt string

	// ReflectionPrompt is the prompt used for generating reflections.
	ReflectionPrompt string

	// QualityThreshold is the minimum quality score (0-10) to accept.
	QualityThreshold float64
}

// Reflection represents a single reflection episode.
type Reflection struct {
	// Query is the original user query.
	Query string `json:"query"`

	// Attempt is the attempt number.
	Attempt int `json:"attempt"`

	// Response is the agent's response.
	Response string `json:"response"`

	// Evaluation contains the self-evaluation.
	Evaluation Evaluation `json:"evaluation"`

	// Feedback is the self-generated feedback for improvement.
	Feedback string `json:"feedback"`
}

// Evaluation represents the result of self-evaluation.
type Evaluation struct {
	// Score is the quality score (0-10).
	Score float64 `json:"score"`

	// Strengths lists what was done well.
	Strengths []string `json:"strengths"`

	// Weaknesses lists areas for improvement.
	Weaknesses []string `json:"weaknesses"`

	// Reasoning explains the evaluation.
	Reasoning string `json:"reasoning"`
}

// NewReflexionAgent creates a new Reflexion agent.
func NewReflexionAgent(llmClient *llm.OpenAIClient, toolRegistry *tools.Registry, config ReflexionConfig) *ReflexionAgent {
	if config.MaxIterations <= 0 {
		config.MaxIterations = 10
	}
	if config.MaxReflections <= 0 {
		config.MaxReflections = 3
	}
	if config.QualityThreshold <= 0 {
		config.QualityThreshold = 7.0
	}
	if config.SystemPrompt == "" {
		config.SystemPrompt = defaultReflexionSystemPrompt
	}
	if config.EvaluationPrompt == "" {
		config.EvaluationPrompt = defaultEvaluationPrompt
	}
	if config.ReflectionPrompt == "" {
		config.ReflectionPrompt = defaultReflectionPrompt
	}

	return &ReflexionAgent{
		llm:            llmClient,
		tools:          toolRegistry,
		config:         config,
		episodicMemory: make([]Reflection, 0),
		maxReflections: config.MaxReflections,
	}
}

const defaultReflexionSystemPrompt = `You are a thoughtful AI assistant that uses tools to help answer questions.

Before responding, carefully consider:
1. What information do I need to answer this question?
2. Which tools can help me gather this information?
3. How can I synthesize the information into a clear, accurate answer?

If you have learned from past mistakes (provided in the reflection context),
apply those lessons to avoid repeating errors.

Be thorough, accurate, and helpful.`

const defaultEvaluationPrompt = `Evaluate the following response to the user's query.

Query: %s

Response: %s

Evaluate the response on these criteria:
1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the query?
3. Clarity: Is it well-organized and easy to understand?
4. Helpfulness: Does it provide actionable/useful information?

Respond with a JSON object:
{
  "score": <0-10>,
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "reasoning": "explanation of the score"
}`

const defaultReflectionPrompt = `Based on this evaluation, generate specific feedback for improvement.

Query: %s
Response: %s
Evaluation: %s

Generate concrete, actionable feedback on how to improve the response.
Focus on specific steps that can be taken to address the weaknesses.`

// Run processes a query with self-reflection.
func (a *ReflexionAgent) Run(ctx context.Context, query string) (*Response, error) {
	return a.RunWithHistory(ctx, nil, query)
}

// RunWithHistory processes a query with conversation history and self-reflection.
func (a *ReflexionAgent) RunWithHistory(ctx context.Context, history []Message, query string) (*Response, error) {
	var bestResponse *Response
	var bestScore float64

	for attempt := 0; attempt < a.maxReflections; attempt++ {
		if a.config.Verbose {
			log.Printf("[Reflexion] Attempt %d/%d", attempt+1, a.maxReflections)
		}

		// Build context with past reflections
		enhancedHistory := a.buildReflectionContext(history, query)

		// Execute using inner ReAct agent
		resp, err := a.executeWithReAct(ctx, enhancedHistory, query)
		if err != nil {
			return nil, fmt.Errorf("execution failed: %w", err)
		}

		// Evaluate the response
		eval, err := a.evaluate(ctx, query, resp.Output)
		if err != nil {
			if a.config.Verbose {
				log.Printf("[Reflexion] Evaluation failed: %v, accepting response", err)
			}
			return resp, nil
		}

		if a.config.Verbose {
			log.Printf("[Reflexion] Score: %.1f (threshold: %.1f)", eval.Score, a.config.QualityThreshold)
		}

		// Track best response
		if eval.Score > bestScore {
			bestScore = eval.Score
			bestResponse = resp
		}

		// Check if quality threshold is met
		if eval.Score >= a.config.QualityThreshold {
			if a.config.Verbose {
				log.Printf("[Reflexion] Quality threshold met, accepting response")
			}
			return resp, nil
		}

		// Generate reflection for next attempt
		feedback, err := a.reflect(ctx, query, resp.Output, eval)
		if err != nil {
			if a.config.Verbose {
				log.Printf("[Reflexion] Reflection failed: %v", err)
			}
			continue
		}

		// Store reflection for learning
		reflection := Reflection{
			Query:      query,
			Attempt:    attempt + 1,
			Response:   resp.Output,
			Evaluation: eval,
			Feedback:   feedback,
		}
		a.episodicMemory = append(a.episodicMemory, reflection)

		if a.config.Verbose {
			log.Printf("[Reflexion] Feedback: %s", truncate(feedback, 100))
		}
	}

	// Return best response after all attempts
	if bestResponse != nil {
		return bestResponse, nil
	}

	return nil, fmt.Errorf("all %d reflection attempts failed", a.maxReflections)
}

// buildReflectionContext enhances history with past reflections.
func (a *ReflexionAgent) buildReflectionContext(history []Message, query string) []Message {
	if len(a.episodicMemory) == 0 {
		return history
	}

	// Find relevant past reflections for this query
	var relevantReflections []Reflection
	for _, r := range a.episodicMemory {
		if strings.Contains(strings.ToLower(query), strings.ToLower(r.Query)) ||
			strings.Contains(strings.ToLower(r.Query), strings.ToLower(query)) {
			relevantReflections = append(relevantReflections, r)
		}
	}

	if len(relevantReflections) == 0 {
		return history
	}

	// Build reflection context message
	var sb strings.Builder
	sb.WriteString("Learn from these past experiences:\n\n")
	for _, r := range relevantReflections {
		sb.WriteString(fmt.Sprintf("Previous attempt (score: %.1f):\n", r.Evaluation.Score))
		sb.WriteString(fmt.Sprintf("- Weaknesses: %s\n", strings.Join(r.Evaluation.Weaknesses, ", ")))
		sb.WriteString(fmt.Sprintf("- Feedback: %s\n\n", r.Feedback))
	}

	// Prepend reflection context
	enhanced := make([]Message, 0, len(history)+1)
	enhanced = append(enhanced, Message{
		Role:    "system",
		Content: sb.String(),
	})
	enhanced = append(enhanced, history...)

	return enhanced
}

// executeWithReAct runs the inner ReAct agent.
func (a *ReflexionAgent) executeWithReAct(ctx context.Context, history []Message, query string) (*Response, error) {
	reactAgent := NewReActAgent(a.llm, a.tools, a.config.Config)
	return reactAgent.RunWithHistory(ctx, history, query)
}

// evaluate assesses the quality of a response.
func (a *ReflexionAgent) evaluate(ctx context.Context, query, response string) (Evaluation, error) {
	prompt := fmt.Sprintf(a.config.EvaluationPrompt, query, response)

	resp, err := a.llm.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a critical evaluator. Respond only with valid JSON."},
			{Role: llm.RoleUser, Content: prompt},
		},
		Temperature: 0.3, // Lower temperature for more consistent evaluation
	})
	if err != nil {
		return Evaluation{}, fmt.Errorf("evaluation request failed: %w", err)
	}

	var eval Evaluation
	if err := json.Unmarshal([]byte(extractJSON(resp.Content)), &eval); err != nil {
		return Evaluation{}, fmt.Errorf("failed to parse evaluation: %w", err)
	}

	return eval, nil
}

// reflect generates feedback for improvement.
func (a *ReflexionAgent) reflect(ctx context.Context, query, response string, eval Evaluation) (string, error) {
	evalJSON, _ := json.Marshal(eval)
	prompt := fmt.Sprintf(a.config.ReflectionPrompt, query, response, string(evalJSON))

	resp, err := a.llm.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "Generate specific, actionable feedback for improvement."},
			{Role: llm.RoleUser, Content: prompt},
		},
		Temperature: 0.5,
	})
	if err != nil {
		return "", fmt.Errorf("reflection request failed: %w", err)
	}

	return resp.Content, nil
}

// GetEpisodicMemory returns the agent's episodic memory.
func (a *ReflexionAgent) GetEpisodicMemory() []Reflection {
	return a.episodicMemory
}

// ClearEpisodicMemory clears the agent's episodic memory.
func (a *ReflexionAgent) ClearEpisodicMemory() {
	a.episodicMemory = make([]Reflection, 0)
}

// extractJSON extracts JSON from a string that may contain markdown code blocks.
func extractJSON(s string) string {
	// Try to find JSON in code block
	if start := strings.Index(s, "```json"); start != -1 {
		s = s[start+7:]
		if end := strings.Index(s, "```"); end != -1 {
			return strings.TrimSpace(s[:end])
		}
	}
	if start := strings.Index(s, "```"); start != -1 {
		s = s[start+3:]
		if end := strings.Index(s, "```"); end != -1 {
			return strings.TrimSpace(s[:end])
		}
	}
	// Try to find raw JSON
	if start := strings.Index(s, "{"); start != -1 {
		if end := strings.LastIndex(s, "}"); end != -1 {
			return s[start : end+1]
		}
	}
	return s
}

// truncate shortens a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
