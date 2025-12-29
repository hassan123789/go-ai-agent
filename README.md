# Go AI Agent

[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Production-ready AI Agent framework in Go. Features ReAct pattern, Function Calling, Hierarchical Memory, Reflexion & Orchestrator agents, and multi-LLM support (OpenAI/Claude/Ollama). Clean Architecture + K8s ready.

## 🎯 Why Go for AI Agents?

- **Performance**: Low latency tool execution with Go's concurrency model
- **Differentiation**: Stand out in the Python-dominated AI landscape
- **Production Quality**: Leverage Go's reliability for enterprise deployments
- **Full Stack Integration**: Seamlessly connect with existing Go microservices

## 🏗️ Architecture

```
go-ai-agent/
├── cmd/
│   └── server/              # Application entry point
├── internal/
│   ├── config/              # Configuration management
│   ├── llm/                 # LLM client abstraction (Multi-provider)
│   │   ├── client.go        # Client interface
│   │   ├── openai.go        # OpenAI implementation
│   │   ├── claude.go        # Claude (Anthropic) implementation
│   │   ├── ollama.go        # Ollama (local models) implementation
│   │   ├── provider.go      # Provider factory & router
│   │   ├── production.go    # Retry, streaming, structured output
│   │   └── tools.go         # Tool definitions
│   ├── handler/             # HTTP handlers
│   ├── agent/               # Agent implementations
│   │   ├── react.go         # ReAct agent pattern
│   │   ├── reflexion.go     # Self-improving Reflexion agent
│   │   └── orchestrator.go  # Multi-agent orchestration
│   ├── memory/              # Memory systems
│   │   └── hierarchical.go  # Working/Episodic/Semantic memory
│   ├── vectorstore/         # Vector storage
│   │   └── raptor.go        # RAPTOR hierarchical retrieval
│   ├── embedding/           # Embedding providers
│   └── tools/               # Function calling tools
├── pkg/
│   └── middleware/          # Shared middleware
└── deploy/                  # Deployment manifests (coming soon)
```

## 🚀 Quick Start

### Prerequisites

- Go 1.22+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/hassan123789/go-ai-agent.git
cd go-ai-agent

# Install dependencies
go mod download

# Copy and edit environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Running

```bash
# Run the server
make run

# Or directly with Go
go run ./cmd/server
```

### API Usage

```bash
# Health check
curl http://localhost:8080/health

# Chat completion
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

# Streaming response
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

## 🛠️ Development

```bash
# Run tests
make test

# Run linter
make lint

# Format code
make fmt

# Pre-push checks
make pre-push
```

## 📋 Roadmap

- [x] **Phase 1**: LLM Client & Basic Chat API
- [x] **Phase 2**: Function Calling & Tool Integration
- [x] **Phase 3**: ReAct Agent Pattern
- [x] **Phase 4**: Advanced Agent Patterns (Reflexion, Orchestrator, Hierarchical Memory, RAPTOR)
- [x] **Phase 5**: Multi-LLM Support (Claude, Ollama local models)
- [ ] **Phase 6**: Kubernetes Deployment

## 🧪 Features

### Current (v0.5)

- ✅ **Multi-LLM Support**: OpenAI, Claude (Anthropic), Ollama (local)
- ✅ **Provider Abstraction**: Factory pattern with intelligent routing
- ✅ **ReAct Agent**: Reasoning + Acting pattern
- ✅ **Reflexion Agent**: Self-improving with evaluation loop
- ✅ **Orchestrator Agent**: Multi-agent coordination with workers
- ✅ **Hierarchical Memory**: Working, Episodic, Semantic memory layers
- ✅ **RAPTOR Store**: Tree-structured hierarchical retrieval
- ✅ **Production LLM**: Retry, streaming, structured output, error handling
- ✅ **Function Calling**: Tool integration with OpenAI-compatible API
- ✅ **Streaming Responses**: SSE support
- ✅ **Clean Architecture**: Separation of concerns

### Coming Soon

- 🔄 Vector Store (pgvector)
- 🔄 gRPC API
- 🔄 Kubernetes Manifests
- 🔄 Web Search Tool Integration

## 📊 Tech Stack

| Category | Technology |
| -------- | ---------- |
| Language | Go 1.22+ |
| HTTP Framework | Echo v4 |
| LLM Clients | go-openai, anthropic-sdk-go |
| LLM Providers | OpenAI, Claude, Ollama |
| Vector DB | pgvector (planned) |
| Deployment | Kubernetes (planned) |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
