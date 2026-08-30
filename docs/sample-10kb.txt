# PromptOps

A small LLMOps learning project: a FastAPI service that wraps a local LLM (via Ollama) behind a `/ask` endpoint, with an agentic LangGraph layer that can decide whether to call tools before answering.

Built as a hands-on project while transitioning from DevOps (Spring Boot/Java) into LLMOps — applying CI/CD, monitoring, and ops discipline to LLM-based systems instead of traditional services.

## Why this project

Most LLMOps tutorials stop at "call an API and print the response." This project instead treats an LLM call like any other production dependency: it gets logged, error-handled, and eventually tested, versioned, and monitored like a normal service.

## Current status: Phase 1 (complete)

- [x] FastAPI service with a `/health` and `/ask` endpoint
- [x] Local LLM via [Ollama](https://ollama.com) (`qwen2.5-coder:7b`) — no API costs during learning/dev
- [x] Structured logging (latency per request)
- [x] Error handling (`HTTPException` on upstream failures)
- [x] LangGraph agent loop: the model decides whether a tool call is needed before answering
- [x] Defensive patterns for local-model quirks (see below)

## Architecture

```
Client
  │
  ▼
FastAPI (/ask)
  │
  ▼
LangGraph agent
  │
  ├── decides: answer directly, or use a tool?
  │
  ├── [tool needed] → ToolNode executes → result fed back to agent
  │
  └── [no tool needed] → final answer
  │
  ▼
Ollama (qwen2.5-coder:7b, local)
```

## Setup

```bash
# 1. Install Ollama and pull the model
brew install ollama
ollama pull qwen2.5-coder:7b
ollama serve

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install fastapi uvicorn langchain langgraph langchain-ollama requests

# 4. Run the service
uvicorn main:app --reload
```

## Usage

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the weather in London?"}'
```

Response:
```json
{
  "answer": "Here's what I found: The weather in London is sunny, 22°C",
  "latency_ms": 1618.93,
  "input_tokens": 0,
  "output_tokens": 0
}
```

## Known limitations (intentional, for now)

- **Token counts are hardcoded to 0** — LangGraph's response doesn't surface Ollama's native token usage yet. Planned fix: extract `prompt_eval_count` / `eval_count` from the underlying model response inside the agent node.
- **Tool result is local-only, dummy data** — `get_current_weather` returns hardcoded values; not wired to a real weather API.
- **Two defensive patterns exist to work around local-model tool-calling quirks:**
  - *Fallback JSON parsing* — `qwen2.5-coder` sometimes emits a tool call as raw JSON text instead of populating LangChain's structured `tool_calls` field. A parser detects and converts this.
  - *Loop guard* — after a tool result comes back, the model sometimes re-emits the same tool call instead of answering. Once a tool has run, the agent now builds the final answer directly instead of re-invoking the model.

These aren't hacks — they're the kind of guardrail a production LLM system needs regardless of which model backs it, since model behavior (local or hosted) is never 100% reliable.

## Roadmap

- **Phase 2** — Retrieval-augmented generation (RAG) over a small document set
- **Phase 3** — Prompt versioning
- **Phase 4** — Eval suite (regression testing for prompt/model changes)
- **Phase 5** — CI/CD (GitHub Actions running evals on every push)
- **Phase 6** — Observability (Langfuse/LangSmith tracing, cost tracking)
- **Phase 7** — Containerized deployment + basic alerting

## Stack

- **FastAPI** — HTTP layer
- **LangGraph** — agent orchestration (tool-use decision loop)
- **Ollama** — local model serving (`qwen2.5-coder:7b`)
- **Pydantic** — request/response validation