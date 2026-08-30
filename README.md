# PromptOps

A small LLMOps learning project: a FastAPI service that wraps a local LLM (via Ollama) behind a `/ask` endpoint, with an agentic LangGraph layer that can decide whether to call tools before answering.


## Why this project

Most LLMOps tutorials stop at "call an API and print the response." This project instead treats an LLM call like any other production dependency: it gets logged, error-handled, and eventually tested, versioned, and monitored like a normal service.

## Current status

### Phase 1 (complete)
- [x] FastAPI service with `/health` and `/ask` endpoints
- [x] Local LLM via [Ollama](https://ollama.com) (`qwen2.5-coder:7b`) — no API costs during learning/dev
- [x] Structured logging (latency per request)
- [x] Error handling (`HTTPException` on upstream failures)
- [x] LangGraph agent loop: the model decides whether a tool call is needed before answering
- [x] Defensive patterns for local-model quirks (see below)

### Phase 2 — RAG (functionally complete, with a documented model-quality limitation)
- [x] Document ingestion pipeline (`ingestor.py`): chunks `.txt`/`.md` files and embeds them with a local embedding model (`nomic-embed-text`)
- [x] Persistent local vector store via Chroma (no external infra required)
- [x] `search_documents` added as a second LangGraph tool alongside `get_current_weather`
- [x] Confirmed: the agent correctly decides *when* retrieval is needed vs. answering directly
- [x] Confirmed: vector search reliably returns relevant, correct chunks from ingested documents
- [ ] **Known limitation:** the local 7B model's ability to *synthesize* a natural-language answer from retrieved chunks is inconsistent — see below

## Project structure

Refactored from a single flat `main.py` into a layered structure — the
Python/FastAPI  package layout:

```
promptops/
├── main.py                  # entrypoint — creates app, includes routers
├── config.py                 # centralized settings 
├── requirements.txt          # pinned dependencies
├── schemas/
│   └── ask.py                # AskRequest / AskResponse 
├── routers/
│   └── ask.py                 # /health, /ask HTTP routes 
├── services/
│   └── agent_service.py       # LangGraph state, graph, agent loop 
├── tools/
│   ├── weather.py              # get_current_weather 
│   └── documents.py            # search_documents + vector store connection
├── rag/
│   ├── ingestor.py             # one-time/on-demand document ingestion script
│   └── chroma_db/              # persistent local vector store (generated)
├── docs/                      # source documents for RAG ingestion
└── dev-reset.sh               # automated local dev cycle (see below)
```

Each layer has one job: `routers/` only handles HTTP concerns, `services/`
holds the actual agent logic, `tools/` are self-contained and independently
testable, and `config.py` centralizes settings that used to be hardcoded
across multiple files.

## Architecture

```
Client
  │
  ▼
FastAPI (/ask)  — routers/ask.py
  │
  ▼
LangGraph agent  — services/agent_service.py
  │
  ├── decides: answer directly, use get_current_weather, or use search_documents?
  │
  ├── [tool needed] → ToolNode executes (tools/) → result fed back to agent
  │
  └── [no tool needed] → final answer
  │
  ▼
Ollama (qwen2.5-coder:7b — chat, nomic-embed-text — embeddings, both local)
  │
  ▼
Chroma (local persistent vector store, rag/chroma_db)
```

## Setup

```bash
# 1. Install Ollama and pull the models
brew install ollama
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
ollama serve

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ingest documents (run once, or whenever docs/ changes)
cd rag
python3 ingestor.py
cd ..

# 5. Run the service
uvicorn main:app --reload
```

## Usage

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Phase 1 of PromptOps?"}'
```

## Dev automation

`dev-reset.sh` automates the full local dev cycle in one command: kill any
running server, wipe and rebuild the vector store from `docs/`, restart the
FastAPI server in the background, wait for it to become healthy, then fire a
smoke-test question at `/ask`.

```bash
chmod +x dev-reset.sh   # one-time
./dev-reset.sh                              # uses a default smoke-test question
./dev-reset.sh "What is Phase 2 about?"     # or pass a custom question
```

Logs go to `server.log`; the script prints the server's PID so you can stop
it (`kill <PID>`) or tail logs (`tail -f server.log`) afterward.

This is effectively a preview of the CI/CD pipeline planned for Phase 5 —
clean state, rebuild, deploy, health-check, smoke-test — just running locally
for now instead of in GitHub Actions.

## Known limitations (documented honestly, not hidden)

- **Token counts are hardcoded to 0** — LangGraph's response doesn't surface Ollama's native token usage yet. Planned fix: extract `prompt_eval_count` / `eval_count` from the underlying model response inside the agent node.
- **RAG answers are echoed from raw retrieved chunks, not fully synthesized.** During Phase 2 development, `qwen2.5-coder:7b` proved unreliable when asked to read retrieved context and compose an answer in its own words:
  - With a permissive prompt, it sometimes claimed "no relevant documents found" even when the correct chunks were retrieved successfully.
  - With a stronger prompt explicitly instructing it to read tool results carefully, it instead fell into an infinite loop, repeatedly re-calling `search_documents` with a malformed literal query instead of answering.
  - **Conclusion:** this is a genuine small-local-model reasoning limitation, not a bug in the retrieval pipeline — the vector search and tool-selection logic were confirmed correct in isolation. Larger hosted models (e.g. Claude, GPT-4) are dramatically more reliable at this specific task (reading tool/retrieval results and synthesizing a grounded answer), which is part of why hosted APIs remain valuable even in a project built primarily for free, local experimentation.
- **Defensive patterns exist to work around local-model tool-calling quirks:**
  - *Fallback JSON parsing* — `qwen2.5-coder` sometimes emits a tool call as raw JSON text instead of populating LangChain's structured `tool_calls` field. A parser detects and converts this.
  - *Safety-net loop guard* — once any tool has returned a result, the agent stops calling the LLM again and builds the answer directly, rather than risking an infinite re-call loop.

These aren't hacks — they're the kind of guardrail a production LLM system needs regardless of which model backs it, since model behavior (local or hosted) is never 100% reliable.

## Roadmap

- **Phase 3** — Prompt versioning
- **Phase 4** — Eval suite (regression testing for prompt/model changes)
- **Phase 5** — CI/CD (GitHub Actions running evals on every push)
- **Phase 6** — Observability (Langfuse/LangSmith tracing, cost tracking)
- **Phase 7** — Containerized deployment + basic alerting

## Stack

- **FastAPI** — HTTP layer
- **LangGraph** — agent orchestration (tool-use decision loop)
- **Ollama** — local model serving (`qwen2.5-coder:7b` for chat, `nomic-embed-text` for embeddings)
- **Chroma** — local persistent vector store
- **Pydantic** — request/response validation