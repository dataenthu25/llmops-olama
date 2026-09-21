# PromptOps

A small LLMOps learning project: a FastAPI service that wraps a local LLM (via Ollama) behind a `/ask` endpoint, with an agentic LangGraph layer that can decide whether to call tools before answering.

Built as a hands-on project while transitioning from DevOps (Spring Boot/Java) into LLMOps — applying CI/CD, monitoring, and ops discipline to LLM-based systems instead of traditional services.

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

### Phase 3 — Prompt versioning (complete)
- [x] System prompts moved out of code into versioned files under `prompts/`
- [x] Active prompt version controlled by a single setting (`ACTIVE_SYSTEM_PROMPT` in `config.py`)
- [x] Every request's logs record which prompt version produced the answer, for traceability
- [x] Confirmed: switching versions is a one-line config change, no code edits needed
- [ ] **Not yet done:** `ACTIVE_SYSTEM_PROMPT` is set in code, not read from an environment variable — acceptable for a solo learning project, but a real deployment would typically make this externally configurable (env var or config service) so it can change without a code deploy

### Phase 4 — Eval suite (complete, 3/3 passing — see Phase 4.5 for how)
- [x] Fixed set of test questions (`evals/test_cases.json`) covering math/reasoning, weather tool, and RAG paths
- [x] Automated eval runner (`evals/run_evals.py`) that hits the running `/ask` endpoint and checks answers against expected/forbidden substrings
- [x] Non-zero exit code on failure — ready to wire into CI (Phase 5)
- [x] Wired into `dev-reset.sh` as a final step, so every dev cycle reset ends with a pass/fail summary
- [x] **Used the suite for real regression testing:** it caught a genuine, repeatable bug — the `math_basic` test case ("What is 2+2?") consistently triggers an unnecessary `search_documents` tool call instead of answering directly
- [x] Tried fixing via a clearer system prompt (explicitly instructing the model to skip tools for math/reasoning questions) — confirmed the prompt change was applied correctly, but it had **zero effect** on the behavior. Confirmed this was a genuine model capability limit, not a prompt-wording issue — see Phase 4.5 for how it was actually fixed.

### Phase 4.5 — Fine-tuning: fixing the Phase 4 limitation for real
Rather than leave the over-eager tool-use bug as a permanent documented limitation, this phase used LoRA fine-tuning to build a small, fast pre-router that fixes the exact behavior prompting couldn't.

- [x] Generated a synthetic training dataset (`finetune_data/generate_dataset.py`) of 110 labeled examples across three classes: `no_tool_needed`, `use_search_documents`, `use_get_current_weather` — varied phrasing per class to encourage generalization, not memorization
- [x] LoRA fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` on Google Colab's free T4 GPU using `peft` + `trl` (training itself took ~26 seconds once environment issues were resolved — only 0.1% of parameters were trainable)
- [x] Validated the fine-tuned classifier standalone before integrating: correctly classified all 5 held-out test questions, including one never seen in training ("What is the capital of Spain?" → correctly `no_tool_needed`), showing genuine pattern generalization rather than memorization
- [x] Integrated the classifier as a **pre-router** in `services/classifier_service.py` and wired it into `agent_service.py`'s `call_agent`: on the first turn, the classifier decides `no_tool_needed` vs. tool-needed *before* the main tool-bound model ever runs, skipping tool-binding entirely when no tool is needed
- [x] **Re-ran the exact same Phase 4 eval suite after integration: 3/3 passing**, including `math_basic` for the first time in the project's history — direct, evaluated proof the fix works, not just an assumption
- [x] Real environment friction encountered and resolved along the way, consistent with this project's pattern: a `peft`-internal `torchao` version mismatch on Colab, an `SFTTrainer` API rename (`tokenizer=` → `processing_class=` in current `trl`), and a silent CPU-fallback (`torch.cuda.is_available()` returning `False`) after an earlier `--force-reinstall` — each diagnosed and fixed rather than worked around
- [ ] **Known limitation:** the classifier is a narrow, single-purpose model (110 training examples, 3 output classes) — it is not a general-purpose replacement for the main agent's reasoning, only a fast gate for one specific known failure mode. It also adds a second model's worth of memory/startup overhead to the service. A production version of this pattern would need a larger, more rigorously held-out evaluation set before being trusted beyond this project's scope.

### Phase 5 — CI/CD (core pipeline working, one job unverified live)
- [x] GitHub Actions workflow (`.github/workflows/ci.yml`) with two jobs:
  - `lint-and-import-check` — runs on every push, confirms the app imports cleanly (catches broken code before it's even run locally)
  - `eval-suite` — installs Ollama, pulls both models, ingests documents, starts the server, and runs the full eval suite — gated to pull requests into `main` only
- [x] Confirmed `lint-and-import-check` runs and passes on every push
- [x] Deliberate two-tier design: fast checks on every push, expensive checks (Ollama install + ~5GB model pull) only on PRs — a practical trade-off for a project with local-model dependencies, avoiding slow/wasteful CI on every commit
- [x] Fixed a real YAML indentation bug during setup (`eval-suite` was accidentally nested inside `lint-and-import-check`'s block) — a good example of a class of bug Java/Spring engineers aren't used to, since YAML's structure is whitespace-significant with no braces to visually anchor nesting
- [ ] **Known limitation:** the `eval-suite` job's `pull_request`-triggered run was never directly observed completing in the Actions UI — the PR for this branch was merged via GitHub's UI, and the corresponding run visible afterward showed as a `push` event (where `eval-suite` correctly shows "Skipped" per its own gating condition), not a `pull_request` event. The job's YAML is correct and the eval suite itself is proven to work (both locally and conceptually equivalent to what CI would run), but a live, fully-completed `eval-suite` CI run showing pass/fail output was not directly confirmed. Documented honestly rather than chased further, given the cost (minutes of runtime, ~5GB download) of repeatedly forcing PR-triggered runs just to verify UI display.

### Phase 6 — Observability (complete)
- [x] Self-hosted [Langfuse](https://langfuse.com) running locally via Docker (`langfuse-local/`, kept as a sibling directory outside the `promptops` repo, not a project dependency)
- [x] Langfuse `CallbackHandler` wired into the LangGraph agent invocation (`routers/ask.py` passes it via `config={"callbacks": [...]}`)
- [x] Every `/ask` request now produces a full trace in the Langfuse UI: the entire `agent` → `should_continue` → `tools` → `agent` decision path, with per-step latency, visible and inspectable per request
- [x] Confirmed traces correctly show tool selection decisions (`search_documents` vs `get_current_weather` vs no tool), matching what was previously only visible through raw log lines
- [x] **Used tracing to sharpen a Phase 4 finding:** a trace showed the `"What is 2+2?"` question being answered *without* triggering a tool call — meaning the over-eager tool-use issue from Phase 4 is intermittent, not fully consistent, a more precise characterization than what log-based debugging alone had shown
- [ ] **Known limitation:** cost/token tracking (`Input Tokens`, `Output Tokens`, `Cost ($)`) shows empty in Langfuse traces. This is expected, not a bug — Langfuse's automatic cost calculation is built around known paid-model pricing tables (OpenAI, Anthropic, etc.), and Ollama doesn't report cost data the same way. This extends the existing Phase 1 known limitation (token counts hardcoded to 0 in the API response) rather than being a new gap.
- [ ] **Setup friction (resolved, documented for future-me):** the Langfuse Python SDK (v4.15.1) changed its import path from `langfuse.callback` (older versions) to `langfuse.langchain` (current). Additionally, `.env` values weren't loading at all initially — `load_dotenv()` was never called anywhere in the codebase (a genuine gap from earlier phases), fixed by adding it to the top of `config.py`, which centralizes all other settings.

## Project structure

Refactored from a single flat `main.py` into a layered structure, separating
HTTP handling, business logic, data models, and tool implementations:

```
promptops/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions: fast import check + gated eval suite
├── main.py                  # entrypoint — creates app, includes routers
├── config.py                 # centralized settings
├── requirements.txt           # pinned dependencies
├── schemas/
│   └── ask.py                  # AskRequest / AskResponse request/response models
├── routers/
│   └── ask.py                  # /health, /ask HTTP routes
├── services/
│   ├── agent_service.py         # LangGraph state, graph, agent loop
│   └── classifier_service.py     # fine-tuned LoRA tool-selection pre-router (Phase 4.5)
├── models/
│   └── lora-tool-selector-final/  # fine-tuned LoRA adapter weights
├── finetune_data/
│   ├── generate_dataset.py         # synthetic training data generator
│   └── training_data.jsonl          # 110 labeled examples for the classifier
├── tools/
│   ├── weather.py                # get_current_weather
│   └── documents.py              # search_documents + vector store connection
├── prompts/
│   ├── system_v1.txt              # versioned system prompts — plain text, one per version
│   └── system_v2.txt
├── evals/
│   ├── test_cases.json             # fixed set of question/expected-answer pairs
│   └── run_evals.py                # eval runner — checks answers, exits non-zero on failure
├── rag/
│   ├── ingestor.py                # one-time/on-demand document ingestion script
│   └── chroma_db/                 # persistent local vector store (generated)
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
  ├── fine-tuned classifier pre-router (services/classifier_service.py)
  │     decides: no_tool_needed vs. tool-needed, before the main model runs
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

Every agent invocation is also traced to a self-hosted Langfuse instance
(separate Docker service, see Observability below) via a LangChain callback
handler — not shown in the request path above since it observes rather than
participates in it.
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

## Observability (Langfuse)

Traces are sent to a self-hosted Langfuse instance, run as a **separate**
service (not part of this repo, not a project dependency):

```bash
# in a sibling directory, NOT inside promptops/
cd /path/to/project-work
git clone https://github.com/langfuse/langfuse.git langfuse-local
cd langfuse-local
docker compose up -d
```

Open `http://localhost:3000`, create a local account/project, and copy the
generated API keys into `promptops/.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Every `/ask` request then appears as a full trace in the Langfuse UI,
showing the entire agent decision path (tool selection, per-step latency,
model input/output at each turn) — see Known Limitations below for what's
not yet captured (cost/token data).

## Usage

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Phase 1 of PromptOps?"}'
```

## Dev automation

`dev-reset.sh` automates the full local dev cycle in one command: kill any
running server, wipe and rebuild the vector store from `docs/`, restart the
FastAPI server in the background, wait for it to become healthy, fire a
smoke-test question at `/ask`, then run the full eval suite as a final
regression check.

```bash
chmod +x dev-reset.sh   # one-time
./dev-reset.sh                              # uses a default smoke-test question
./dev-reset.sh "What is Phase 2 about?"     # or pass a custom question
```

Logs go to `server.log`; the script prints the server's PID so you can stop
it (`kill <PID>`) or tail logs (`tail -f server.log`) afterward.

This same sequence is now also automated in CI (`.github/workflows/ci.yml`,
Phase 5) — a fast version runs on every push, and the full version
(including the eval suite) runs on pull requests into `main`.

## Known limitations (documented honestly, not hidden)

- **Token counts are hardcoded to 0** — LangGraph's response doesn't surface Ollama's native token usage yet. Planned fix: extract `prompt_eval_count` / `eval_count` from the underlying model response inside the agent node.
- **RAG answers are echoed from raw retrieved chunks, not fully synthesized.** During Phase 2 development, `qwen2.5-coder:7b` proved unreliable when asked to read retrieved context and compose an answer in its own words:
  - With a permissive prompt, it sometimes claimed "no relevant documents found" even when the correct chunks were retrieved successfully.
  - With a stronger prompt explicitly instructing it to read tool results carefully, it instead fell into an infinite loop, repeatedly re-calling `search_documents` with a malformed literal query instead of answering.
  - **Conclusion:** this is a genuine small-local-model reasoning limitation, not a bug in the retrieval pipeline — the vector search and tool-selection logic were confirmed correct in isolation. Larger hosted models (e.g. Claude, GPT-4) are dramatically more reliable at this specific task (reading tool/retrieval results and synthesizing a grounded answer), which is part of why hosted APIs remain valuable even in a project built primarily for free, local experimentation.
- **Defensive patterns exist to work around local-model tool-calling quirks:**
  - *Fallback JSON parsing* — `qwen2.5-coder` sometimes emits a tool call as raw JSON text instead of populating LangChain's structured `tool_calls` field. A parser detects and converts this.
  - *Safety-net loop guard* — once any tool has returned a result, the agent stops calling the LLM again and builds the answer directly, rather than risking an infinite re-call loop.
  - *Fine-tuned classifier pre-router (Phase 4.5)* — a small LoRA-tuned model decides `no_tool_needed` vs. tool-needed before the main model runs, fixing a specific over-eager tool-use bug that prompting alone couldn't resolve.

These aren't hacks — they're the kind of guardrail a production LLM system needs regardless of which model backs it, since model behavior (local or hosted) is never 100% reliable. Where the eval suite has caught a real regression, that's the eval suite doing its job — and where fine-tuning fixed what prompting couldn't, that's a genuine, evaluated engineering result, not a workaround.

## Roadmap

- ~~**Phase 3** — Prompt versioning~~ (complete, see above)
- ~~**Phase 4** — Eval suite (regression testing for prompt/model changes)~~ (complete, see above)
- ~~**Phase 5** — CI/CD (GitHub Actions running evals on every push)~~ (core pipeline complete, see above)
- ~~**Phase 6** — Observability (Langfuse/LangSmith tracing, cost tracking)~~ (complete, see above)
- **Phase 7** — Containerized deployment + basic alerting

## Stack

- **FastAPI** — HTTP layer
- **LangGraph** — agent orchestration (tool-use decision loop)
- **Ollama** — local model serving (`qwen2.5-coder:7b` for chat, `nomic-embed-text` for embeddings)
- **Chroma** — local persistent vector store
- **Pydantic** — request/response validation
- **Langfuse** (self-hosted) — LLM observability and tracing
- **peft / trl** — LoRA fine-tuning of the tool-selection classifier (Phase 4.5)