import json
import logging
import time
from typing import Annotated, Sequence, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promptops")


# ─────────────────────────────────────────────────────────────
# Pydantic request/response models
# ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


# ─────────────────────────────────────────────────────────────
# 1. Define tools — must all be defined BEFORE `tools = [...]`
#    and before llm.bind_tools(tools) is called below.
# ─────────────────────────────────────────────────────────────

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location.

    Use this when the user asks about weather conditions.

    Args:
        location: City name (e.g., "San Francisco", "Amsterdam")

    Returns:
        Weather description string
    """
    return f"The weather in {location} is sunny, 22°C"


# Connect to the vector store built by ingestor.py.
# chroma_db lives at promptops/chroma_db — same level as this main.py file.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


@tool
def search_documents(query: str) -> str:
    """Search internal documents for relevant information.

    Use this when the user asks a question that might be answered by
    the ingested documents/notes, rather than general knowledge or weather.

    Args:
        query: The search query

    Returns:
        Relevant document excerpts
    """
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in results)


tools = [get_current_weather, search_documents]


# ─────────────────────────────────────────────────────────────
# 2. Agent State
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Conversation history"]


# ─────────────────────────────────────────────────────────────
# 3. Model + prompt (built once at startup, reused per request)
# ─────────────────────────────────────────────────────────────

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.2)
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    MessagesPlaceholder(variable_name="messages"),
])


def try_parse_fallback_tool_call(text: str):
    """Some local models emit a tool call as raw JSON text instead of
    populating LangChain's structured tool_calls field. Detect and
    convert that pattern manually."""
    text = text.strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(data, dict) and "name" in data and "arguments" in data:
        return [{
            "name": data["name"],
            "args": data["arguments"],
            "id": "fallback_call_1",
        }]
    return None


async def call_agent(state: AgentState):
    messages = state["messages"]

    # Safety net: as soon as ANY tool has returned a result, stop calling
    # the LLM again and build the final answer directly. Earlier testing
    # showed qwen2.5-coder can loop indefinitely re-calling tools instead
    # of answering, so this guard is intentionally strict (threshold=1)
    # rather than waiting for 2+ results to confirm a real loop.
    if any(isinstance(m, ToolMessage) for m in messages):
        last_tool_msg = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
        return {"messages": [AIMessage(content=f"Based on what I found: {last_tool_msg.content}")]}

    formatted = prompt.format_messages(messages=messages)
    response = await llm_with_tools.ainvoke(formatted)

    if not getattr(response, "tool_calls", None) and response.content:
        fallback_calls = try_parse_fallback_tool_call(response.content)
        if fallback_calls:
            response.tool_calls = fallback_calls
            response.content = ""

    logger.debug(f"DEBUG tool_calls={getattr(response, 'tool_calls', None)} content={response.content!r}")

    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────────────────────
# 4. Assemble the graph ONCE at import/startup time
# ─────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

agent = workflow.compile()


# ─────────────────────────────────────────────────────────────
# 5. FastAPI routes
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    start = time.perf_counter()

    try:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=request.question)]
        })
        answer_text = result["messages"][-1].content

        latency_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"question_len={len(request.question)} "
            f"latency_ms={latency_ms:.2f}"
        )

        return AskResponse(
            answer=answer_text,
            latency_ms=round(latency_ms, 2),
            input_tokens=0,   # LangGraph/Ollama doesn't surface this directly yet
            output_tokens=0,
        )

    except Exception:
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="Failed to generate an answer from the agent.",
        ) from None