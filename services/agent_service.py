"""
The core agent logic: LangGraph state, graph assembly, and the
agent/tool-routing loop. Equivalent to a @Service class in Spring —
this is where the actual business logic lives, separate from the
HTTP layer (routers/) and the tool implementations (tools/).
"""

import json
import logging
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import CHAT_MODEL, CHAT_TEMPERATURE, LOGGER_NAME
from tools.weather import get_current_weather
from tools.documents import search_documents
from services.prompt_service import load_prompt
from config import ACTIVE_SYSTEM_PROMPT
from langfuse.langchain import CallbackHandler
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

langfuse_handler = CallbackHandler()

logger = logging.getLogger(LOGGER_NAME)


# ─────────────────────────────────────────────────────────────
# Tools available to the agent — add new tools here as they're built
# ─────────────────────────────────────────────────────────────

tools = [get_current_weather, search_documents]


# ─────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Conversation history"]


# ─────────────────────────────────────────────────────────────
# Model + prompt (built once at import time, reused across requests)
# ─────────────────────────────────────────────────────────────

llm = ChatOllama(model=CHAT_MODEL, temperature=CHAT_TEMPERATURE)
llm_with_tools = llm.bind_tools(tools)

system_prompt_text = load_prompt(ACTIVE_SYSTEM_PROMPT)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
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
    # the LLM again and build the final answer directly. Local models
    # (e.g. qwen2.5-coder) can loop indefinitely re-calling tools instead
    # of answering, so this guard is intentionally strict.
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

    logger.debug(f"tool_calls={getattr(response, 'tool_calls', None)} content={response.content!r}")

    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────────────────────
# Assemble the graph ONCE at import time
# ─────────────────────────────────────────────────────────────

_workflow = StateGraph(AgentState)
_workflow.add_node("agent", call_agent)
_workflow.add_node("tools", ToolNode(tools))
_workflow.set_entry_point("agent")
_workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
_workflow.add_edge("tools", "agent")

agent = _workflow.compile()
