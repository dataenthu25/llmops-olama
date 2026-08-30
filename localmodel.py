"""
LangGraph "hello world" agent — wired to a LOCAL Ollama model instead of OpenAI/Claude.
Free to run, no API key or credits needed.

Prerequisites:
    brew install ollama          (or download from ollama.com)
    ollama pull llama3.1         (a tool-calling-capable model)
    ollama serve                 (usually auto-starts)
    pip install langchain langgraph langchain-ollama

Run with: python3 langgraph_ollama_hello_world.py
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ─────────────────────────────────────────────────────────────
# 1. Define a dummy tool (same pattern as the weather example)
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
    # Dummy data — in real life you'd call a weather API here
    return f"The weather in {location} is sunny, 22°C"


tools = [get_current_weather]


# ─────────────────────────────────────────────────────────────
# 2. Define Agent State
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Conversation history"]


# ─────────────────────────────────────────────────────────────
# 3. Set up the LOCAL model via Ollama (this replaces ChatOpenAI)
# ─────────────────────────────────────────────────────────────

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    MessagesPlaceholder(variable_name="messages"),
])


# ─────────────────────────────────────────────────────────────
# 4. Agent node — the LLM decides whether to call a tool
# ─────────────────────────────────────────────────────────────

import json


def try_parse_fallback_tool_call(text: str):
    """
    Some local models (via Ollama) emit a tool call as raw JSON text in
    `content` instead of populating LangChain's structured `tool_calls`
    field. This is a known integration gap between LangChain and Ollama,
    not a bug in our graph logic.

    This function detects that pattern and manually converts it into the
    same shape LangGraph's ToolNode expects, so should_continue and the
    tools node work correctly regardless of the quirk.
    """
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


from langchain_core.messages import AIMessage, ToolMessage


async def call_agent(state: AgentState):
    messages = state["messages"]

    # Loop guard for flaky local models: qwen2.5-coder sometimes ignores
    # the tool result and just re-emits the same tool-call JSON forever
    # instead of answering. If the last message is already a ToolMessage,
    # the tool has run — skip calling the LLM again and build the final
    # answer directly from the tool's result.
    if isinstance(messages[-1], ToolMessage):
        final_answer = AIMessage(
            content=f"Here's what I found: {messages[-1].content}"
        )
        print("DEBUG: loop guard triggered, skipping re-invoke")
        return {"messages": [final_answer]}

    formatted = prompt.format_messages(messages=messages)
    response = await llm_with_tools.ainvoke(formatted)

    # Fallback: if the model put a tool call in plain text content instead
    # of structured tool_calls, parse it out and attach it manually.
    if not getattr(response, "tool_calls", None) and response.content:
        fallback_calls = try_parse_fallback_tool_call(response.content)
        if fallback_calls:
            response.tool_calls = fallback_calls
            response.content = ""  # clear so it isn't shown as a text answer

    print("DEBUG tool_calls:", getattr(response, "tool_calls", None))
    print("DEBUG content:", repr(response.content))
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────
# 5. Router — decides: go to tools, or finish?
# ─────────────────────────────────────────────────────────────

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────────────────────
# 6. Assemble the graph
# ─────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")

agent = workflow.compile()


# ─────────────────────────────────────────────────────────────
# 7. Run it
# ─────────────────────────────────────────────────────────────

async def main():
    result = await agent.ainvoke({
        "messages": [HumanMessage(content="What's the weather like in Amsterdam?")]
    })
    final_response = result["messages"][-1].content
    print("\n--- FINAL ANSWER ---")
    print(final_response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())