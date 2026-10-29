# =============================================================================
# LangGraph agent that PAUSES (interrupt) to ask a human, then RESUMES.
#
# Big picture of the flow:
#
#   START -> chatbot --(model asked for a tool?)--> tools -> chatbot -> ...
#                |                                    |
#                |                                    +-- if the tool is
#                +--(no tool call)--> END                 human_assistance, the
#                                                         graph PAUSES here and
#                                                         waits for a human.
#
# Pause/resume needs two things:
#   1. interrupt()      - called inside a node/tool to stop the graph.
#   2. a checkpointer   - saves the graph's state so it can be loaded later.
# =============================================================================

# `Annotated` lets us attach extra info (a "reducer") to a type hint.
# We use it below to tell LangGraph HOW to merge new messages into the state.
from typing import Annotated

# Reads key=value pairs from a `.env` file into environment variables.
# The Groq and Tavily clients read GROQ_API_KEY / TAVILY_API_KEY from there.
from dotenv import load_dotenv

# `@tool` turns a plain Python function into a tool the LLM can call.
from langchain_core.tools import tool

# Chat model client for Groq's hosted models.
from langchain_groq import ChatGroq

# Web search tool (needs TAVILY_API_KEY). Gives the agent something to call.
from langchain_tavily import TavilySearch

# In-memory checkpointer: stores graph state in a Python dict while the
# program runs. Lost when the process exits (use a DB-backed one for that).
from langgraph.checkpoint.memory import MemorySaver

# StateGraph: the graph builder.  START: the special entry point.
from langgraph.graph import START, StateGraph

# Reducer for chat messages: APPENDS new messages instead of overwriting.
from langgraph.graph.message import add_messages

# ToolNode: a ready-made node that runs whatever tool(s) the model requested.
# tools_condition: a ready-made router; sends to "tools" if the last AI message
# has tool_calls, otherwise to END.
from langgraph.prebuilt import ToolNode, tools_condition

# interrupt: pauses the graph.  Command: lets us RESUME the graph with a value.
from langgraph.types import Command, interrupt

# TypedDict (from typing_extensions for compatibility) defines the State shape.
from typing_extensions import TypedDict

# Load the .env file now, before any client below needs the API keys.
load_dotenv()


# -----------------------------------------------------------------------------
# 1) STATE: the data that flows through the graph and gets checkpointed.
# -----------------------------------------------------------------------------
class State(TypedDict):
    # One field, `messages`: the whole conversation (human, AI, tool messages).
    # `add_messages` is the reducer: when a node returns {"messages": [msg]},
    # LangGraph APPENDS msg to the existing list rather than replacing it.
    # NOTE: the invoke() input key must be "messages".
    messages: Annotated[list, add_messages]


# -----------------------------------------------------------------------------
# 2) TOOLS: what the agent is allowed to call.
# -----------------------------------------------------------------------------
# @tool turns a plain function into a tool object the LLM can call. It builds:
#   - name        from the function name
#   - description from the docstring (the model reads it to decide when to call)
#   - arg schema  from the parameters and type hints
# llm.bind_tools() tells the model about the tool; ToolNode actually runs it.
#
# Notes:
#   - Without a docstring, @tool raises an error (the description is required).
#   - The result is an object, not a function: call it with
#     human_assistance.invoke({"query": "hi"}), not human_assistance("hi").
#   - TavilySearch is already a tool class, so it needs no decorator; @tool is
#     for your own functions.
#   - Override the defaults with @tool("custom_name") or @tool(description="...").
#   - @tool is optional: bind_tools() and ToolNode both accept plain functions
#     too, reading the name, docstring and type hints the same way. A docstring
#     and JSON-schema-friendly type hints (str, int, list[str], Pydantic models)
#     are still required. @tool adds explicit control (name, description,
#     args_schema, return_direct), .invoke()/.name/.args for testing, and makes
#     it clear the function is a tool. For a quick demo, plain functions are fine.
@tool
def human_assistance(query: str) -> str:
    # The docstring below is sent to the LLM as the tool description. The model
    # reads it to decide WHEN to call this tool, so keep it meaningful.
    """Request assistance from a human"""

    # interrupt() STOPS the graph right here.
    #  - The dict we pass (the "payload") is surfaced to the caller under
    #    result["__interrupt__"], so the outside world knows what is asked.
    #  - The state is saved to the checkpointer.
    #  - Later, when the caller resumes with Command(resume=X), the graph
    #    re-enters this node and interrupt() RETURNS X instead of pausing.
    human_response = interrupt({"query": query})

    # `human_response` is whatever we pass as `resume=` further below, i.e.
    # {"data": "<the human's answer>"}. Return the text part to the model as
    # the tool's result.
    return human_response["data"]


# A web search tool returning at most 2 results (keeps token usage small).
# The list of tools is shared by the model (so it knows them) and ToolNode
# (so it can run them).
tools = [TavilySearch(max_results=2), human_assistance]

# Create the chat model, then `bind_tools` tells it the tool names, arguments
# and descriptions. Without this the model never produces `tool_calls`, and
# tools_condition would always route straight to END.
llm_with_tools = ChatGroq(model="openai/gpt-oss-120b").bind_tools(tools)


# -----------------------------------------------------------------------------
# 3) NODES: functions that take the state and return a state UPDATE.
# -----------------------------------------------------------------------------
def chatbot(state: State):
    # Send the full conversation so far to the model. It replies with an
    # AIMessage, which either contains text (final answer) or `tool_calls`
    # (a request to run tools).
    # Returning {"messages": [reply]} is a partial update; the add_messages
    # reducer appends `reply` to the existing history.
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# -----------------------------------------------------------------------------
# 4) GRAPH WIRING: connect nodes and edges.
# -----------------------------------------------------------------------------
# Create a builder that knows the state schema.
graph_builder = StateGraph(State)

# Register nodes by name. The names are what the edges refer to.
graph_builder.add_node("chatbot", chatbot)
# ToolNode looks at the last AI message, runs each requested tool, and appends
# the results as ToolMessages. The name "tools" must match what
# tools_condition returns.
graph_builder.add_node("tools", ToolNode(tools=tools))

# Entry point: the first node to run is "chatbot".
graph_builder.add_edge(START, "chatbot")

# Conditional edge out of "chatbot", decided at RUNTIME by tools_condition:
#   last message has tool_calls -> go to "tools"
#   otherwise                   -> go to END (this is the chatbot->END path,
#                                  so no separate add_edge to END is needed)
graph_builder.add_conditional_edges("chatbot", tools_condition)

# After tools run, always go back to "chatbot" so the model can read the tool
# results and either answer or call more tools. This creates the agent loop.
graph_builder.add_edge("tools", "chatbot")

# compile() turns the builder into a runnable graph.
# checkpointer=MemorySaver() is REQUIRED for interrupt(): at the pause, the
# state is stored under a thread_id, and resuming loads it back from there.
graph = graph_builder.compile(checkpointer=MemorySaver())


# -----------------------------------------------------------------------------
# 5) RUNNER: start the graph, handle each pause, resume until finished.
# -----------------------------------------------------------------------------
def run(user_input: str, thread_id: str = "1") -> None:
    # `thread_id` identifies one conversation/run in the checkpointer. Using
    # the SAME id for the first call and the resume calls is how LangGraph
    # finds the paused run. A different id would start a fresh conversation.
    config = {"configurable": {"thread_id": thread_id}}

    # First call: give the graph the user's message in the "messages" key.
    # A dict with role/content is converted to a HumanMessage by add_messages.
    # If the graph hits interrupt(), invoke() returns early (it does not raise)
    # with the state so far plus an "__interrupt__" entry.
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config)

    # As long as the result contains "__interrupt__", the graph is paused and
    # waiting for a human. A loop also handles the model asking more than once.
    while "__interrupt__" in result:
        # "__interrupt__" is a list of Interrupt objects. `.value` is the exact
        # payload we gave to interrupt(), i.e. {"query": "..."}.
        query = result["__interrupt__"][0].value["query"]
        print(f"\n[Graph paused] The agent asks a human: {query}")

        # Block until the human types something in the terminal.
        answer = input("Your answer: ")

        # RESUME: pass Command(resume=...) instead of new input, with the same
        # config/thread_id. LangGraph loads the checkpoint, re-runs the paused
        # "tools" node, and this time interrupt() returns {"data": answer}.
        # The tool then returns `answer`, and the graph continues to
        # "chatbot". The result may be another interrupt, or the final state.
        result = graph.invoke(Command(resume={"data": answer}), config)

    # No more interrupts: the run finished. The last message is the model's
    # final reply.
    print("\nFinal answer:\n", result["messages"][-1].content)


# Runs only when executed directly (python file.py), not when imported.
if __name__ == "__main__":
    # The prompt explicitly asks for human help so the model is likely to call
    # the human_assistance tool, which triggers the pause/resume demo.
    run(
        "I need some expert guidance for building an AI agent. "
        "Could you request assistance for me?"
    )
