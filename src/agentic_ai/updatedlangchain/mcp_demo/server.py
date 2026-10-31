# =============================================================================
# A minimal MCP (Model Context Protocol) SERVER, for learning.
#
# What is MCP?
#   An open protocol that lets an AI app (the CLIENT / host) discover and use
#   capabilities exposed by a separate program (the SERVER). Think "USB-C for
#   AI tools": one standard plug, so any client can talk to any server.
#
# A server can expose three kinds of things (called "primitives"):
#   1. TOOLS     - functions the model can CALL (do something: add, search...).
#                  Controlled by the model.
#   2. RESOURCES - read-only DATA addressed by a URI (files, records, docs).
#                  Controlled by the application.
#   3. PROMPTS   - reusable prompt templates the user can pick.
#                  Controlled by the user.
#
# Messages are JSON-RPC 2.0. How they travel is the TRANSPORT. This file can
# run with two of them:
#   - stdio : the client LAUNCHES this script as a child process and talks over
#             its stdin/stdout. Simple, local only, one client per process.
#   - http  : "streamable HTTP". The server runs on its own, listening on a
#             URL (http://127.0.0.1:8000/mcp). Can be remote, many clients.
#
# What "stdio" means:
#   stdio = STANDARD INPUT/OUTPUT, the three text streams every process gets
#   from the operating system:
#       stdin  (standard input)  - data coming in (keyboard, or a pipe)
#       stdout (standard output) - normal output (what print() writes)
#       stderr (standard error)  - error and log messages
#   In a terminal, stdin is your keyboard and stdout is your screen, but they
#   are just streams, so another program can be connected to them instead of a
#   human. That is how shell pipes work: in `ls | grep py`, the stdout of `ls`
#   feeds the stdin of `grep`.
#
#   In MCP, the stdio transport has the client start the server as a child
#   process and connect the two through those streams:
#       client  --writes JSON-RPC requests-->  server's stdin
#       client  <--reads JSON-RPC responses--  server's stdout
#                                              server's stderr = logs only
#   Each message is one line of JSON. No network, port or URL is involved.
#
#   Consequences:
#       - Local only: the server must run on the same machine as the client.
#       - One client per server process: the client owns the process and stops
#         it when it disconnects.
#       - No print() in the server: stdout is the protocol channel, so stray
#         text breaks the JSON stream. Log to stderr instead.
#   HTTP is the alternative: the server runs independently at a URL and
#   clients connect over the network.
#
# All transports (the spec defines the first two; the rest are SDK features):
#   - stdio           STANDARD. Child process, JSON-RPC over stdin/stdout.
#                     Use for local tools launched by the client.
#   - Streamable HTTP STANDARD. Client sends HTTP POSTs to ONE endpoint (/mcp).
#                     The server answers with plain JSON, or streams the reply
#                     as Server-Sent Events. Use for remote or shared servers
#                     with many clients.
#   - SSE             DEPRECATED. The old HTTP transport: a long-lived SSE stream
#                     for server->client messages plus separate POSTs for
#                     client->server (two endpoints). Only for older servers or
#                     clients. mcp.run(transport="sse") still exists for them.
#   - In-memory       Client and server run in the same Python process with no
#                     I/O (the v2 Client accepts an MCPServer instance
#                     directly). Use for unit tests.
#   - Custom          The SDK has a Transport abstraction, so you can plug in
#                     your own (WebSockets, message queues, ...).
#   mcp.run() accepts exactly three string values: "stdio", "sse" and
#   "streamable-http".
#
# Why Streamable HTTP replaced SSE:
#   SSE needed a permanently open connection and two endpoints, which is hard
#   to scale and awkward behind load balancers. Streamable HTTP uses a single
#   endpoint, can run without server-side session state
#   (stateless_http=True), and only opens a stream when the server actually
#   needs to stream something.
#
# Rule of thumb:
#   - server runs on the user's own machine        -> stdio
#   - server runs elsewhere or serves many users   -> Streamable HTTP, with
#                                                     authentication
#   - writing tests                                -> in-memory
#
# Under every transport the messages are the same JSON-RPC 2.0, so tools,
# resources and prompts behave identically. That is why demo() in client.py
# works over both stdio and HTTP unchanged.
#
# Run it (from the project root):
#   stdio : normally you do NOT run it by hand; client.py launches it.
#   http  : .venv/bin/python src/agentic_ai/updatedlangchain/mcp_demo/server.py --transport http
#
# Note: this uses the mcp 2.x SDK, where the high-level server class is
# `MCPServer` (it was called `FastMCP` in mcp 1.x).
# =============================================================================

import argparse

# MCPServer is the high-level API: you register plain Python functions with
# decorators, and it builds the protocol messages and JSON schemas for you.
from mcp.server.mcpserver import MCPServer

# The name is sent to clients during the handshake (the "initialize" step), so
# they know which server they connected to.
mcp = MCPServer("learning-demo")


# -----------------------------------------------------------------------------
# 1) TOOL: a function the model can call.
# -----------------------------------------------------------------------------
# @mcp.tool() registers the function. The SDK builds the tool definition from:
#   - function name  -> tool name
#   - docstring      -> description (the model reads this to decide when to use it)
#   - type hints     -> JSON schema of the arguments
# This is the same idea as LangChain's @tool, but exposed over a protocol so
# any MCP client can use it, not just your own Python code.
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city (fake data, for demo purposes)."""
    fake_data = {"london": "rainy, 14C", "delhi": "sunny, 33C", "tokyo": "cloudy, 21C"}
    return fake_data.get(city.lower(), f"no data for {city}")


# -----------------------------------------------------------------------------
# 2) RESOURCE: read-only data identified by a URI.
# -----------------------------------------------------------------------------
# The URI template "greeting://{name}" has a placeholder. A client reading
# "greeting://Pankaj" makes the SDK call get_greeting(name="Pankaj").
# Use a resource for data to READ; use a tool for actions or computation.
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """A personalised greeting."""
    return f"Hello, {name}! Welcome to MCP."


# -----------------------------------------------------------------------------
# 3) PROMPT: a reusable template the user (not the model) can choose.
# -----------------------------------------------------------------------------
# Returns text that a client can insert into a conversation. Clients usually
# show prompts as things like slash commands.
@mcp.prompt()
def review_code(code: str) -> str:
    """Ask the model to review a piece of code."""
    return f"Please review this code and point out bugs and improvements:\n\n{code}"


# -----------------------------------------------------------------------------
# Entry point: choose the transport.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        # IMPORTANT for stdio: stdout IS the protocol channel. Never use
        # print() in a stdio server, because it would inject non-JSON text into
        # the stream and break the client. Log to stderr instead:
        #   print("debug", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        # Starts a web server (uvicorn) and serves MCP at http://127.0.0.1:PORT/mcp.
        # It binds to 127.0.0.1 by default, so only your machine can reach it.
        # Do not expose it on a network without authentication.
        mcp.run(transport="streamable-http", port=args.port)
