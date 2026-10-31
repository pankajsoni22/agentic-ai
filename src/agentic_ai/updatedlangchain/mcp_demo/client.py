# =============================================================================
# A minimal MCP CLIENT, for learning. It talks to server.py over both
# transports and exercises tools, resources and prompts.
#
# Lifecycle of every MCP connection:
#   1. connect   - open the transport (spawn a process, or open an HTTP URL)
#   2. initialize - handshake: client and server exchange name, version and
#                   capabilities (which primitives each side supports)
#   3. use       - list_tools / call_tool / read_resource / get_prompt ...
#   4. close     - leaving the `async with` block shuts everything down
# `async with Client(...)` does steps 1, 2 and 4 for you.
#
# Run it (from the project root):
#   stdio : .venv/bin/python src/agentic_ai/updatedlangchain/mcp_demo/client.py stdio
#   http  : start the server first in another terminal:
#             .venv/bin/python src/agentic_ai/updatedlangchain/mcp_demo/server.py --transport http
#           then:
#             .venv/bin/python src/agentic_ai/updatedlangchain/mcp_demo/client.py http
# =============================================================================

import asyncio
import sys
from pathlib import Path

# Client is the high-level API. StdioServerParameters describes a command to
# launch as a child process for the stdio transport.
from mcp import Client, StdioServerParameters

SERVER_FILE = Path(__file__).parent / "server.py"


async def demo(client: Client) -> None:
    """Everything below is transport-independent: the same calls work whether
    the client is connected over stdio or HTTP. That is the point of MCP."""

    # --- Discovery ---------------------------------------------------------
    # The client does not need to know the server's tools in advance; it asks.
    # This is how an LLM app learns what it can do at runtime.
    tools = await client.list_tools()
    print("Tools:")
    for t in tools.tools:
        # inputSchema is the JSON schema the model would see for the arguments.
        print(f"  - {t.name}: {t.description}  args={list(t.input_schema['properties'])}")

    # --- Calling a tool ----------------------------------------------------
    # You pass the tool name and a dict of arguments. In an agent, the LLM
    # produces these (a "tool call") and your code forwards them here.
    result = await client.call_tool("add", {"a": 2, "b": 3})
    # The result is a list of content blocks (text, image...). A text block has
    # a `.text` field. There can also be structured_content for typed results.
    print("\nadd(2, 3) ->", result.content[0].text)

    result = await client.call_tool("get_weather", {"city": "Delhi"})
    print("get_weather('Delhi') ->", result.content[0].text)

    # --- Reading a resource ------------------------------------------------
    # Resources are fetched by URI, not called by name.
    resource = await client.read_resource("greeting://Pankaj")
    print("\nresource greeting://Pankaj ->", resource.contents[0].text)

    # --- Getting a prompt --------------------------------------------------
    # The server fills the template with the arguments we pass.
    prompt = await client.get_prompt("review_code", {"code": "print(1/0)"})
    print("\nprompt review_code ->", prompt.messages[0].content.text)


async def run_stdio() -> None:
    print("=== stdio transport ===")
    # With stdio the client OWNS the server: it spawns it as a subprocess and
    # speaks JSON-RPC over the child's stdin/stdout. When we leave the `async
    # with` block, the process is shut down. sys.executable makes sure the
    # child uses the same Python (and venv) as this script.
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_FILE), "--transport", "stdio"],
    )
    async with Client(params) as client:
        await demo(client)


async def run_http() -> None:
    print("=== streamable HTTP transport ===")
    # With HTTP the server is already running somewhere else; the client only
    # needs its URL. Passing a plain string URL selects streamable HTTP.
    # Many clients can connect to the same server at once.
    async with Client("http://127.0.0.1:8000/mcp") as client:
        await demo(client)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    asyncio.run(run_stdio() if mode == "stdio" else run_http())
