"""Real FastMCP server shared by the harness MCP transport tests."""

import argparse
import asyncio
import os
import threading
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from mcp.types import ImageContent

server = FastMCP("harness-tests", list_page_size=2, dereference_schemas=False)
started = threading.Event()
cancelled = threading.Event()
echo_calls: list[tuple[str, dict[str, Any]]] = []
IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAIAAAA2iEnWAAAAEUlEQVR4nGP8zwACTGASRgEAFEAB"
    "BVDVLjgAAAAASUVORK5CYII="
)

COMPLEX_SCHEMA = {
    "$defs": {
        "Item": {
            "type": "object",
            "properties": {"value": {"type": "integer"}, "label": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }
    },
    "type": "object",
    "properties": {
        "_id": {"type": "string"},
        "user-id": {"type": "integer"},
        "class": {"enum": ["card", "text"]},
        "payload": {"$ref": "#/$defs/Item"},
    },
    "required": ["_id", "user-id", "class", "payload"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"class": {"const": "card"}}},
            "then": {"properties": {"user-id": {"minimum": 1}}},
        }
    ],
}

DRAFT7_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "credit_card": {"type": "integer"},
        "billing_address": {"type": "string"},
    },
    "dependencies": {"credit_card": ["billing_address"]},
    "additionalProperties": False,
}


class EchoTool(Tool):
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        echo_calls.append((self.name, arguments))
        return ToolResult(content=[], structured_content=arguments)


server.add_tool(
    EchoTool(
        name="schema.echo",
        description="Echo complex JSON arguments.",
        parameters=COMPLEX_SCHEMA,
    )
)
server.add_tool(
    EchoTool(
        name="draft7", description="Echo Draft 7 arguments.", parameters=DRAFT7_SCHEMA
    )
)


@server.tool(name="math.add")
def add(a: int, b: int) -> int:
    """Add two numbers remotely."""
    return a + b


@server.tool(name="math_add")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers remotely."""
    return a * b


@server.tool
def describe(label: str, prefix: str = "default") -> dict[str, str]:
    """Describe a label with an optional prefix."""
    return {"label": label, "prefix": prefix}


@server.tool
def fail(message: str) -> str:
    """Fail with the supplied message."""
    raise ValueError(message)


@server.tool
def structured_fail() -> ToolResult:
    """Fail with structured content only."""
    return ToolResult(
        content=[],
        structured_content={"error": "quota_exceeded", "retryAfterSeconds": 60},
        is_error=True,
    )


class BadOutput(Tool):
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[], structured_content={"count": "invalid"})


server.add_tool(
    BadOutput(
        name="bad_output",
        description="Return a malformed structured result.",
        parameters={"type": "object", "additionalProperties": False},
        output_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    )
)


@server.tool(
    output_schema={
        "type": "object",
        "properties": {"width": {"type": "integer"}},
        "required": ["width"],
    }
)
def image_result() -> ToolResult:
    """Return an image and its structured result."""
    return ToolResult(
        content=[ImageContent(type="image", data=IMAGE_BASE64, mime_type="image/png")],
        structured_content={"width": 2},
    )


@server.tool
def identify() -> dict[str, int]:
    """Identify the server process."""
    return {"pid": os.getpid()}


@server.tool
async def slow() -> str:
    """Wait until cancelled."""
    started.set()
    try:
        await asyncio.sleep(30)
    except asyncio.CancelledError:
        cancelled.set()
        raise
    return "finished"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.port is None:
        server.run(transport="stdio", show_banner=False, log_level="ERROR")
    else:
        server.run(
            transport="http",
            host="127.0.0.1",
            port=args.port,
            show_banner=False,
            log_level="ERROR",
        )
