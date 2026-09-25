"""Exercise the actual harness codecs and real MCP transports without an LLM API."""

import asyncio
import base64
import concurrent.futures
import dataclasses
import datetime
import inspect
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import litellm
import mcp.types
import pydantic
import pytest
import tenacity
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from fastmcp.server.providers import LocalProvider
from fastmcp.tools import ToolResult
from PIL import Image

from effectful.handlers.llm.harness import harness
from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
from effectful.handlers.llm.harness.hooks import (
    ToolCallExecutionError,
    call_agent,
    call_assistant,
    call_tool,
    completion,
)
from effectful.handlers.llm.harness.legibility.mcp import (
    MCPTools,
    _MCPTool,
    background_loop,
)
from effectful.handlers.llm.harness.serialization import (
    _NAME2TOOL_KEY,
    DecodedToolCall,
    _advertised_names,
    _NameAndTool,
)
from effectful.handlers.llm.types import Encodable, Skill, Tool
from effectful.ops.semantics import fwd, handler
from tests.fixtures.mcp_server import (
    EchoTool,
    cancelled,
    echo_calls,
    server,
    started,
)

SERVER = Path(__file__).parent / "fixtures" / "mcp_server.py"
pytestmark = pytest.mark.timeout(30)
VALID_ARGUMENTS = {
    "_id": "card-1",
    "user_id": 7,
    "class_": "card",
    "payload": {"value": 42},
}


@Tool.define
def local_math(a: int, b: int) -> int:
    """Subtract two numbers locally."""
    return a - b


local_math.__name__ = "math_add"


@Skill.define
def answer() -> str:
    """Use the available tools, then summarize."""


def decoded(tool, arguments):
    return pydantic.TypeAdapter(Encodable[DecodedToolCall]).validate_python(
        {
            "type": "function",
            "id": "call_probe",
            "function": {"name": tool.__name__, "arguments": json.dumps(arguments)},
        },
        context={_NAME2TOOL_KEY: {tool.__name__: tool}},
    )


def offered_tools(mcp):
    offered = set()

    def capture(messages, response_type, env, tools=frozenset()):
        offered.update(tools)
        return {"role": "assistant", "content": "ok"}, [], "ok"

    with handler({call_assistant: capture}), handler(mcp):
        call_assistant([], str, {})
    return frozenset(offered)


def find_tool(mcp, name):
    return next(tool for tool in offered_tools(mcp) if tool.__name__ == name)


def call_with_tools(mcp: MCPTools, check):
    with handler({call_agent: lambda skill: check(mcp)}), handler(mcp):
        return answer()


def run_with_client(client: Client, check):
    async def main():
        result = await asyncio.to_thread(call_with_tools, MCPTools(client), check)
        assert not client.is_connected()
        return result

    return asyncio.run(main())


def result_server(content, structured_content, output_schema=None):
    result = FastMCP("prepared-result", dereference_schemas=False)

    @result.tool(output_schema=output_schema)
    def prepared() -> ToolResult:
        """Return a prepared result."""
        return ToolResult(content=content, structured_content=structured_content)

    return result


@pytest.fixture(scope="module")
def http_url(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("mcp-http") / "server.log"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(SERVER), "--port", str(port)],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                if process.poll() is not None:
                    raise AssertionError(log_path.read_text())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise AssertionError("HTTP server did not start")
                    time.sleep(0.02)
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize("transport", ["memory", "stdio", "http"])
@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_harness_roundtrip(transport, mode, request, tmp_path):
    if transport == "memory":
        source = server
    elif transport == "stdio":
        source = StdioTransport(
            sys.executable,
            [str(SERVER)],
            log_file=tmp_path / "stdio.log",
        )
    else:
        source = request.getfixturevalue("http_url")
    client = Client(source, mode=mode, timeout=5)
    rounds = []

    def fake_completion(**kwargs):
        rounds.append(kwargs)
        functions = [entry["function"] for entry in kwargs["tools"]]
        lookup = {}
        for label, description in {
            "add": "Add two numbers remotely.",
            "multiply": "Multiply two numbers remotely.",
            "subtract": "Subtract two numbers locally.",
            "echo": "Echo complex JSON arguments.",
            "describe": "Describe a label with an optional prefix.",
            "draft7": "Echo Draft 7 arguments.",
            "image": "Return an image and its structured result.",
        }.items():
            lookup[label] = next(
                fn for fn in functions if description in fn["description"]
            )
        assert (
            len({lookup[key]["name"] for key in ("add", "multiply", "subtract")}) == 3
        )
        echo_schema = lookup["echo"]["parameters"]
        assert set(echo_schema["properties"]) == set(VALID_ARGUMENTS)
        assert echo_schema["properties"]["_id"]["type"] == "string"
        assert echo_schema["properties"]["user_id"]["type"] == "integer"
        assert set(echo_schema["required"]) == set(VALID_ARGUMENTS)
        assert not lookup["draft7"]["parameters"].get("required")
        assert lookup["echo"]["strict"] is False
        assert lookup["subtract"]["strict"] is True
        assert "prefix" not in lookup["describe"]["parameters"]["required"]
        if len(rounds) == 1:
            arguments = {
                "add": {"a": 19, "b": 23},
                "multiply": {"a": 3, "b": 4},
                "subtract": {"a": 9, "b": 7},
                "echo": VALID_ARGUMENTS,
                "describe": {"label": "probe"},
                "draft7": {},
                "image": {},
            }
            calls = [
                {
                    "id": f"call_{key}",
                    "type": "function",
                    "function": {
                        "name": lookup[key]["name"],
                        "arguments": json.dumps(value),
                    },
                }
                for key, value in arguments.items()
            ]
            message = {"role": "assistant", "content": None, "tool_calls": calls}
            reason = "tool_calls"
        else:
            results = {
                message["tool_call_id"]: message
                for message in kwargs["messages"]
                if message["role"] == "tool"
            }
            assert set(results) == {f"call_{key}" for key in lookup}

            def result_value(label):
                return json.loads(
                    "".join(
                        block["text"] for block in results[f"call_{label}"]["content"]
                    )
                )

            assert result_value("add")["structuredContent"] == {"result": 42}
            assert result_value("multiply")["structuredContent"] == {"result": 12}
            assert result_value("subtract") == 2
            echo_arguments = {
                "_id": "card-1",
                "user-id": 7,
                "class": "card",
                "payload": {"value": 42, "label": None},
            }
            assert result_value("echo")["structuredContent"] == echo_arguments
            assert result_value("describe")["structuredContent"] == {
                "label": "probe",
                "prefix": "default",
            }
            image_blocks = results["call_image"]["content"]
            image = next(
                block for block in image_blocks if block["type"] == "image_url"
            )
            decoded_image = pydantic.TypeAdapter(
                Encodable[Image.Image]
            ).validate_python(image)
            assert decoded_image.size == (2, 3)
            assert decoded_image.getpixel((0, 0)) == (255, 0, 0)
            image_text = "".join(
                block["text"] for block in image_blocks if block["type"] == "text"
            )
            assert '"structuredContent": {"width": 2}' in image_text
            # The assistant history retains the original model response.
            assistant = next(
                message for message in kwargs["messages"] if message.get("tool_calls")
            )
            echoed = next(
                call for call in assistant["tool_calls"] if call["id"] == "call_echo"
            )
            assert json.loads(echoed["function"]["arguments"]) == VALID_ARGUMENTS
            assert echoed["function"]["name"] == lookup["echo"]["name"]
            message = {"role": "assistant", "content": "All tools completed."}
            reason = "stop"
        return litellm.ModelResponse(
            choices=[{"message": message, "finish_reason": reason}]
        )

    async def main():
        mcp = MCPTools(client)
        pid = None

        def check(skill):
            nonlocal pid
            # Eleven tools and a page size of two exercise real pagination.
            assert len(offered_tools(mcp)) == 11
            assert client.protocol_version == (
                "2026-07-28" if mode == "auto" else "2025-11-25"
            )
            result = fwd(skill)
            assert client.is_connected()
            if transport == "stdio":
                pid = find_tool(mcp, "identify")().structuredContent["pid"]
            return result

        def add_local_tool(messages, response_type, env, tools=frozenset()):
            return fwd(messages, response_type, env, tools | {local_math})

        with (
            handler(harness(num_retries=0)),
            handler({call_agent: check, call_assistant: add_local_tool}),
            handler(mcp),
            handler({completion: fake_completion}),
        ):
            assert await asyncio.to_thread(answer) == "All tools completed."
        assert len(rounds) == 2
        assert not client.is_connected()
        return pid

    pid = asyncio.run(main())
    assert not client.is_connected()
    if transport == "stdio":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_named_typed_parameters():
    def check(remote):
        add = find_tool(remote, "math.add")
        parameters = inspect.signature(add).parameters
        for name in ("a", "b"):
            assert parameters[name].annotation is int
            assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is inspect.Parameter.empty
        assert add(a=19, b=23).structuredContent.result == 42
        with pytest.raises(pydantic.ValidationError):
            decoded(add, {"a": "not an integer", "b": 2})
        with pytest.raises(TypeError, match="missing a required argument"):
            add(a=1)

        describe = find_tool(remote, "describe")
        prefix = inspect.signature(describe).parameters["prefix"]
        assert prefix.annotation is str
        assert prefix.default == "default"
        call = decoded(describe, {"label": "probe"})
        assert "prefix" not in call.bound_args.arguments
        assert call_tool(call)[1].structuredContent["prefix"] == "default"

    run_with_client(Client(server), check)


def test_typed_extra_parameters():
    echo_server = FastMCP("typed-extras")
    echo_server.add_tool(
        EchoTool(
            name="echo",
            parameters={"type": "object", "additionalProperties": {"type": "integer"}},
        )
    )

    def check(remote):
        echo = find_tool(remote, "echo")
        spec = pydantic.TypeAdapter(Encodable[_NameAndTool]).dump_python(
            _NameAndTool("echo", echo)
        )["function"]
        assert spec["parameters"]["additionalProperties"] == {"type": "integer"}
        assert spec["strict"] is False
        arguments = {"user-id": 7, "class": 42, "_extra": 3}
        call = decoded(echo, arguments)
        assert call_tool(call)[1].structuredContent == arguments
        encoded = pydantic.TypeAdapter(Encodable[DecodedToolCall]).dump_python(call)
        assert json.loads(encoded["function"]["arguments"]) == arguments
        with pytest.raises(pydantic.ValidationError):
            decoded(echo, {"user-id": "invalid"})

    run_with_client(Client(echo_server), check)


@pytest.mark.parametrize("include_null", [False, True])
def test_generated_parameter_types_are_serialized_to_json(include_null):
    schema = {
        "type": "object",
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "label": {"type": ["string", "null"]},
                },
                "required": ["value"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
            "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "when": {"type": "string", "format": "date-time"},
            "arguments": {"type": "integer"},
            "note": {"type": ["string", "null"]},
            "user-id": {"type": "integer"},
            "_extra": {"type": "string"},
        },
        "required": ["items", "arguments"],
        "additionalProperties": True,
    }
    echo_server = FastMCP("typed-arguments", dereference_schemas=False)
    echo_server.add_tool(EchoTool(name="echo", parameters=schema))
    arguments = {
        "items": [{"value": 1}, {"value": 2, "label": None}],
        "tags": ["second", "first"],
        "when": "2026-09-25T09:30:00-04:00",
        "arguments": 3,
        "user-id": 7,
        "_extra": "a declared parameter",
        "_extra_": {"nested": [True, 1]},
    }
    if include_null:
        arguments["note"] = None

    def check(remote):
        echo = find_tool(remote, "echo")
        assert inspect.signature(echo).parameters["arguments"].annotation is int
        call = decoded(echo, arguments)
        converted = call.bound_args.arguments
        assert dataclasses.is_dataclass(converted["items"][0])
        assert converted["items"][0].label is None
        assert converted["tags"] == {"first", "second"}
        assert isinstance(converted["when"], datetime.datetime)
        message, result, final = call_tool(call)
        expected = {
            **arguments,
            "items": [{"value": 1, "label": None}, {"value": 2, "label": None}],
        }
        returned = result.structuredContent
        assert set(returned["tags"]) == {"first", "second"}
        assert {**returned, "tags": arguments["tags"]} == expected
        assert not final
        encoded = pydantic.TypeAdapter(Encodable[DecodedToolCall]).dump_python(call)
        assert json.loads(encoded["function"]["arguments"]) == returned

    run_with_client(Client(echo_server), check)


def test_parameters_named_like_operation_receivers():
    schema = {
        "type": "object",
        "properties": {name: {"type": "integer"} for name in ("self", "op", "app")},
        "required": ["self", "op", "app"],
    }
    echo_server = FastMCP("receiver-names")
    echo_server.add_tool(EchoTool(name="echo", parameters=schema))
    arguments = {"self": 1, "op": 2, "app": 3}

    def check(remote):
        message, result, final = call_tool(
            decoded(find_tool(remote, "echo"), arguments)
        )
        assert result.structuredContent == arguments

    run_with_client(Client(echo_server), check)


def test_generated_field_aliases_keep_wire_names():
    schema = {
        "type": "object",
        "properties": {
            "user-id": {"type": "integer"},
            "user_id": {"type": "string"},
            "class": {"type": "string"},
            "payload": {
                "type": "object",
                "properties": {"entry-id": {"type": "integer"}},
                "required": ["entry-id"],
            },
        },
        "required": ["user-id", "user_id", "class", "payload"],
    }
    echo_server = FastMCP("field-aliases")
    echo_server.add_tool(EchoTool(name="echo", parameters=schema))
    arguments = {
        "user_id": 7,
        "user_id_2": "seven",
        "class_": "card",
        "payload": {"entry-id": 42},
    }
    wire_arguments = {
        "user-id": 7,
        "user_id": "seven",
        "class": "card",
        "payload": {"entry-id": 42},
    }

    def check(remote):
        echo = find_tool(remote, "echo")
        assert inspect.signature(echo).parameters["user_id"].annotation is int
        assert inspect.signature(echo).parameters["user_id_2"].annotation is str
        spec = pydantic.TypeAdapter(Encodable[_NameAndTool]).dump_python(
            _NameAndTool("echo", echo)
        )["function"]
        assert set(spec["parameters"]["properties"]) == set(arguments)
        assert set(spec["parameters"]["required"]) == set(arguments)
        call = decoded(echo, arguments)
        assert call_tool(call)[1].structuredContent == wire_arguments
        encoded = pydantic.TypeAdapter(Encodable[DecodedToolCall]).dump_python(call)
        assert json.loads(encoded["function"]["arguments"]) == arguments

    run_with_client(Client(echo_server), check)


def test_argument_errors_from_shared_codec():
    def check(mcp):
        echo = find_tool(mcp, "schema.echo")
        before = len(echo_calls)
        with pytest.raises(pydantic.ValidationError):
            decoded(echo, dict(VALID_ARGUMENTS, payload={"value": "wrong"}))
        assert len(echo_calls) == before

        with pytest.raises(pydantic.ValidationError, match="Unexpected argument extra"):
            decoded(find_tool(mcp, "math.add"), {"a": 1, "b": 2, "extra": True})

    run_with_client(Client(server), check)


def test_error_payloads_and_output_validation():
    def check(mcp):
        for name, arguments, match in [
            ("fail", {"message": "expected failure"}, "expected failure"),
            ("structured_fail", {}, "quota_exceeded"),
        ]:
            with handler(TenacityRetryer(stop=tenacity.stop_after_attempt(1))):
                message, error, final = call_tool(
                    decoded(find_tool(mcp, name), arguments)
                )
            assert isinstance(error, ToolCallExecutionError)
            assert isinstance(error.original_error, ToolError)
            assert match in json.dumps(message)
            assert message["tool_call_id"] == "call_probe"
            assert final is False
            if name == "structured_fail":
                assert "retryAfterSeconds" in str(error) and "60" in str(error)
        with pytest.raises(ToolCallExecutionError) as caught:
            call_tool(decoded(find_tool(mcp, "bad_output"), {}))
        assert isinstance(caught.value.original_error, RuntimeError)
        assert "Invalid structured content" in str(caught.value)

    run_with_client(Client(server), check)


def test_output_schema_types_structured_content_and_preserves_aliases():
    schema = {
        "type": "object",
        "properties": {
            "created-at": {"type": "string", "format": "date-time"},
            "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "item": {
                "type": "object",
                "properties": {
                    "entry-id": {"type": "integer"},
                    "label": {"type": "string", "default": "untitled"},
                },
                "required": ["entry-id"],
            },
        },
        "required": ["created-at", "tags", "item"],
    }
    value = {
        "created-at": "2026-09-25T09:30:00-04:00",
        "tags": ["second", "first"],
        "item": {"entry-id": 42},
    }

    def check(remote):
        tool = find_tool(remote, "prepared")
        description = pydantic.TypeAdapter(Encodable[_NameAndTool]).dump_python(
            _NameAndTool("prepared", tool)
        )["function"]["description"]
        advertised = json.loads(
            description.split("Annotated JSON schema of return type: ")[1]
        )
        structured = advertised["properties"]["structuredContent"]
        properties = advertised["$defs"][structured["$ref"].rsplit("/", 1)[1]][
            "properties"
        ]
        assert properties["created-at"]["format"] == "date-time"
        assert properties["tags"]["uniqueItems"] is True
        assert '"entry-id"' in json.dumps(advertised)

        message, result, final = call_tool(decoded(tool, {}))
        structured = result.structuredContent
        assert dataclasses.is_dataclass(structured)
        assert structured.created_at == datetime.datetime.fromisoformat(
            value["created-at"]
        )
        assert structured.tags == {"first", "second"}
        assert structured.item.entry_id == 42
        assert structured.item.label == "untitled"
        assert not final
        encoded = json.loads("".join(block["text"] for block in message["content"]))
        assert encoded["content"] == []
        assert set(encoded["structuredContent"]["tags"]) == {"first", "second"}
        assert {**encoded["structuredContent"], "tags": value["tags"]} == {
            **value,
            "item": {"entry-id": 42, "label": "untitled"},
        }

    run_with_client(Client(result_server([], value, schema)), check)


@pytest.mark.parametrize(
    "schema,value",
    [
        ({"type": "object"}, {}),
        (None, {}),
        (None, None),
        (None, {"untyped": [0, False, "", [], None]}),
    ],
)
def test_empty_and_untyped_structured_results(schema, value):
    def check(remote):
        message, result, final = call_tool(decoded(find_tool(remote, "prepared"), {}))
        assert result.structuredContent == value
        assert type(result.structuredContent) is type(value)
        assert not final
        encoded = json.loads("".join(block["text"] for block in message["content"]))
        assert encoded["structuredContent"] == value

    run_with_client(Client(result_server([], value, schema)), check)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_multimodal_results_decode_and_reencode_harness_content(image_format):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), (255, 0, 0)).save(buffer, format=image_format)
    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    image_mime = f"image/{image_format.lower()}"
    original_image = Image.open(io.BytesIO(buffer.getvalue()))
    wav = "UklGRjQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YRAAAAAAAAAAAAAAAAAAAAAAAAAA"
    svg = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4="
    pdf = "JVBERi0xLjQKJSVFT0YK"
    binary = "AAECAwQ="
    content = [
        mcp.types.TextContent(type="text", text="These are the result's attachments."),
        mcp.types.ImageContent(type="image", data=image_data, mime_type=image_mime),
        mcp.types.AudioContent(type="audio", data=wav, mime_type="audio/wav"),
        mcp.types.EmbeddedResource(
            type="resource",
            resource=mcp.types.TextResourceContents(
                uri="resource://notes", mime_type="text/plain", text="A resource note."
            ),
        ),
        mcp.types.ResourceLink(
            type="resource_link", name="Report", uri="https://example.invalid/report"
        ),
    ]
    for name, data, mime in [
        ("image", image_data, image_mime),
        ("audio", wav, "audio/x-wav"),
        ("pdf", pdf, "application/pdf"),
        ("unknown", binary, "application/octet-stream"),
    ]:
        content.append(
            mcp.types.EmbeddedResource(
                type="resource",
                resource=mcp.types.BlobResourceContents(
                    uri=f"resource://{name}", blob=data, mime_type=mime
                ),
            )
        )
    content.append(
        mcp.types.AudioContent(type="audio", data=binary, mime_type="audio/ogg")
    )
    content.append(
        mcp.types.ImageContent(type="image", data=svg, mime_type="image/svg+xml")
    )
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }

    def check(remote):
        message, result, final = call_tool(decoded(find_tool(remote, "prepared"), {}))
        assert not final
        assert result.structuredContent.count == len(content)
        assert result.content[0] == "These are the result's attachments."
        assert all(
            isinstance(item, str | Image.Image)
            or item["type"] in {"input_audio", "file"}
            for item in result.content
        )
        decoded_images = [
            item for item in result.content if isinstance(item, Image.Image)
        ]
        assert len(decoded_images) == 2
        for image in decoded_images:
            assert image.size == original_image.size
            assert image.tobytes() == original_image.tobytes()
        blocks = message["content"]
        images = [block for block in blocks if block["type"] == "image_url"]
        assert len(images) == 2
        for image in images:
            assert image["image_url"]["url"].startswith("data:image/png;base64,")
            decoded_image = pydantic.TypeAdapter(
                Encodable[Image.Image]
            ).validate_python(image)
            assert decoded_image.size == original_image.size
            assert decoded_image.tobytes() == original_image.tobytes()
        audio = [
            block["input_audio"] for block in blocks if block["type"] == "input_audio"
        ]
        assert audio == [{"data": wav, "format": "wav"}] * 2
        assert [
            item["input_audio"]
            for item in result.content
            if isinstance(item, dict) and item["type"] == "input_audio"
        ] == audio
        file = next(block["file"] for block in blocks if block["type"] == "file")
        assert file["file_data"] == f"data:application/pdf;base64,{pdf}"
        assert (
            next(
                item["file"]
                for item in result.content
                if isinstance(item, dict) and item["type"] == "file"
            )
            == file
        )
        text = "".join(block["text"] for block in blocks if block["type"] == "text")
        for expected in (
            "These are the result's attachments.",
            "A resource note.",
            "resource://notes",
            "https://example.invalid/report",
            "resource://image",
            "resource://audio",
            "resource://pdf",
            "resource://unknown",
            "Unsupported audio MIME type for OpenAI: 'audio/ogg'",
            "Unsupported image MIME type for OpenAI: 'image/svg+xml'",
            "Unsupported MCP content type 'application/octet-stream'",
            f'"structuredContent": {{"count": {len(content)}}}',
        ):
            assert expected in text
        for payload in (image_data, wav, svg, pdf, binary):
            assert payload not in text

    run_with_client(
        Client(result_server(content, {"count": len(content)}, schema)), check
    )


@pytest.mark.parametrize("position", ["input", "output"])
def test_unknown_schema_dialect_uses_dependency_behavior(position):
    unknown = FastMCP("unknown-dialect")
    schema = {"$schema": "https://example.invalid/dialect", "type": "object"}
    unknown.add_tool(
        EchoTool(
            name="unknown",
            description="Unsupported dialect.",
            parameters=schema if position == "input" else {"type": "object"},
            output_schema=schema if position == "output" else None,
        )
    )

    def check(remote):
        echo = find_tool(remote, "unknown")
        if position == "output":
            with pytest.warns(DeprecationWarning, match="metaschema"):
                result = echo(value=42)
        else:
            result = echo(value=42)
        assert result.structuredContent == {"value": 42}

    run_with_client(Client(unknown), check)


@pytest.mark.parametrize("raw", ["[]", "null", '"string"'])
def test_nonobject_arguments_rejected_by_codec(raw):
    def check(mcp):
        mapping = _advertised_names(offered_tools(mcp))
        name = next(
            name for name, tool in mapping.items() if tool.__name__ == "schema.echo"
        )
        adapter = pydantic.TypeAdapter(Encodable[DecodedToolCall])
        with pytest.raises(TypeError, match="JSON object"):
            adapter.validate_python(
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": name, "arguments": raw},
                },
                context={_NAME2TOOL_KEY: mapping},
            )

    run_with_client(Client(server), check)


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_catalog_changes_between_assistant_calls(mode):
    provider = LocalProvider(on_duplicate="replace")
    provider.add_tool(EchoTool(name="stable", parameters={"type": "object"}))
    changing = FastMCP("changing-tools", providers=[provider], cache_ttl=60)

    def check(remote):
        def snapshot():
            offered = {}

            def capture(messages, response_type, env, tools=frozenset()):
                offered.update({tool.__name__: tool for tool in tools})
                return {"role": "assistant", "content": "ok"}, [], "ok"

            with handler({call_assistant: capture}), handler(remote):
                call_assistant([], str, {}, {local_math})
            return offered

        first = snapshot()
        assert set(first) == {"stable", "math_add"}
        assert first["math_add"] is local_math

        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        provider.add_tool(EchoTool(name="echo", parameters=schema))
        second = snapshot()
        assert set(second) == {"stable", "math_add", "echo"}
        assert second["echo"](value=7).structuredContent == {"value": 7}

        updated_schema = {**schema, "properties": {"value": {"type": "string"}}}
        provider.add_tool(EchoTool(name="echo", parameters=updated_schema))
        third = snapshot()
        assert inspect.signature(third["echo"]).parameters["value"].annotation is str
        with pytest.raises(pydantic.ValidationError):
            decoded(third["echo"], {"value": 7})
        assert third["echo"](value="new").structuredContent == {"value": "new"}

        provider.remove_tool("echo")
        assert set(snapshot()) == set(first)
        provider.remove_tool("stable")
        assert snapshot() == {"math_add": local_math}

    run_with_client(Client(changing, mode=mode, cache=True), check)


@pytest.mark.parametrize("explicit_loop", [False, True])
def test_discovery_requires_a_worker_and_an_open_connection(explicit_loop):
    async def main():
        client = Client(server)
        remote = (
            await asyncio.to_thread(MCPTools, client, asyncio.get_running_loop())
            if explicit_loop
            else MCPTools(client)
        )

        def check(mcp):
            identify = find_tool(mcp, "identify")
            assert identify().structuredContent["pid"] == os.getpid()
            return identify

        with pytest.raises(RuntimeError, match="worker thread"):
            offered_tools(remote)
        with pytest.raises(RuntimeError, match="not connected"):
            await asyncio.to_thread(offered_tools, remote)
        identify = await asyncio.to_thread(call_with_tools, remote, check)
        assert not client.is_connected()
        with pytest.raises(RuntimeError, match="not connected"):
            await asyncio.to_thread(identify)
        return remote, identify

    remote, identify = asyncio.run(main())
    with pytest.raises(RuntimeError, match="event loop must be running"):
        offered_tools(remote)
    with pytest.raises(RuntimeError, match="event loop must be running"):
        identify()


def test_interpretation_can_follow_a_reconnected_client_to_another_loop():
    client = Client(server)
    remote = None

    async def main():
        nonlocal remote
        if remote is None:
            remote = MCPTools(client)
        else:
            remote.loop = asyncio.get_running_loop()

        def check(mcp):
            return find_tool(mcp, "identify")().structuredContent["pid"]

        for _ in range(2):
            results = await asyncio.gather(
                asyncio.to_thread(call_with_tools, remote, check),
                asyncio.to_thread(call_with_tools, remote, check),
            )
            assert results == [os.getpid(), os.getpid()]
            assert not client.is_connected()

    asyncio.run(main())
    asyncio.run(main())


@pytest.mark.parametrize("fail", [False, True])
def test_skill_owns_stdio_lifetime(tmp_path, fail):
    client = Client(
        StdioTransport(sys.executable, [str(SERVER)], log_file=tmp_path / "stdio.log")
    )

    pids = []

    def check(mcp):
        pids.append(find_tool(mcp, "identify")().structuredContent["pid"])
        if fail:
            raise ValueError("caller error")

    async def main():
        remote = MCPTools(client)
        for _ in range(2):
            if fail:
                with pytest.raises(ValueError, match="caller error"):
                    await asyncio.to_thread(call_with_tools, remote, check)
            else:
                await asyncio.to_thread(call_with_tools, remote, check)
            assert not client.is_connected()
            with pytest.raises(ProcessLookupError):
                os.kill(pids[-1], 0)

    asyncio.run(main())


@pytest.mark.parametrize("mode", ["auto", "legacy"])
@pytest.mark.parametrize("transport", ["memory", "stdio"])
def test_concurrent_skills_share_client_and_failure_preserves_other_call(
    mode, transport, tmp_path
):
    source = (
        server
        if transport == "memory"
        else StdioTransport(
            sys.executable, [str(SERVER)], log_file=tmp_path / "stdio.log"
        )
    )
    client = Client(source, mode=mode, timeout=5)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    sessions = []
    pids = []

    def check(remote, index):
        assert remote._is_client_active()
        identify = find_tool(remote, "identify")
        sessions.append(client.session)
        pids.append(identify().structuredContent["pid"])
        entered[index].set()
        assert release[index].wait(5)
        if index == 0:
            raise ValueError("first call failed")
        return identify().structuredContent["pid"]

    async def main():
        remote = MCPTools(client)
        first = asyncio.create_task(
            asyncio.to_thread(call_with_tools, remote, lambda mcp: check(mcp, 0))
        )
        second = asyncio.create_task(
            asyncio.to_thread(call_with_tools, remote, lambda mcp: check(mcp, 1))
        )
        try:
            for event in entered:
                assert await asyncio.to_thread(event.wait, 5)
            # Activation belongs to each worker's handler scope, not the host.
            assert not remote._is_client_active()
            assert sessions[0] is sessions[1]
            assert pids[0] == pids[1]
            release[0].set()
            with pytest.raises(ValueError, match="first call failed"):
                await first
            assert client.is_connected()
            release[1].set()
            assert await second == pids[0]
            assert not client.is_connected()
        finally:
            for event in release:
                event.set()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(main())
    if transport == "stdio":
        with pytest.raises(ProcessLookupError):
            os.kill(pids[0], 0)


def test_nested_skills_reuse_enclosing_connection():
    @Skill.define
    def inner() -> str:
        """Perform a nested call."""

    async def main():
        client = Client(server)
        remote = MCPTools(client)

        def execute(skill):
            assert remote._is_client_active()
            session = client.session
            identify = find_tool(remote, "identify")
            assert identify().structuredContent["pid"] == os.getpid()
            if skill is inner:
                raise ValueError("nested failure")
            with pytest.raises(ValueError, match="nested failure"):
                inner()
            assert client.session is session
            assert identify().structuredContent["pid"] == os.getpid()
            return "done"

        with handler({call_agent: execute}), handler(remote):
            assert not remote._is_client_active()
            assert await asyncio.to_thread(answer) == "done"
            assert not remote._is_client_active()
        assert not client.is_connected()

    asyncio.run(main())


def test_activation_is_scoped_to_each_interpretation_instance():
    @Skill.define
    def inner() -> str:
        """Use a second MCP interpretation in a nested call."""

    async def main():
        first = MCPTools(Client(server))
        second = MCPTools(Client(server))

        def execute(skill):
            assert first._is_client_active()
            assert first.client.is_connected()
            if skill is inner:
                assert second._is_client_active()
                assert second.client.is_connected()
                raise ValueError("second client failed")
            assert not second._is_client_active()
            assert not second.client.is_connected()
            with (
                handler(second),
                pytest.raises(ValueError, match="second client failed"),
            ):
                inner()
            assert not second._is_client_active()
            assert not second.client.is_connected()
            assert first._is_client_active()
            assert (
                find_tool(first, "identify")().structuredContent["pid"] == os.getpid()
            )
            return "done"

        with handler({call_agent: execute}), handler(first):
            assert await asyncio.to_thread(answer) == "done"
        for remote in (first, second):
            assert not remote._is_client_active()
            assert not remote.client.is_connected()

    asyncio.run(main())


def test_existing_client_context_is_preserved():
    async def main():
        client = Client(server)
        remote = MCPTools(client)
        async with client:

            def check(mcp):
                assert (
                    find_tool(mcp, "identify")().structuredContent["pid"] == os.getpid()
                )

            await asyncio.to_thread(call_with_tools, remote, check)
            assert client.is_connected()
            assert await client.list_tools()
        await client.close()

    asyncio.run(main())


def test_failed_context_entry_closes_client_and_allows_reconnection():
    class FailOnceClient(Client):
        failed = False

        async def __aenter__(self):
            await super().__aenter__()
            if not self.failed:
                self.failed = True
                raise ValueError("connection setup failed")
            return self

    async def main():
        client = FailOnceClient(server)
        remote = MCPTools(client)
        with pytest.raises(ValueError, match="connection setup failed"):
            await asyncio.to_thread(
                call_with_tools, remote, lambda mcp: offered_tools(mcp)
            )
        assert not client.is_connected()
        tools = await asyncio.to_thread(
            call_with_tools, remote, lambda mcp: offered_tools(mcp)
        )
        assert len(tools) == 11
        assert not client.is_connected()

    asyncio.run(main())


def test_failed_entry_does_not_exit_another_calls_context():
    class FailNextClient(Client):
        fail_next = False

        async def __aenter__(self):
            if self.fail_next:
                self.fail_next = False
                raise ValueError("failed before entering")
            return await super().__aenter__()

    ready = threading.Event()
    release = threading.Event()

    def survivor(remote):
        identify = find_tool(remote, "identify")
        ready.set()
        assert release.wait(5)
        return identify().structuredContent["pid"]

    async def main():
        client = FailNextClient(server)
        remote = MCPTools(client)
        running = asyncio.create_task(
            asyncio.to_thread(call_with_tools, remote, survivor)
        )
        try:
            assert await asyncio.to_thread(ready.wait, 5)
            client.fail_next = True
            with pytest.raises(ValueError, match="failed before entering"):
                await asyncio.to_thread(call_with_tools, remote, offered_tools)
            assert client.is_connected()
            release.set()
            assert await running == os.getpid()
            assert not client.is_connected()
        finally:
            release.set()
            await running

    asyncio.run(main())


def test_skill_call_on_application_loop_fails_without_deadlocking():
    async def main():
        client = Client(server)
        remote = MCPTools(client)
        with pytest.raises(RuntimeError, match="worker thread"):
            call_with_tools(remote, lambda mcp: None)
        await asyncio.to_thread(call_with_tools, remote, lambda mcp: offered_tools(mcp))
        assert not client.is_connected()

    asyncio.run(main())


@pytest.mark.parametrize("explicit_loop", [False, True])
def test_tool_call_on_client_loop_fails_without_deadlocking(explicit_loop):
    async def main():
        async with Client(server) as client:
            schemas = await client.list_tools()
            schema = next(schema for schema in schemas if schema.name == "identify")
            if explicit_loop:
                # An explicit loop also allows construction from a worker.
                identify = await asyncio.to_thread(
                    _MCPTool.define, schema, client, asyncio.get_running_loop()
                )
            else:
                identify = _MCPTool.define(schema, client)
            with pytest.raises(RuntimeError, match="worker thread"):
                identify()
            result = await asyncio.to_thread(identify)
            assert result.structuredContent["pid"] == os.getpid()

    asyncio.run(main())


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_interrupt_cancels_inflight_tool(mode):
    started.clear()
    cancelled.clear()
    ready: concurrent.futures.Future = concurrent.futures.Future()
    survivor_entered = threading.Event()
    release_survivor = threading.Event()

    async def host():
        client = Client(server, mode=mode, timeout=5)
        mcp = MCPTools(client)
        stop = asyncio.Event()
        ready.set_result((mcp, asyncio.get_running_loop(), stop))
        await stop.wait()
        assert not client.is_connected()

    def survivor(remote):
        identify = find_tool(remote, "identify")
        survivor_entered.set()
        assert release_survivor.wait(10)
        return identify().structuredContent["pid"]

    # Keep the harness call on the main thread so SIGINT interrupts its wait.
    # The host supplies only an event loop; MCPTools manages the connection.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        serving = executor.submit(asyncio.run, host())
        mcp, loop, stop = ready.result(timeout=5)
        surviving = executor.submit(call_with_tools, mcp, survivor)
        try:
            assert survivor_entered.wait(5)

            def interrupt_when_started():
                if started.wait(5):
                    os.kill(os.getpid(), signal.SIGINT)

            interrupter = threading.Thread(target=interrupt_when_started, daemon=True)
            interrupter.start()
            try:
                with pytest.raises(KeyboardInterrupt):
                    call_with_tools(mcp, lambda remote: find_tool(remote, "slow")())
            finally:
                interrupter.join(timeout=5)
            assert not interrupter.is_alive()
            assert cancelled.wait(2), (
                "Server continued running after local cancellation"
            )
            assert mcp.client.is_connected()
            release_survivor.set()
            assert surviving.result(timeout=5) == os.getpid()
            assert not mcp.client.is_connected()
            # A later skill can reconnect using the same interpretation.
            assert (
                call_with_tools(
                    mcp, lambda remote: find_tool(remote, "identify")()
                ).structuredContent["pid"]
                == os.getpid()
            )
        finally:
            release_survivor.set()
            surviving.result(timeout=5)
            loop.call_soon_threadsafe(stop.set)
            serving.result(timeout=5)
    assert cancelled.wait(2)


def test_tool_timeout_is_reported_and_client_reusable():
    def check(mcp):
        start = time.monotonic()
        with pytest.raises(ToolCallExecutionError):
            call_tool(decoded(find_tool(mcp, "slow"), {}))
        assert time.monotonic() - start < 3
        assert find_tool(mcp, "identify")().structuredContent["pid"] == os.getpid()

    run_with_client(Client(server, timeout=0.2), check)


def test_wait_can_abandon_an_inflight_call():
    started.clear()
    cancelled.clear()
    abandon = threading.Event()

    class Abandoned(Exception):
        pass

    def wait(future):
        while not abandon.is_set():
            try:
                return future.result(timeout=0.05)
            except concurrent.futures.TimeoutError:
                pass
        raise Abandoned

    def abandon_when_started():
        if started.wait(5):
            abandon.set()

    mcp = MCPTools(Client(server, timeout=5), loop=background_loop(), wait=wait)
    abandoner = threading.Thread(target=abandon_when_started, daemon=True)
    abandoner.start()
    # A synchronous caller on the main thread, with no event loop of its own.
    with pytest.raises(Abandoned):
        call_with_tools(mcp, lambda remote: find_tool(remote, "slow")())
    abandoner.join(timeout=5)
    assert cancelled.wait(2)
    assert not mcp.client.is_connected()


@pytest.mark.parametrize("source", ["mapping", "path"])
def test_harness_mcp_config_offers_every_servers_tools(source, tmp_path):
    config = {
        "mcpServers": {
            "a": {"command": sys.executable, "args": [str(SERVER)]},
            "b": {"command": sys.executable, "args": [str(SERVER)]},
            "broken": {"command": str(tmp_path / "missing")},
        }
    }
    if source == "path":
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps(config))
        mcp_config: object = str(path)
    else:
        mcp_config = config
    rounds = []

    def fake_completion(**kwargs):
        rounds.append(kwargs)
        functions = {
            entry["function"]["name"]: entry["function"] for entry in kwargs["tools"]
        }
        assert {"a_describe", "b_describe"} <= set(functions)
        assert not any(name.startswith("broken_") for name in functions)
        # Offered as a JSON tool, not wrapped by the code-calling pathway.
        assert "label" in functions["a_describe"]["parameters"]["properties"]
        if len(rounds) == 1:
            call = {
                "id": "call_describe",
                "type": "function",
                "function": {
                    "name": "a_describe",
                    "arguments": json.dumps({"label": "probe"}),
                },
            }
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
            reason = "tool_calls"
        else:
            result = next(
                message
                for message in kwargs["messages"]
                if message.get("tool_call_id") == "call_describe"
            )
            text = "".join(block["text"] for block in result["content"])
            assert json.loads(text)["structuredContent"] == {
                "label": "probe",
                "prefix": "default",
            }
            message = {"role": "assistant", "content": "described"}
            reason = "stop"
        return litellm.ModelResponse(
            choices=[{"message": message, "finish_reason": reason}]
        )

    # No event loop and no worker thread: the harness supplies the loop.
    with (
        handler(harness(num_retries=0, mcp_config=mcp_config)),
        handler({completion: fake_completion}),
    ):
        assert answer() == "described"
    assert len(rounds) == 2
