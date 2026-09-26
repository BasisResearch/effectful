"""Expose MCP tools and their structured and multimodal results to the harness.

The interpretation connects the FastMCP client around skill calls, using the
application's event loop from the synchronous harness's worker threads.
"""

import asyncio
import concurrent.futures
import dataclasses
import functools
import inspect
import json
import threading
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence, Set
from typing import Any, Self, cast, get_args, get_origin

import fastmcp
import fastmcp.exceptions
import mcp.types
import pydantic
from fastmcp.client.sampling.handlers.openai import (
    _audio_content_to_openai_part,
    _image_content_to_openai_part,
)
from fastmcp.utilities.json_schema_type import json_schema_to_type
from litellm.types.llms.openai import (
    ChatCompletionAudioObject,
    ChatCompletionFileObject,
)
from PIL import Image

from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    Message,
    call_agent,
    call_assistant,
)
from effectful.handlers.llm.types import Encodable, Skill, Tool
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation

type Wait = Callable[[concurrent.futures.Future[Any]], Any]
"""Block until a future scheduled on the client's loop finishes, and return its result."""


@functools.cache
@functools.cache
def background_loop() -> asyncio.AbstractEventLoop:
    """A process-wide event loop on a daemon thread, for callers that have none."""
    loop = asyncio.new_event_loop()
    running = threading.Event()
    loop.call_soon(running.set)
    threading.Thread(
        target=loop.run_forever, name="mcp-background-loop", daemon=True
    ).start()
    running.wait()
    return loop


def _call_on_loop[**P, T](
    loop: asyncio.AbstractEventLoop,
    wait: Wait,
    function: Callable[P, Coroutine[Any, Any, T]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    if not loop.is_running():
        raise RuntimeError("The MCP client's event loop must be running")
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        raise RuntimeError(
            "Run skills and MCP tools from a worker thread using asyncio.to_thread"
        )
    future = asyncio.run_coroutine_threadsafe(function(*args, **kwargs), loop)
    try:
        return wait(future)
    except BaseException:
        future.cancel()
        raise


class _MCPResult[T](pydantic.BaseModel):
    """Keep the field codecs and structured type available during serialization."""

    content: Encodable[
        Sequence[
            str | Image.Image | ChatCompletionAudioObject | ChatCompletionFileObject
        ]
    ]
    structuredContent: T


def _content_blocks(content: mcp.types.ContentBlock) -> Iterator[str | dict[str, Any]]:
    try:
        if isinstance(content, mcp.types.TextContent):
            yield content.text
        elif isinstance(content, mcp.types.ImageContent):
            yield dict(_image_content_to_openai_part(content))
        elif isinstance(content, mcp.types.AudioContent):
            yield dict(_audio_content_to_openai_part(content))
        elif isinstance(content, mcp.types.EmbeddedResource) and isinstance(
            content.resource, mcp.types.BlobResourceContents
        ):
            # Keep the resource's URI and metadata alongside its rendered content.
            metadata = content.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                exclude={"resource": {"blob"}},
            )
            yield json.dumps(metadata)
            resource = content.resource
            mime = (
                (resource.mime_type or "application/octet-stream")
                .partition(";")[0]
                .strip()
                .lower()
            )
            if mime.startswith("image/"):
                yield dict(
                    _image_content_to_openai_part(
                        mcp.types.ImageContent(data=resource.blob, mime_type=mime)
                    )
                )
            elif mime.startswith("audio/"):
                yield dict(
                    _audio_content_to_openai_part(
                        mcp.types.AudioContent(data=resource.blob, mime_type=mime)
                    )
                )
            elif mime == "application/pdf":
                yield {
                    "type": "file",
                    "file": {
                        "filename": "resource.pdf",
                        "file_data": f"data:{mime};base64,{resource.blob}",
                    },
                }
            else:
                yield f"Unsupported MCP content type {mime!r}; binary payload omitted."
        else:
            # Text resources and resource links remain readable, including their URI.
            yield content.model_dump_json(by_alias=True, exclude_none=True)
    except ValueError as error:
        yield f"{error} Binary payload omitted."


class _MCPTool(Tool[..., _MCPResult[Any]]):
    """A remote tool with named parameters derived from its MCP schema."""

    @classmethod
    def define(  # type: ignore[override]
        cls,
        schema: mcp.types.Tool,
        client: fastmcp.Client,
        loop: asyncio.AbstractEventLoop | None = None,
        wait: Wait = concurrent.futures.Future.result,
    ) -> Self:
        """Wrap a discovered tool, using the current event loop unless supplied."""
        event_loop = loop if loop is not None else asyncio.get_running_loop()
        parameter_type = json_schema_to_type(schema.input_schema)
        # Schemas without named properties convert to dict[str, T].
        signature = (
            inspect.Signature(
                [
                    inspect.Parameter(
                        "_extra",
                        inspect.Parameter.VAR_KEYWORD,
                        annotation=get_args(parameter_type)[1],
                    )
                ]
            )
            if get_origin(parameter_type) is dict
            else inspect.signature(parameter_type)
        )
        output_type = (
            json_schema_to_type(schema.output_schema)
            if schema.output_schema is not None
            else Any
        )
        result_type = _MCPResult[output_type]  # type: ignore[valid-type]
        result_adapter = pydantic.TypeAdapter(result_type)
        signature = signature.replace(return_annotation=result_type)
        aliases = (
            {
                field.name: field.metadata.get("alias", field.name)
                for field in dataclasses.fields(parameter_type)
            }
            if dataclasses.is_dataclass(parameter_type)
            else {}
        )
        encoders = {
            parameter.name: pydantic.TypeAdapter(
                dict[str, parameter.annotation]  # type: ignore[name-defined]
                if parameter.kind == inspect.Parameter.VAR_KEYWORD
                else parameter.annotation
            )
            for parameter in signature.parameters.values()
        }

        def invoke(**arguments):
            bound = signature.bind(**arguments)
            arguments = {}
            for name, value in bound.arguments.items():
                encoded = encoders[name].dump_python(value, mode="json", by_alias=True)
                if signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
                    arguments.update(encoded)
                else:
                    # FastMCP sanitizes dataclass fields into Python identifiers.
                    arguments[aliases.get(name, name)] = encoded
            result = _call_on_loop(
                event_loop, wait, client.call_tool_mcp, schema.name, arguments
            )
            if result.is_error:
                payload = result.model_dump(
                    mode="json",
                    by_alias=True,
                    include={"content", "structured_content"},
                    exclude_none=True,
                )
                raise fastmcp.exceptions.ToolError(
                    f"{schema.name}: {json.dumps(payload)}"
                )
            return result_adapter.validate_python(
                {
                    "content": [
                        block
                        for item in result.content
                        for block in _content_blocks(item)
                    ],
                    "structuredContent": result.structured_content,
                }
            )

        invoke.__name__ = invoke.__qualname__ = schema.name
        invoke.__doc__ = schema.description or schema.name
        invoke.__signature__ = signature  # type: ignore[attr-defined]
        return cast(Self, super().define(invoke))


@dataclasses.dataclass
class MCPTools(ObjectInterpretation):
    """Offer one harness ``Tool`` per tool discovered by a FastMCP client.

    Use one interpretation per FastMCP client. Construct it on the application's
    running event loop, or pass that loop explicitly. Configure authentication,
    timeouts, and callbacks on the client itself.

    Run synchronous skills in worker threads. The interpretation enters the
    client's async context for each outer skill call. Concurrent calls share
    the connection, and nested skills reuse the scope of the same interpretation.
    An enclosing client context held by the application remains open. Otherwise,
    the last call closes the client and its transport, including on failure. The
    application keeps its event loop running until the skill calls finish.
    Between groups of calls, ``loop`` can be updated to another running
    application loop.

    For an existing ``Skill`` named ``answer``::

        import asyncio

        from fastmcp import Client
        from effectful.handlers.llm.harness import harness
        from effectful.handlers.llm.harness.legibility.mcp import MCPTools
        from effectful.ops.semantics import handler

        async def main():
            remote = MCPTools(Client("http://localhost:8000/mcp"))
            with handler(harness()), handler(remote):
                return await asyncio.to_thread(answer)

        asyncio.run(main())

    Multiple calls through the same interpretation can run concurrently with
    ``asyncio.gather(asyncio.to_thread(first), asyncio.to_thread(second))``.

    Each assistant call fetches the current server catalog, including additions,
    removals, and schema changes.

    Tools use the constructor signature of FastMCP's generated parameter type,
    including its Python parameter names, defaults, and ``**kwargs``. The shared
    harness serializer advertises that signature; constraints not represented
    by the generated types are not advertised. Argument conversion follows
    FastMCP and Pydantic, including nested defaults, dataclasses, sets, and datetime
    values. Values and field aliases are serialized back to JSON before calling
    the server. Schema support and validation errors follow the dependencies
    and the server.

    Calls return a model with ``content`` and ``structuredContent`` attributes. When
    an output schema is supplied, it defines the structured field's advertised
    type and its conversion to Python values, using the same FastMCP converter.
    Otherwise the structured value is returned as received.

    Content is a sequence of strings, PIL images, and the harness's typed audio
    and file blocks. FastMCP's OpenAI sampling helpers adapt image and audio
    blocks; the shared ``Encodable`` codecs decode them to Python values and
    encode them for the model, including PNG encoding for images. Resource
    text, links, and metadata become strings. Other binary formats produce a
    text notice for that item, preserving the rest of the result.
    Tool errors use the harness's normal error feedback.

    Synchronous applications with no event loop can pass ``background_loop()``.
    ``wait`` blocks on catalog requests and tool calls, and may raise to abandon
    one, for example on cancellation; connecting and disconnecting always run to
    completion.
    """

    client: fastmcp.Client
    loop: asyncio.AbstractEventLoop = dataclasses.field(
        default_factory=asyncio.get_running_loop
    )
    wait: Wait = concurrent.futures.Future.result
    _client_lock: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = (
        dataclasses.field(default=None, init=False, repr=False, compare=False)
    )

    @Operation.define
    def _is_client_active(self) -> bool:
        """Whether this interpretation activated the client in the current scope."""
        return False

    @implements(call_agent)
    def call_agent[**P, T](
        self, skill: Skill[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        if self._is_client_active():
            return fwd(skill, *args, **kwargs)

        client, loop = self.client, self.loop
        lock = None
        entered = False

        async def enter():
            nonlocal lock, entered
            if self._client_lock is None or self._client_lock[0] is not loop:
                self._client_lock = (loop, asyncio.Lock())
            lock = self._client_lock[1]
            async with lock:
                was_connected = client.is_connected()
                try:
                    await client.__aenter__()
                except BaseException:
                    if not was_connected:
                        await client.close()
                    raise
                entered = True

        async def exit():
            if lock is None:
                return
            async with lock:
                if entered:
                    try:
                        await client.__aexit__(None, None, None)
                    finally:
                        # FastMCP disconnects when its last context exits. Close
                        # kept-alive transports before another call can enter.
                        if not client.is_connected():
                            await client.close()

        try:
            _call_on_loop(loop, concurrent.futures.Future.result, enter)
            with handler({self._is_client_active: lambda: True}):
                return fwd(skill, *args, **kwargs)
        finally:
            _call_on_loop(loop, concurrent.futures.Future.result, exit)

    @implements(call_assistant)
    def call_assistant(
        self,
        messages: Sequence[Message],
        response_type: type,
        env: Mapping[str, Any],
        tools: Set[Tool] = frozenset(),
    ) -> AssistantResult:
        schemas = _call_on_loop(
            self.loop, self.wait, self.client.list_tools, cache_mode="refresh"
        )
        mcp_tools = frozenset(
            _MCPTool.define(schema, self.client, self.loop, self.wait)
            for schema in schemas
        )
        return fwd(messages, response_type, env, tools | mcp_tools)
