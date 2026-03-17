import collections
import collections.abc
import inspect
import json
import textwrap
import typing
import uuid

import pydantic

from effectful.handlers.llm.completions import (
    Message,
    MessageResult,
    ResultDecodingError,
    ToolCallDecodingError,
    _get_history,
    _make_message,
    append_message,
    call_assistant,
    call_system,
    call_tool,
    call_user,
)
from effectful.handlers.llm.encoding import DecodedToolCall, Encodable
from effectful.handlers.llm.template import Template, Tool
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements

try:
    import anthropic
    from anthropic.types import Message as AnthropicMessage

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def _get_oauth_token_from_keychain() -> str | None:
    """Extract Claude Code OAuth token from macOS Keychain or Linux keyring."""
    import platform
    import subprocess

    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                capture_output=True,
                text=True,
                check=True,
            )
            creds = json.loads(result.stdout.strip())
            oauth = creds.get("claudeAiOauth", {})
            return oauth.get("accessToken")
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            return None
    elif system == "Linux":
        try:
            import secretstorage

            connection = secretstorage.dbus_init()
            collection = secretstorage.get_default_collection(connection)
            for item in collection.get_all_items():
                if item.get_label() == "Claude Code-credentials":
                    creds = json.loads(item.get_secret().decode())
                    oauth = creds.get("claudeAiOauth", {})
                    return oauth.get("accessToken")
        except Exception:
            return None
    return None


def _get_anthropic_client(
    api_key: str | None = None,
    base_url: str | None = None,
    max_subscription: bool = False,
    **kwargs,
) -> "anthropic.Anthropic":
    """Create an Anthropic client with support for API key or Claude Max subscription.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        base_url: Custom base URL for the API.
        max_subscription: If True, configure for Claude Max subscription usage.
            This uses OAuth-based authentication. Requires the user to have
            authenticated via ``claude auth login`` or have a valid OAuth token
            stored in the system keychain by Claude Code.
        **kwargs: Additional keyword arguments passed to the Anthropic client.
    """
    if not HAS_ANTHROPIC:
        raise ImportError(
            "The 'anthropic' package is required. Install it with: pip install anthropic"
        )

    client_kwargs: dict[str, typing.Any] = {**kwargs}

    if max_subscription:
        oauth_token = _get_oauth_token_from_keychain()
        if not oauth_token:
            raise RuntimeError(
                "Could not obtain Claude Max OAuth token from system keychain. "
                "Please run 'claude auth login' first or provide an api_key."
            )
        client_kwargs["api_key"] = oauth_token
    elif api_key:
        client_kwargs["api_key"] = api_key

    if base_url:
        client_kwargs["base_url"] = base_url

    return anthropic.Anthropic(**client_kwargs)


def _tools_to_anthropic_format(
    tools: collections.abc.Mapping[str, Tool],
) -> list[dict[str, typing.Any]]:
    """Convert effectful Tool specs to Anthropic API tool format.

    Uses the same encoding system as LiteLLM (ToolEncodable) to generate
    schemas, ensuring tool call decoding is compatible.
    """
    tool_specs = []
    for name, tool_obj in tools.items():
        # Use the existing ToolEncodable to get the OpenAI-format tool spec
        encoded = Encodable.define(type(tool_obj), tools).encode(tool_obj)
        spec = encoded.model_dump(exclude_none=True)
        # Convert from OpenAI format {"type": "function", "function": {...}}
        # to Anthropic format {"name": ..., "description": ..., "input_schema": ...}
        func = spec.get("function", {})
        tool_specs.append({
            "name": func.get("name", name),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        })
    return tool_specs


def _resolve_refs(schema: typing.Any, defs: dict[str, typing.Any]) -> typing.Any:
    """Recursively resolve $ref pointers in a JSON schema."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_path = schema["$ref"]
            # Handle #/$defs/Name format
            if ref_path.startswith("#/$defs/"):
                ref_name = ref_path[len("#/$defs/"):]
                if ref_name in defs:
                    return _resolve_refs(defs[ref_name], defs)
            return schema
        return {k: _resolve_refs(v, defs) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [_resolve_refs(item, defs) for item in schema]
    return schema


def _messages_to_anthropic_format(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, typing.Any]]]:
    """Convert internal messages to Anthropic API format.

    Returns (system_prompt, messages) where system_prompt is extracted
    from any system message.
    """
    system_prompt = None
    anthropic_messages: list[dict[str, typing.Any]] = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                system_prompt = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            continue

        if role == "assistant":
            blocks: list[dict[str, typing.Any]] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        # Convert to Anthropic image format
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                        if url.startswith("data:"):
                            media_type, _, data = url.partition(";base64,")
                            media_type = media_type.replace("data:", "")
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            })

            # Include tool_calls as tool_use blocks
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc if isinstance(tc, dict) else tc.model_dump() if hasattr(tc, "model_dump") else vars(tc)
                    func = fn.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": fn.get("id", str(uuid.uuid4())),
                        "name": func.get("name", ""),
                        "input": args,
                    })

            if blocks:
                anthropic_messages.append({"role": "assistant", "content": blocks})

        elif role == "user":
            if isinstance(content, str):
                anthropic_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                blocks = []
                for part in content:
                    if part.get("type") == "text":
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                        if url.startswith("data:"):
                            media_type, _, data = url.partition(";base64,")
                            media_type = media_type.replace("data:", "")
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            })
                anthropic_messages.append({"role": "user", "content": blocks})

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = content if isinstance(content, str) else json.dumps(content)
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_content,
                }],
            })

    return system_prompt, anthropic_messages


class AnthropicProvider(ObjectInterpretation):
    """Implements templates using the Anthropic API directly.

    Supports both standard API key authentication and Claude Max subscription
    (OAuth-based) authentication.

    Args:
        model: Model name (e.g. "claude-sonnet-4-5-20250514", "claude-haiku-4-5-20250514").
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        base_url: Custom base URL for the API.
        max_subscription: If True, use Claude Max subscription (OAuth) authentication.
        max_tokens: Maximum number of tokens to generate.
        **config: Additional keyword arguments passed to the Anthropic messages.create call.

    Example::

        from effectful.handlers.llm import Template
        from effectful.handlers.llm.anthropic import AnthropicProvider
        from effectful.ops.semantics import handler
        from effectful.ops.types import NotHandled

        @Template.define
        def greet(name: str) -> str:
            \"\"\"Say hello to {name}.\"\"\"
            raise NotHandled

        # Using API key (default)
        with handler(AnthropicProvider(model="claude-haiku-4-5-20250514")):
            result = greet("world")

        # Using Claude Max subscription
        with handler(AnthropicProvider(model="claude-sonnet-4-5-20250514", max_subscription=True)):
            result = greet("world")
    """

    client: "anthropic.Anthropic"
    config: dict[str, typing.Any]

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        max_subscription: bool = False,
        max_tokens: int = 4096,
        **config,
    ):
        self.client = _get_anthropic_client(
            api_key=api_key,
            base_url=base_url,
            max_subscription=max_subscription,
        )
        self.config = {
            "model": model,
            "max_tokens": max_tokens,
            **config,
        }

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)

        response_model = Encodable.define(template.__signature__.return_annotation, env)

        history: collections.OrderedDict[str, Message] = getattr(
            template, "__history__", collections.OrderedDict()
        )
        history_copy = history.copy()

        with handler({_get_history: lambda: history_copy}):
            call_system(template)
            message: Message = call_user(template.__prompt_template__, env)

            tool_calls: list[DecodedToolCall] = []
            result: T | None = None
            while message["role"] != "assistant" or tool_calls:
                message, tool_calls, result = call_assistant(
                    template.tools, response_model, **self.config
                )
                for tool_call in tool_calls:
                    message = call_tool(tool_call)

        try:
            _get_history()
        except NotImplementedError:
            history.clear()
            history.update(history_copy)
        return typing.cast(T, result)

    @implements(call_assistant)
    def _call_assistant[T, U](
        self,
        tools: collections.abc.Mapping[str, Tool],
        response_format: Encodable[T, U],
        model: str,
        max_tokens: int = 4096,
        **kwargs,
    ) -> MessageResult[T]:
        # Build tool specs in Anthropic format
        tool_specs = _tools_to_anthropic_format(tools)

        messages = list(_get_history().values())
        system_prompt, anthropic_messages = _messages_to_anthropic_format(messages)

        # Build the API call kwargs
        create_kwargs: dict[str, typing.Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            **{k: v for k, v in kwargs.items() if k not in ("response_format",)},
        }

        if system_prompt:
            create_kwargs["system"] = system_prompt

        all_tools = list(tool_specs)

        # For non-string structured output, add a _respond tool so the model
        # returns structured data via tool_use (Anthropic has no response_format).
        has_respond_tool = False
        if response_format.enc is not str:
            result_schema = pydantic.create_model(
                "Result", value=(response_format.enc, ...)
            ).model_json_schema()
            if "$defs" in result_schema:
                result_schema["properties"] = {
                    k: _resolve_refs(v, result_schema.get("$defs", {}))
                    for k, v in result_schema.get("properties", {}).items()
                }
                del result_schema["$defs"]
            all_tools.append({
                "name": "_respond",
                "description": "Return the final structured response. Call this when you are ready to answer.",
                "input_schema": result_schema,
            })
            has_respond_tool = True
            # Force _respond when no other tools are defined
            if not tool_specs:
                create_kwargs["tool_choice"] = {"type": "tool", "name": "_respond"}

        if all_tools:
            create_kwargs["tools"] = all_tools

        response: AnthropicMessage = self.client.messages.create(**create_kwargs)

        # Convert response to internal format
        content_parts: list[dict[str, typing.Any]] = []
        raw_tool_calls: list[dict[str, typing.Any]] = []
        text_content = ""

        for block in response.content:
            if block.type == "text":
                text_content += block.text
                content_parts.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                raw_tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Check if _respond tool was called (structured output)
        respond_call = None
        other_tool_calls: list[dict[str, typing.Any]] = []
        for tc in raw_tool_calls:
            if has_respond_tool and tc.get("name") == "_respond":
                respond_call = tc
            else:
                other_tool_calls.append(tc)

        # If _respond was called, decode structured result and ignore other tool calls
        if respond_call is not None:
            respond_input = respond_call["input"]
            result_value = respond_input.get("value", respond_input)
            serialized = json.dumps(result_value) if not isinstance(result_value, str) else result_value

            raw_message = _make_message({
                "role": "assistant",
                "content": serialized,
            })
            append_message(raw_message)

            try:
                result = response_format.decode(
                    response_format.deserialize(serialized)
                )
            except (pydantic.ValidationError, TypeError, ValueError, SyntaxError) as e:
                raise ResultDecodingError(e, raw_message=raw_message) from e

            return (raw_message, [], result)

        # Build the raw message in internal format
        raw_msg_content: typing.Any = text_content
        if content_parts and not other_tool_calls:
            raw_msg_content = text_content
        elif content_parts:
            raw_msg_content = content_parts

        # Build tool_calls in litellm-compatible format for the raw message
        litellm_tool_calls = []
        for tc in other_tool_calls:
            litellm_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["input"]),
                },
            })

        raw_message_dict: dict[str, typing.Any] = {
            "role": "assistant",
            "content": raw_msg_content,
        }
        if litellm_tool_calls:
            raw_message_dict["tool_calls"] = litellm_tool_calls

        raw_message = _make_message(raw_message_dict)
        append_message(raw_message)

        # Decode tool calls.
        # Anthropic returns raw values matching the schema, but the effectful
        # encoding system may wrap types in _BoxEncoding ({"value": ...}).
        # We wrap the raw input to match what ToolCallEncodable.decode expects.
        from effectful.handlers.llm.encoding import BaseEncodable, _BoxEncoding
        tool_call_encoding = Encodable.define(DecodedToolCall, dict(tools))

        tool_calls: list[DecodedToolCall] = []
        for tc in other_tool_calls:
            try:
                from litellm import ChatCompletionMessageToolCall

                # Wrap param values in {"value": ...} when the encoding expects it
                tool_name = tc["name"]
                raw_input = tc["input"]
                if tool_name in tools:
                    sig = inspect.signature(tools[tool_name])
                    wrapped_input = {}
                    for pname, param in sig.parameters.items():
                        if pname in raw_input:
                            param_enc = Encodable.define(param.annotation)
                            if isinstance(param_enc, BaseEncodable):
                                wrapped_input[pname] = {"value": raw_input[pname]}
                            else:
                                wrapped_input[pname] = raw_input[pname]
                        elif pname not in raw_input and param.default is not inspect.Parameter.empty:
                            pass  # optional param not provided
                    raw_input = wrapped_input

                litellm_tc = ChatCompletionMessageToolCall.model_validate({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(raw_input),
                    },
                })
                tool_calls.append(tool_call_encoding.decode(litellm_tc))
            except Exception as e:
                from litellm import ChatCompletionMessageToolCall

                mock_tc = ChatCompletionMessageToolCall.model_validate({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"]),
                    },
                })
                raise ToolCallDecodingError(
                    raw_tool_call=mock_tc,
                    original_error=e,
                    raw_message=raw_message,
                ) from e

        result = None
        if not tool_calls:
            serialized_result = text_content
            assert isinstance(serialized_result, str), (
                "final response from the model should be a string"
            )
            try:
                result = response_format.decode(
                    response_format.deserialize(serialized_result)
                )
            except (pydantic.ValidationError, TypeError, ValueError, SyntaxError) as e:
                raise ResultDecodingError(e, raw_message=raw_message) from e

        return (raw_message, tool_calls, result)
