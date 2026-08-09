import collections
import collections.abc
import inspect
import typing

import litellm

from effectful.handlers.llm.harness.contextualization import _tools_in_scope
from effectful.handlers.llm.harness.hooks import (
    Message,
    call_assistant,
    call_system,
    call_tool,
    call_user,
    completion,
)
from effectful.handlers.llm.types import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class LiteLLMConfigurer(ObjectInterpretation):
    """Configures the LiteLLM API."""

    config: collections.abc.Mapping[str, typing.Any]

    def __init__(self, model="gpt-4o", **config):
        self.config = {
            "model": model,
            **inspect.signature(litellm.completion).bind_partial(**config).kwargs,
        }

    def _add_cache_control(
        self,
        messages: collections.abc.Sequence[Message],
    ) -> list[Message]:
        """Mark the last user/tool message of a request for prompt caching.

        A `cache_control` breakpoint caches the request prefix -- tools, system,
        and messages, in that order -- up to and including the block it sits on,
        and a later request that shares that prefix reads it back at a fraction
        of the input price. Putting the breakpoint on the *last* input message
        therefore caches as much of a growing conversation as possible: an
        agent's accumulated history, or the rounds of a single tool-use loop.

        Exactly one breakpoint is added, and it moves to the newest message on
        every request. That matters in both directions: providers cap how many
        breakpoints a request may carry (Anthropic allows four), and a
        breakpoint pinned to an old position stops extending the cached prefix
        as the conversation grows. The assembled system prompt carries its own,
        separately (see `call_system`).

        Returns a new list, leaving the caller's messages untouched, so this
        transport-level annotation never reaches the stored history -- and so
        never reaches an `Agent`'s checkpointed transcript.
        """
        out = list(messages)
        for i in reversed(range(len(out))):
            msg = out[i]
            if msg["role"] not in ("user", "tool"):
                continue
            content = msg.get("content")
            if not isinstance(content, list) or not content:
                continue
            last_block = content[-1]
            if not isinstance(last_block, dict):
                continue
            if "cache_control" not in last_block:
                out[i] = typing.cast(
                    Message,
                    {
                        **msg,
                        "content": [
                            *content[:-1],
                            {**last_block, "cache_control": {"type": "ephemeral"}},
                        ],
                    },
                )
            break
        return out

    @implements(completion)
    def _completion(self, *args, **kwargs):
        """Inject the provider's configuration (model and bound litellm kwargs)
        into the low-level request before delegating."""
        kwargs = {**self.config, **kwargs}
        if kwargs.get("messages"):
            kwargs["messages"] = self._add_cache_control(kwargs["messages"])
        return fwd(*args, **kwargs)


class LiteLLMProvider(LiteLLMConfigurer):
    """Implements templates using the LiteLLM API."""

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        message: Message = call_system(template)

        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)
        message = call_user(template, env)

        result: T | None = None
        is_final: bool = False
        while not is_final:
            message, tool_calls, result = call_assistant(
                env,
                template.__signature__.return_annotation,
                _tools_in_scope(env) - {template},
                anchor=template,
                force_tool=self.config.get("tool_choice") == "required",
            )
            if tool_calls:
                for tool_call in tool_calls:
                    message, result, is_final = call_tool(tool_call)
            else:
                is_final = True

        return typing.cast(T, result)
