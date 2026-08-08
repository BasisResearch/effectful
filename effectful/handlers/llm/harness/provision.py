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

    # TODO reinstate this (and check, it seems wrong)
    def _add_cache_control_to_history(
        self,
        history: collections.abc.Sequence[Message],
    ) -> None:
        """Add cache_control to the last user/tool message in an agent's history.

        This enables prompt caching on providers that support it (e.g. Anthropic).
        Providers that don't support it (e.g. OpenAI) have cache_control stripped
        by litellm's request transformation, so this is always safe to apply.
        """
        if not history:
            return
        for msg in history:
            if msg["role"] not in ("user", "tool", "assistant"):
                continue
            content = msg.get("content")
            if isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict) and "cache_control" not in last_block:
                    new_content = list(content)
                    new_content[-1] = {
                        **last_block,
                        "cache_control": {"type": "ephemeral"},
                    }
                    msg["content"] = new_content

    @implements(completion)
    def _completion(self, *args, **kwargs):
        """Inject the provider's configuration (model and bound litellm kwargs)
        into the low-level request before delegating."""
        return fwd(*args, **{**self.config, **kwargs})


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
