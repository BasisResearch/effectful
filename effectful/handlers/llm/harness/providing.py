import collections
import collections.abc
import inspect
import typing

import litellm

from effectful.handlers.llm.harness.hooks import (
    Message,
    _add_cache_control_to_history,
    _get_history,
    _tools_in_scope,
    call_assistant,
    call_system,
    call_tool,
    call_user,
    completion,
    new_agent_call_scope,
)
from effectful.handlers.llm.types import Template
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements

_history_scope = new_agent_call_scope()


class LiteLLMProvider(ObjectInterpretation):
    """Implements templates using the LiteLLM API."""

    config: collections.abc.Mapping[str, typing.Any]

    def __init__(self, model="gpt-4o", **config):
        self.config = {
            "model": model,
            **inspect.signature(litellm.completion).bind_partial(**config).kwargs,
        }

    @implements(completion)
    def _completion(self, *args, **kwargs):
        """Inject the provider's configuration (model and bound litellm kwargs)
        into the low-level request before delegating."""
        return fwd(*args, **{**self.config, **kwargs})

    @implements(Template.__apply__)
    def _call[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        # encode arguments
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)

        history: collections.OrderedDict[str, Message] = getattr(
            template, "__history__", collections.OrderedDict()
        )  # type: ignore
        agent_id: str | None = getattr(template, "__agent_id__", None)
        is_agent = agent_id is not None
        history_copy = history.copy()

        # Only the outermost call for a given agent writes back to its shared
        # history; nested calls (e.g. a tool invoking another template on the
        # same agent) work on a private copy that is discarded on return. This
        # is scoped per-agent (via __agent_id__) rather than globally, so a
        # *different* agent invoked mid-call (e.g. via tool delegation) is
        # still correctly treated as outermost for itself.
        with (
            _history_scope(agent_id) as is_outermost,
            handler({_get_history: lambda: history_copy}),
        ):
            if (
                not _get_history()
                or next(iter(_get_history().values()))["role"] != "system"
            ):
                message: Message = call_system(template)

            message = call_user(template, env)

            # For agents with persistent history, add cache_control to the
            # last user message so the growing prefix gets cached on providers
            # that support it (Anthropic). litellm strips it for OpenAI.
            if is_agent:
                _add_cache_control_to_history(history_copy)

            # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
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

        if is_agent and is_outermost:
            history.clear()
            history.update(history_copy)
        return typing.cast(T, result)
