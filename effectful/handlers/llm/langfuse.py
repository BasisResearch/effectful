import functools
import typing

import litellm
from langfuse import get_client, observe

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import (
    call_assistant,
    call_system,
    call_user,
    completion,
)
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


def _extract_generation_meta(result) -> dict[str, typing.Any]:
    usage = result.usage
    if usage is None:
        return {}
    meta: dict[str, typing.Any] = {"model": getattr(result, "model", None)}
    usage_details: dict[str, int] = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if v is not None:
            usage_details[k] = v
    if usage_details:
        meta["usage_details"] = usage_details
    try:
        cost = litellm.completion_cost(completion_response=result)
        meta["cost_details"] = {"total": cost}
    except Exception:
        pass
    meta["metadata"] = {"response_id": getattr(result, "id", None)}
    return meta


def _make_instrumented(op, as_type):
    @observe(as_type=as_type)
    @functools.wraps(op)
    def wrapper(*args, **kwargs):
        return fwd(op, *args, **kwargs)

    return wrapper


class LangfuseProvider(ObjectInterpretation):
    """Traces Tool, Template, and completion calls with Langfuse.

    Compose with a provider via :func:`~effectful.ops.semantics.handler`
    to add tracing::

        with handler(provider), handler(LangfuseProvider()):
            print(limerick(theme))
    """

    def __init__(self):
        self.langfuse = get_client()
        # cache each template instead of repeatedly instrumenting it
        self._get_instrumented = functools.cache(_make_instrumented)

    @implements(completion)
    @observe(as_type="generation")
    def completion(self, *args, **kwargs):
        result = fwd(*args, **kwargs)
        self.langfuse.update_current_generation(**_extract_generation_meta(result))
        return result

    @implements(call_user)
    @observe()
    def call_user(self, template, env):
        return fwd(template, env)

    @implements(call_system)
    @observe()
    def call_system(self, template):
        return fwd(template)

    @implements(call_assistant)
    @observe()
    def call_assistant(self, tools, response_format, model, **kwargs):
        return fwd(tools, response_format, model, **kwargs)

    @implements(Tool.__apply__)
    def call_tool(self, tool, *args, **kwargs):
        return self._get_instrumented(tool, "tool")(*args, **kwargs)

    @implements(Template.__apply__)
    def call_template(self, template, *args, **kwargs):
        return self._get_instrumented(template, "generation")(*args, **kwargs)
