import functools

import weave

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import (
    call_assistant,
    call_system,
    call_user,
)
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


def _make_instrumented(op):
    @weave.op()
    @functools.wraps(op)
    def wrapper(*args, **kwargs):
        return fwd(op, *args, **kwargs)

    return wrapper


class WeaveProvider(ObjectInterpretation):
    """Traces Tool, Template, and message-level calls with Weights & Biases Weave.

    Compose with a provider via :func:`~effectful.ops.semantics.handler`
    to add tracing::

        weave.init("my-project")
        with handler(provider), handler(WeaveProvider()):
            print(limerick(theme))
    """

    def __init__(self):
        # cache each template instead of repeatedly instrumenting it
        self._get_instrumented = functools.cache(_make_instrumented)

    @implements(call_user)
    @weave.op()
    def call_user(self, template, env):
        return fwd(template, env)

    @implements(call_system)
    @weave.op()
    def call_system(self, template):
        return fwd(template)

    @implements(call_assistant)
    @weave.op()
    def call_assistant(self, tools, response_format, model, **kwargs):
        return fwd(tools, response_format, model, **kwargs)

    @implements(Tool.__apply__)
    def call_tool(self, tool, *args, **kwargs):
        return self._get_instrumented(tool)(*args, **kwargs)

    @implements(Template.__apply__)
    def call_template(self, template, *args, **kwargs):
        return self._get_instrumented(template)(*args, **kwargs)
