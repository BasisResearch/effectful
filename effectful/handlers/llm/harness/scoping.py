import collections.abc
import typing

import pydantic

from effectful.handlers.llm.harness.hooks import call_assistant, call_system
from effectful.handlers.llm.types import Agent, Encodable, Template, Tool
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

if typing.TYPE_CHECKING:
    from effectful.handlers.llm.harness.hooks import AssistantResult


class LexicalReaders(ObjectInterpretation):
    """Intercept `call_assistant` to also expose plain values from the
    lexical context as zero-argument read-only Tools.  Each non-Tool,
    non-Template, non-Agent value bound to a valid identifier is
    wrapped via `_LexicalVariableTool` if `Encodable[T]` accepts it;
    schema-generation failures cause the symbol to be skipped.
    """

    @typing.final
    class _LexicalVariableTool[T](Tool[[], T]):
        """## Reading lexical variables

        Some of the tools below take no arguments and simply return the current
        value of a named variable from this Template's lexical scope (see the
        *Lexical scope* table for the available names and their types). Call such a
        reader when your answer depends on the concrete value of an in-scope
        variable that has not already been spliced into the prompt — it lets you
        fetch that value on demand instead of guessing it. Each reader's description
        names the variable it reads.
        """

        @classmethod
        def define(cls, value: typing.Any, *, name: str) -> "Tool[[], typing.Any]":
            """Construct a synthetic reader Tool that returns `value`.

            Raises if `Encodable[nested_type(value)]` cannot be generated.
            The caller is responsible for catching the failure and deciding
            whether to skip the symbol.
            """
            assert not isinstance(value, Tool), (
                "Tools are real tools and must not be re-wrapped as lexical readers."
            )
            typ: typing.Any = nested_type(value).value
            # Probe schema generation; raises if `Encodable[typ]` is not implemented.
            pydantic.TypeAdapter(Encodable[typ]).json_schema()

            def tool_fn():
                return value

            tool_fn.__name__ = name
            tool_fn.__qualname__ = name
            tool_fn.__module__ = type(value).__module__
            tool_fn.__doc__ = "Reads lexical variable of the same name"
            tool_fn.__annotations__ = {"return": typ}
            return super().define(tool_fn)

    @implements(call_system)
    def _call_system(self, template, tool_types=frozenset()):
        return fwd(template, tool_types=tool_types | {self._LexicalVariableTool})

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        env: collections.abc.Mapping[str, typing.Any],
        response_type: type[T],
        tools: collections.abc.Set[Tool] = frozenset(),
        anchor: "Template | None" = None,
        force_tool: bool = False,
    ) -> AssistantResult[T]:
        readers: set[Tool] = set(tools)
        taken = {t.__name__ for t in tools}
        for name, obj in env.items():
            if (
                name in taken
                or not name.isidentifier()
                or isinstance(obj, Tool)
                or (name.startswith("__") and name.endswith("__"))
            ):
                continue
            try:
                readers.add(self._LexicalVariableTool.define(obj, name=name))
                taken.add(name)
            except Exception:
                continue
        return fwd(env, response_type, readers, anchor=anchor, force_tool=force_tool)


def _tools_in_scope(
    env: collections.abc.Mapping[str, typing.Any],
) -> collections.abc.Set[Tool]:
    """Return the tools available to a Template given its lexical context.

    Default rule: real `Tool` and `Template` values bound directly in
    `env`, plus `Tool` methods discovered through the MRO of any
    `Agent` instance in `env`.

    Tools are identified by object, so the same `Tool` visible under
    several bindings appears once.  Names are derived from each tool's
    `__name__` by :func:`call_assistant`, not from the binding name.
    """
    result: set[Tool] = set()

    for obj in env.values():
        if isinstance(obj, Tool | Template):
            result.add(obj)
        elif isinstance(obj, Agent):
            for cls in type(obj).__mro__:
                for attr_name in vars(cls):
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, Tool):
                        result.add(attr)

    return result
