from __future__ import annotations

import dataclasses
import functools
import inspect
import types
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


class LexicalContext(NamedTuple):
    """Pair of (globals [live], locals [snapshot]) MappingProxyTypes."""

    globals_: types.MappingProxyType[str, Any]
    locals_: types.MappingProxyType[str, Any]


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __prompt_template__: str
    __signature__: inspect.Signature
    __name__: str
    lexical_context: LexicalContext = dataclasses.field(
        default=LexicalContext(types.MappingProxyType({}), types.MappingProxyType({})),
        repr=False,
    )
    _caller_module: str = dataclasses.field(default="", repr=False)
    # Explicit tools (empty by default, "__auto__" sentinel for auto-capture)
    _explicit_tools: tuple[Operation | Template, ...] = dataclasses.field(
        default=(), repr=False
    )

    @property
    def tools(self) -> tuple[Operation | Template, ...]:
        """Operations and Templates available as tools.

        By default, no tools are exposed. Use tools= parameter in define():
            @Template.define(tools=[my_tool])  # explicit list
            @Template.define(tools="auto")     # auto-capture from lexical scope
        """
        if self._explicit_tools == ("__auto__",):
            # Auto-capture from lexical context
            result: list[Operation | Template] = []
            for scope in (self.lexical_context.globals_, self.lexical_context.locals_):
                for name, obj in scope.items():
                    if name.startswith("_") or obj in result:
                        continue
                    if isinstance(obj, Operation):
                        if getattr(obj, "__module__", None) == self._caller_module:
                            result.append(obj)
                    elif isinstance(obj, Template):
                        if obj._caller_module == self._caller_module:
                            result.append(obj)
            return tuple(result)
        return self._explicit_tools

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotHandled

    def __get__(self, instance, _owner):
        if instance is not None:
            return functools.partial(self, instance)
        else:
            return self

    @classmethod
    def define(
        cls,
        _func=None,
        *,
        tools: Iterable[Operation | Template] | str | None = None,
    ):
        """Define a prompt template.

        Args:
            tools: Tools to expose to the LLM:
                   - None (default): no tools
                   - "auto": auto-capture from lexical scope
                   - list: explicit list of Operations/Templates
        """
        frame: types.FrameType = inspect.currentframe().f_back  # type: ignore
        caller_module = frame.f_globals.get("__name__", "")
        globals_proxy = types.MappingProxyType(frame.f_globals)
        locals_proxy = types.MappingProxyType(frame.f_locals)
        if tools == "auto":
            explicit_tools: tuple[Any, ...] = ("__auto__",)
        elif tools is not None:
            explicit_tools = tuple(tools)  # type: ignore
        else:
            explicit_tools = ()  # no tools by default

        def decorator(body: Callable[P, T]):
            if not body.__doc__:
                raise ValueError("Expected a docstring on body")

            return cls(
                __prompt_template__=body.__doc__,
                __signature__=inspect.signature(body),
                __name__=body.__name__,
                lexical_context=LexicalContext(globals_proxy, locals_proxy),
                _caller_module=caller_module,
                _explicit_tools=explicit_tools,
            )

        if _func is None:
            return decorator
        return decorator(_func)
