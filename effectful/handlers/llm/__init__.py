from __future__ import annotations

import dataclasses
import functools
import inspect
import types
from collections import ChainMap
from collections.abc import Callable, Iterable
from typing import Any

from effectful.ops.semantics import evaluate
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


class LexicalContext(ChainMap):
    """ChainMap subclass for Template lexical scope.

    This avoids recursive evaluation of circular Template references.
    """

    pass


@evaluate.register(LexicalContext)
def _evaluate_lexical_context(expr: LexicalContext, **kwargs) -> LexicalContext:
    return expr


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __prompt_template__: str
    __signature__: inspect.Signature
    __name__: str
    lexical_context: LexicalContext = dataclasses.field(
        default_factory=lambda: LexicalContext(
            types.MappingProxyType({}), types.MappingProxyType({})
        ),  # type: ignore
        repr=False,
    )

    @staticmethod
    def _get_excluded_operations() -> frozenset[Operation]:
        """Get the set of internal operations to exclude from auto-capture."""
        from effectful.handlers.llm import providers
        from effectful.ops import semantics

        excluded: set[Operation] = set()
        for module in (providers, semantics):
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, Operation):
                    excluded.add(obj)
        return frozenset(excluded)

    @property
    def tools(self) -> tuple[Operation | Template, ...]:
        """Operations and Templates available as tools. Auto-capture from lexical context."""
        excluded_ops = self._get_excluded_operations()
        result: list[Operation | Template] = []
        # ChainMap.items() respects shadowing (locals shadow globals)
        for name, obj in self.lexical_context.items():
            if name.startswith("_") or obj in result:
                continue
            if isinstance(obj, Operation):
                # Exclude internal operations from providers and semantics modules
                if obj in excluded_ops:
                    continue
                result.append(obj)
            elif isinstance(obj, Template):
                result.append(obj)
        return tuple(result)

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
        globals_proxy: types.MappingProxyType[str, Any] = types.MappingProxyType(
            frame.f_globals
        )
        locals_proxy: types.MappingProxyType[str, Any] = types.MappingProxyType(
            frame.f_locals
        )
        # LexicalContext: locals first (shadow globals), then globals
        lexical_context = LexicalContext(locals_proxy, globals_proxy)

        def decorator(body: Callable[P, T]):
            if not body.__doc__:
                raise ValueError("Expected a docstring on body")

            return cls(
                __prompt_template__=body.__doc__,
                __signature__=inspect.signature(body),
                __name__=body.__name__,
                lexical_context=lexical_context,
            )

        if _func is None:
            return decorator
        return decorator(_func)
