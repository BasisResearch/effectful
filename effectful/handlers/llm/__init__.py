import dataclasses
import functools
import inspect
import types
import weakref
from collections import ChainMap
from collections.abc import Callable, Mapping
from typing import Any

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __signature__: inspect.Signature
    __prompt_template__: str
    lexical_context: Mapping[str, Any] = dataclasses.field(
        default_factory=weakref.WeakValueDictionary
    )

    @property
    def tools(self) -> tuple[Operation, ...]:
        """Operations from lexical context, available as tools for LLM calls."""
        return tuple(
            obj for obj in self.lexical_context.values() if isinstance(obj, Operation)
        )

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotHandled

    def __get__(self, instance, _owner):
        if instance is not None:
            return functools.partial(self, instance)
        else:
            return self

    @classmethod
    def define(cls, _func=None):
        frame: types.FrameType = inspect.currentframe().f_back  # type: ignore[union-attr]
        caller_module = frame.f_globals.get("__name__", "")
        caller_scope = ChainMap(frame.f_locals, frame.f_globals)
        lexical_ctx: weakref.WeakValueDictionary[str, Any] = (
            weakref.WeakValueDictionary()
        )
        for name, obj in caller_scope.items():
            if name.startswith("_"):
                continue
            if not isinstance(obj, (Template, Operation)):
                continue
            # Only capture Operations/Templates defined in the same module as caller
            # This filters out imported library operations (completion, tool_call, etc.)
            obj_module = getattr(obj, "__module__", None)
            if obj_module != caller_module:
                continue
            lexical_ctx[name] = obj

        def decorator(body: Callable[P, T]):
            if not body.__doc__:
                raise ValueError("Expected a docstring on body")

            return cls(
                __signature__=inspect.signature(body),
                __prompt_template__=body.__doc__,
                lexical_context=lexical_ctx,
            )

        if _func is None:
            return decorator
        return decorator(_func)
