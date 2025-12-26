import inspect
import types
import typing
from collections import ChainMap
from collections.abc import Callable, Mapping
from typing import Any

from effectful.ops.types import NotHandled, Operation


class Tool[**P, T](Operation[P, T]):
    def __init__(
        self, signature: inspect.Signature, name: str, default: Callable[P, T]
    ):
        if not default.__doc__:
            raise ValueError("Tools must have docstrings.")

        for param_name, param in signature.parameters.items():
            # allow self to be unannotated
            if param_name != "self" and param.annotation is inspect.Parameter.empty:
                raise ValueError(
                    f"Parameter '{param_name}' of '{default.__name__}' has no type annotation"
                )

        super().__init__(signature, name, default)

    @classmethod
    def define(cls, *args, **kwargs) -> "Tool[P, T]":
        return typing.cast("Tool[P, T]", super().define(*args, **kwargs))


class Template[**P, T](Tool[P, T]):
    __context__: Mapping[str, Any]

    @property
    def __prompt_template__(self) -> str:
        return self.__default__.__doc__

    @property
    def tools(self) -> Mapping[str, Tool]:
        """Operations and Templates available as tools. Auto-capture from lexical context."""
        result = {
            name: obj
            for (name, obj) in self.__context__.items()
            if isinstance(obj, Tool)
        }
        return result

    def __get__[T](self, instance: T | None, owner: type[T] | None = None):
        if hasattr(self, "_name_on_instance") and hasattr(
            instance, self._name_on_instance
        ):
            return getattr(instance, self._name_on_instance)

        result = super().__get__(instance, owner)

        self_context = {}
        for k in instance.__dir__():
            if k.startswith("_"):
                continue

            v = getattr(instance, k)
            if isinstance(v, Tool):
                self_context[k] = v

        context = self.__grandparent_context__.new_child(self_context)
        result.__context__ = context
        return result

    @staticmethod
    def _frame_context(offset: int) -> ChainMap[str, Any] | None:
        """Return the lexical context of a stack frame. `offset` is the number
        of frames to travel up the stack.

        """
        frame = inspect.currentframe()

        for _ in range(offset + 1):  # include this function's frame
            if frame is None:
                return None
            frame = frame.f_back

        globals_proxy = types.MappingProxyType(frame.f_globals)
        locals_proxy = types.MappingProxyType(frame.f_locals)
        context: ChainMap[str, Any] = ChainMap(locals_proxy, globals_proxy)  # type: ignore[arg-type]
        return context

    @classmethod
    def define[**Q, V](
        cls, default: Callable[Q, V], *args, **kwargs
    ) -> "Template[Q, V]":
        """Define a prompt template.

        `define` takes a function, and can be used as a decorator. The
        function's docstring should be a prompt, which may be templated in the
        function arguments. The prompt will be provided with any instances of
        `Tool` that exist in the lexical context as callable tools.

        """
        context = Template._frame_context(1)
        gp_context = Template._frame_context(2)
        op = super().define(default, *args, **kwargs)
        op.__context__ = context  # type: ignore[attr-defined]
        op.__grandparent_context__ = gp_context  # type: ignore[attr-defined]
        return typing.cast(Template[Q, V], op)

    def replace(
        self,
        signature: inspect.Signature | None = None,
        prompt_template: str | None = None,
        name: str | None = None,
    ) -> "Template":
        signature = signature or self.__signature__
        prompt_template = prompt_template or self.__prompt_template__
        name = name or self.__name__

        if prompt_template:

            def default(*args, **kwargs):
                raise NotHandled

            default.__doc__ = prompt_template
        else:
            default = self.__default__

        op = Template(signature, name, default)
        op.__context__ = self.__context__
        return op
