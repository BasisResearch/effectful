import inspect
import types
import typing
from collections import ChainMap
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from effectful.ops.types import NotHandled, Operation


class Tool[**P, T](Operation[P, T]):
    def __init__(
        self, signature: inspect.Signature, name: str, default: Callable[P, T]
    ):
        if not default.__doc__:
            raise ValueError("Tools must have docstrings.")

        super().__init__(signature, name, default)

    @classmethod
    def define(cls, *args, **kwargs) -> "Tool[P, T]":
        return typing.cast("Tool[P, T]", super().define(*args, **kwargs))


class Template[**P, T](Tool[P, T]):
    __context__: ChainMap[str, Any]

    @property
    def __prompt_template__(self) -> str:
        assert self.__default__.__doc__ is not None
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

    def __get__[S](self, instance: S | None, owner: type[S] | None = None):
        if hasattr(self, "_name_on_instance") and hasattr(
            instance, self._name_on_instance
        ):
            return getattr(instance, self._name_on_instance)

        result = super().__get__(instance, owner)

        self_context = {}
        for k in instance.__dir__():
            v = getattr(instance, k)
            if isinstance(v, Tool):
                self_context[k] = v

        result.__context__ = self.__context__.new_child(self_context)
        return result

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
        frame = inspect.currentframe()
        assert frame is not None
        frame = frame.f_back
        assert frame is not None

        # Check if we're in a class definition by looking for __qualname__
        qualname = frame.f_locals.get("__qualname__")
        n_frames = qualname.count(".") if qualname is not None else 1

        contexts = []
        for offset in range(n_frames):
            assert frame is not None
            locals_proxy: types.MappingProxyType[str, Any] = types.MappingProxyType(
                frame.f_locals
            )
            globals_proxy: types.MappingProxyType[str, Any] = types.MappingProxyType(
                frame.f_globals
            )
            contexts.append(locals_proxy)
            frame = frame.f_back
        contexts.append(globals_proxy)
        context: ChainMap[str, Any] = ChainMap(
            *typing.cast(list[MutableMapping[str, Any]], contexts)
        )

        op = super().define(default, *args, **kwargs)
        op.__context__ = context  # type: ignore[attr-defined]
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
