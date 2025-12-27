import inspect
import types
import typing
from collections import ChainMap
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from effectful.ops.types import INSTANCE_OP_PREFIX, NotHandled, Operation


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


@dataclass
class _BoundInstance[T]:
    instance: T


class Template[**P, T](Tool[P, T]):
    __context__: ChainMap[str, Any]

    @property
    def __prompt_template__(self) -> str:
        assert self.__default__.__doc__ is not None
        return self.__default__.__doc__

    @property
    def tools(self) -> Mapping[str, Tool]:
        """Operations and Templates available as tools. Auto-capture from lexical context."""
        result = {}

        for name, obj in self.__context__.items():
            # Collect tools in context
            if isinstance(obj, Tool):
                result[name] = obj

            if isinstance(obj, staticmethod) and isinstance(obj.__func__, Tool):
                result[name] = obj.__func__

            # Collect tools as methods on any bound instances
            if isinstance(obj, _BoundInstance):
                for instance_name in obj.instance.__dir__():
                    if instance_name.startswith(INSTANCE_OP_PREFIX):
                        continue
                    instance_obj = getattr(obj.instance, instance_name)
                    if isinstance(instance_obj, Tool):
                        result[instance_name] = instance_obj

        return result

    def __get__[S](self, instance: S | None, owner: type[S] | None = None):
        if hasattr(self, "_name_on_instance") and hasattr(
            instance, self._name_on_instance
        ):
            return getattr(instance, self._name_on_instance)

        result = super().__get__(instance, owner)
        self_param_name = list(self.__signature__.parameters.keys())[0]
        self_context = {self_param_name: _BoundInstance(instance)}
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
        n_frames = 1
        if qualname is not None:
            name_components = qualname.split(".")
            for name in reversed(name_components):
                if name == "<locals>":
                    break
                n_frames += 1

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
