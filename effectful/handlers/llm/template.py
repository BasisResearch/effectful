import inspect
import typing
from collections.abc import Callable, Iterable

import pydantic

from effectful.handlers.llm.encoding import type_to_encodable_type
from effectful.ops.types import NotHandled, Operation


class Tool[**P, T](Operation[P, T]):
    def __init__(
        self, signature: inspect.Signature, name: str, default: Callable[P, T]
    ):
        if not body.__doc__:
            raise ValueError("Tools must have docstrings.")

        for param_name, param in signature.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                raise ValueError(
                    f"Parameter '{param_name}' in '{obj.__name__}' has no type annotation"
                )

        super().__init__(signature, name, default)

    @classmethod
    def define(cls, *args, **kwargs) -> "Tool[P, T]":
        return typing.cast("Tool[P, T]", super().define(*args, **kwargs))

    @functools.cached_property
    def parameter_annotations(self) -> dict[str, type]:
        for param_name, param in self.__signature__.parameters.items():
            # Skip parameters without type annotations
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"Parameter '{param_name}' in '{obj.__name__}' "
                    "does not have a type annotation"
                )
            # get_type_hints might not include the parameter if annotation is invalid
            if param_name not in hints:
                raise TypeError(
                    f"Parameter '{param_name}' in '{obj.__name__}' "
                    "does not have a valid type annotation"
                )
            parameter_annotations[param_name] = hints[param_name]

    @functools.cached_property
    def parameter_model(self) -> type[pydantic.BaseModel]:
        fields = {
            param_name: type_to_encodable_type(param_type).t
            for param_name, param_type in self.parameter_annotations.items()
        }
        parameter_model = pydantic.create_model(
            "Params",
            __config__={"extra": "forbid"},
            **fields,  # type: ignore
        )
        return parameter_model


class Template[**P, T](Tool[P, T]):
    __prompt_template__: str
    __context__: Mapping[str, Any]

    def __init__(
        self,
        signature: inspect.Signature,
        name: str,
        default: Callable[P, T],
        context: Mapping[str, Any],
    ):
        super().__init__(signature, name, default)
        self.__context__ = context

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
        return frozenset(getatt)

    @property
    def tools(self) -> tuple[Tool, ...]:
        """Operations and Templates available as tools. Auto-capture from lexical context."""
        result = (
            set(
                obj
                for (name, obj) in self.__context__.items()
                if not name.startswith("_") and isinstance(obj, Tool)
            )
            - self._get_excluded_operations()
        )
        return tuple(result)

    @classmethod
    def define(cls, _func=None, **kwargs) -> "Template[P, T]":
        """Define a prompt template."""

        current_frame = inspect.currentframe()
        parent_frame = current_frame.f_back if current_frame else None
        if parent_frame:
            globals_proxy = types.MappingProxyType(parent_frame.f_globals)
            locals_proxy = types.MappingProxyType(parent_frame.f_locals)

            # LexicalContext: locals first (shadow globals), then globals
            context = LexicalContext(locals_proxy, globals_proxy)  # type: ignore[arg-type]
        else:
            context = LexicalContext()

        def decorator(body: Callable[P, T]):
            return cls(inspect.signature(body), body.__name__, body, context)

        if _func is None:
            return decorator
        return decorator(_func)
