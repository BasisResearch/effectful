import functools
import inspect
import types
import typing
from collections import ChainMap
from collections.abc import Callable, Mapping
from typing import Any

import litellm
import pydantic
from litellm import OpenAIChatCompletionToolParam, OpenAIMessageContent

from effectful.handlers.llm.encoding import type_to_encodable_type
from effectful.ops.semantics import evaluate
from effectful.ops.types import Operation


class LexicalContext(ChainMap):
    """ChainMap subclass for Template lexical scope.

    This avoids recursive evaluation of circular Template references.
    """

    pass


@evaluate.register(LexicalContext)
def _evaluate_lexical_context(expr: LexicalContext, **kwargs) -> LexicalContext:
    return expr


class Tool[**P, T](Operation[P, T]):
    def __init__(
        self, signature: inspect.Signature, name: str, default: Callable[P, T]
    ):
        if not default.__doc__:
            raise ValueError("Tools must have docstrings.")

        for param_name, param in signature.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                raise ValueError(
                    f"Parameter '{param_name}' of '{default.__name__}' has no type annotation"
                )

        super().__init__(signature, name, default)

    @classmethod
    def define(cls, *args, **kwargs) -> "Tool[P, T]":
        return typing.cast("Tool[P, T]", super().define(*args, **kwargs))

    @functools.cached_property
    def parameter_model(self) -> type[pydantic.BaseModel]:
        fields = {
            name: type_to_encodable_type(param.annotation).t
            for name, param in self.__signature__.parameters.items()
        }
        parameter_model = pydantic.create_model(
            "Params",
            __config__={"extra": "forbid"},
            **fields,  # type: ignore
        )
        return parameter_model

    @property
    def function_definition(self) -> OpenAIChatCompletionToolParam:
        response_format = litellm.utils.type_to_response_format_param(
            self.parameter_model
        )
        assert response_format is not None
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": response_format["json_schema"][
                    "schema"
                ],  # extract the schema
                "strict": True,
            },
        }

    def call_with_json_args(self, json_str: str) -> OpenAIMessageContent:
        """Implements a roundtrip call to a python function. Input is a json
        string representing an LLM tool call request parameters. The output is
        the serialised response to the model.

        """
        try:
            # build dict of raw encodable types U
            raw_args = self.parameter_model.model_validate_json(json_str)

            # use encoders to decode Us to python types T
            params: dict[str, Any] = {
                param_name: type_to_encodable_type(
                    self.parameter_annotations[param_name]
                ).decode(getattr(raw_args, param_name))
                for param_name in raw_args.model_fields_set
            }

            # call tool with python types
            result = self(**params)

            # serialize back to U using encoder for return type
            sig = self.__signature__
            encoded_ty = type_to_encodable_type(sig.return_annotation)
            encoded_value = encoded_ty.encode(result)

            # serialise back to Json
            return encoded_ty.serialize(encoded_value)
        except Exception as exn:
            return str({"status": "failure", "exception": str(exn)})


class Template[**P, T](Tool[P, T]):
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

    @property
    def __prompt_template__(self):
        return self.__default__.__doc__

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
