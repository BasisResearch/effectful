import inspect
import typing
from collections.abc import Callable, Iterable

from effectful.ops.types import NotHandled, Operation


class Tool[**P, T](Operation[P, T]):
    @classmethod
    def define(cls, *args, **kwargs) -> "Tool[P, T]":
        return typing.cast(Tool[P, T], super().define(*args, **kwargs))


class Template[**P, T](Operation[P, T]):
    __prompt_template__: str
    tools: tuple[Tool, ...]

    def __init__(
        self,
        signature: inspect.Signature,
        name: str,
        prompt_template: str,
        tools: tuple[Tool, ...],
    ):
        self.__prompt_template__ = prompt_template
        self.tools = tools

        def default(*args, **kwargs):
            raise NotHandled

        super().__init__(signature, name, default)

    @classmethod
    def define(cls, _func=None, *, tools: Iterable[Tool] = (), **kwargs):
        def decorator(body: Callable[P, T]):
            if not body.__doc__:
                raise ValueError("Expected a docstring on body")

            return cls(
                inspect.signature(body), body.__name__, body.__doc__, tuple(tools)
            )

        if _func is None:
            return decorator
        return decorator(_func)
