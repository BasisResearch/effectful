import dataclasses
import functools
import inspect
from collections.abc import Callable, Iterable

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


class Template[**P, T](Operation[P, T]):
    __prompt_template__: str
    tools: tuple[Operation, ...]

    def __init__(
        self,
        body: Callable[P, T],
        __prompt_template__: str,
        tools: tuple[Operation, ...],
    ):
        self.__prompt_template__ = __prompt_template__
        self.tools = tools
        super().__init__(inspect.signature(body), body.__name__, body)

    @classmethod
    def define(cls, _func=None, *, tools: Iterable[Operation] = ()):
        def decorator(body: Callable[P, T]):
            if not body.__doc__:
                raise ValueError("Expected a docstring on body")

            return cls(body, body.__doc__, tuple(tools))

        if _func is None:
            return decorator
        return decorator(_func)
