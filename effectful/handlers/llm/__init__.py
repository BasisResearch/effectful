import dataclasses
import inspect
from collections.abc import Callable

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __signature__: inspect.Signature
    __prompt_template__: str

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotHandled

    @classmethod
    def define(cls, body: Callable[P, T]) -> "Template[P, T]":
        if not body.__doc__:
            raise ValueError("Expected a docstring on body")

        return cls(
            __signature__=inspect.signature(body), __prompt_template__=body.__doc__
        )
