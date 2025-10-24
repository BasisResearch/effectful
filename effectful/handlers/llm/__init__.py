import dataclasses
import functools
import inspect
from collections.abc import Callable, Iterable

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __signature__: inspect.Signature
    __prompt_template__: str
    tools: tuple[Operation, ...]

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotHandled

    def __get__(self, instance, _owner):
        if instance is not None:
            return functools.partial(self, instance)
        else:
            return self

    @classmethod
    def define(cls, _func=None, *, tools: Iterable[Operation] = ()):
        def decorator(body: Callable[P, T]):
            if not body.__doc__:
                raise ValueError("Expected a docstring on body")

            return cls(
                __signature__=inspect.signature(body),
                __prompt_template__=body.__doc__,
                tools=tuple(tools),
            )

        if _func is None:
            return decorator
        return decorator(_func)
