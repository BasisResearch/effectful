import dataclasses
import inspect
import typing
from collections.abc import Callable

from effectful.ops.syntax import defop


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __signature__: inspect.Signature
    __prompt_template__: str

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotImplementedError

    @staticmethod
    def define(body: Callable[P, T]) -> "Template[P, T]":
        if not body.__doc__:
            raise ValueError("Expected a docstring on body")

        return Template(inspect.signature(body), body.__doc__)


class DecodeError(RuntimeError):
    """Raised when decoding an LLM response fails."""

    def __init__(self, t: type, response: str):
        super().__init__()
        self.type_ = t
        self.response = response

    def __repr__(self):
        return f"DecodeError({self.type_}, {self.response})"


@defop
def decode[T](t: type[T], content: str) -> T:
    """Decode `content` as an instance of `t`. Used to consume the output of an
    LLM.

    """
    if t is str:
        return typing.cast(T, content)
    elif t is bool:
        match content.strip().lower():
            case "true":
                return typing.cast(T, True)
            case "false":
                return typing.cast(T, False)
            case _:
                raise DecodeError(t, content)
    elif t in (int, float, complex, bool):
        try:
            result = t(content)  # type: ignore
        except ValueError:
            raise DecodeError(t, content)
        return typing.cast(T, result)

    raise DecodeError(t, content)
