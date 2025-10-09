import typing

from effectful.ops.syntax import defop


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
