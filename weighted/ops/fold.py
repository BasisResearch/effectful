import collections.abc
import functools
import itertools
from collections.abc import Callable, Generator, Mapping
from typing import Annotated, Any, TypeAlias, TypeVar

import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import ObjectInterpretation, Scoped, deffn, defop, implements
from effectful.ops.types import Operation

from .semiring import Semiring

S = TypeVar("S")
T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")

Runner: TypeAlias = Mapping[Operation[..., T], collections.abc.Iterable[T]]


@defop
def fold(
    semiring: Semiring[S],
    streams: Annotated[Runner, Scoped[A]],
    body: Annotated[S, Scoped[A | B]],
) -> Annotated[S, Scoped[B]]:
    raise NotImplementedError


Body: TypeAlias = Generator[T] | Callable[..., T] | T | Mapping[Any, "Body"]


def _promote_add(add, a, b):
    if isinstance(b, Generator) or isinstance(a, Generator):
        a = a if isinstance(a, Generator) else (a,)
        b = b if isinstance(b, Generator) else (b,)
        return (v for v in (*a, *b))
    elif isinstance(b, Mapping):
        result = {
            k: a[k]
            if k not in b
            else b[k]
            if k not in a
            else _promote_add(add, a[k], b[k])
            for k in set(a) | set(b)
        }
        return result
    elif isinstance(b, Callable):
        return lambda *args, **kwargs: _promote_add(
            add, a(*args, **kwargs), b(*args, **kwargs)
        )
    else:
        return add(a, b)


class BaselineFold(ObjectInterpretation):
    @implements(fold)
    def fold(self, semiring: Semiring[T], streams: Runner, body: Body[T]) -> Body[T]:
        def generator():
            stream_values = list(streams.values())
            all_vals = itertools.product(*stream_values)
            for vals in all_vals:
                keys = list(streams.keys())
                with handler({k: deffn(v) for (k, v) in zip(keys, vals, strict=True)}):
                    yield evaluate(body)

        return functools.reduce(
            functools.partial(_promote_add, semiring.add), generator()
        )
