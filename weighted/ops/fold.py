import functools
from collections.abc import Callable, Generator, Iterable, Mapping
from graphlib import TopologicalSorter
from typing import Annotated, Any

import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import ObjectInterpretation, Scoped, deffn, defop, implements
from effectful.ops.types import Interpretation, Operation

from .monoid import Monoid

type Streams[T] = Mapping[Operation[[], T], Iterable[T]]

type Body[T] = (
    Iterable[T]
    | Callable[..., T]
    | T
    | Mapping[Any, Body[T]]
    | Interpretation[T, Body[T]]
)


@defop
def fold[A, B, S, T, U: Body](
    monoid: Monoid[S],
    streams: Annotated[Streams[T], Scoped[A]],
    body: Annotated[U, Scoped[A | B]],
) -> Annotated[U, Scoped[B]]:
    raise NotImplementedError


def _body_value(body: Body, intp: Interpretation) -> Body:
    if isinstance(body, Interpretation):
        # TODO: This should be a product, but the implementation of product isn't quite correct.
        return {op: handler(coproduct(intp, body))(impl) for op, impl in body.items()}
    elif callable(body):
        return handler(intp)(body)
    elif isinstance(body, Mapping):
        return {k: _body_value(v, intp) for (k, v) in body.items()}
    elif isinstance(body, Generator):
        return (_body_value(v, intp) for v in body)
    else:
        return evaluate(body, intp=intp)


def order_streams[T](streams: Streams[T]) -> list[Operation[[], T]]:
    """Determine an order to evaluate the streams based on their dependencies"""
    stream_vars = set(streams.keys())
    topo = TopologicalSorter({k: fvsof(v) & stream_vars for k, v in streams.items()})
    loop_order = list(topo.static_order())
    return loop_order


class BaselineFold(ObjectInterpretation):
    @implements(fold)
    def fold[T](self, monoid: Monoid[T], streams: Streams[T], body: Body[T]) -> Body[T]:
        def generator(loop_order):
            if loop_order:
                stream_key = loop_order[0]
                stream_values = evaluate(streams[stream_key])
                for val in stream_values:
                    intp = {stream_key: deffn(val)}
                    with handler(intp):
                        for intp2 in generator(loop_order[1:]):
                            yield coproduct(intp, intp2)
            else:
                yield {}

        loop_order = order_streams(streams)
        values = (_body_value(body, intp) for intp in generator(loop_order))
        result = functools.reduce(monoid.add, values)  # type: ignore
        return result
