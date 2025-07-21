import functools
from collections.abc import Callable, Generator, Iterable, Mapping
from graphlib import TopologicalSorter
from typing import Annotated, Any

import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import ObjectInterpretation, Scoped, deffn, defop, implements
from effectful.ops.types import Interpretation, Operation

from .semiring import Semiring

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
    semiring: Semiring[S],
    streams: Annotated[Streams[T], Scoped[A]],
    body: Annotated[U, Scoped[A | B]],
) -> Annotated[U, Scoped[B]]:
    raise NotImplementedError


def _promote_add(add, a: Body, b: Body) -> Body:
    if isinstance(a, Generator):
        assert isinstance(b, Generator)
        return (v for v in (*a, *b))
    elif isinstance(a, Mapping):
        assert isinstance(b, Mapping)
        result = {
            k: a[k]
            if k not in b
            else b[k]
            if k not in a
            else _promote_add(add, a[k], b[k])
            for k in set(a) | set(b)
        }
        return result
    elif callable(a):
        assert callable(b)
        return lambda *args, **kwargs: _promote_add(
            add, a(*args, **kwargs), b(*args, **kwargs)
        )
    elif isinstance(a, Interpretation):
        assert isinstance(b, Interpretation)
        assert a.keys() == b.keys()
        result = {k: _promote_add(add, handler(a)(a[k]), handler(b)(b[k])) for k in a}
        return result
    else:
        return add(a, b)


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


class BaselineFold(ObjectInterpretation):
    @implements(fold)
    def fold[T](
        self, semiring: Semiring[T], streams: Streams[T], body: Body[T]
    ) -> Body[T]:
        # Determine an order to evaluate the streams based on their dependencies
        topo = TopologicalSorter(
            {k: set(fvsof(v)) & set(streams) for k, v in streams.items()}
        )
        loop_order = list(topo.static_order())

        def generator(loop_order):
            if loop_order:
                stream_key = loop_order[0]
                stream_values = streams[stream_key]
                for val in stream_values:
                    intp = {stream_key: deffn(val)}
                    with handler(intp):
                        for intp2 in generator(loop_order[1:]):
                            yield coproduct(intp, intp2)
            else:
                yield {}

        values = (_body_value(body, intp) for intp in generator(loop_order))
        result = functools.reduce(functools.partial(_promote_add, semiring.add), values)
        return result
