import functools
from collections.abc import Callable, Generator, Iterable, Mapping
from graphlib import TopologicalSorter
from typing import Annotated, Any

from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import ObjectInterpretation, Scoped, deffn, defop, implements
from effectful.ops.types import Interpretation, NotHandled, Operation

from .monoid import Monoid

# Note: The streams value type should be something like Iterable[T], but some of
# our target stream types (e.g. jax.Array) are not subtypes of Iterable
type Streams[T] = Mapping[Operation[[], T], Any]

type Body[T] = (
    Iterable[T]
    | Callable[..., T]
    | T
    | Mapping[Any, Body[T]]
    | Interpretation[T, Body[T]]
)


@defop
def reduce[A, B, S, U: Body](
    monoid: Monoid[S],
    streams: Annotated[Streams, Scoped[A]],
    body: Annotated[U, Scoped[A | B]],
) -> Annotated[U, Scoped[B]]:
    raise NotHandled


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


def order_streams[T](streams: Streams[T]) -> Iterable[Operation[[], T]]:
    """Determine an order to evaluate the streams based on their dependencies"""
    stream_vars = set(streams.keys())
    dependencies = {k: fvsof(v) & stream_vars for k, v in streams.items()}
    topo = TopologicalSorter(dependencies)
    topo.prepare()
    while topo.is_active():
        node_group = topo.get_ready()
        yield from sorted(node_group, key=str)
        topo.done(*node_group)


class BaselineReduce(ObjectInterpretation):
    @implements(reduce)
    def reduce[T](self, monoid: Monoid[T], streams: Streams[T], body: Body[T]) -> Body[T]:
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

        loop_order = list(order_streams(streams))
        values = (_body_value(body, intp) for intp in generator(loop_order))
        result = functools.reduce(monoid.add, values)  # type: ignore
        return result
