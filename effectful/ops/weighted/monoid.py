import collections.abc
import functools
import itertools
import numbers
import typing
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any, Protocol

from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import Scoped, _IteratorTerm, _NumberTerm
from effectful.ops.types import Interpretation, NotHandled, Operation, Term

# Note: The streams value type should be something like Iterable[T], but some of
# our target stream types (e.g. jax.Array) are not subtypes of Iterable
type Streams[T] = Mapping[Operation[[], T], Any]

type Body[T] = (
    Iterable[T]
    | Callable[..., Body[T]]
    | T
    | Mapping[Any, Body[T]]
    | Interpretation[T, Body[T]]
)


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


class Add[T](Protocol):
    def __call__(self, *args: T) -> T: ...


class Monoid[T]:
    kernel: Operation[[T, T], T]
    add: Add[T]
    identity: T

    def __init__(
        self, kernel: Callable[[T, T], T], identity: T, add: Add[T] | None = None
    ):
        self.identity = identity
        self.kernel = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )

        if add is None:

            def _add(*args: T) -> T:
                return functools.reduce(self.kernel, args)

            self.add = _add
        else:
            self.add = add

    @functools.singledispatchmethod
    def __call__[S: Body[T]](self, a: S, *bs: S) -> S:
        """Monoid addition with broadcasting over common collection types,
        callables, and interpretations.

        """
        if callable(a):
            for b in bs:
                if not callable(b):
                    raise TypeError(f"Expected callable but got {b}")

            result = lambda *args, **kwargs: self(
                a(*args, **kwargs),
                *[b(*args, **kwargs) for b in bs],  # type: ignore[operator]
            )
            return typing.cast(S, result)

        # Base case: a: T, *bs: T
        return typing.cast(
            S, self.add(typing.cast(T, a), *typing.cast(tuple[T, ...], bs))
        )

    # TODO: This case should be covered by a more general use of jax.tree or
    # similar.
    @__call__.register(Sequence)
    def _(self, a: Sequence, *bs: Sequence):
        result = [self(*vs) for vs in zip(*(a, *bs), strict=True)]
        return type(a)(result)  # type: ignore[call-arg]

    # TODO: This case should be covered by a more general use of jax.tree or
    # similar.
    @__call__.register(Mapping)
    def _(self, a: Mapping, *bs: Mapping):
        # singledispatch doesn't recognize Interpretation as a more specific type than Mapping
        if isinstance(a, Interpretation):
            for b in bs:
                if not isinstance(b, Interpretation):
                    raise TypeError(f"Expected interpretation but got {b}")

            a_keys = a.keys()
            for b in bs:
                b_keys = b.keys()
                if not a_keys == b_keys:
                    raise ValueError(
                        f"Expected interpretation of {a_keys} but got {b_keys}"
                    )

            result = {
                k: self(handler(a)(a[k]), *[handler(b)(b[k]) for b in bs])
                for k in a_keys
            }
            return result
        else:
            all_values = collections.defaultdict(list)
            for d in (a, *bs):
                assert isinstance(d, Mapping)
                for k, v in d.items():
                    all_values[k].append(v)
            result = {k: self(*vs) for (k, vs) in all_values.items()}
            return result

    def scalar_mul(self, v: T, x: int) -> T:
        """
        Returns the scalar multiplication w.r.t. a monoid.

        The scalar multiplication of a monoid (A, ⊕, e) is
        a function (⋅): A × ℕ → A, inductively defined as
            a⋅0 = e
            a⋅n = a⋅(n-1) + a

        Warning: scalar multiplication is not commutative,
            the scalar is always the second argument.
        """
        if x < 0:
            raise ValueError("Expected x >= 0")
        if x == 0:
            return self.identity
        return self(*itertools.repeat(v, x))

    @Operation.define
    def reduce[A, B, U: Body](
        self, streams: Annotated[Streams, Scoped[A]], body: Annotated[U, Scoped[A | B]]
    ) -> Annotated[U, Scoped[B]]:

        def generator(loop_order):
            if loop_order:
                stream_key = loop_order[0]
                stream_values = evaluate(streams[stream_key])
                stream_values_iter = iter(stream_values)

                # If we try to iterate and get a term instead of a real
                # iterator, give up
                if (
                    isinstance(stream_values_iter, Term)
                    and stream_values_iter.op is _IteratorTerm.__iter__
                ):
                    raise NotHandled

                for val in stream_values:
                    intp = {stream_key: functools.partial(lambda v: v, val)}
                    with handler(intp):
                        for intp2 in generator(loop_order[1:]):
                            yield coproduct(intp, intp2)
            else:
                yield {}

        def body_value(body: Body, intp: Interpretation) -> Body:
            if isinstance(body, Interpretation):
                # TODO: This should be a product, but the implementation of product isn't quite correct.
                return {
                    op: handler(coproduct(intp, body))(impl)
                    for op, impl in body.items()
                }
            elif callable(body):
                return handler(intp)(body)
            elif isinstance(body, Mapping):
                return {k: body_value(v, intp) for (k, v) in body.items()}
            elif isinstance(body, Generator):
                return (body_value(v, intp) for v in body)
            else:
                return handler(intp)(evaluate)(body)

        loop_order = list(order_streams(streams))
        values = [body_value(body, intp) for intp in generator(loop_order)]
        result = self(*values)
        return result


class IdempotentMonoid[T](Monoid[T]):
    def scalar_mul(self, v: T, x: int) -> T:
        if x < 0:
            raise ValueError("Expected x >= 0")
        if x == 0:
            return self.identity
        return v


class CommutativeMonoid[T](Monoid[T]): ...


@dataclass
class CommutativeMonoidWithZero[T](CommutativeMonoid[T]):
    zero: T

    def __init__(
        self,
        kernel: Callable[[T, T], T],
        identity: T,
        zero: T,
        add: Add[T] | None = None,
    ):
        super().__init__(kernel, identity, add=add)
        self.zero = zero


class Semilattice[T](IdempotentMonoid[T], CommutativeMonoid[T]): ...


class SupportsRichComparison(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...


@Operation.define
def _min[T: SupportsRichComparison](*args: T) -> T:
    if any(isinstance(x, Term) for x in args):
        raise NotHandled
    return min(*args)


Min = Semilattice(_min, identity=float("inf"), add=_min)


@Operation.define
def _max[T: SupportsRichComparison](*args: T) -> T:
    if any(isinstance(x, Term) for x in args):
        raise NotHandled
    return max(*args)


Max = Semilattice(_max, identity=float("-inf"), add=_max)


@Operation.define
def _arg_min[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] < a[0] else a  # type: ignore


ArgMin: Monoid[tuple[float, Any]] = Monoid(_arg_min, identity=(float("inf"), None))


@Operation.define
def _arg_max[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] > a[0] else a  # type: ignore


ArgMax: Monoid[tuple[float, Any]] = Monoid(_arg_max, identity=(float("-inf"), None))


class _SumMonoid[N: int | float | complex](CommutativeMonoid[N]):
    def scalar_mul(self, v: N, x: int) -> N:
        return typing.cast(N, v * x)


Sum = _SumMonoid(_NumberTerm.__add__, identity=0)


class _ProductMonoid[N: int | float | complex](CommutativeMonoidWithZero[N]):
    def scalar_mul(self, v: N, x: int) -> N:
        return typing.cast(N, v**x)


Product = _ProductMonoid(_NumberTerm.__mul__, identity=1, zero=0)


@dataclass
class _ExtensibleBinaryRelation[S, T]:
    tuples: set[tuple[S, T]]

    def register(self, s: S, t: T) -> None:
        self.tuples.add((s, t))

    def __call__(self, s: S, t: T) -> bool:
        return (s, t) in self.tuples


distributes_over = _ExtensibleBinaryRelation(
    {
        (max, min),
        (min, max),
        (_NumberTerm.__add__, min),
        (_NumberTerm.__add__, max),
        (_NumberTerm.__mul__, _NumberTerm.__add__),
    }
)
