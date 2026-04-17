import collections.abc
import functools
import itertools
import numbers
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any

from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import Scoped, _NumberTerm, deffn
from effectful.ops.types import Interpretation, NotHandled, Operation, Term

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


@dataclass
class Monoid[T]:
    kernel: Operation[[T, T], T]
    add: Callable[[T, ...], T]
    identity: T

    @classmethod
    def from_binary(cls, kernel: Callable[[T, T], T], identity: T):
        kernel_op = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )

        def _add(*args: T) -> T:
            return functools.reduce(kernel_op, args)

        return cls(kernel_op, _add, identity)

    @classmethod
    def from_nary(cls, kernel: Callable[[T, ...], T], identity: T):
        kernel_op = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )
        return cls(kernel_op, kernel_op, identity)

    @functools.singledispatchmethod
    def __call__(self, a: Body[T], *bs: Body[T]) -> Body[T]:
        """Monoid addition with broadcasting over common collection types,
        callables, and interpretations.

        """
        if callable(a):
            result = lambda *args, **kwargs: self(
                a(*args, **kwargs), *[b(*args, **kwargs) for b in bs]
            )
            return result
        return self.add(a, *bs)

    # TODO: This case should be covered by a more general use of jax.tree or
    # similar.
    @__call__.register(Sequence)
    def _(self, a: Sequence, *bs: Sequence):
        result = [self(*vs) for vs in zip(*(a, *bs), strict=True)]
        return type(a)(result)

    # TODO: This case should be covered by a more general use of jax.tree or
    # similar.
    @__call__.register(Mapping)
    def _(self, a: Mapping, *bs: Mapping):
        all_values = collections.defaultdict(list)
        for d in (a, *bs):
            assert isinstance(d, Mapping)
            for k, v in d.items():
                all_values[k].append(v)
        result = {k: self(*vs) for (k, vs) in all_values.items()}
        return result

    @__call__.register(Interpretation)
    def _(self, a: Interpretation, *bs: Interpretation):
        a_keys = a.keys()
        for b in bs:
            b_keys = b.keys()
            if not a_keys == b_keys:
                raise ValueError(
                    f"Expected interpretation of {a_keys} but got {b_keys}"
                )

        result = {
            k: self(handler(a)(a[k]), *[handler(b)(b[k]) for b in bs]) for k in a_keys
        }
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
        return functools.reduce(self, itertools.repeat(v, x))

    @Operation.define
    def reduce[A, B, U: Body](
        self, streams: Annotated[Streams, Scoped[A]], body: Annotated[U, Scoped[A | B]]
    ) -> Annotated[U, Scoped[B]]:

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
                return evaluate(body, intp=intp)

        loop_order = list(order_streams(streams))
        values = (body_value(body, intp) for intp in generator(loop_order))
        result = self(*values)  # type: ignore
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

    @classmethod
    def from_binary(cls, kernel: Callable[[T, T], T], identity: T, zero: T):
        kernel_op = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )

        def _add(*args: T) -> T:
            return functools.reduce(kernel_op, args)

        return cls(kernel_op, _add, identity, zero)

    @classmethod
    def from_nary(cls, kernel: Callable[[T, ...], T], identity: T, zero: T):
        kernel_op = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )
        return cls(kernel_op, kernel_op, identity, zero)


class Semilattice[T](IdempotentMonoid[T], CommutativeMonoid[T]): ...


@Operation.define
def _min[T](*args: T) -> T:
    if any(isinstance(x, Term) for x in args):
        raise NotHandled
    return min(*args)


Min = Semilattice.from_nary(_min, float("inf"))


@Operation.define
def _max[T](*args: T) -> T:
    if any(isinstance(x, Term) for x in args):
        raise NotHandled
    return max(*args)


Max = Semilattice.from_nary(_max, float("-inf"))


@Operation.define
def _arg_min[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] < a[0] else a


ArgMin: Monoid[tuple[float, Any]] = Monoid.from_binary(_arg_min, (float("inf"), None))


@Operation.define
def _arg_max[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] > a[0] else a


ArgMax: Monoid[tuple[float, Any]] = Monoid.from_binary(_arg_max, (float("-inf"), None))

Sum = CommutativeMonoid.from_binary(_NumberTerm.__add__, 0)
Product = CommutativeMonoidWithZero.from_binary(_NumberTerm.__mul__, 1, 0)


StreamChain: Monoid[collections.abc.Generator] = Monoid.from_binary(
    lambda a, b: (v for v in itertools.chain(a, b)), ()
)

# note: empty tuple is not a valid identity
StreamProd: Monoid[collections.abc.Generator] = Monoid.from_binary(
    lambda a, b: ((v1, v2) for (v1, v2) in itertools.product(a, b)), ()
)


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
