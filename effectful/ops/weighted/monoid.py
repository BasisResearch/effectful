import collections.abc
import functools
import itertools
import numbers
import operator
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any

import jax
import jax.tree as tree

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax._handlers import is_eager_array
from effectful.ops.semantics import coproduct, evaluate, fvsof, handler
from effectful.ops.syntax import Scoped, deffn, defop
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


class Monoid[T]:
    kernel: Operation[[T, ...], T]
    identity: T

    def __init__(self, kernel: Callable[[T, ...], T], identity: T):
        self.kernel = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )
        self.identity = identity

    @classmethod
    def from_binary(cls, kernel: Callable[[T, T], T], identity: T):
        return cls(lambda *a: functools.reduce(kernel, a), identity)

    @functools.singledispatchmethod
    def __call__(self, *bodies: Body[T]) -> Body[T]:
        """Monoid addition with broadcasting over common collection types,
        callables, and interpretations.

        """
        return self.kernel(*bodies)

    # TODO: This case should be covered by a more general use of jax.tree or
    # similar.
    @__call__.register
    def _(self, a: Sequence, *bs: Sequence):
        result = [self(*vs) for vs in zip(*(a, *bs), strict=True)]
        return result

    # TODO: This case should be covered by a more general use of jax.tree or
    # similar.
    @__call__.register
    def _(self, a: Mapping, *bs: Mapping):
        all_values = collections.defaultdict(list)
        for d in (a, *bs):
            for k, v in d.items():
                all_values[k].append(v)
        result = {k: self(*vs) for (k, vs) in all_values.items()}
        return result

    @__call__.register
    def _(self, a: callable, *bs: callable):
        result = lambda *args, **kwargs: self(
            a(*args, **kwargs), *[b(*args, **kwargs) for b in bs]
        )
        return result

    @__call__.register
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


class Semilattice[T](IdempotentMonoid[T], CommutativeMonoid[T]): ...


# Semiring laws:
# (R, +) is a commutative monoid with identity 0
# (R, *) is a monoid with identity 1
# a * 0 = 0 = 0 * a
# a * (b + c) = (a * b) + (a * c)
# (b + c) * a = (b * a) + (c * a)


@defop
def add[T](a: T, b: T) -> T:
    if any(isinstance(x, Term) and not is_eager_array(x) for x in (a, b)):
        raise NotHandled
    return a + b  # type: ignore


@defop
def mul[T](a: T, b: T) -> T:
    if (isinstance(a, numbers.Number) and a == 0) or (
        isinstance(b, numbers.Number) and b == 0
    ):
        return 0  # type: ignore
    if any(isinstance(x, Term) and not is_eager_array(x) for x in (a, b)):
        raise NotHandled
    return a * b  # type: ignore


@defop
def min[T](a: T, b: T) -> T:
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotHandled
    return a if a < b else b  # type: ignore


@defop
def max[T](a: T, b: T) -> T:
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotHandled
    return a if a > b else b  # type: ignore


@defop
def arg_min(a, b):
    if (
        isinstance(a, tuple)
        and isinstance(a[0], numbers.Number)
        and a[0] == float("inf")
    ):
        return b
    if (
        isinstance(b, tuple)
        and isinstance(b[0], numbers.Number)
        and b[0] == float("inf")
    ):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotHandled
    return a if a[0] < b[0] else b


@defop
def arg_max(a, b):
    if (
        isinstance(a, tuple)
        and isinstance(a[0], numbers.Number)
        and a[0] == float("-inf")
    ):
        return b
    if (
        isinstance(b, tuple)
        and isinstance(b[0], numbers.Number)
        and b[0] == float("-inf")
    ):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotHandled
    return a if a[0] > b[0] else b


@defop
def logaddexp(a, b):
    return jnp.logaddexp(a, b)


@defop
def jax_cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


SumMonoid = CommutativeMonoid(operator.add, 0.0)
ProdMonoid = CommutativeMonoid(operator.mul, 1.0)
MinMonoid = Semilattice(min, float("inf"))
MaxMonoid = Semilattice(max, float("-inf"))
LogSumMonoid: Monoid[float] = Monoid(logaddexp, float("-inf"))
ArgMinMonoid: Monoid[tuple[float, Any]] = Monoid(arg_min, (float("inf"), None))
ArgMaxMonoid: Monoid[tuple[float, Any]] = Monoid(arg_max, (float("-inf"), None))
JaxCartesianProdMonoid: Monoid[jax.Array] = Monoid(jax_cartesian_prod, jnp.array([]))

StreamChainMonoid: Monoid[collections.abc.Generator] = Monoid(
    lambda a, b: (v for v in itertools.chain(a, b)),
    (),
)

# note: empty tuple is not a valid identity
StreamProdMonoid: Monoid[collections.abc.Generator] = Monoid(
    lambda a, b: ((v1, v2) for (v1, v2) in itertools.product(a, b)),
    (),
)


@dataclass
class _ExtensibleBinaryRelation[S, T]:
    tuples: set[tuple[S, T]]

    def register(self, s: S, t: T) -> None:
        self.tuples.add((s, t))

    def __call__(self, s: S, t: T) -> bool:
        return (s, t) in self.tuples


distributes_with = _ExtensibleBinaryRelation({})
