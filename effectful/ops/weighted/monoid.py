import collections.abc
import functools
import itertools
import numbers
import typing
from collections import defaultdict
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


@dataclass
class Monoid[T]:
    kernel: Operation[[T, ...], T]
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

    @classmethod
    def from_binary(cls, kernel: Callable[[T, T], T], identity: T):
        kernel_op = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )

        def _add(*args: T) -> T:
            return functools.reduce(kernel_op, args)

        return cls(kernel_op, _add, identity)

    @classmethod
    def from_nary(cls, kernel: Add[T], identity: T):
        kernel_op = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )
        return cls(kernel_op, kernel_op, identity)

    def __call__[S: Body[T]](self, x: S, *xs: S) -> S:
        return self.plus(x, *xs)

    @Operation.define
    @functools.singledispatchmethod
    def plus[S: Body[T]](self, a: S, *bs: S) -> S:
        """Monoid addition with broadcasting over common collection types,
        callables, and interpretations.

        """
        # eliminate identity elements
        bs = tuple(b for b in bs if b is not self.identity)

        # eliminate singleton plus
        if not bs:
            return a

        # eliminate nested plus
        breakpoint()
        args = []
        for t in (a, *bs):
            if isinstance(t, Term) and t.op is self.plus:
                args += t.args
            else:
                args.append(t)
        a = args[0]
        bs = args[1:]

        if callable(a):
            for b in bs:
                if not callable(b):
                    raise TypeError(f"Expected callable but got {b}")

            result = lambda *args, **kwargs: self.plus(
                a(*args, **kwargs),
                *[b(*args, **kwargs) for b in bs],  # type: ignore[operator]
            )
            return typing.cast(S, result)

        # group terms by head operation
        by_head_op = defaultdict(list)
        for t in (a, *bs):
            by_head_op[t.op].append(t)

        # distribute over each group
        final_sum = []
        progress = False
        for op, terms in by_head_op.items():
            if distributes_over(self.plus, op):
                progress = True

                term_args = tuple(t.args for t in terms)
                dist_terms = tuple(
                    self.plus(*args) for args in itertools.product(*term_args)
                )
                final_sum.append(op(*dist_terms))
            else:
                final_sum += terms
        if progress:
            return self.plus(*final_sum)
        raise NotHandled

        # Base case: a: T, *bs: T
        return typing.cast(
            S, self.kernel(typing.cast(T, a), *typing.cast(tuple[T, ...], bs))
        )

    # TODO: This case should be covered by a more general use of jax.tree or
    # similar.
    @plus.register
    def _(self, a: Sequence, *bs: Sequence):
        result = [self(*vs) for vs in zip(*(a, *bs), strict=True)]
        return type(a)(result)  # type: ignore[call-arg]

    # TODO: This case should be covered by a more general use of jax.tree or
    # similar.
    @plus.register
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

    @classmethod
    def from_binary_with_zero(cls, kernel: Callable[[T, T], T], identity: T, zero: T):
        kernel_op = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )

        def _add(*args: T) -> T:
            return functools.reduce(kernel_op, args)

        return cls(kernel_op, _add, identity, zero)

    @classmethod
    def from_nary_with_zero(cls, kernel: Add[T], identity: T, zero: T):
        kernel_op = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )
        return cls(kernel_op, kernel_op, identity, zero)


class Semilattice[T](IdempotentMonoid[T], CommutativeMonoid[T]): ...


class SupportsRichComparison(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...


@Operation.define
def _min[T: SupportsRichComparison](*args: T) -> T:
    if any(isinstance(x, Term) for x in args):
        raise NotHandled
    return min(*args)


Min = Semilattice.from_nary(_min, float("inf"))


@Operation.define
def _max[T: SupportsRichComparison](*args: T) -> T:
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
    return b if b[0] < a[0] else a  # type: ignore


ArgMin: Monoid[tuple[float, Any]] = Monoid.from_binary(_arg_min, (float("inf"), None))


@Operation.define
def _arg_max[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] > a[0] else a  # type: ignore


ArgMax: Monoid[tuple[float, Any]] = Monoid.from_binary(_arg_max, (float("-inf"), None))


class _SumMonoid[N: int | float | complex](CommutativeMonoid[N]):
    def scalar_mul(self, v: N, x: int) -> N:
        return typing.cast(N, v * x)


Sum = _SumMonoid.from_binary(_NumberTerm.__add__, 0)


class _ProductMonoid[N: int | float | complex](CommutativeMonoidWithZero[N]):
    def scalar_mul(self, v: N, x: int) -> N:
        return typing.cast(N, v**x)


Product = _ProductMonoid.from_binary_with_zero(_NumberTerm.__mul__, 1, 0)


@dataclass
class _ExtensibleBinaryRelation[S, T]:
    tuples: set[tuple[S, T]]

    def register(self, s: S, t: T) -> None:
        self.tuples.add((s, t))

    def __call__(self, s: S, t: T) -> bool:
        return (s, t) in self.tuples


distributes_over = _ExtensibleBinaryRelation(
    {
        (Max.plus, Min.plus),
        (Min.plus, Max.plus),
        (Sum.plus, Min.plus),
        (Sum.plus, Max.plus),
        (Product.plus, Sum.plus),
    }
)
