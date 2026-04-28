import collections.abc
import functools
import itertools
import numbers
import typing
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any, Protocol

from effectful.internals.disjoint_set import DisjointSet
from effectful.ops.semantics import fvsof, handler, typeof
from effectful.ops.syntax import (
    Scoped,
    _NumberTerm,
    defdata,
    syntactic_eq,
    syntactic_hash,
)
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


def order_streams[T](streams: Streams[T]) -> Iterable[tuple[Operation[[], T], Any]]:
    """Determine an order to evaluate the streams based on their dependencies"""
    stream_vars = set(streams.keys())
    dependencies = {k: fvsof(v) & stream_vars for k, v in streams.items()}
    topo = TopologicalSorter(dependencies)
    topo.prepare()
    while topo.is_active():
        node_group = topo.get_ready()
        for op in sorted(node_group):
            yield (op, streams[op])
        topo.done(*node_group)


def transitive_dependencies(graph: dict, dependents: set) -> set:
    """
    Compute the transitive dependencies of a set of dependents.

    Args:
        graph: Dict mapping each dependent to an iterable of its direct dependencies.
        dependents: Set of dependents whose transitive dependencies to compute.

    Returns:
        Set of all transitive dependencies (excluding the input dependents themselves
        unless they appear as dependencies of other items in the input set).
    """
    result = set()
    stack = [dep for d in dependents for dep in graph.get(d, ())]

    while stack:
        node = stack.pop()
        if node in result:
            continue
        result.add(node)
        stack.extend(graph.get(node, ()))

    return result


def independent_terms(factors: Term, vs: set[Operation]) -> Iterable[set[Operation]]:
    var_ids = {v: i for (i, v) in enumerate(vs)}
    var_sets = DisjointSet(len(vs))

    for factor in factors:
        var_sets.union(*(var_ids[v] for v in (fvsof(factor) & vs)))

    result_sets = {}
    for v, i in var_ids.items():
        set_i = var_sets.find(i)
        if set_i in result_sets:
            result_sets[set_i].add(v)
        else:
            result_sets[set_i] = {v}

    return result_sets.values()


class Add[T](Protocol):
    def __call__(self, *args: T) -> T: ...

    # @Operation.define
    # def reduce[A, B, U: Body](
    #     self, streams: Annotated[Streams, Scoped[A]], body: Annotated[U, Scoped[A | B]]
    # ) -> Annotated[U, Scoped[B]]:

    #     def generator(loop_order):
    #         if loop_order:
    #             stream_key = loop_order[0]
    #             stream_values = evaluate(streams[stream_key])
    #             stream_values_iter = iter(stream_values)

    #             # If we try to iterate and get a term instead of a real
    #             # iterator, give up
    #             if (
    #                 isinstance(stream_values_iter, Term)
    #                 and stream_values_iter.op is _IteratorTerm.__iter__
    #             ):
    #                 raise NotHandled

    #             for val in stream_values:
    #                 intp = {stream_key: functools.partial(lambda v: v, val)}
    #                 with handler(intp):
    #                     for intp2 in generator(loop_order[1:]):
    #                         yield coproduct(intp, intp2)
    #         else:
    #             yield {}

    #     def body_value(body: Body, intp: Interpretation) -> Body:
    #         if isinstance(body, Interpretation):
    #             # TODO: This should be a product, but the implementation of product isn't quite correct.
    #             return {
    #                 op: handler(coproduct(intp, body))(impl)
    #                 for op, impl in body.items()
    #             }
    #         elif callable(body):
    #             return handler(intp)(body)
    #         elif isinstance(body, Mapping):
    #             return {k: body_value(v, intp) for (k, v) in body.items()}
    #         elif isinstance(body, Generator):
    #             return (body_value(v, intp) for v in body)
    #         else:
    #             return handler(intp)(evaluate)(body)

    #     loop_order = list(order_streams(streams))
    #     values = [body_value(body, intp) for intp in generator(loop_order)]
    #     result = self(*values)
    #     return result


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

    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        """Monoid addition with broadcasting over common collection types,
        callables, and interpretations.

        """
        if not args:
            return self.identity

        # elim single arg plus
        if len(args) == 1:
            return args[0]

        # elim identity
        if any(x is self.identity for x in args):
            return self.plus(*(x for x in args if x is not self.identity))

        # elim associativity
        if any(isinstance(x, Term) and x.op is self.plus for x in args):
            flat_args = sum(
                (
                    t.args if isinstance(t, Term) and t.op is self.plus else (t,)
                    for t in args
                ),
                start=(),
            )
            assert len(args) > 0
            return self.plus(*flat_args)

        # elim distributivity
        if any(isinstance(x, Term) and distributes_over(self.plus, x.op) for x in args):
            # group terms by head operation
            by_head_op = defaultdict(list)
            for t in args:
                by_head_op[t.op].append(t)

            # distribute over each group
            progress = False
            final_sum = []
            for op, terms in by_head_op.items():
                if (
                    len(terms) > 1
                    and distributes_over(self.plus, op)
                    and not distributes_over(op, self.plus)
                ):
                    progress = True
                    term_args = (t.args for t in terms)
                    dist_terms = (
                        self.plus(*args) for args in itertools.product(*term_args)
                    )
                    final_sum.append(op(*dist_terms))
                else:
                    final_sum += terms
            if progress:
                return self.plus(*final_sum)

        if any(isinstance(x, Term) for x in args):
            return defdata(self.plus, *args)

        if callable(args[0]):
            for b in args[1:]:
                if not callable(b):
                    raise TypeError(f"Expected callable but got {b}")

            result = lambda *args, **kwargs: self.plus(
                *(x(*args, **kwargs) for x in args),  # type: ignore[operator]
            )
            return typing.cast(S, result)

        if isinstance(args[0], Sequence):
            return type(args[0])(self.plus(*vs) for vs in zip(*args, strict=True))

        if isinstance(args[0], Interpretation):
            keys = args[0].keys()

            for b in args[1:]:
                if not isinstance(b, Interpretation):
                    raise TypeError(f"Expected interpretation but got {b}")

                b_keys = b.keys()
                if not keys == b_keys:
                    raise ValueError(
                        f"Expected interpretation of {keys} but got {b_keys}"
                    )

            result = {k: self.plus(*(handler(b)(b[k]) for b in args)) for k in keys}
            return result

        if isinstance(args[0], Mapping):
            for b in args[1:]:
                if not isinstance(b, Mapping):
                    raise TypeError(f"Expected mapping but got {b}")

            all_values = collections.defaultdict(list)
            for d in args:
                for k, v in d.items():
                    all_values[k].append(v)
            result = {k: self.plus(*vs) for (k, vs) in all_values.items()}
            return result

        # Base case: a: T, *bs: T
        return typing.cast(S, self.kernel(*args))

    @Operation.define
    @functools.singledispatchmethod
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        # elim empty streams dict
        if not streams:
            return self.identity

        # elim empty stream
        if any(
            issubclass(typeof(v), Sequence)
            and not isinstance(iter(v), Term)
            and len(v) == 0
            for v in streams.values()
        ):
            return self.identity

        # elim reduce body
        if isinstance(body, Term) and body.op == self.reduce:
            return self.reduce(body.args[0], streams | body.args[1])

        if callable(body):
            return lambda *a, **k: self.reduce(body(*a, *k), streams)

        return defdata(self.reduce, body, streams)

    @reduce.register
    def _(self, body: Mapping, streams):
        return {k: self.reduce(v, streams) for (k, v) in body.items()}

    @reduce.register
    def _(self, body: Sequence, streams):
        return type(body)(self.reduce(x, streams) for x in body)

    @reduce.register
    def _(self, body: Generator, streams):
        return (self.reduce(x, streams) for x in body)

    @Operation.define
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
        if isinstance(x, Term):
            return defdata(self.scalar_mul, v, x)

        if x < 0:
            raise ValueError("Expected x >= 0")
        if x == 0:
            return self.identity
        return self.plus(*itertools.repeat(v, x))


class IdempotentMonoid[T](Monoid[T]):
    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        # elim consecutive duplicates
        dedup_args = [
            args[i]
            for i in range(len(args))
            if i == 0 or not syntactic_eq(args[i - 1], args[i])
        ]
        return super().plus(*dedup_args)

    @Operation.define
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        # elim unused streams
        stream_deps = {k: fvsof(v) for (k, v) in streams.items()}
        body_vars = fvsof(body)
        used_streams = (
            transitive_dependencies(stream_deps, body_vars) | body_vars
        ) & set(streams.keys())

        if len(used_streams) == 0:
            return body

        if len(used_streams) < len(streams):
            return super().reduce(
                body, {k: v for (k, v) in streams.items() if k in used_streams}
            )

        return super().reduce(body, streams)

    @Operation.define
    def scalar_mul(self, v: T, x: int) -> T:
        if isinstance(x, Term):
            return defdata(self.scalar_mul, v, x)

        if x < 0:
            raise ValueError("Expected x >= 0")
        if x == 0:
            return self.identity
        return v


class CommutativeMonoid[T](Monoid[T]):
    @Operation.define
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        # elim plus body
        if isinstance(body, Term) and body.op == self.plus:
            return self.plus(*(self.reduce(x, streams) for x in body.args))

        # elim unused streams
        #
        # Implements the identity
        #     reduce(R, S × S', body) = |S'| ⋅ reduce(R, S, body)
        #         where fvsof(body) ∩ S' = ∅
        #         and `⋅` is the scalar product of the monoid addition
        stream_deps = {k: fvsof(v) for (k, v) in streams.items()}
        body_vars = fvsof(body)
        stream_vars = streams.keys()
        used_stream_vars = (
            transitive_dependencies(stream_deps, body_vars) | body_vars
        ) & set(stream_vars)

        if len(used_stream_vars) < len(streams) and not (self is Sum and body == 1):
            used_streams = {k: v for (k, v) in streams.items() if k in used_stream_vars}
            unused_streams = {
                k: v for (k, v) in streams.items() if k not in used_stream_vars
            }

            if len(used_stream_vars) > 0:
                result = super().reduce(body, used_streams)
            else:
                result = body
            mult = Sum.reduce(1, unused_streams)
            return self.scalar_mul(result, mult)

        # Elim factors
        #
        # Implements factorization of independent terms.
        #
        # For example, when having two independent distributions, we can rewrite
        # their marginalization as: ∫p(x)⋅q(y)dxdy => ∫p(x)dx ⋅ ∫q(y)dy
        #
        # More specifically, in terms of reduces we are performing:
        #     reduce(R, (S₁ × ... × Sₖ), A₁ * ... * Aₖ)
        #     => reduce(R, S₁, A₁) * ... * reduce(R, Sₖ, Aₖ)
        #     where free(Aᵢ) ∩ free(Aⱼ) ∩ S = ∅
        #       and free(Aᵢ) ∩ Sᵢ ≠ ∅
        #
        # (The implementation is a little more general than this, as each
        # independent component can have an arbitrary number of streams.)
        if isinstance(body, Term) and distributes_over(body.op, self.plus):
            factors = body.args

            stream_ids = {v: i for (i, v) in enumerate(stream_vars)}
            ds = DisjointSet(len(streams))

            # streams are in the same partition as their dependencies
            for stream_id, (stream_var, stream_body) in enumerate(streams.items()):
                deps = {stream_ids[v] for v in fvsof(stream_body) & stream_vars}
                ds.union(stream_id, *deps)

            # factors are in the same partition as their dependencies
            for factor in factors:
                ds.union(*(stream_ids[v] for v in (fvsof(factor) & stream_vars)))

            partitions = {}
            for v, i in stream_ids.items():
                p = ds.find(i)
                part_streams = partitions.get(p, {})
                part_streams[v] = streams[v]
                partitions[p] = part_streams

            if len(partitions) > 1:
                new_reduces = []
                for partition_streams in partitions.values():
                    partition_terms = (
                        t for t in factors if (fvsof(t) & set(partition_streams.keys()))
                    )
                    partition_term = body.op(*partition_terms)
                    new_reduces.append(self.reduce(partition_term, partition_streams))

                return body.op(*new_reduces)

        return super().reduce(body, streams)


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

    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        # elim zero
        if any(x is self.zero for x in args):
            return self.zero

        return super().plus(*args)


@dataclass
class _HashableTerm:
    term: Term

    def __eq__(self, other):
        return syntactic_eq(self, other)

    def __hash__(self):
        return syntactic_hash(self)


class Semilattice[T](IdempotentMonoid[T], CommutativeMonoid[T]):
    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        # elim dups
        args_count = Counter(_HashableTerm(t) for t in args)
        if len(args_count) < len(args):
            dedup_args = []
            for t in args:
                ht = _HashableTerm(t)
                if ht in args_count:
                    dedup_args.append(t)
                    del args_count[ht]
            args = tuple(dedup_args)

        return super().plus(*args)


class SupportsRichComparison(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...


@Operation.define
def _min[T: SupportsRichComparison](*args: T) -> T:
    if any(isinstance(x, Term) for x in args):
        raise NotHandled
    return min(*args)


Min = Semilattice(kernel=_min, identity=float("inf"), add=_min)


@Operation.define
def _max[T: SupportsRichComparison](*args: T) -> T:
    if any(isinstance(x, Term) for x in args):
        raise NotHandled
    return max(*args)


Max = Semilattice(kernel=_max, identity=float("-inf"), add=_min)


@Operation.define
def _arg_min[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] < a[0] else a  # type: ignore


ArgMin: Monoid[tuple[float, Any]] = Monoid(
    kernel=_arg_min, identity=(float("inf"), None)
)


@Operation.define
def _arg_max[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] > a[0] else a  # type: ignore


ArgMax: Monoid[tuple[float, Any]] = Monoid(
    kernel=_arg_max, identity=(float("-inf"), None)
)


class _SumMonoid[N: int | float | complex](CommutativeMonoid[N]):
    def scalar_mul(self, v: N, x: int) -> N:
        if not isinstance(x, Term) and x == 0:
            return self.identity
        if not isinstance(x, Term) and x == 1:
            return v
        return typing.cast(N, v * x)


Sum = _SumMonoid(kernel=_NumberTerm.__add__, identity=0)


class _ProductMonoid[N: int | float | complex](CommutativeMonoidWithZero[N]):
    def scalar_mul(self, v: N, x: int) -> N:
        if not isinstance(x, Term) and x == 0:
            return self.identity
        if not isinstance(x, Term) and x == 1:
            return v
        return typing.cast(N, v**x)


Product = _ProductMonoid(kernel=_NumberTerm.__mul__, identity=1, zero=0)


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
