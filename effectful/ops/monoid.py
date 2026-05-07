import collections.abc
import functools
import itertools
import numbers
import typing
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any

from effectful.internals.disjoint_set import DisjointSet
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler
from effectful.ops.syntax import (
    ObjectInterpretation,
    Scoped,
    _NumberTerm,
    defdata,
    deffn,
    implements,
    iter_,
    syntactic_eq,
    syntactic_hash,
)
from effectful.ops.types import Expr, Interpretation, NotHandled, Operation, Term

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


class Monoid[T]:
    kernel: Operation[[T, T], T]
    identity: T

    def __init__(self, kernel: Callable[[T, T], T], identity: T):
        self.identity = identity
        self.kernel = (
            kernel if isinstance(kernel, Operation) else Operation.define(kernel)
        )

    def __repr__(self):
        return f"{type(self)}({self.kernel}, {self.identity})"

    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        """Monoid addition with broadcasting over common collection types,
        callables, and interpretations.

        """
        if not args:
            return typing.cast(S, self.identity)

        if any(isinstance(x, Term) for x in args):
            return typing.cast(S, defdata(self.plus, *args))

        return self._plus(*args)

    @functools.singledispatchmethod
    def _plus[S](self, *args: S) -> S:
        return typing.cast(S, functools.reduce(self.kernel, args, self.identity))

    @_plus.register(tuple)
    def _(self, *args):
        return tuple(self.plus(*vs) for vs in zip(*args, strict=True))

    @_plus.register(Generator)
    def _(self, *args):
        return (self.plus(*vs) for vs in zip(*args, strict=True))

    @_plus.register(Mapping)
    def _(self, *args):
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

        for b in args[1:]:
            if not isinstance(b, Mapping):
                raise TypeError(f"Expected mapping but got {b}")

        all_values = collections.defaultdict(list)
        for d in args:
            for k, v in d.items():
                all_values[k].append(v)
        result = {k: self.plus(*vs) for (k, vs) in all_values.items()}
        return result

    @Operation.define
    @functools.singledispatchmethod
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        if callable(body):
            return typing.cast(U, lambda *a, **k: self.reduce(body(*a, **k), streams))

        def generator(loop_order) -> Iterator[Interpretation]:
            if len(loop_order) == 0:
                return

            stream_key = loop_order[0][0]
            stream_values = evaluate(streams[stream_key])
            stream_values_iter = iter(stream_values)  # type: ignore[arg-type]

            # If we try to iterate and get a term instead of a real
            # iterator, give up
            if isinstance(stream_values_iter, Term) and stream_values_iter.op is iter_:
                raise NotHandled

            if len(loop_order) == 1:
                for val in stream_values_iter:
                    yield {stream_key: functools.partial(lambda v: v, val)}
            else:
                for val in stream_values_iter:
                    intp: Interpretation = {
                        stream_key: functools.partial(lambda v: v, val)
                    }
                    with handler(intp):
                        for intp2 in generator(loop_order[1:]):
                            yield coproduct(intp, intp2)

        loop_order = list(order_streams(streams))
        try:
            return self.plus(
                *(handler(intp)(evaluate)(body) for intp in generator(loop_order))
            )
        except NotHandled:
            return typing.cast(U, defdata(self.reduce, body, streams))

    @reduce.register  # type: ignore[attr-defined]
    def _(self, body: Mapping, streams):
        return {k: self.reduce(v, streams) for (k, v) in body.items()}

    @reduce.register  # type: ignore[attr-defined]
    def _(self, body: tuple, streams):
        return tuple(self.reduce(x, streams) for x in body)

    @reduce.register  # type: ignore[attr-defined]
    def _(self, body: Generator, streams):
        return (self.reduce(x, streams) for x in body)


class IdempotentMonoid[T](Monoid[T]):
    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        return super().plus(*args)

    @Operation.define
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        return super().reduce(body, streams)


class CommutativeMonoid[T](Monoid[T]):
    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        return super().plus(*args)

    @Operation.define
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        return super().reduce(body, streams)


class CommutativeMonoidWithZero[T](CommutativeMonoid[T]):
    zero: T

    def __init__(self, kernel: Callable[[T, T], T], identity: T, zero: T):
        super().__init__(kernel, identity)
        self.zero = zero

    def __repr__(self):
        return f"{type(self)}({self.kernel}, {self.identity}, {self.zero})"

    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        return super().plus(*args)

    @Operation.define
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        return super().reduce(body, streams)


class Semilattice[T](IdempotentMonoid[T], CommutativeMonoid[T]):
    @Operation.define
    def plus[S: Body[T]](self, *args: S) -> S:
        return super().plus(*args)

    @Operation.define
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        return super().reduce(body, streams)


@Operation.define
def _arg_min[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] < a[0] else a  # type: ignore


@Operation.define
def _arg_max[T](
    a: tuple[numbers.Number, T | None], b: tuple[numbers.Number, T | None]
) -> tuple[numbers.Number, T | None]:
    if isinstance(a[0], Term) or isinstance(b[0], Term):
        raise NotHandled
    return b if b[0] > a[0] else a  # type: ignore


@Operation.define
def product[T](
    a: Iterable[tuple[T, ...] | T], b: Iterable[tuple[T, ...] | T]
) -> Iterable[tuple[T, ...]]:
    if isinstance(a, Term) or isinstance(b, Term):
        raise NotHandled

    def to_tuple(x):
        return x if isinstance(x, tuple) else (x,)

    return [to_tuple(x) + to_tuple(y) for (x, y) in itertools.product(a, b)]


Min = Semilattice(kernel=min, identity=float("inf"))
Max = Semilattice(kernel=max, identity=float("-inf"))
ArgMin = Monoid(kernel=_arg_min, identity=(float("inf"), None))
ArgMax = Monoid(kernel=_arg_max, identity=(float("-inf"), None))
Sum = CommutativeMonoid(kernel=_NumberTerm.__add__, identity=0)
Product = CommutativeMonoidWithZero(kernel=_NumberTerm.__mul__, identity=1, zero=0)
CartesianProduct = Monoid(kernel=product, identity=[()])


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


class PlusEmpty(ObjectInterpretation):
    """plus() = 0"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not args:
            return monoid.identity
        return fwd()


class PlusSingle(ObjectInterpretation):
    """plus(x) = x"""

    @implements(Monoid.plus)
    def plus(self, _, *args):
        if len(args) == 1:
            return args[0]
        return fwd()


class PlusIdentity(ObjectInterpretation):
    """x₁ + ... + 0 + ... + xₙ = x₁ + ... + xₙ"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if any(x is monoid.identity for x in args):
            return monoid.plus(*(x for x in args if x is not monoid.identity))
        return fwd()


class PlusAssoc(ObjectInterpretation):
    """x + (y + z) = (x + y) + z = x + y + z"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if any(isinstance(x, Term) and x.op is monoid.plus for x in args):
            flat_args = itertools.chain.from_iterable(
                t.args if isinstance(t, Term) and t.op is monoid.plus else (t,)
                for t in args
            )
            assert len(args) > 0
            return monoid.plus(*flat_args)
        return fwd()


class PlusDistr(ObjectInterpretation):
    """x + (y * z) = x * y + x * z"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if any(
            isinstance(x, Term) and distributes_over(monoid.plus, x.op) for x in args
        ):
            non_terms = []

            # group terms by head operation
            by_head_op = defaultdict(list)
            for t in args:
                if isinstance(t, Term):
                    by_head_op[t.op].append(t)
                else:
                    non_terms.append(t)

            # distribute over each group
            progress = False
            final_sum = []
            for op, terms in by_head_op.items():
                if (
                    len(terms) > 1
                    and distributes_over(monoid.plus, op)
                    and not distributes_over(op, monoid.plus)
                ):
                    progress = True
                    term_args = (t.args for t in terms)
                    dist_terms = (
                        monoid.plus(*args) for args in itertools.product(*term_args)
                    )
                    final_sum.append(op(*dist_terms))
                else:
                    final_sum += terms
            if progress:
                return monoid.plus(*non_terms, *final_sum)
        return fwd()


class PlusZero(ObjectInterpretation):
    """x₁ * ... * 0 * ... * xₙ = 0"""

    @implements(CommutativeMonoidWithZero.plus)
    def plus(self, monoid, *args):
        if any(x is monoid.zero for x in args):
            return monoid.zero
        return fwd()


class PlusConsecutiveDups(ObjectInterpretation):
    """x ⊕ x ⊕ y = x ⊕ y"""

    @implements(IdempotentMonoid.plus)
    def plus(self, monoid, *args):
        dedup_args = (
            args[i]
            for i in range(len(args))
            if i == 0 or not syntactic_eq(args[i - 1], args[i])
        )
        return fwd(monoid, *dedup_args)


class PlusDups(ObjectInterpretation):
    """x ⊕ y ⊕ x = x ⊕ y"""

    @dataclass
    class _HashableTerm:
        term: Term

        def __eq__(self, other):
            return syntactic_eq(self, other)

        def __hash__(self):
            return syntactic_hash(self)

    @implements(Semilattice.plus)
    def plus(self, monoid, *args):
        # elim dups
        args_count = Counter(self._HashableTerm(t) for t in args)
        if len(args_count) < len(args):
            dedup_args = []
            for t in args:
                ht = self._HashableTerm(t)
                if ht in args_count:
                    dedup_args.append(t)
                    del args_count[ht]
            return fwd(monoid, *dedup_args)
        return fwd()


NormalizePlusIntp = functools.reduce(
    coproduct,
    typing.cast(
        list[Interpretation],
        [
            PlusEmpty(),
            PlusSingle(),
            PlusIdentity(),
            PlusAssoc(),
            PlusDistr(),
            PlusZero(),
            PlusConsecutiveDups(),
            PlusDups(),
        ],
    ),
)


class ReduceNoStreams(ObjectInterpretation):
    """Implements the identity
    reduce(R, ∅, body) = 0
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, _, streams):
        if len(streams) == 0:
            return monoid.identity
        return fwd()


class ReduceFusion(ObjectInterpretation):
    """Implements the identity
    reduce(R, S1, reduce(R, S2, body)) = reduce(R, S1 ∪ S2, body)
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if isinstance(body, Term) and body.op == monoid.reduce:
            return monoid.reduce(body.args[0], streams | body.args[1])
        return fwd()


class ReduceSplit(ObjectInterpretation):
    """Implements the identity
    reduce(R, S, b1 + ... + bn) = reduce(R, S, b1) + ... + reduce(R, S, bn)
    """

    @implements(CommutativeMonoid.reduce)
    def reduce(self, monoid, body, streams):
        if isinstance(body, Term) and body.op == monoid.plus:
            return monoid.plus(*(monoid.reduce(x, streams) for x in body.args))
        return fwd()


class ReduceFactorization(ObjectInterpretation):
    """
    Implements factorization of independent terms.
    For example, when having two independent distributions,
    we can rewrite their marginalization as:
        ∫p(x)⋅q(y)dxdy => ∫p(x)dx ⋅ ∫q(y)dy

    More specifically, in terms of reduces we are performing:
        reduce(R, (S₁ × ... × Sₖ) , A₁ * ... * Aₖ)
        => reduce(R, S₁, A₁) * ... * reduce(R, Sₖ, Aₖ)
        where free(Aᵢ) ∩ free(Aⱼ) ∩ S = ∅
          and free(Aᵢ) ∩ S ⊆ Sᵢ
    """

    @implements(CommutativeMonoid.reduce)
    def reduce(self, monoid, body, streams):
        if isinstance(body, Term) and distributes_over(body.op, monoid.plus):
            stream_vars = set(streams.keys())
            factors = [(arg, fvsof(arg)) for arg in body.args]
            stream_ids = {v: i for (i, v) in enumerate(stream_vars)}
            ds = DisjointSet(len(streams))

            # streams are in the same partition as their dependencies
            for stream_var, stream_id in stream_ids.items():
                stream_body = streams[stream_var]
                deps = sorted([stream_ids[v] for v in fvsof(stream_body) & stream_vars])
                ds.union(stream_id, *deps)

            # factors are in the same partition as their dependencies
            for factor, factor_fvs in factors:
                factor_streams = sorted(
                    [stream_ids[v] for v in (factor_fvs & stream_vars)]
                )
                ds.union(*factor_streams)

            placed_streams = set()
            new_reduces = []
            for stream_key in streams:
                if stream_key in placed_streams:
                    continue

                partition = ds.find(stream_ids[stream_key])
                partition_streams = {
                    k: v
                    for (k, v) in streams.items()
                    if ds.find(stream_ids[k]) == partition
                }
                partition_stream_keys = set(partition_streams.keys())

                partition_factors = [
                    t for t in factors if (t[1] & partition_stream_keys)
                ]

                assert all(
                    (t[1] & stream_vars) <= partition_stream_keys
                    for t in partition_factors
                ), "partition contains all streams required by factor"

                partition_term = body.op(*(t[0] for t in partition_factors))
                new_reduces.append((partition_term, partition_streams))
                placed_streams |= partition_stream_keys

            constant_factors = [t for (t, fvs) in factors if not (fvs & stream_vars)]

            if len(new_reduces) > 1:
                result = body.op(
                    *constant_factors, *(monoid.reduce(*args) for args in new_reduces)
                )
                return result

        return fwd()


def outer_stream(
    streams: dict[Operation, Expr],
) -> Iterable[tuple[Operation, Expr, dict[Operation, Expr]]]:
    """Returns the streams that can be ordered outermost in the loop nest as
    well as the remaining streams in the nest.

    """
    stream_vars = set(streams.keys())
    pred = {k: fvsof(v) & stream_vars for k, v in streams.items()}
    topo = TopologicalSorter(pred)
    topo.prepare()
    return (
        (op, streams[op], {k: v for (k, v) in streams.items() if k != op})
        for op in topo.get_ready()
    )


def inner_stream(
    streams: dict[Operation, Expr],
) -> Iterable[tuple[dict[Operation, Expr], Operation, Expr]]:
    """Returns the streams that can be ordered innermost in the loop nest as
    well as the remaining streams in the nest.

    """
    stream_vars = set(streams.keys())

    no_dependents = set()
    succ = defaultdict(set)
    for k, v in streams.items():
        for pred in fvsof(v) & stream_vars:
            succ[pred].add(k)
        else:
            no_dependents.add(k)

    topo = TopologicalSorter(succ)
    topo.prepare()
    return (
        ({k: v for (k, v) in streams.items() if k != op}, op, streams[op])
        for op in set(topo.get_ready()) | no_dependents
    )


def match_reduce(term: Term) -> tuple | None:
    reduce_args = None

    def set_reduce_args(*args, **kwargs):
        nonlocal reduce_args
        reduce_args = args

    with interpreter({Monoid.reduce: set_reduce_args}):
        term.op(*term.args, **term.kwargs)
    return reduce_args


class ReduceDistributeCartesianProduct(ObjectInterpretation):
    """Eliminates a reduce over a cartesian product.
        ∑_x₁ ∑_x₂ ... ∑_xₙ ∏_i f(xᵢ) = ∏_i ∑_xᵢ f(xᵢ)
    This transform is also called inversion in the lifting
    literature (e.g. [1]).

    More specifically, this transform implements the identity
    reduce(⨁, reduce(⨂, body2, {vv: v()}), {v: reduce(×, body1, S1)} ∪ S2)
        = reduce(⨁, reduce(⨂, reduce(⨁, body2, {vv: v()}), S1), S2)
            where × is the cartesian product and ⨂ distributes over ⨁.

    Note: This could be generalized to grouped inversion [2].

    [1] Braz, Rd, Eyal Amir, and Dan Roth. "Lifted first-order
    probabilistic inference." IJCAI. 2005.
    [2] Taghipour, Nima, et al. "Completeness results for lifted
    variable elimination." AISTATS. 2013.
    """

    @implements(CommutativeMonoid.reduce)
    def reduce(self, sum_monoid: Monoid, sum_body, sum_streams):
        if not (isinstance(sum_body, Term)):
            return fwd()

        # body is a product or multiplication of products
        if distributes_over(sum_body.op, sum_monoid.plus):
            prod_reduces = sum_body.args
        else:
            prod_reduces = [sum_body]

        products: list[tuple[Monoid, Callable, Operation, Term]] = []
        for prod_reduce in prod_reduces:
            prod_args = match_reduce(prod_reduce)
            if prod_args is None:
                return fwd()
            (prod_monoid, prod_body, prod_streams) = prod_args
            if not (
                distributes_over(prod_monoid.plus, sum_monoid.plus)
                and (len(products) == 0 or products[-1][0] == prod_monoid)
            ):
                return fwd()

            if len(prod_streams) > 1 or len(prod_streams) == 0:
                return fwd()
            (prod_op, prod_stream) = next(iter(prod_streams.items()))
            products.append(
                (prod_monoid, deffn(prod_body, prod_op), prod_op, prod_stream)
            )

        assert len(products) > 0

        for outer_sum_streams, cprod_op, cprod_term in inner_stream(sum_streams):
            if not (
                isinstance(cprod_term, Term)
                and cprod_term.op == CartesianProduct.reduce
            ):
                continue
            (cprod_body, cprod_streams) = cprod_term.args

            if not all(
                prod_stream.op == cprod_op for (_, _, _, prod_stream) in products
            ):
                continue

            prod_op = Operation.define(products[0][2])
            inner_sum = sum_monoid.reduce(
                Product.plus(
                    *(prod_body(prod_op()) for (_, prod_body, _, _) in products)
                ),
                {prod_op: cprod_body},
            )
            prod = prod_monoid.reduce(inner_sum, cprod_streams)
            outer_sum = (
                sum_monoid.reduce(prod, outer_sum_streams)
                if outer_sum_streams
                else prod
            )
            return outer_sum

        return fwd()


NormalizeReduceIntp = functools.reduce(
    coproduct,
    typing.cast(
        list[Interpretation],
        [
            ReduceNoStreams(),
            ReduceFusion(),
            ReduceSplit(),
            ReduceFactorization(),
            ReduceDistributeCartesianProduct(),
        ],
    ),
)

NormalizeIntp = coproduct(NormalizePlusIntp, NormalizeReduceIntp)
