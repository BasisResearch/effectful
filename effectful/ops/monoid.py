import collections.abc
import functools
import itertools
import typing
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any

from effectful.internals.disjoint_set import DisjointSet
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler
from effectful.ops.syntax import (
    ObjectInterpretation,
    Scoped,
    _NumberTerm,
    deffn,
    implements,
    iter_,
    syntactic_eq,
    syntactic_hash,
)
from effectful.ops.types import (
    Expr,
    Interpretation,
    NotHandled,
    Operation,
    Term,
    _CustomSingleDispatchMethod,
)

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


def outer_stream(
    streams: Streams,
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


class Monoid[T]:
    _name: str

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"Monoid({self._name!r})"

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    @Operation.define
    def kernel(self, _x: T, _y: T) -> T:
        raise NotHandled

    @Operation.define
    def identity(self) -> T:
        raise NotHandled

    @Operation.define
    @_CustomSingleDispatchMethod
    def plus[S](self, dispatch, *args: S) -> S:
        """Monoid addition with broadcasting over common collection types,
        callables, and interpretations.
        """
        if not args:
            return typing.cast(S, self.identity())
        return dispatch(type(args[0]))(self, *args)

    @plus.register(object)  # type: ignore[attr-defined]
    def _(self, *args):
        if any(isinstance(x, Term) for x in args):
            raise NotHandled
        return functools.reduce(self.kernel, args, self.identity())

    @plus.register(tuple)  # type: ignore[attr-defined]
    def _(self, *args):
        return tuple(self.plus(*vs) for vs in zip(*args, strict=True))

    @plus.register(Generator)  # type: ignore[attr-defined]
    def _(self, *args):
        return (self.plus(*vs) for vs in zip(*args, strict=True))

    @plus.register(Mapping)  # type: ignore[attr-defined]
    def _(self, *args):
        if isinstance(args[0], Interpretation):
            keys = args[0].keys()
            for b in args[1:]:
                if not isinstance(b, Interpretation):
                    raise TypeError(f"Expected interpretation but got {b}")
                if not keys == b.keys():
                    raise ValueError(
                        f"Expected interpretation of {keys} but got {b.keys()}"
                    )
            return {k: self.plus(*(handler(b)(b[k]) for b in args)) for k in keys}

        for b in args[1:]:
            if not isinstance(b, Mapping):
                raise TypeError(f"Expected mapping but got {b}")
        all_values = collections.defaultdict(list)
        for d in args:
            for k, v in d.items():
                all_values[k].append(v)
        return {k: self.plus(*vs) for (k, vs) in all_values.items()}

    @Operation.define
    @functools.singledispatchmethod
    def reduce[A, B, U: Body](
        self, body: Annotated[U, Scoped[A | B]], streams: Annotated[Streams, Scoped[A]]
    ) -> Annotated[U, Scoped[B]]:
        if not streams:
            return self.identity()

        # find and reduce a ground stream
        for stream_key, stream_body, streams_tail in outer_stream(streams):
            if isinstance(stream_body, Term):
                continue

            stream_values_iter = iter(stream_body)

            # if we iterate and get a term instead of a real iterator, skip
            if isinstance(stream_values_iter, Term) and stream_values_iter.op is iter_:
                continue

            new_reduces = []
            for stream_val in stream_values_iter:
                with handler({stream_key: deffn(stream_val)}):
                    eval_args = evaluate((body, streams_tail))
                    assert isinstance(eval_args, tuple)
                    new_reduces.append(self.reduce(*eval_args))

            return self.plus(*new_reduces)

        raise NotHandled

    @reduce.register(Callable)  # type: ignore[attr-defined]
    def _reduce_callable(self, body: Callable, streams):
        return lambda *a, **k: self.reduce(body(*a, **k), streams)

    @reduce.register(Mapping)  # type: ignore[attr-defined]
    def _reduce_mapping(self, body: Mapping, streams):
        return {k: self.reduce(v, streams) for (k, v) in body.items()}

    @reduce.register(tuple)  # type: ignore[attr-defined]
    def _reduce_sequence(self, body: tuple, streams):
        return tuple(self.reduce(x, streams) for x in body)  # type:ignore[call-arg]

    @reduce.register(Generator)  # type: ignore[attr-defined]
    def _reduce_generator(self, body: Generator, streams):
        return (self.reduce(x, streams) for x in body)


def _is_monoid_plus(op: Operation) -> bool:
    """True if ``op`` is the ``plus`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.plus


def _is_monoid_reduce(op: Operation) -> bool:
    """True if ``op`` is the ``reduce`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.reduce


class MonoidWithZero[T](Monoid[T]):
    @Operation.define
    @staticmethod
    def zero() -> T:
        raise NotHandled


@Operation.define
def product[T](
    a: Iterable[tuple[T, ...] | T], b: Iterable[tuple[T, ...] | T]
) -> Iterable[tuple[T, ...]]:
    if isinstance(a, Term) or isinstance(b, Term):
        raise NotHandled

    def to_tuple(x):
        return x if isinstance(x, tuple) else (x,)

    return [to_tuple(x) + to_tuple(y) for (x, y) in itertools.product(a, b)]


Min = Monoid("Min")
Max = Monoid("Max")
ArgMin = Monoid("ArgMin")
ArgMax = Monoid("ArgMax")
Sum = Monoid("Sum")
Product = MonoidWithZero("Product")
CartesianProduct = Monoid("CartesianProduct")


@dataclass
class _ExtensiblePredicate[T]:
    elems: set[T]

    def register(self, t: T) -> None:
        self.elems.add(t)

    def __call__(self, t: T) -> bool:
        return t in self.elems


is_commutative = _ExtensiblePredicate({Max, Min, Sum, Product})
is_idempotent = _ExtensiblePredicate({Max, Min})


@dataclass
class _ExtensiblePredicate[T]:
    elems: set[T]

    def register(self, t: T) -> None:
        self.elems.add(t)

    def __call__(self, t: T) -> bool:
        return t in self.elems


is_commutative = _ExtensiblePredicate({Max, Min, Sum, Product})
is_idempotent = _ExtensiblePredicate({Max, Min})


@dataclass
class _ExtensibleBinaryRelation[S, T]:
    tuples: set[tuple[S, T]]

    def register(self, s: S, t: T) -> None:
        self.tuples.add((s, t))

    def __call__(self, s: S, t: T) -> bool:
        return (s, t) in self.tuples


distributes_over = _ExtensibleBinaryRelation(
    {(Max, Min), (Min, Max), (Sum, Min), (Sum, Max), (Product, Sum)}
)


class SumKernel(ObjectInterpretation):
    @implements(Sum.identity)
    def identity(self):
        return 0

    @implements(Sum.kernel)
    def kernel(self, x, y):
        return x + y


class ProductKernel(ObjectInterpretation):
    @implements(Product.identity)
    def identity(self):
        return 1

    @implements(Product.zero)
    def zero(self):
        return 0

    @implements(Product.kernel)
    def kernel(self, x, y):
        return x * y


class MinKernel(ObjectInterpretation):
    @implements(Min.identity)
    def identity(self):
        return float("inf")

    @implements(Min.kernel)
    def kernel(self, x, y):
        if isinstance(x, int | float) and isinstance(y, int | float):
            return min(x, y)
        return fwd()


class MaxKernel(ObjectInterpretation):
    @implements(Min.identity)
    def identity(self):
        return -float("inf")

    @implements(Min.kernel)
    def kernel(self, x, y):
        if isinstance(x, int | float) and isinstance(y, int | float):
            return max(x, y)
        return fwd()


class ArgMinKernel(ObjectInterpretation):
    @implements(ArgMin.identity)
    def identity(self):
        return (float("inf"), None)

    @implements(ArgMin.kernel)
    def kernel(self, a, b):
        if isinstance(a[0], Term) or isinstance(b[0], Term):
            return fwd()
        if isinstance(a[0], int | float) and isinstance(b[0], int | float):
            return b if b[0] < a[0] else a
        return fwd()


class ArgMaxKernel(ObjectInterpretation):
    @implements(ArgMax.identity)
    def identity(self):
        return (-float("inf"), None)

    @implements(ArgMax.kernel)
    def kernel(self, a, b):
        if isinstance(a[0], Term) or isinstance(b[0], Term):
            return fwd()
        if isinstance(a[0], int | float) and isinstance(b[0], int | float):
            return b if b[0] < a[0] else a
        return fwd()


class CartesianProductKernel(ObjectInterpretation):
    @implements(CartesianProduct.kernel)
    def kernel(self, a, b):
        if isinstance(a, Term) or isinstance(b, Term):
            raise NotHandled

        if isinstance(a, Iterable) and isinstance(b, Iterable):

            def to_tuple(x):
                return x if isinstance(x, tuple) else (x,)

            return [to_tuple(x) + to_tuple(y) for (x, y) in itertools.product(a, b)]

        return fwd()


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
        def is_nested_plus(x):
            return isinstance(x, Term) and x.op is monoid.plus

        if any(is_nested_plus(x) for x in args):
            flat_args = itertools.chain.from_iterable(
                t.args if is_nested_plus(t) else (t,) for t in args
            )
            assert len(args) > 0
            return monoid.plus(*flat_args)
        return fwd()


class PlusDistr(ObjectInterpretation):
    """x + (y * z) = x * y + x * z"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if any(
            isinstance(x, Term)
            and _is_monoid_plus(x.op)
            and distributes_over(monoid, x.op.__self__)
            for x in args
        ):
            non_terms = []

            # group terms by their monoid
            by_monoid: dict[Monoid, list[Term]] = defaultdict(list)
            for t in args:
                if isinstance(t, Term) and _is_monoid_plus(t.op):
                    by_monoid[t.op.__self__].append(t)
                else:
                    non_terms.append(t)

            # distribute over each group
            progress = False
            final_sum = []
            for m, terms in by_monoid.items():
                if (
                    len(terms) > 1
                    and distributes_over(monoid, m)
                    and not distributes_over(m, monoid)
                ):
                    progress = True
                    term_args = (t.args for t in terms)
                    dist_terms = (
                        monoid.plus(*args) for args in itertools.product(*term_args)
                    )
                    final_sum.append(m.plus(*dist_terms))
                else:
                    final_sum += terms
            if progress:
                return monoid.plus(*non_terms, *final_sum)
        return fwd()


class PlusZero(ObjectInterpretation):
    """x₁ * ... * 0 * ... * xₙ = 0"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not (isinstance(monoid, MonoidWithZero)):
            return fwd()
        if any(x is monoid.zero for x in args):
            return monoid.zero
        return fwd()


class PlusConsecutiveDups(ObjectInterpretation):
    """x ⊕ x ⊕ y = x ⊕ y"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not is_idempotent(monoid):
            return fwd()

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

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not (is_idempotent(monoid) and is_commutative(monoid)):
            return fwd()

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
        if isinstance(body, Term) and body.op is monoid.reduce:
            return monoid.reduce(body.args[0], streams | body.args[1])
        return fwd()


class ReduceSplit(ObjectInterpretation):
    """Implements the identity
    reduce(R, S, b1 + ... + bn) = reduce(R, S, b1) + ... + reduce(R, S, bn)
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not is_commutative(monoid):
            return fwd()
        if isinstance(body, Term) and body.op is monoid.plus:
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

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not is_commutative(monoid):
            return fwd()
        if (
            isinstance(body, Term)
            and _is_monoid_plus(body.op)
            and distributes_over(body.op.__self__, monoid)
        ):
            inner_monoid: Monoid = body.op.__self__
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
            for _, factor_fvs in factors:
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

                partition_term = inner_monoid.plus(*(t[0] for t in partition_factors))
                new_reduces.append((partition_term, partition_streams))
                placed_streams |= partition_stream_keys

            constant_factors = [t for (t, fvs) in factors if not (fvs & stream_vars)]

            if len(new_reduces) > 1:
                result = inner_monoid.plus(
                    *constant_factors, *(monoid.reduce(*args) for args in new_reduces)
                )
                return result

        return fwd()


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
        preds = fvsof(v) & stream_vars
        if preds:
            for pred in preds:
                succ[pred].add(k)
        else:
            no_dependents.add(k)

    topo = TopologicalSorter(succ)
    topo.prepare()
    return (
        ({k: v for (k, v) in streams.items() if k != op}, op, streams[op])
        for op in set(topo.get_ready()) | no_dependents
    )


class ReduceDistributeCartesianProduct(ObjectInterpretation):
    """Eliminates a reduce over a cartesian product.
        ∑_x₁ ∑_x₂ ... ∑_xₙ ∏_i f(xᵢ) = ∏_i ∑_xᵢ f(xᵢ)
    This transform is also called inversion in the lifting
    literature (e.g. [1]).

    More specifically, this transform implements the identity
    reduce(⨁, reduce(⨂, body2, {vv: v()}), {v: reduce(×, body1, S1)} ∪ S2)
        = reduce(⨁, reduce(⨂, reduce(⨁, body2, {vv: body1}), S1), S2)
    where × is the cartesian product and ⨂ distributes over ⨁.

    Note: This could be generalized to grouped inversion [2].

    [1] Braz, Rd, Eyal Amir, and Dan Roth. "Lifted first-order
    probabilistic inference." IJCAI. 2005.
    [2] Taghipour, Nima, et al. "Completeness results for lifted
    variable elimination." AISTATS. 2013.
    """

    @implements(Monoid.reduce)
    def reduce(self, sum_monoid: Monoid, sum_body, sum_streams):
        if not (is_commutative(sum_monoid) and isinstance(sum_body, Term)):
            return fwd()

        # body is a product or multiplication of products
        if _is_monoid_plus(sum_body.op) and distributes_over(
            sum_body.op.__self__, sum_monoid
        ):
            prod_reduces = sum_body.args
        else:
            prod_reduces = [sum_body]

        products: list[tuple[Monoid, Callable, Operation, Term]] = []
        for prod_reduce in prod_reduces:
            if not (
                isinstance(prod_reduce, Term) and _is_monoid_reduce(prod_reduce.op)
            ):
                return fwd()
            prod_monoid: Monoid = prod_reduce.op.__self__
            prod_body = prod_reduce.args[0]
            prod_streams = typing.cast(Mapping, prod_reduce.args[1])
            if not (
                distributes_over(prod_monoid, sum_monoid)
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
                and cprod_term.op is CartesianProduct.reduce
            ):
                continue
            (cprod_body, cprod_streams) = cprod_term.args

            if not all(
                prod_stream.op == cprod_op for (_, _, _, prod_stream) in products
            ):
                continue

            prod_op = Operation.define(products[0][2])
            prod_monoid = products[0][0]
            inner_sum = sum_monoid.reduce(
                prod_monoid.plus(
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
