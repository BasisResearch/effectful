import collections.abc
import functools
import itertools
import operator
import typing
from collections import Counter, UserDict, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any

from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import (
    ObjectInterpretation,
    Scoped,
    defdata,
    deffn,
    implements,
    syntactic_eq,
    syntactic_hash,
)
from effectful.ops.types import Expr, Interpretation, NotHandled, Operation, Term

type Stream[T] = Iterable[T]

type Streams = Mapping[Operation[[], Any], Stream[Any]]

type Body[T] = (
    Iterable[T]
    | Callable[..., Body[T]]
    | T
    | Mapping[Any, Body[T]]
    | Interpretation[T, Body[T]]
)


def outer_stream(streams: Streams) -> Iterable[tuple[Operation, Stream, Streams]]:
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


def inner_streams_first(streams: dict[Operation, Expr]) -> Iterable[Operation]:
    """Iterable over streams where dependent streams precede their dependencies."""
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
    return topo.static_order()


class Monoid[W]:
    """A monoid with ``plus`` and ``reduce`` :class:`Operation` s."""

    _name: str
    identity: W

    def __init__(self, identity: W, name: str):
        self._name = name
        self.identity = identity

    def __repr__(self):
        return f"Monoid({self._name!r})"

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    @Operation.define
    def plus(self, *args: W) -> W:
        """Monoid addition. Handlers supply per-monoid and broadcasting
        behavior; the default rule only handles identity and zero cases (for
        monoids that have a zero).

        """
        if hasattr(self, "zero") and any(a is self.zero for a in args):
            return self.zero

        nonident_args = [a for a in args if a is not self.identity]
        if len(nonident_args) != len(args):
            return self.plus(*nonident_args)

        return defdata(self.plus, *nonident_args)  # type: ignore[return-value]

    @Operation.define
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        """Reduce ``body`` over ``streams``. Handlers supply per-monoid and
        broadcasting behavior; the default rule only handles the empty-stream
        case.
        """
        for stream_key, stream_body, streams_tail in outer_stream(streams):
            if isinstance(stream_body, Term):
                continue
            stream_values_iter = iter(stream_body)

            # if we iterate and get a term instead of a real iterator, skip
            if isinstance(stream_values_iter, Term):
                continue

            new_reduces = []
            for stream_val in stream_values_iter:
                with handler({stream_key: deffn(stream_val)}):
                    eval_args = evaluate((body, streams_tail))
                    assert isinstance(eval_args, tuple)
                    new_reduces.append(
                        self.reduce(*eval_args) if streams_tail else eval_args[0]
                    )
            return self.plus(*new_reduces)
        raise NotHandled

    @Operation.define
    def weighted[T](
        self, stream: Stream[T], weight: Callable[[T], W] | Operation[[T], W]
    ) -> Stream[T]:
        """A stream paired with a per-element weight. ``var`` is an
        :class:`Operation` standing for "an element of ``stream``"; ``weight``
        is an expression that uses ``var`` and evaluates to the weight of that
        element.

        """
        raise NotHandled


class MonoidWithZero[T](Monoid[T]):
    zero: T

    def __init__(self, name: str, identity: T, zero: T):
        super().__init__(name=name, identity=identity)
        self.zero = zero


Min = Monoid(name="Min", identity=float("inf"))
Max = Monoid(name="Max", identity=-float("inf"))
ArgMin = Monoid(name="ArgMin", identity=(Min.identity, None))
ArgMax = Monoid(name="ArgMax", identity=(Max.identity, None))
Sum = Monoid(name="Sum", identity=0)
Product = MonoidWithZero(name="Product", identity=1, zero=0)
# CartesianProduct values are "two-level indexable" (rows × positions). The
# identity ``[()]`` is one row of zero positions (composing with it preserves
# shape); the zero ``[]`` is no rows (absorbs under product).
CartesianProduct = MonoidWithZero(name="CartesianProduct", identity=[()], zero=[])


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


def _is_monoid_plus(op: Operation) -> bool:
    """True if ``op`` is the ``plus`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.plus


def _is_monoid_reduce(op: Operation) -> bool:
    """True if ``op`` is the ``reduce`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.reduce


def _is_monoid_weighted(op: Operation) -> bool:
    """True if ``op`` is the ``weighted`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.weighted


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
    def plus(self, monoid: Monoid, *args):
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
    """reduce(⊗(F_v ∪ F_rest), {v} ∪ S) = reduce(⊗F_rest ⊗ reduce(⊗F_v, {v}), S)

    where F_v = factors mentioning v, F_rest = the others. Fires only when
    v has no dependents among the remaining streams (so it can be innermost)
    and F_rest is nonempty (universal variables stay in the outer core).
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not (
            is_commutative(monoid)
            and isinstance(body, Term)
            and _is_monoid_plus(body.op)
            and distributes_over(body.op.__self__, monoid)
        ):
            return fwd()

        inner = body.op.__self__
        stream_keys = set(streams)
        factors = [(a, fvsof(a)) for a in body.args]

        # candidates: innermost-eligible (no remaining stream depends on v),
        # non-universal (some factor doesn't mention v)
        support: dict = {}
        for v in streams:
            if any(v in fvsof(s) for k, s in streams.items() if k is not v):
                continue
            f_v = frozenset(i for i, (_, fvs) in enumerate(factors) if v in fvs)
            if len(f_v) == len(factors):
                continue  # v is universal: leave it in the outer core
            support[v] = f_v

        # eliminate a variable with subset-minimal factor support
        # (leaves-first; canonical on hierarchical/laminar supports)
        inner_stream = None
        inner_factor_ids = None
        for v, f_v in support.items():
            if any(u_sup < f_v for u, u_sup in support.items() if u is not v):
                continue
            inner_stream = v
            inner_factor_ids = f_v
            break

        if not inner_stream or not inner_factor_ids:
            return fwd()

        inner_factors = [factors[i][0] for i in sorted(inner_factor_ids)]
        inner_stream_keys = {inner_stream}
        inner_deps = set().union(
            *(factors[i][1] for i in f_v), fvsof(streams[v]) & stream_keys
        )

        outer_factors = [a for i, (a, _) in enumerate(factors) if i not in f_v]
        outer_stream_keys = stream_keys - inner_stream_keys
        outer_factor_deps = set().union(
            *(vars for i, (_, vars) in enumerate(factors) if i not in f_v)
        )

        # find all streams that are used in the inner factors/streams and are
        # not used by the outer factors/streams
        # this has to be done iteratively, because moving a stream inward
        # reduces the outer dependency set
        # ensures that no future factorization application creates a reduce that
        # fuses with with the inner reduce
        for s in inner_streams_first(streams):
            outer_stream_deps = (
                set().union(*(fvsof(streams[k]) for k in outer_stream_keys))
                & stream_keys
            )
            outer_deps = outer_factor_deps | outer_stream_deps
            if s in inner_deps and s not in outer_deps:
                inner_stream_keys |= {s}
                inner_deps |= stream_keys & fvsof(streams[s])
                outer_stream_keys -= {s}

        inner_streams = {k: v for (k, v) in streams.items() if k in inner_stream_keys}
        inner_red = monoid.reduce(inner.plus(*inner_factors), inner_streams)

        rest_streams = {k: s for k, s in streams.items() if k in outer_stream_keys}
        new_body = inner.plus(*outer_factors, inner_red)
        return monoid.reduce(new_body, rest_streams) if rest_streams else new_body


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


class ReduceWeightedStream(ObjectInterpretation):
    """reduce(M, body, {x: WM.weighted(s, v, w), ...}) = reduce(M, WM.plus(w[v:=x()], body), {x: s, ...})

    requires distributes_over(WM, M).

    The substitution ``v -> x`` is done by beta-reducing ``deffn(w, v)`` on
    ``x()`` — symbolic, no Python dispatch on the weight expression.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        for k, v in streams.items():
            if isinstance(v, Term) and _is_monoid_weighted(v.op):
                v_stream, v_weight = v.args
                v_monoid = v.op.__self__
                if not distributes_over(v_monoid, monoid):
                    continue
                w_at_k = v_weight(k())
                weighted_body = v_monoid.plus(w_at_k, body)
                new_streams = {**streams, k: v_stream}
                return monoid.reduce(weighted_body, new_streams)
        return fwd()


class ReduceCartesianWeightedStream(ObjectInterpretation):
    """``CartesianProduct.reduce`` over a :func:`weighted` body whose
    ``weight`` is independent of the plate (product-index) streams::

        CartesianProduct.reduce(M.weighted(s, w), plates)
          = M.weighted(
              CartesianProduct.reduce(s, plates),
              deffn(M.reduce(w, {e: row()}), row),
            )

    Reuses ``body``'s element binder ``e`` (already typed by construction);
    introduces a fresh ``row`` binder typed as ``Iterable[elem_type]``.

    Only fires when ``w`` is independent of the plate vars.
    """

    @Operation.define
    @staticmethod
    def _iterable_elem[T](iter: Iterable[T]) -> T:
        raise NotHandled

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if monoid is not CartesianProduct:
            return fwd()
        if not (isinstance(body, Term) and _is_monoid_weighted(body.op)):
            return fwd()

        s, w = body.args
        if not isinstance(s, Term) and len(s) == 0:
            return CartesianProduct.reduce([], streams)

        if set(streams.keys()) & fvsof(w):
            return fwd()

        elem_typ = typeof(self._iterable_elem(s))
        elem_op = Operation.define(elem_typ, name="elem")
        row_op = Operation.define(Iterable[elem_typ], name="row")

        weight_monoid = body.op.__self__
        joint_weight = deffn(
            weight_monoid.reduce(w(elem_op()), {elem_op: row_op()}), row_op
        )
        joint_stream = CartesianProduct.reduce(s, streams)

        return weight_monoid.weighted(joint_stream, joint_weight)


class MonoidOverCallable(ObjectInterpretation):
    """``monoid.reduce(f, streams) = lambda *a: monoid.reduce(f(*a), streams)``."""

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if isinstance(body, Term) or not isinstance(body, Callable):
            return fwd()
        return lambda *a, **k: monoid.reduce(body(*a, **k), streams)

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not args or any(
            isinstance(arg, Term) or not isinstance(arg, Callable) for arg in args
        ):
            return fwd()
        return lambda *a, **k: monoid.plus(*(arg(*a, **k) for arg in args))


class MonoidOverMapping(ObjectInterpretation):
    """``monoid.reduce({k: v_k}, streams) = {k: monoid.reduce(v_k, streams)}``."""

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if isinstance(body, Term) or not isinstance(body, Mapping):
            return fwd()
        return {k: monoid.reduce(v, streams) for (k, v) in body.items()}

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not args or not isinstance(args[0], Mapping):
            return fwd()

        if isinstance(args[0], Interpretation):
            keys = args[0].keys()
            for b in args[1:]:
                if not isinstance(b, Interpretation):
                    raise TypeError(f"Expected interpretation but got {b}")
                if not keys == b.keys():
                    raise ValueError(
                        f"Expected interpretation of {keys} but got {b.keys()}"
                    )
            return {k: monoid.plus(*(handler(b)(b[k]) for b in args)) for k in keys}

        for b in args[1:]:
            if not isinstance(b, Mapping):
                raise TypeError(f"Expected mapping but got {b}")
        all_values = collections.defaultdict(list)
        for d in args:
            for k, v in d.items():
                all_values[k].append(v)
        return {k: monoid.plus(*vs) for (k, vs) in all_values.items()}


def _scalar_args(args):
    """True iff ``args`` is non-empty and every arg is a concrete int/float."""
    return (
        bool(args)
        and not any(isinstance(x, Term) for x in args)
        and all(isinstance(x, int | float) for x in args)
    )


class SumPlus(ObjectInterpretation):
    """Scalar implementation of :data:`Sum`."""

    @implements(Sum.plus)
    def plus(self, *args):
        if not _scalar_args(args):
            return fwd()
        return sum(args)


class MinPlus(ObjectInterpretation):
    """Scalar implementation of :data:`Min`."""

    @implements(Min.plus)
    def plus(self, *args):
        if not _scalar_args(args):
            return fwd()
        return min(args)


class MaxPlus(ObjectInterpretation):
    """Scalar implementation of :data:`Max`."""

    @implements(Max.plus)
    def plus(self, *args):
        if not _scalar_args(args):
            return fwd()
        return max(args)


class ProductPlus(ObjectInterpretation):
    """Scalar implementation of :data:`Product`."""

    @implements(Product.plus)
    def plus(self, *args):
        if not _scalar_args(args):
            return fwd()
        return functools.reduce(operator.mul, args)


class ArgMinPlus(ObjectInterpretation):
    """Scalar score implementation of :data:`ArgMin`."""

    @implements(ArgMin.plus)
    def plus(self, *args):
        if not args or not all(isinstance(a, tuple) for a in args):
            return fwd()
        if any(isinstance(a[0], Term) for a in args):
            return fwd()
        if not all(isinstance(a[0], int | float) for a in args):
            return fwd()
        return min(args, key=lambda a: a[0])


class ArgMaxPlus(ObjectInterpretation):
    """Scalar score implementation of :data:`ArgMax`."""

    @implements(ArgMax.plus)
    def plus(self, *args):
        if not args or not all(isinstance(a, tuple) for a in args):
            return fwd()
        if any(isinstance(a[0], Term) for a in args):
            return fwd()
        if not all(isinstance(a[0], int | float) for a in args):
            return fwd()
        return max(args, key=lambda a: a[0])


class CartesianProductPlus(ObjectInterpretation):
    """Pure-Python implementation of :data:`CartesianProduct`."""

    @implements(CartesianProduct.plus)
    def plus(self, *args):
        if not args:
            return fwd()
        if any(isinstance(x, Term) for x in args):
            return fwd()
        if not all(isinstance(x, Iterable) for x in args):
            return fwd()

        def to_tuple(x):
            return x if isinstance(x, tuple) else (x,)

        return [
            sum((to_tuple(v) for v in vals), ()) for vals in itertools.product(*args)
        ]


is_scalar = _ExtensiblePredicate({Min, Max, Sum, Product})


class MonoidOverSequence(ObjectInterpretation):
    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if (
            not is_scalar(monoid)
            or not args
            or not isinstance(args[0], tuple | list | Generator)
        ):
            return fwd()
        zipped = zip(*args, strict=True)
        result = (monoid.plus(*vs) for vs in zipped)
        if isinstance(args[0], tuple | list):
            return type(args[0])(result)
        return result

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not is_scalar(monoid) or not isinstance(body, tuple | list | Generator):
            return fwd()
        result = (monoid.reduce(x, streams) for x in body)
        if isinstance(body, tuple | list):
            return type(body)(result)
        return result


class PlusInf(ObjectInterpretation):
    """Workaround for the inability to give Monoid.plus(x, inf) a type."""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if monoid in (Sum, Max) and any(
            not isinstance(x, Term) and x == float("inf") for x in args
        ):
            return float("inf")
        if monoid in (Sum, Min) and any(
            not isinstance(x, Term) and x == -float("inf") for x in args
        ):
            return -float("inf")
        return fwd()


class _ExtensibleInterpretation(UserDict, Interpretation):
    def extend(self, *intps: Interpretation) -> typing.Self:
        for intp in intps:
            self.data = coproduct(self.data, intp)  # type: ignore[assignment]
        return self


NormalizeIntp = _ExtensibleInterpretation().extend(
    MonoidOverSequence(),
    MonoidOverMapping(),
    MonoidOverCallable(),
    ReduceNoStreams(),
    ReduceFusion(),
    ReduceSplit(),
    ReduceFactorization(),
    ReduceDistributeCartesianProduct(),
    ReduceWeightedStream(),
    ReduceCartesianWeightedStream(),
    PlusEmpty(),
    PlusSingle(),
    PlusAssoc(),
    PlusDistr(),
    PlusConsecutiveDups(),
    PlusDups(),
    SumPlus(),
    MinPlus(),
    MaxPlus(),
    ProductPlus(),
    ArgMinPlus(),
    ArgMaxPlus(),
    CartesianProductPlus(),
    PlusInf(),
)
"""``NormalizeIntp``applies pure-Term rewrites (associativity, distributivity,
identity elimination, fusion, factorization, etc.).

"""
