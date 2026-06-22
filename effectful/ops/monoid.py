import collections.abc
import functools
import itertools
import operator
import typing
from collections import Counter, UserDict, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any

from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import (
    Array,
    ObjectInterpretation,
    Scoped,
    _NumberTerm,
    defdata,
    deffn,
    getitem,
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

    __name__: str
    identity: W

    def __init__(self, identity: W, name: str):
        self.__name__ = name
        self.identity = identity

    def __repr__(self):
        return f"Monoid({self.__name__!r}, {self.identity!r})"

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

    @Operation.define
    def mask(self, value: W, cond: bool = True) -> W:
        raise NotHandled

    @Operation.define
    def delta[K](self, index: K, weight: W) -> Mapping[K, W]:
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
CartesianProduct = MonoidWithZero(name="CartesianProduct", identity=[{}], zero=[])
Union = Monoid(name="Union", identity=[])
And = MonoidWithZero(name="And", identity=True, zero=False)
Or = Monoid(name="And", identity=False)


@dataclass
class _ExtensiblePredicate[T]:
    elems: set[T]

    def register(self, t: T) -> None:
        self.elems.add(t)

    def __call__(self, t: T) -> bool:
        return t in self.elems


is_commutative = _ExtensiblePredicate({Max, Min, Sum, Product, And, Or})
is_idempotent = _ExtensiblePredicate({Max, Min, And, Or})


@dataclass
class _ExtensibleBinaryRelation[S, T]:
    tuples: set[tuple[S, T]]

    def register(self, s: S, t: T) -> None:
        self.tuples.add((s, t))

    def __call__(self, s: S, t: T) -> bool:
        return (s, t) in self.tuples


distributes_over = _ExtensibleBinaryRelation(
    {
        (Max, Min),
        (Min, Max),
        (Sum, Min),
        (Sum, Max),
        (Product, Sum),
        (CartesianProduct, Union),
        (And, Or),
    }
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


def _is_monoid_mask(op: Operation) -> bool:
    """True if ``op`` is the ``mask`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.mask


def _is_monoid_delta(op: Operation) -> bool:
    """True if ``op`` is the ``delta`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.delta


class DeltaEmpty(ObjectInterpretation):
    """delta((), weight) ≡ weight"""

    @implements(Monoid.delta)
    def _(self, monoid, index, weight):
        if not index:
            return weight
        return fwd()


class DeltaFusion(ObjectInterpretation):
    """delta(i1, delta(i2, weight)) ≡ delta(i1 ++ i2, weight)"""

    @implements(Monoid.delta)
    def _(self, monoid, index, weight):
        if (
            isinstance(weight, Term)
            and _is_monoid_delta(weight.op)
            and weight.op.__self__ == monoid
        ):
            return monoid.delta(index + weight.args[0], weight.args[1])
        return fwd()


class MaskTrue(ObjectInterpretation):
    """mask(value, True) ≡ value"""

    @implements(Monoid.mask)
    def _(self, monoid, value, cond):
        if not isinstance(cond, Term) and cond:
            return value
        return fwd()


class MaskFalse(ObjectInterpretation):
    """M.mask(value, False) ≡ M.identity"""

    @implements(Monoid.mask)
    def _(self, monoid, value, cond):
        if not isinstance(cond, Term) and not cond:
            return monoid.identity
        return fwd()


class MaskFusion(ObjectInterpretation):
    """M.mask(M.mask(value, i1), i2) ≡ M.mask(value, And.plus(i1, i2))"""

    @implements(Monoid.mask)
    def _(self, monoid, value, cond):
        if (
            isinstance(value, Term)
            and _is_monoid_mask(value.op)
            and value.op.__self__ == monoid
        ):
            return monoid.mask(value.args[0], And.plus(value.args[1], cond))
        return fwd()


class MaskDistr(ObjectInterpretation):
    @implements(Monoid.mask)
    def _(self, monoid, value, cond):
        if (
            isinstance(value, Term)
            and _is_monoid_mask(value.op)
            and value.op.__self__ == monoid
        ):
            return monoid.mask(value.args[0], And.plus(value.args[1], cond))
        return fwd()


class ReduceEqualityMaskRange(ObjectInterpretation):
    """M.reduce(M.mask(v, And.plus(i = x, *m)), {i: range(N)} ∪ S) ≡
    M.mask(M.reduce(M.mask(v, *m), {i: [x]} ∪ S), And.plus(0 <= x, x < N))

    """

    @implements(Monoid.reduce)
    def _(self, monoid, body, streams):
        if not (
            isinstance(body, Term)
            and _is_monoid_mask(body.op)
            and body.op.__self__ == monoid
        ):
            return fwd()

        (value, mask) = body.args
        match mask:
            case Term(And.plus, conds, {}):
                ...
            case cond:
                conds = (cond,)

        for i, cond in enumerate(conds):

            def test(stream_op, mask_key):
                return (
                    stream_op in streams
                    and isinstance(stream := streams[stream_op], range)
                    and stream.start == 0
                    and stream.step == 1
                    and not (fvsof(mask_key) & set(streams))
                )

            match cond:
                case Term(
                    _NumberTerm.__eq__, ((Term(stream_op, (), {}), mask_key)), {}
                ) if test(stream_op, mask_key):
                    ...
                case Term(
                    _NumberTerm.__eq__, ((mask_key, Term(stream_op, (), {}))), {}
                ) if test(stream_op, mask_key):
                    ...
                case _:
                    continue

            stream = streams[stream_op]
            return monoid.mask(
                monoid.reduce(
                    monoid.mask(
                        value, And.plus(*(c for (j, c) in enumerate(conds) if i != j))
                    ),
                    {stream_op: (mask_key,)}
                    | {k: v for (k, v) in streams.items() if k != stream_op},
                ),
                And.plus(stream.start <= mask_key, mask_key < stream.stop),
            )
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


class ReducePartial(ObjectInterpretation):
    @implements(Monoid.reduce)
    def _(self, monoid, body, streams):
        if not streams:
            return monoid.identity

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
                        monoid.reduce(*eval_args) if streams_tail else eval_args[0]
                    )
            return monoid.plus(*new_reduces)
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


@Operation.define
def choose_contraction(factors: Sequence[Any], streams: Streams) -> Operation:
    """Used by `ReduceFactorization` to choose a contraction when there is
    ambiguity. Takes the factors and streams that are eligible for contraction
    (innermost and non-universal).

    The default behavior is to return the first support-minimal stream in the
    streams dictionary.

    """
    assert len(streams) > 0

    factors = [(a, fvsof(a)) for a in factors]
    support: dict = {
        k: frozenset(i for i, (_, fvs) in enumerate(factors) if k in fvs)
        for k in streams
    }
    for v, f_v in support.items():
        if any(u_sup < f_v for u, u_sup in support.items() if u is not v):
            continue
        return v
    assert False, "expected at least one subset-minimal stream"


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
        eligible = {}
        for k, v in streams.items():
            if any(k in fvsof(vv) for kk, vv in streams.items() if k is not kk):
                continue
            if len({i for i, (_, fvs) in enumerate(factors) if k in fvs}) == len(
                factors
            ):
                continue  # v is universal: leave it in the outer core
            eligible[k] = v

        if not eligible:
            return fwd()
        if len(eligible) == 1:
            inner_stream = next(iter(eligible))
        else:
            inner_stream = choose_contraction(body.args, eligible)

        inner_factor_ids = frozenset(
            i for i, (_, fvs) in enumerate(factors) if inner_stream in fvs
        )

        inner_factors = [factors[i][0] for i in sorted(inner_factor_ids)]
        inner_stream_keys = {inner_stream}
        inner_deps = set().union(
            *(factors[i][1] for i in inner_factor_ids),
            fvsof(streams[inner_stream]) & stream_keys,
        )

        outer_factors = [
            a for i, (a, _) in enumerate(factors) if i not in inner_factor_ids
        ]
        outer_stream_keys = stream_keys - inner_stream_keys
        outer_factor_deps = set().union(
            *(vars for i, (_, vars) in enumerate(factors) if i not in inner_factor_ids)
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
    def reduce(self, monoid: Monoid, body, streams):
        if not (isinstance(body, Term)):
            return fwd()

        if not _is_monoid_reduce(body.op):
            return fwd()

        inner_monoid = body.op.__self__
        if not distributes_over(inner_monoid, monoid):
            return fwd()
        inner_body, inner_streams = body.args

        for stream_key, stream_body in streams.items():
            # stream is cartesian
            if not (
                isinstance(stream_body, Term)
                and stream_body.op is CartesianProduct.reduce
            ):
                continue

            (cprod_body, cprod_streams) = stream_body.args

            # plates are rectangular
            if not all(
                isinstance(plate_stream, range)
                for plate_stream in cprod_streams.values()
            ):
                continue

            if stream_key in fvsof(inner_streams):
                continue

            # stream body is a sequence of mappings from plate index to domain value
            match cprod_body:
                case Term(
                    Union.reduce,
                    ([Term(Union.delta, (idx, union_body), {})], union_streams),
                    {},
                ) if set(i.op for i in idx if isinstance(i, Term)) >= set(
                    cprod_streams
                ):
                    pass
                case _:
                    continue

            assert len(idx) > 0

            # inner product folds over all plates
            plate_index, plate_op = next(
                (j, i.op) for (j, i) in enumerate(idx) if i.op in cprod_streams
            )
            plate_range = cprod_streams[plate_op]
            inner_plate_op = None

            class InvalidIndexError(Exception): ...

            def drop_elem(ls, index):
                return tuple(x for (i, x) in enumerate(ls) if i != index)

            # substitute all instances of row[i, *rest] -> row[*rest]
            def _getitem(mapping, idx1):
                nonlocal inner_plate_op

                if isinstance(mapping, Term) and mapping.op == stream_key:
                    if not (
                        isinstance(idx1, Sequence)
                        and len(idx1) > plate_index
                        and isinstance(idx1[plate_index], Term)
                        and idx1[plate_index].op in inner_streams
                        and syntactic_eq(
                            inner_streams[idx1[plate_index].op], plate_range
                        )
                    ):
                        raise InvalidIndexError()

                    if inner_plate_op is None:
                        inner_plate_op = idx1[plate_index].op
                    elif inner_plate_op != idx1[plate_index].op:
                        raise InvalidIndexError()

                    return fwd(mapping, drop_elem(idx1, plate_index))
                return fwd()

            row_subst = {getitem: _getitem}
            try:
                subst_inner_body = handler(row_subst)(evaluate)(inner_body)
            except InvalidIndexError:
                continue

            peeled_body = Union.reduce(
                [Union.delta(drop_elem(idx, plate_index), union_body)], union_streams
            )
            peeled_cprod_streams = {
                k: v for (k, v) in cprod_streams.items() if k != plate_op
            }
            if not peeled_cprod_streams:
                peeled_cprod = peeled_body
            else:
                peeled_cprod = CartesianProduct.reduce(
                    peeled_body, peeled_cprod_streams
                )

            # include any extra inner product streams
            inner_tail_streams = {
                k: v for (k, v) in inner_streams.items() if k != inner_plate_op
            }
            if inner_tail_streams:
                inner_reduce = inner_monoid.reduce(
                    subst_inner_body,
                    inner_tail_streams,
                )
            else:
                inner_reduce = subst_inner_body

            peeled_reduce = inner_monoid.reduce(
                monoid.reduce(inner_reduce, {stream_key: peeled_cprod}),
                {k: v for (k, v) in inner_streams.items() if k == inner_plate_op},
            )

            # include any extra sum streams outermost
            tail_streams = {k: v for (k, v) in streams.items() if k != stream_key}
            if tail_streams:
                result = monoid.reduce(peeled_reduce, tail_streams)
            else:
                result = peeled_reduce

            return result

        return fwd()


class ReduceUnion(ObjectInterpretation):
    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        for k, v in streams.items():
            if isinstance(v, Term) and v.op == Union.reduce:
                union_body, union_streams = v.args
                return monoid.reduce(body, streams | {k: union_body} | union_streams)
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


def _disjoint_merge[K, V](*dicts: Mapping[K, V]) -> Mapping[K, V]:
    merged = {}
    for d in dicts:
        for key, value in d.items():
            if key in merged:
                raise ValueError(f"Duplicate key found: '{key}'")
        merged[key] = value
    return merged


class CartesianProductPlus(ObjectInterpretation):
    """Pure-Python implementation of :data:`CartesianProduct`."""

    @implements(CartesianProduct.plus)
    def plus(self, *args):
        assert args
        if any(isinstance(x, Term) for x in args):
            return fwd()
        if not all(isinstance(x, Iterable) for x in args):
            return fwd()
        return [_disjoint_merge(*vals) for vals in itertools.product(*args)]


class UnionPlus(ObjectInterpretation):
    @implements(Union.plus)
    def plus(self, *args):
        assert args
        if any(isinstance(x, Term) for x in args):
            return fwd()
        if not all(isinstance(x, Iterable) for x in args):
            return fwd()
        return list(itertools.chain(*args))


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


@Operation.define
def as_float(x: int) -> float:
    if isinstance(x, Term):
        raise NotHandled
    return float(x)


class PlusCastFloat(ObjectInterpretation):
    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        typs = [typeof(a) for a in args]
        if any(issubclass(t, float) for t in typs) and any(
            issubclass(t, int) for t in typs
        ):
            args = [
                as_float(a) if issubclass(t, int) else a
                for (a, t) in zip(args, typs, strict=True)
            ]
            return monoid.plus(*args)
        return fwd()


class EliminateSingletonStreams(ObjectInterpretation):
    """Eliminate a length-1 stream by substituting its sole element.

    reduce(M, body, {k: (v,)} ∪ S) = reduce(M, body[k := v], S[k := v])

    Fires only when the sole element ``v`` is a :class:`Term`, i.e. a *symbolic*
    singleton. This is exactly the form ``ReduceArrayGather`` produces (a gather
    ``(a[i()],)``) and, more generally, every dependent singleton that
    :class:`ReducePartial` cannot peel -- a non-outermost stream whose element
    references another stream var. Concrete enumerated streams (``[0]``,
    ``range(1)``) and monoid sentinels (``CartesianProduct.identity == [()]``)
    have non-``Term`` elements and are left to ``ReducePartial`` / the
    per-monoid rules.

    Unlike ``ReducePartial``, this peels the stream wherever it sits in the loop
    nest and substitutes symbolically rather than unrolling, leaving a
    vectorized index range (e.g. the gather's range) intact instead of
    materializing it.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        # Eliminate *all* symbolic length-1 streams in one pass via a
        # simultaneous substitution. Doing them together (rather than one per
        # invocation) keeps an interleaving reduction rule -- e.g.
        # ``ReduceArray`` consuming a now-live index range -- from firing
        # between eliminations, so sibling index ranges stay together and fuse
        # into a single reduction.
        singletons = {
            k: vs[0]
            for k, vs in streams.items()
            if not isinstance(vs, Term)
            and isinstance(vs, collections.abc.Sequence)
            and len(vs) == 1
            and isinstance(vs[0], Term)
        }
        if not singletons:
            return fwd()

        subs = {k: deffn(v) for k, v in singletons.items()}
        new_body = handler(subs)(evaluate)(body)
        new_streams = {
            kk: handler(subs)(evaluate)(vv)
            for kk, vv in streams.items()
            if kk not in singletons
        }
        # reduce over no streams is a single (empty) assignment, i.e. the body
        # itself -- not the monoid identity.
        return monoid.reduce(new_body, new_streams) if new_streams else new_body


class _ExtensibleInterpretation(UserDict, Interpretation):
    def extend(self, *intps: Interpretation) -> typing.Self:
        for intp in intps:
            self.data = coproduct(self.data, intp)  # type: ignore[assignment]
        return self


EvaluateIntp = _ExtensibleInterpretation().extend(
    ReducePartial(),
    SumPlus(),
    MinPlus(),
    MaxPlus(),
    ProductPlus(),
    ArgMinPlus(),
    ArgMaxPlus(),
    CartesianProductPlus(),
    UnionPlus(),
)

NormalizeIntp = _ExtensibleInterpretation().extend(
    DeltaEmpty(),
    DeltaFusion(),
    MonoidOverSequence(),
    MonoidOverMapping(),
    MonoidOverCallable(),
    ReduceFusion(),
    ReduceUnion(),
    # ReduceSplit(),
    ReduceDistributeCartesianProduct(),
    ReduceFactorization(),
    ReduceWeightedStream(),
    ReduceCartesianWeightedStream(),
    ReduceEqualityMaskRange(),
    EliminateSingletonStreams(),
    PlusEmpty(),
    PlusSingle(),
    PlusAssoc(),
    PlusDistr(),
    PlusConsecutiveDups(),
    PlusDups(),
    PlusCastFloat(),
    MaskTrue(),
    MaskFalse(),
    MaskFusion(),
)
"""``NormalizeIntp``applies pure-Term rewrites (associativity, distributivity,
identity elimination, fusion, factorization, etc.).

"""
