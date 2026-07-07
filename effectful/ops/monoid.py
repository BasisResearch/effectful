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
    _ArrayTerm,
    _NumberTerm,
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
    def mask(self, value: W, cond: Any = True) -> W:
        raise NotHandled

    @Operation.define
    def delta(self, index: tuple[int, ...], weight: W) -> Array:
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
Or = Monoid(name="Or", identity=False)


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


def _leaf_vars(expr: Any) -> set[Operation]:
    """The free variables that appear as nullary leaves (``var()``) in ``expr``.

    Unlike :func:`fvsof`, this excludes operator/function symbols (``__eq__``,
    getitem, a factor's callable head, ...) and keeps only the value-carrying
    leaf variables -- every variable occurrence is a nullary call, so this is
    exactly the set of variables the expression *mentions*. Used to decide which
    factors a mask conjunct actually constrains.
    """
    result: set[Operation] = set()

    def walk(e: Any) -> None:
        if isinstance(e, Term):
            if not e.args and not e.kwargs:
                result.add(e.op)
            else:
                for a in e.args:
                    walk(a)
                for v in e.kwargs.values():
                    walk(v)
        elif isinstance(e, tuple | list):
            for x in e:
                walk(x)
        elif isinstance(e, Mapping):
            for x in e.values():
                walk(x)

    walk(expr)
    return result


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


class GetitemDelta(ObjectInterpretation):
    """M.delta(i, v)[q] ≡ M.mask(i, v == q)"""

    @implements(_ArrayTerm.__getitem__)
    def _(self, value, index):
        if isinstance(value, Term) and _is_monoid_delta(value.op):
            return value.op.__self__.mask(value.args[1], value.args[0] == index)
        return fwd()


class MaskBool(ObjectInterpretation):
    @implements(Monoid.mask)
    def _(self, monoid, value, cond):
        if isinstance(cond, bool):
            return value if cond else monoid.identity
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


is_equality = _ExtensiblePredicate({_NumberTerm.__eq__})


class ReduceEqualityMaskRange(ObjectInterpretation):
    """M.reduce(M.mask(v, And.plus(i = x, *m)), {i: range(N)} ∪ S) ≡
    M.mask(M.reduce(M.mask(v, *m), {i: [x]} ∪ S), And.plus(0 <= x, x < N))

    The equality constraint ``i = x`` on a range-stream reduce is discharged by
    a gather (the stream becomes the singleton ``[x]``) guarded by a bounds
    check.

    When the reduce body is a ``plus`` of the same monoid, the rule distributes
    the reduce over the plus -- but only when doing so exposes an eliminable
    equality mask in some summand. This is a *targeted* split (it leaves
    ``ReduceSplit`` conservative): summands whose reduced index appears in an
    equality become gathers, while the rest stay as ordinary masked reduces.
    """

    @staticmethod
    def _conds(mask):
        match mask:
            case Term(And.plus, conds, {}):
                return conds
            case cond:
                return (cond,)

    @staticmethod
    def _match_eq(cond, streams):
        """If ``cond`` is ``stream_op == key`` (either order) where ``stream_op``
        is a ``range(0, N)`` stream and ``key`` is stream-independent, return
        ``(stream_op, key)``; otherwise ``None``."""

        def test(op, stream_op, mask_key):
            return (
                is_equality(op)
                and stream_op in streams
                and isinstance(stream := streams[stream_op], range)
                and stream.start == 0
                and stream.step == 1
                and not (fvsof(mask_key) & set(streams))
            )

        match cond:
            case Term(op, (Term(stream_op, (), {}), mask_key), {}) if test(
                op, stream_op, mask_key
            ):
                return (stream_op, mask_key)
            case Term(op, (mask_key, Term(stream_op, (), {})), {}) if test(
                op, stream_op, mask_key
            ):
                return (stream_op, mask_key)
            case _:
                return None

    def _eliminate(self, monoid, value, mask, streams):
        """Discharge one eliminable equality constraint via a gather, or return
        ``None`` if no constraint is eliminable."""
        conds = self._conds(mask)
        for i, cond in enumerate(conds):
            matched = self._match_eq(cond, streams)
            if matched is None:
                continue
            stream_op, mask_key = matched
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
        return None

    def _summand_eliminable(self, monoid, summand, streams):
        return (
            isinstance(summand, Term)
            and _is_monoid_mask(summand.op)
            and summand.op.__self__ == monoid
            and self._eliminate(monoid, summand.args[0], summand.args[1], streams)
            is not None
        )

    @implements(Monoid.reduce)
    def _(self, monoid, body, streams):
        if not isinstance(body, Term):
            return fwd()

        # single mask body: discharge an equality constraint directly
        if _is_monoid_mask(body.op) and body.op.__self__ == monoid:
            result = self._eliminate(monoid, body.args[0], body.args[1], streams)
            return result if result is not None else fwd()

        # plus body: distribute the reduce only when it exposes an eliminable
        # equality mask in some summand
        if _is_monoid_plus(body.op) and body.op.__self__ == monoid:
            if any(self._summand_eliminable(monoid, s, streams) for s in body.args):
                return monoid.plus(*(monoid.reduce(s, streams) for s in body.args))

        return fwd()


class ReduceMaskHoist(ObjectInterpretation):
    """M.reduce(M.mask(v, c), S) ≡ M.mask(M.reduce(v, S), c) when ``c`` does not
    depend on any stream in ``S``.

    A reduce-stream-independent condition gates the whole reduction uniformly,
    so it can be lifted out: when ``c`` holds both sides are ``reduce(v, S)``,
    and when it fails both are the identity (``reduce(identity, S) = identity``).
    This holds for any monoid -- no commutativity or distributivity required.
    """

    @implements(Monoid.reduce)
    def _(self, monoid, body, streams):
        if not (
            streams
            and isinstance(body, Term)
            and _is_monoid_mask(body.op)
            and body.op.__self__ == monoid
        ):
            return fwd()
        value, cond = body.args
        if fvsof(cond) & set(streams):
            return fwd()
        return monoid.mask(monoid.reduce(value, streams), cond)


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
    """x * (y + z) = x * y + x * z"""

    @implements(Monoid.plus)
    def plus(self, monoid: Monoid, *args):
        if len(args) < 2:
            return fwd()

        for i, a in enumerate(args):
            if (
                isinstance(a, Term)
                and _is_monoid_plus(a.op)
                and distributes_over(monoid, (inner_monoid := a.op.__self__))
                and not distributes_over(inner_monoid, monoid)
            ):
                if i > 0:
                    return monoid.plus(
                        *args[: i - 1],
                        inner_monoid.plus(
                            *(monoid.plus(args[i - 1], x) for x in a.args)
                        ),
                        *args[i + 1 :],
                    )
                else:
                    return monoid.plus(
                        inner_monoid.plus(
                            *(monoid.plus(x, args[i + 1]) for x in a.args)
                        ),
                        *args[i + 2 :],
                    )
        return fwd()


class PlusOrder(ObjectInterpretation):
    """Normalize plus ordering for commutative monoids.

    x ⊕ y ⊕ x = x ⊕ x ⊕ y

    """

    @staticmethod
    def _term_sort_key(t: Term) -> tuple[int, int]:
        return (syntactic_hash(t), id(t))

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not is_commutative(monoid):
            return fwd()

        sorted_args = tuple(
            sorted(range(len(args)), key=lambda i: self._term_sort_key(args[i]))
        )
        if sorted_args == tuple(range(len(args))):
            return fwd()
        return monoid.plus(*(args[i] for i in sorted_args))


class PlusConsecutiveDups(ObjectInterpretation):
    """Normalize duplicate arguments for idempotent monoids.

    x ⊕ x ⊕ y = x ⊕ y

    """

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not is_idempotent(monoid):
            return fwd()

        dedup_args = tuple(
            i
            for i in range(len(args))
            if i == 0 or not syntactic_eq(args[i - 1], args[i])
        )
        if dedup_args == tuple(range(len(args))):
            return fwd()
        return monoid.plus(*(args[i] for i in dedup_args))


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
    """Confines a stream to the summands that use it, factorization-style.

    For an innermost stream ``v`` used by a strict, non-empty subset of the
    summands, this implements

        reduce(R, ⊕(B_T, B_R), {v} ∪ S)
            = reduce(R, ⊕(reduce(R, B_T, {v}), reduce(R, B_R, {v})), S)

    where ``B_T`` are the summands that mention ``v`` and ``B_R`` the rest. Both
    sides equal ``⊕_S ⊕_v (B_T ⊕ B_R)`` so this is correct for any commutative
    monoid (no cardinality term is needed -- both groups stay under the
    ``{v}``-reduce). The outer reduce sheds ``v`` and the ``v``-users are
    isolated for further factorization.

    The rule deliberately does *not* fire when ``v`` is used by every summand:
    such reduces stay fused, which is what :class:`ReduceDistributeCartesianProduct`
    expects. Repeated application confines each subset-used stream in turn and
    terminates once every remaining stream is used by all (or no) summands.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not is_commutative(monoid):
            return fwd()
        if not (isinstance(body, Term) and body.op is monoid.plus):
            return fwd()

        summand_fvs = [fvsof(arg) for arg in body.args]

        for v in streams:
            # innermost-eligible: no other stream depends on v
            if any(v in fvsof(s) for k, s in streams.items() if k is not v):
                continue
            # used by a strict, non-empty subset of the summands
            users = [v in fvs for fvs in summand_fvs]
            if all(users) or not any(users):
                continue

            inner_streams = {v: streams[v]}
            outer_streams = {k: s for k, s in streams.items() if k is not v}

            def group(args):
                return args[0] if len(args) == 1 else monoid.plus(*args)

            using = [arg for arg, used in zip(body.args, users) if used]
            rest = [arg for arg, used in zip(body.args, users) if not used]
            body2 = monoid.plus(
                monoid.reduce(group(using), inner_streams),
                monoid.reduce(group(rest), inner_streams),
            )
            return monoid.reduce(body2, outer_streams) if outer_streams else body2

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


class Factor(ObjectInterpretation):
    @implements(Monoid.mask)
    def _(self, monoid, value, cond):
        """Scope a mask to the factors its condition constrains -- the ``mask``
        analog of :class:`ReduceFactorization`.

        Peel one conjunct ``c`` that *some but not all* factors mention, wrapping
        just those factors::

            M.mask(WM.plus(F_c ∪ F_rest), And.plus(c, *rest))
              = M.mask(WM.plus(F_rest, M.mask(WM.plus(F_c), c)), And.plus(*rest))

        where ``F_c`` are the factors mentioning a variable of ``c`` and ``F_rest``
        the rest. Sound by the same annihilator argument as the old ``MaskPushPlus``:
        for every registered distributing pair the additive identity ``M.identity``
        is the zero of ``WM``, so when ``c`` fails the inner mask collapses to
        ``M.identity == WM.zero`` and annihilates the product, matching the
        masked-away left side.

        Like :class:`ReduceFactorization`, this peels one conjunct per firing and
        re-enters; the outer conjunct set strictly shrinks, so it terminates. The
        inner ``M.mask(WM.plus(F_c), c)`` has a single conjunct that all its factors
        mention, so it is immediately ineligible -- terminal by construction.

        A conjunct every factor mentions (``F_rest`` empty) or none mentions
        (``F_c`` empty) is not eligible and stays put: a cross-factor equality is
        left grouped on the whole product (the shape
        :class:`ReduceEqualityMaskRange` discharges with a single gather), and an
        orphan condition stays as an outer mask -- dropping it would be unsound. Two
        conjuncts on the same factor land as nested masks that :class:`MaskFusion`
        re-merges.

        Landing a mask adjacent to a single array factor is what lets it fuse with a
        gather during stream elimination. This is the inverse of hoisting a mask out
        of a plus.
        """
        if not (
            isinstance(value, Term)
            and _is_monoid_plus(value.op)
            and distributes_over(value.op.__self__, monoid)
        ):
            return fwd()

        plus_monoid = value.op.__self__
        factors = value.args
        conds = cond.args if isinstance(cond, Term) and cond.op is And.plus else (cond,)
        factor_fvs = [_leaf_vars(f) for f in factors]

        for i, c in enumerate(conds):
            c_fvs = _leaf_vars(c)
            inside = {j for j, ffvs in enumerate(factor_fvs) if c_fvs & ffvs}
            # eligible iff c constrains some but not all factors, so F_rest can
            # be pulled out. all-factors (grouped equality) and no-factor
            # (orphan) conjuncts stay put.
            if not inside or len(inside) == len(factors):
                continue

            f_c = [f for j, f in enumerate(factors) if j in inside]
            f_rest = [f for j, f in enumerate(factors) if j not in inside]
            rest = tuple(d for k, d in enumerate(conds) if k != i)

            inner = monoid.mask(plus_monoid.plus(*f_c) if len(f_c) > 1 else f_c[0], c)
            outer_val = plus_monoid.plus(*f_rest, inner)
            if not rest:
                return outer_val
            return monoid.mask(
                outer_val, rest[0] if len(rest) == 1 else And.plus(*rest)
            )

        return fwd()

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        """reduce(⊗(F_v ∪ F_rest), {v} ∪ S) = reduce(⊗F_rest ⊗ reduce(⊗F_v, {v}), S)

        where F_v = factors mentioning v, F_rest = the others. Fires only when
        v has no dependents among the remaining streams (so it can be innermost)
        and F_rest is nonempty (universal variables stay in the outer core).

        A reduce-monoid mask wrapping the plus is handled too:

            reduce(M.mask(⊗(F_v ∪ F_rest), c), {v} ∪ S)
                = reduce(M.mask(⊗F_rest ⊗ reduce(⊗F_v, {v}), c), S)

        Because the mask gates the whole product, every factor effectively depends
        on the condition's variables; folding ``fvsof(c)`` into each factor's free
        variables keeps those streams in the outer core (a stream the mask depends
        on is treated as universal and never pulled into the inner reduce). The
        mask therefore stays outside the inner reduce unchanged. Soundness relies
        on ``M.identity`` annihilating the inner monoid's plus (the semiring zero),
        so masking distributes over the inner product.
        """
        if not (is_commutative(monoid) and isinstance(body, Term)):
            return fwd()

        # Optionally peel an outer mask of the reduce monoid.
        cond = None
        plus_term = body
        if _is_monoid_mask(body.op) and body.op.__self__ is monoid:
            plus_term, cond = body.args

        if not (
            isinstance(plus_term, Term)
            and _is_monoid_plus(plus_term.op)
            and distributes_over(plus_term.op.__self__, monoid)
        ):
            return fwd()

        inner = plus_term.op.__self__
        stream_keys = set(streams)
        cond_fvs = fvsof(cond) if cond is not None else set()
        factors = [(a, fvsof(a) | cond_fvs) for a in plus_term.args]

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
            inner_stream = choose_contraction(plus_term.args, eligible)

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
        if cond is not None:
            new_body = monoid.mask(new_body, cond)
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

            row_subst = {_ArrayTerm.__getitem__: _getitem}
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
        if isinstance(d, Term):
            return fwd()
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


is_scalar = _ExtensiblePredicate({Min, Max, Sum, Product, And, Or})


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


class SplitDisjointProduct(ObjectInterpretation):
    def _is_var(self, term: Term) -> bool:
        return not term.args and not term.kwargs

    def _var_comps(
        self, terms: list[Term], op: Operation
    ) -> tuple[set[tuple[Operation, Operation]], list[Term]]:
        comps = set()
        resid = []
        for t in terms:
            if (
                isinstance(t, Term)
                and t.op == op
                and all(self._is_var(a) for a in t.args)
            ):
                comps.add(tuple(sorted(a.op for a in t.args)))
            else:
                resid.append(t)
        return comps, resid

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        match body:
            case Term(body_op, (lhs, rhs), {}) if _is_monoid_plus(
                body_op
            ) and distributes_over(monoid, body.op.__self__):
                pass

            case _:
                return fwd()

        def match(lhs, rhs):
            if not (
                isinstance(lhs, Term)
                and _is_monoid_mask(lhs.op)
                and lhs.op.__self__ == body.op.__self__
            ):
                return None
            lhs_op, (lhs_value, lhs_mask) = lhs.op, lhs.args
            lhs_mask_terms = (
                lhs_mask.args
                if isinstance(lhs_mask, Term) and lhs_mask.op == And.plus
                else [lhs_mask]
            )
            (eqs, lhs_residual) = self._var_comps(lhs_mask_terms, _NumberTerm.__eq__)

            if not (
                isinstance(rhs, Term)
                and _is_monoid_mask(rhs.op)
                and rhs.op.__self__ == body.op.__self__
            ):
                return None
            rhs_op, (rhs_value, rhs_mask) = rhs.op, rhs.args
            rhs_mask_terms = (
                rhs_mask.args
                if isinstance(rhs_mask, Term) and rhs_mask.op == Or.plus
                else [rhs_mask]
            )
            (nes, rhs_residual) = self._var_comps(rhs_mask_terms, _NumberTerm.__ne__)

            return (
                lhs_op,
                lhs_value,
                eqs,
                lhs_residual,
                rhs_op,
                rhs_value,
                nes,
                rhs_residual,
            )

        pat = match(lhs, rhs) or match(rhs, lhs)
        if not pat:
            return fwd()

        lhs_op, lhs_value, eqs, lhs_residual, rhs_op, rhs_value, nes, rhs_residual = pat
        shared = eqs & nes

        if not shared or (
            (fvsof(lhs_residual) | fvsof(rhs_residual))
            & set().union(*(set(x) for x in shared))
        ):
            return fwd()

        rhs_inner = [*rhs_residual, *(x() != y() for (x, y) in nes - shared)]
        result = monoid.reduce(
            monoid.plus(
                monoid.mask(
                    lhs_op.__self__.mask(
                        lhs_value,
                        And.plus(
                            *lhs_residual, *(x() == y() for (x, y) in eqs - shared)
                        ),
                    ),
                    And.plus(*(x() == y() for (x, y) in shared)),
                ),
                monoid.mask(
                    rhs_op.__self__.mask(
                        rhs_value,
                        Or.plus(*rhs_inner) if rhs_inner else True,
                    ),
                    Or.plus(*(x() != y() for (x, y) in shared)),
                ),
            ),
            streams,
        )
        return result


class MaskOrderStreamOps(ObjectInterpretation):
    """Rearrange mask terms so that stream operations appear as the first
    argument to binary operations.

    """

    flipped_ops = {
        _NumberTerm.__eq__: _NumberTerm.__eq__,
        _NumberTerm.__ne__: _NumberTerm.__ne__,
        _NumberTerm.__lt__: _NumberTerm.__gt__,
        _NumberTerm.__gt__: _NumberTerm.__lt__,
        _NumberTerm.__le__: _NumberTerm.__ge__,
        _NumberTerm.__ge__: _NumberTerm.__le__,
    }

    @implements(Monoid.reduce)
    def _(self, monoid, body, streams):
        match body:
            case Term(mask_op, (value, mask), {}) if (
                _is_monoid_mask(mask_op) and mask_op.__self__ == monoid
            ):
                pass
            case _:
                return fwd()

        match mask:
            case Term(And.plus, mask_elems, {}):
                pass
            case _:
                mask_elems = (mask,)

        def is_stream_op(x):
            return isinstance(x, Term) and x.op in streams

        new_mask_elems = []
        progress = False
        for elem in mask_elems:
            match elem:
                case Term(op, (lhs, rhs), {}) if (
                    op in self.flipped_ops
                    and is_stream_op(rhs)
                    and not is_stream_op(lhs)
                ):
                    new_mask_elems.append(self.flipped_ops[op](rhs, lhs))
                    progress = True
                case _:
                    new_mask_elems.append(elem)

        return (
            monoid.reduce(monoid.mask(value, And.plus(*new_mask_elems)), streams)
            if progress
            else fwd()
        )


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
    ReduceEqualityMaskRange(),
)

NormalizeIntp = _ExtensibleInterpretation().extend(
    DeltaEmpty(),
    DeltaFusion(),
    GetitemDelta(),
    MonoidOverSequence(),
    MonoidOverMapping(),
    MonoidOverCallable(),
    ReduceFusion(),
    ReduceUnion(),
    ReduceSplit(),
    SplitDisjointProduct(),
    ReduceDistributeCartesianProduct(),
    Factor(),
    ReduceWeightedStream(),
    ReduceCartesianWeightedStream(),
    ReduceMaskHoist(),
    EliminateSingletonStreams(),
    PlusEmpty(),
    PlusSingle(),
    PlusAssoc(),
    PlusDistr(),
    PlusConsecutiveDups(),
    PlusOrder(),
    PlusCastFloat(),
    MaskFusion(),
    MaskBool(),
    MaskOrderStreamOps(),
)
"""``NormalizeIntp`` applies pure-Term rewrites (associativity, distributivity,
identity elimination, fusion, factorization, etc.) that drive a reduce
expression toward a *normal form*.

Normal form
===========

The rules collectively push expressions toward an einsum / variable-elimination
shape: a factored sum-of-products over independent index ranges, with weights
carried symbolically by ``mask`` and ``delta`` rather than lowered to concrete
array ops. Throughout, "sum" means a ``reduce``/``plus`` of an additive monoid
``R`` and "product" a ``plus`` of a multiplicative monoid ``M`` with
``distributes_over(M, R)``.

A fully normalized leaf expression matches the grammar (for additive ``R`` /
multiplicative ``M``)::

    nf     ::= container[nf]                    # lambdas/dicts/sequences outermost
             | M.mask(prod, orphan_cond)       # only conjuncts no factor mentions
             | prod
    prod   ::= M.plus(factor, ...)             # flattened, n-ary
             | factor
    factor ::= M.mask(core, cond)             # conjuncts the factor mentions
             | core
    core   ::= atom                            # array, getitem, delta(idx, w), ...
             | R.reduce(prod, ranges)          # every factor in prod uses a range var
    cond   ::= And.plus(cmp, ...)              # comparison atoms, stream-var first
    ranges ::= { var: range(0, N, 1), ... }

The invariants, grouped by the rules that establish them:

A. Sums are factored over products (the einsum shape).
   - Stream-quantified sums are factored, not expanded: under any
     ``R.reduce(M.plus(...), streams)`` every remaining factor mentions a
     reduced stream variable. :class:`ReduceFactorization` pulls stream-invariant
     factors out as outer product factors; :class:`ReduceSplit` confines a stream
     to just the summands that use it. So no factor sits under a reduce whose
     streams it does not use.
   - Explicit (materialized) products-of-sums are expanded to sums-of-products
     by :class:`PlusDistr`. (This is the opposite direction from factorization,
     but it acts on explicit ``plus`` terms, never on ``reduce`` nodes, so the
     two never fight.)

B. Streams are independent ranges ``range(0, N, 1)``.
   Every non-range binding has a rule that removes it: eager arrays
   (:class:`~effectful.handlers.jax.monoid.ReduceArrayGather` +
   :class:`EliminateSingletonStreams`), unions (:class:`ReduceUnion`), weighted
   streams (:class:`ReduceWeightedStream`), cartesian products (eliminated by
   inversion in :class:`ReduceDistributeCartesianProduct`), dependent ranges
   (:class:`~effectful.handlers.jax.monoid.ReduceDependentRangeMask`), and
   symbolic length-1 streams (:class:`EliminateSingletonStreams`). In particular
   ``CartesianProduct.reduce`` is fully eliminated.

C. ``plus``, ``mask``, and ``delta`` stay symbolic (not lowered).
   - ``plus`` is only normalized structurally: flattened/associative
     (:class:`PlusAssoc`), with nullary/unary collapsed
     (:class:`PlusEmpty`/:class:`PlusSingle`) and int/float operands unified
     (:class:`PlusCastFloat`). The scalar/array implementations live in
     ``EvaluateIntp``, not here.
   - ``mask`` floats to a canonical position: fused to a single layer with a
     conjunctive (``And.plus``) condition (:class:`MaskFusion`), constant-bool
     conditions discharged (:class:`MaskBool`), pushed *down* onto the factors
     of a ``plus`` (:class:`MaskPushPlus`) -- each conjunct landing on the
     factors that mention its variables, so a mask sits adjacent to a single
     product factor where it can fuse with a gather (a conjunct no factor
     mentions stays as a residual outer mask) -- and out of a ``reduce`` when
     the condition is stream-independent (:class:`ReduceMaskHoist`). A mask
     remaining inside a reduce body therefore depends on a reduced stream.
     Comparison atoms are oriented stream-variable-first
     (:class:`MaskOrderStreamOps`).
   - ``delta`` is flattened: nested deltas merge (:class:`DeltaFusion`),
     empty-index deltas collapse to their weight (:class:`DeltaEmpty`), and
     subscripting a delta becomes a mask (:class:`GetitemDelta`). A normal-form
     delta has a non-empty index, a single layer, and is never subscripted.

D. Structural invariants.
   - No same-monoid ``reduce`` is nested directly in a ``reduce`` body: they are
     fused into one reduce over the combined stream set (:class:`ReduceFusion`),
     so each reduce is maximal in its stream set per monoid.
   - Containers are outermost: :class:`MonoidOverCallable`,
     :class:`MonoidOverMapping`, and :class:`MonoidOverSequence` push
     ``reduce``/``plus`` through lambdas, mappings, and sequences down to the
     scalar/array leaves, so a monoid op never wraps a container.

Termination and confluence
===========================

The rules run as a deterministic, priority-ordered, innermost normalization
(``evaluate`` rewrites args before parents, and each constructed replacement
re-enters the interpretation to a fixpoint). For a fixed *syntactic* input the
result is therefore unique.

The normal form is **not canonical**, however: semantically equal but
differently-presented inputs can reach distinct (still semantically equal)
forms. There is no AC-normalization (commutative ``plus`` arguments are never
sorted and duplicate-elimination is disabled -- see the commented-out
:class:`PlusDups`/:class:`PlusConsecutiveDups`), and the variable-elimination
order depends on ``streams`` insertion order via
:func:`choose_contraction`/:func:`outer_stream`. So summand order, stream
insertion order, and factor order all survive normalization. Downstream code
relies only on the *semantic* normal form (the result is ultimately evaluated to
a concrete array), not on syntactic canonicity.

Termination rests on per-family progress (each "lowering" rule strictly
consumes a resource -- a cartesian/weighted/union/array/singleton stream, a
nested reduce/mask/delta, or a liftable factor) plus redex-shape disjointness
that keeps the families from cycling (e.g. :class:`ReduceSplit` deliberately
does not fire when a stream is used by *every* summand, leaving those for
:class:`ReduceDistributeCartesianProduct`/:class:`ReduceFactorization`). There
is no single global measure and no confluence theorem: the system is
normalizing by construction, so a new rule that overlaps an existing redex shape
can introduce a loop or shift which normal form is reached. Note also that
:class:`PlusDistr` is exponential in the number of distributed sums.

"""
