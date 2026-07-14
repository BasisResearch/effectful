import collections.abc
import functools
import itertools
import operator
import typing
from collections import UserDict, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Any

from effectful.internals.runtime import interpreter
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import (
    ObjectInterpretation,
    Scoped,
    _MappingTerm,
    _NumberTerm,
    defdata,
    deffn,
    implements,
    ite,
    range_,
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
CartesianProduct: MonoidWithZero[Sequence[Mapping]] = MonoidWithZero(
    name="CartesianProduct", identity=[{}], zero=[]
)
Union: Monoid[Sequence[Mapping]] = Monoid(name="Union", identity=[])
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


class _ExtensiblePartialInvolution[S](_ExtensibleBinaryRelation[S, S]):
    def register(self, s: S, t: S) -> None:
        for existing, image in self.tuples:
            if existing == s and image != t:
                raise ValueError(f"{s!r} already matched to {image!r}")
            if existing == t and image != s:
                raise ValueError(f"{t!r} already matched to {image!r}")
        super().register(s, t)
        super().register(t, s)

    def of(self, t: S) -> S | None:
        for a, b in self.tuples:
            if a == t:
                return b
        return None


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


class GetitemDelta(ObjectInterpretation):
    """M.delta(i, v)[q] ≡ M.mask(v, i == q)"""

    @implements(_MappingTerm.__getitem__)
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
            return monoid.reduce(
                monoid.mask(
                    monoid.mask(
                        value,
                        And.plus(stream.start <= mask_key, mask_key < stream.stop),
                    ),
                    And.plus(*(c for (j, c) in enumerate(conds) if i != j)),
                ),
                {stream_op: (mask_key,)}
                | {k: v for (k, v) in streams.items() if k != stream_op},
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
    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not is_commutative(monoid):
            return fwd()
        if not (isinstance(body, Term) and body.op is monoid.plus):
            return fwd()

        return monoid.plus(*(monoid.reduce(a, streams) for a in body.args))


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


class ReduceUnfactor(ObjectInterpretation):
    """Undo one :class:`Factor` layer beneath a compatible product.

    This rule is deliberately kept out of ``NormalizeIntp``: together with
    ``Factor`` it would form a rewrite cycle.  It is used only while preparing
    a candidate for cartesian-product inversion.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid: Monoid, body, streams):
        if not is_commutative(monoid):
            return fwd()
        if not (isinstance(body, Term) and _is_monoid_plus(body.op)):
            return fwd()

        product_monoid = body.op.__self__
        if not distributes_over(product_monoid, monoid):
            return fwd()

        for i, factor in enumerate(body.args):
            if not (isinstance(factor, Term) and factor.op is monoid.reduce):
                continue

            inner_body, inner_streams = factor.args
            if not isinstance(inner_streams, Mapping):
                continue
            inner_keys = set(inner_streams)
            if set(streams) & inner_keys:
                continue
            merged_streams = dict(streams) | dict(inner_streams)

            # A reduction's stream bindings must still form an acyclic
            # dependency graph after they are brought into the same scope.
            dependencies = {
                key: fvsof(stream) & set(merged_streams)
                for key, stream in merged_streams.items()
            }
            try:
                tuple(TopologicalSorter(dependencies).static_order())
            except CycleError:
                continue

            inner_factors = (
                inner_body.args
                if isinstance(inner_body, Term) and inner_body.op is product_monoid.plus
                else (inner_body,)
            )
            return monoid.reduce(
                product_monoid.plus(
                    *body.args[:i], *inner_factors, *body.args[i + 1 :]
                ),
                merged_streams,
            )
        return fwd()


class ReduceDistributeCartesianProduct(ObjectInterpretation):
    """Eliminates a reduce over a cartesian product.
        ∑_x₁ ∑_x₂ ... ∑_xₙ ∏_i f(xᵢ) = ∏_i ∑_xᵢ f(xᵢ)
    This transform is also called inversion in the lifting
    literature (e.g. [1]).

    More specifically, this transform implements the identity
    reduce(⨁, reduce(⨂, body2, {vv: v()}), {v: reduce(×, body1, S1)} ∪ S2)
        = reduce(⨁, reduce(⨂, reduce(⨁, body2, {vv: body1}), S1), S2)
    where × is the cartesian product and ⨂ distributes over ⨁.

    The body may also be a ``⨂``-plus of such reductions. Each reduction is
    row-substituted independently, and the row positions determine which plate
    variables are unified. Ordinary row-independent factors remain outside the
    peeled plate reduction.

    Note: This could be generalized to grouped inversion [2].

    [1] Braz, Rd, Eyal Amir, and Dan Roth. "Lifted first-order
    probabilistic inference." IJCAI. 2005.
    [2] Taghipour, Nima, et al. "Completeness results for lifted
    variable elimination." AISTATS. 2013.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid: Monoid, body, streams):
        if not isinstance(body, Term):
            return fwd()

        if not any(
            isinstance(v, Term) and v.op == CartesianProduct.reduce
            for v in streams.values()
        ):
            return fwd()

        # ``Factor`` may have moved product factors into nested additive
        # reductions. Normalize the whole candidate so ReduceUnfactor can see
        # and merge both stream bundles. This isolated interpretation cannot
        # cycle with Factor. Build the candidate outside the active handler;
        # otherwise invoking ``monoid.reduce`` here would recursively redispatch
        # this rule.
        with interpreter(CartesianProductNormalizeIntp):
            candidate = monoid.reduce(body, streams)
        body, streams = candidate.args

        inner_reduces: tuple[Term, ...]
        if isinstance(body, Term) and _is_monoid_reduce(body.op):
            inner_reduces = (body,)
            outer_factors: tuple = ()
            inner_monoid = body.op.__self__
        elif isinstance(body, Term) and _is_monoid_plus(body.op):
            inner_monoid = body.op.__self__
            inner_reduces = tuple(
                arg
                for arg in body.args
                if isinstance(arg, Term) and arg.op is inner_monoid.reduce
            )
            if not inner_reduces:
                return fwd()
            outer_factors = tuple(
                arg
                for arg in body.args
                if not (isinstance(arg, Term) and arg.op is inner_monoid.reduce)
            )
        else:
            return fwd()

        if not distributes_over(inner_monoid, monoid):
            return fwd()

        class InvalidIndexError(Exception): ...

        def drop_elem(ls, index):
            return tuple(x for (i, x) in enumerate(ls) if i != index)

        for stream_key, stream_body in streams.items():
            # stream is cartesian
            if not (
                isinstance(stream_body, Term)
                and stream_body.op is CartesianProduct.reduce
            ):
                continue

            # Product reductions that do not use this row are ordinary outer
            # factors. Every factor that does use it must be plate-reduced.
            row_inner_reduces = tuple(
                reduce for reduce in inner_reduces if stream_key in fvsof(reduce)
            )
            row_outer_factors = outer_factors + tuple(
                reduce for reduce in inner_reduces if stream_key not in fvsof(reduce)
            )
            if not row_inner_reduces or stream_key in fvsof(row_outer_factors):
                continue

            (cprod_body, cprod_streams) = stream_body.args
            assert isinstance(cprod_streams, dict)

            # plates are rectangular
            if not all(
                isinstance(plate_stream, range)
                for plate_stream in cprod_streams.values()
            ):
                continue

            # stream body is a sequence of mappings from plate index to domain value
            match cprod_body:
                case Term(
                    Union.reduce,
                    ([Term(Union.delta, (idx, union_body), {})], union_streams),
                    {},
                ) if isinstance(idx, Sequence) and set(
                    i.op for i in idx if isinstance(i, Term)
                ) >= set(cprod_streams):
                    pass
                case _:
                    continue

            assert len(idx) > 0

            # inner product folds over all plates
            plate_index, plate_op = next(
                (j, i.op) for (j, i) in enumerate(idx) if i.op in cprod_streams
            )
            plate_range = cprod_streams[plate_op]

            def row_substitute(inner_body, inner_streams):
                """Peel ``plate_index`` off every ``stream_key[...]`` in one
                summand. Asserts the summand's bundle contains a stream that
                folds over the full cartesian product row and returns that plate
                variable alongside the substituted body."""
                if stream_key in fvsof(inner_streams):
                    raise InvalidIndexError()

                inner_plate_op = None

                # substitute all instances of row[i, *rest] -> row[*rest]
                def _getitem(mapping, idx1):
                    nonlocal inner_plate_op

                    idx1 = idx1 if isinstance(idx1, Sequence) else (idx1,)
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

                subst = handler({_MappingTerm.__getitem__: _getitem})(evaluate)(
                    inner_body
                )
                if inner_plate_op is None and inner_streams:
                    # A nontrivial reduction that does not fold over the row
                    # cannot be unified with the peeled plate.  Streamless
                    # reductions are ordinary product factors introduced by
                    # sum-of-products expansion and are retained unchanged.
                    raise InvalidIndexError()
                return subst, inner_plate_op

            try:
                substituted = [
                    row_substitute(*reduce.args) for reduce in row_inner_reduces
                ]
            except InvalidIndexError:
                continue

            # Unify reductions by the variable used at this row position, not
            # merely by equal ranges. Equal-sized axes are otherwise ambiguous.
            shared_plate_op = plate_op
            combined_factors = []
            for (subst_body, inner_plate_op), inner_reduce in zip(
                substituted, row_inner_reduces
            ):
                _, inner_streams = inner_reduce.args
                assert isinstance(inner_streams, Mapping)
                assert inner_plate_op is not None
                if inner_plate_op is not shared_plate_op:
                    subst_body = handler({inner_plate_op: shared_plate_op})(evaluate)(
                        subst_body
                    )

                inner_tail_streams = {
                    k: v for (k, v) in inner_streams.items() if k != inner_plate_op
                }
                if inner_tail_streams:
                    subst_body = inner_monoid.reduce(subst_body, inner_tail_streams)
                combined_factors.append(subst_body)

            combined = (
                combined_factors[0]
                if len(combined_factors) == 1
                else inner_monoid.plus(*combined_factors)
            )

            peeled_idx = drop_elem(idx, plate_index)
            peeled_cprod_streams = {
                k: v for (k, v) in cprod_streams.items() if k != plate_op
            }
            if not peeled_cprod_streams and not peeled_idx:
                # Base case: the plate was the only cartesian product stream and
                # the row was fully peeled, so ``Union.delta((), union_body)``
                # collapses to ``union_body``. Rather than emit a bare
                # ``Union.reduce`` stream and defer to
                # ``ReduceUnion``/``EliminateSingletonStreams`` (which would leave
                # an empty-index ``Union.delta`` behind), substitute the union
                # body directly for each empty ``stream_key[()]`` subscript --
                # equivalently ``Union.delta((), union_body)[()]`` -- and reduce
                # directly over ``union_streams``.
                def _to_body(mapping, key):
                    if isinstance(mapping, Term) and mapping.op == stream_key:
                        return union_body
                    return fwd()

                subst_combined = handler({_MappingTerm.__getitem__: _to_body})(
                    evaluate
                )(combined)
                inner_reduce_body = monoid.reduce(subst_combined, union_streams)
            else:
                peeled_body = Union.reduce(
                    [Union.delta(peeled_idx, union_body)], union_streams
                )
                if not peeled_cprod_streams:
                    peeled_cprod = peeled_body
                else:
                    peeled_cprod = CartesianProduct.reduce(
                        peeled_body, peeled_cprod_streams
                    )
                inner_reduce_body = monoid.reduce(combined, {stream_key: peeled_cprod})

            peeled_reduce = inner_monoid.reduce(
                inner_reduce_body,
                {shared_plate_op: plate_range},
            )

            result_body = (
                inner_monoid.plus(peeled_reduce, *row_outer_factors)
                if row_outer_factors
                else peeled_reduce
            )

            # Include any extra sum streams outermost.  In particular, the
            # non-reduce product factors above remain outside the plate fold.
            tail_streams = {k: v for (k, v) in streams.items() if k != stream_key}
            if tail_streams:
                result = monoid.reduce(result_body, tail_streams)
            else:
                result = result_body

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


class DeltaConcrete(ObjectInterpretation):
    @implements(Monoid.delta)
    def _(self, _, k, v):
        if not fvsof((k, v)):
            return {k: v}
        return fwd()


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
        if not args:
            return fwd()
        if any(isinstance(x, Term) for x in args):
            return fwd()
        if not all(isinstance(x, Iterable) for x in args):
            return fwd()
        return [_disjoint_merge(*vals) for vals in itertools.product(*args)]


class UnionPlus(ObjectInterpretation):
    @implements(Union.plus)
    def plus(self, *args):
        if not args:
            return fwd()
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


class WhereHoist(ObjectInterpretation):
    """Hoist :func:`ite` out of monoid ``reduce`` and ``plus``.

    A stream-independent selection commutes with reduction, while monoid
    addition distributes pointwise over either selected branch.
    """

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if len(args) < 2:
            return fwd()

        for i, arg in enumerate(args):
            if not (
                isinstance(arg, Term)
                and arg.op is ite
                and len(arg.args) == 3
                and not arg.kwargs
            ):
                continue

            cond, when_true, when_false = arg.args
            return ite(
                cond,
                monoid.plus(*args[:i], when_true, *args[i + 1 :]),
                monoid.plus(*args[:i], when_false, *args[i + 1 :]),
            )

        return fwd()

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not (
            streams
            and isinstance(body, Term)
            and body.op is ite
            and len(body.args) == 3
            and not body.kwargs
        ):
            return fwd()

        cond, when_true, when_false = body.args
        if fvsof(cond) & set(streams):
            return fwd()

        return ite(
            cond,
            monoid.reduce(when_true, streams),
            monoid.reduce(when_false, streams),
        )


class ReduceWhereEqualityPeel(ObjectInterpretation):
    """Peel stream-independent conjuncts off a ``where`` equality guard.

    Given a condition ``outer & inner`` where ``outer`` is independent of the
    reduced streams and ``inner`` contains an equality selecting one of them::

        M.reduce(ite(outer & inner, x, y), S)
          == ite(outer,
                 M.reduce(ite(inner, x, y), S),
                 M.reduce(y, S))

    This exposes the stream equality to gather/elimination rules while moving
    the remaining output guard above the reduction.
    """

    @staticmethod
    def _stream_equality(cond, streams):
        def matches(op, stream_term, other):
            return (
                is_equality(op)
                and isinstance(stream_term, Term)
                and not stream_term.args
                and not stream_term.kwargs
                and stream_term.op in streams
                and not (fvsof(other) & set(streams))
            )

        return (
            isinstance(cond, Term)
            and len(cond.args) == 2
            and (
                matches(cond.op, cond.args[0], cond.args[1])
                or matches(cond.op, cond.args[1], cond.args[0])
            )
        )

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not (
            streams
            and isinstance(body, Term)
            and body.op is ite
            and len(body.args) == 3
            and not body.kwargs
        ):
            return fwd()

        cond, when_true, when_false = body.args
        if not (isinstance(cond, Term) and cond.op is And.plus):
            return fwd()

        stream_ops = set(streams)
        dependent = [c for c in cond.args if fvsof(c) & stream_ops]
        independent = [c for c in cond.args if not (fvsof(c) & stream_ops)]
        if not independent or not any(
            self._stream_equality(c, streams) for c in dependent
        ):
            return fwd()

        return ite(
            And.plus(*independent),
            monoid.reduce(ite(And.plus(*dependent), when_true, when_false), streams),
            monoid.reduce(when_false, streams),
        )


complement = _ExtensiblePartialInvolution(
    {
        (_NumberTerm.__ne__, _NumberTerm.__eq__),
    }
)


class ReduceWhereToMasks(ObjectInterpretation):
    """Split an equality-guarded ``where`` reduction into masked reductions.

    For a conjunction of stream-dependent equalities ``eqs``::

        M.reduce(where(eqs, a, b), S)
          == M.plus(M.reduce(M.mask(a, eqs), S),
                    M.reduce(M.mask(b, not(eqs)), S))

    De Morgan's law represents ``not(eqs)`` as the disjunction of the
    corresponding disequalities. The two masks are complementary and hence
    partition the stream assignments without double counting.
    """

    @staticmethod
    def _combine(op, terms):
        return terms[0] if len(terms) == 1 else op(*terms)

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not (
            streams
            and isinstance(body, Term)
            and body.op is ite
            and len(body.args) == 3
            and not body.kwargs
        ):
            return fwd()

        cond, when_true, when_false = body.args
        equalities = (
            cond.args if isinstance(cond, Term) and cond.op is And.plus else (cond,)
        )
        stream_ops = set(streams)
        if not equalities or not all(
            isinstance(eq, Term)
            and is_equality(eq.op)
            and complement.of(eq.op) is not None
            and bool(fvsof(eq) & stream_ops)
            for eq in equalities
        ):
            return fwd()

        disequalities = tuple(
            complement.of(eq.op)(*eq.args, **eq.kwargs) for eq in equalities
        )
        return monoid.plus(
            monoid.reduce(monoid.mask(when_true, cond), streams),
            monoid.reduce(
                monoid.mask(when_false, self._combine(Or.plus, disequalities)),
                streams,
            ),
        )


class ReduceDisjunctiveDisequalityMask(ObjectInterpretation):
    """Partition a disjunctive disequality mask into disjoint reductions.

    For example, the overlapping regions ``i != x`` and ``j != y`` are
    rewritten as ``i != x`` and ``i == x and j != y``::

        reduce(mask(v, (i != x) or (j != y)), streams)
        == plus(
            reduce(mask(v, i != x), streams),
            reduce(mask(v, (i == x) and (j != y)), streams),
        )

    More generally, disjunct ``k`` is conjoined with the complements of all
    preceding disjuncts.  The resulting masks are pairwise disjoint, so their
    reductions can be combined with the reduction monoid.  In particular,
    each result has a conjunctive mask that :class:`ReduceArrayScan` can
    eliminate.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams: Streams):
        match body:
            case Term(mask_op, (value, Term(Or.plus, disjuncts, {})), {}) if (
                _is_monoid_mask(mask_op) and mask_op.__self__ == monoid
            ):
                pass
            case _:
                return fwd()

        if len(disjuncts) < 2 or not all(
            isinstance(disjunct, Term) and is_equality(complement.of(disjunct.op))
            for disjunct in disjuncts
        ):
            return fwd()

        preceding_complements: list = []
        reductions = []
        for disjunct in disjuncts:
            assert isinstance(disjunct, Term)
            reductions.append(
                monoid.reduce(
                    monoid.mask(value, And.plus(*preceding_complements, disjunct)),
                    streams,
                )
            )
            comp = complement.of(disjunct.op)
            assert comp is not None
            preceding_complements.append(comp(*disjunct.args, **disjunct.kwargs))

        return monoid.plus(*reductions)


class ReduceDependentRangeMask(ObjectInterpretation):
    """Eliminate a dependent range by masking.

    reduce(M, streams ∪ {u: range(N), v: range(u())}, body)
    ═══════════════════════════════════════════════════════════════════════════
    reduce(M, streams ∪ {u: range(N), v: range(N)}, where(v() < u(), body, M.identity))

    Currently recognises only the lower-triangular form ``v: range(u())``:
    constant start of 0, dependent stop equal to a bare call of another
    stream var.

    Not yet supported:

    - **Upper-triangular** (``v: range(u(), N)`` — constant stop, dependent
      start): bbox becomes ``range(0, N)`` (or ``range(0, bbox_N)``), guard
      becomes ``v() >= u()``. Same shape of rewrite as lower-tri; differs
      only in which side of the range carries the stream-var reference and
      in the predicate direction.
    - **Banded** (``v: range(u() - k, u() + k + 1)`` — two-sided dependent
      bounds with constant width): bbox is ``range(0, N + k)`` (or similar
      bounded by both endpoints' extents), guard is
      ``(v() >= u() - k) & (v() < u() + k + 1)``. Needs both-sides
      affine-bound recognition.
    - **Strided dependent** (``v: range(0, u(), k)`` for ``k != 1``): bbox
      stays ``range(0, N)`` and guard becomes
      ``(v() < u()) & (v() % k == 0)`` (or equivalent), or alternatively
      embed in a smaller bbox ``range(0, ceil(N/k))`` and remap the index.
    - **Affine bounds** (``v: range(a*u() + b, c*u() + d)`` for affine
      coefficients): bbox computed from ``ub(c*u() + d)`` over ``u``'s
      range; guard is the conjunction of the two affine constraints. This
      subsumes the upper/banded/strided cases under one affine recogniser.
    - **Multi-stream-var dependent** (``v: range(u() + w())`` referencing
      more than one outer stream var): bbox is the affine combination over
      both referents' ranges; guard threads through all dependencies.
    - **Reverse-order dependent ranges**: e.g. ``v: range(u(), 0, -1)``;
      needs to handle negative step and the corresponding reverse
      enumeration.
    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        stream_vars = set(streams.keys())

        for u, u_stream in streams.items():
            # streams of the form k: range(X)
            if not (
                isinstance(u_stream, range)
                and isinstance(u_stream.start, int)
                and u_stream.start == 0
                and isinstance(u_stream.step, int)
                and u_stream.step == 1
                and not fvsof(u_stream) & stream_vars
            ):
                continue

            for v, v_stream in streams.items():
                if (
                    isinstance(v_stream, Term)
                    and v_stream.op == range_
                    and isinstance(v_stream.stop, Term)  # type: ignore[attr-defined]
                    and v_stream.stop.op == u  # type: ignore[attr-defined]
                ):
                    fresh_streams = {
                        a: (u_stream if a == v else b) for (a, b) in streams.items()
                    }
                    fresh_body = monoid.mask(v() < u(), body)
                    return monoid.reduce(fresh_body, fresh_streams)

        return fwd()


class ContractLongestStream(ObjectInterpretation):
    @implements(choose_contraction)
    def _(self, factors, streams):
        lengths = {
            k: len(v) if not isinstance(v, Term) and isinstance(v, Sized) else 0
            for (k, v) in streams.items()
        }
        longest = max(lengths.values())
        longest_streams = {k: v for (k, v) in streams.items() if lengths[k] == longest}
        if len(longest_streams) == len(streams):
            return fwd()
        return choose_contraction(factors, longest_streams)


class _ExtensibleInterpretation(UserDict, Interpretation):
    def extend(self, *intps: Interpretation) -> typing.Self:
        for intp in intps:
            self.data = coproduct(self.data, intp)  # type: ignore[assignment]
        return self


EvaluateIntp = _ExtensibleInterpretation().extend(
    # ReducePartial(),
    DeltaConcrete(),
    SumPlus(),
    MinPlus(),
    MaxPlus(),
    ProductPlus(),
    ArgMinPlus(),
    ArgMaxPlus(),
    CartesianProductPlus(),
    UnionPlus(),
    ReduceEqualityMaskRange(),
    ReduceWhereToMasks(),
)

CartesianProductNormalizeIntp = functools.reduce(
    coproduct,
    typing.cast(
        tuple[Interpretation, ...],
        (
            PlusEmpty(),
            PlusSingle(),
            PlusAssoc(),
            ReduceUnfactor(),
        ),
    ),
)
"""Structural preprocessing used exclusively for cartesian-product inversion.

It intentionally excludes ``Factor`` and sum-of-products rewrites, which
would either cycle with ``ReduceUnfactor`` or move factors across a product
fold.
"""


NormalizeIntp = _ExtensibleInterpretation().extend(
    GetitemDelta(),
    MonoidOverSequence(),
    MonoidOverMapping(),
    MonoidOverCallable(),
    ReduceFusion(),
    ReduceUnion(),
    ReduceSplit(),
    Factor(),
    ReduceDistributeCartesianProduct(),
    ReduceWeightedStream(),
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
    WhereHoist(),
    ReduceWhereEqualityPeel(),
    ReduceDisjunctiveDisequalityMask(),
    ReduceDependentRangeMask(),
    ContractLongestStream(),
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
