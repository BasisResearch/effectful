import functools
import logging
import typing
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.core
import opt_einsum
from opt_einsum import get_symbol

import effectful.handlers.jax.lax as lax
import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, unbind_dims
from effectful.handlers.jax._handlers import JaxOperation, is_eager_array
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    And,
    CartesianProduct,
    EvaluateIntp,
    Max,
    Min,
    Monoid,
    NormalizeIntp,
    Or,
    Product,
    Streams,
    Sum,
    _is_monoid_mask,
    _is_monoid_plus,
    choose_contraction,
    distributes_over,
    is_equality,
)
from effectful.ops.monoid import Union as UnionM
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import (
    Array,
    ObjectInterpretation,
    _ArrayTerm,
    _NumberTerm,
    deffn,
    implements,
)
from effectful.ops.types import Expr, Interpretation, Operation, Term

logger = logging.getLogger(__name__)


LogSumExp = Monoid(name="LogSumExp", identity=jnp.asarray(float("-inf")))

# ``Sum`` in log space is multiplication, which distributes over ``LogSumExp``:
#   a + logsumexp(b, c) = logsumexp(a + b, a + c)
distributes_over.register(Sum, LogSumExp)

is_equality.register(jnp.equal)


def _jax_args(args):
    """True iff ``args`` is non-empty and every arg is a concrete
    :class:`jax.typing.ArrayLike` or named tensor. At least one argument must be
    a jax-related type.

    """
    if not args:
        return False

    types = tuple(typeof(a) for a in args)
    return all(
        issubclass(t, jax.typing.ArrayLike | jax.core.Tracer) for t in types
    ) and any(issubclass(t, jax.Array | jax.core.Tracer) for t in types)


class PlusCastArray(ObjectInterpretation):
    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        arg_types = [typeof(a) for a in args]

        def _is_jax(t):
            return issubclass(t, jax.Array | jax.core.Tracer)

        # exists array valued and non-array-valued args
        if any(_is_jax(t) for t in arg_types) and any(
            not _is_jax(t) for t in arg_types
        ):
            return monoid.plus(
                *(
                    a if _is_jax(t) else jnp.asarray(a)
                    for (a, t) in zip(args, arg_types, strict=True)
                )
            )

        return fwd()


class SumPlusJax(ObjectInterpretation):
    @implements(Sum.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.add, args)


class ProductPlusJax(ObjectInterpretation):
    @implements(Product.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.multiply, args)


class MinPlusJax(ObjectInterpretation):
    @implements(Min.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.minimum, args)


class MaxPlusJax(ObjectInterpretation):
    @implements(Max.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.maximum, args)


class LogSumExpPlusJax(ObjectInterpretation):
    @implements(LogSumExp.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.logaddexp, args)


class AndPlusJax(ObjectInterpretation):
    @implements(And.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.logical_and, args)


class OrPlusJax(ObjectInterpretation):
    @implements(Or.plus)
    def plus(self, *args):
        if not _jax_args(args):
            return fwd()
        return functools.reduce(jnp.logical_or, args)


class MaskJax(ObjectInterpretation):
    @implements(Monoid.mask)
    def mask(self, monoid, value, mask):
        if not (is_eager_array(value) and is_eager_array(mask)):
            return fwd()
        return jnp.where(mask, value, monoid.identity)


class ReduceArrayGather(ObjectInterpretation):
    """Split an array-valued stream into an index range and a length-1 stream:

    M.reduce(body, {k: a} ∪ S) ≡ M.reduce(body, {i: range(a.shape[0]), k: (a[i()],)} ∪ S)

    where ``i`` is fresh and ``a[i()] = unbind_dims(a, i)``. The length-1 stream
    ``{k: (a[i()],)}`` is then eliminated by
    :class:`~effectful.ops.monoid.EliminateSingletonStreams`, which substitutes
    ``k := a[i()]`` into the body and the remaining streams. Together the two
    steps perform the gather
    ``M.reduce(body[k := a[i()]], {i: range(a.shape[0])} ∪ S)``.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if typeof(body) is not jax.Array:
            return fwd()

        if isinstance(body, Term) and body.op is monoid.delta:
            return fwd()

        body_fvs = fvsof(body)
        stream_keys = set(streams)

        new_streams: dict = {}
        progress = False
        for k, v in streams.items():
            if is_eager_array(v) and k in body_fvs and not (fvsof(v) & stream_keys):
                index = Operation.define(k)
                new_streams[index] = range(v.shape[0])
                new_streams[k] = (unbind_dims(v, index),)
                progress = True
            else:
                new_streams[k] = v

        if not progress:
            return fwd()

        return monoid.reduce(body, new_streams)


class Reductor(Protocol):
    def __call__(
        self, arr: jax.Array, axis: int | tuple[int, ...] | None = None
    ) -> jax.Array: ...


ARRAY_REDUCTORS: dict[Monoid, Reductor] = {}
for monoid, func in [
    (Sum, jnp.sum),
    (Product, jnp.prod),
    (Min, jnp.min),
    (Max, jnp.max),
]:
    assert isinstance(monoid, Monoid)
    assert callable(func)
    ARRAY_REDUCTORS[monoid] = functools.partial(func, initial=monoid.identity)

ARRAY_REDUCTORS[LogSumExp] = logsumexp


class ReduceArray(ObjectInterpretation):
    """Reduce an array body over range streams."""

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        reductor = ARRAY_REDUCTORS.get(monoid, None)
        if reductor is None:
            return fwd()

        if typeof(body) is not jax.Array:
            return fwd()

        pos_dims = (
            {d.op for d in body.args[0] if isinstance(d, Term) and d.op in streams}
            if isinstance(body, Term) and body.op == monoid.delta
            else {}
        )
        body_fvs = fvsof(body)
        used = {
            k
            for k, v in streams.items()
            if k in body_fvs and k not in pos_dims and isinstance(v, range)
        }
        if not used:
            return fwd()

        index = tuple(k() for k in streams if k in used)
        arr = monoid.reduce(monoid.delta(index, body), streams)
        reduced_body = reductor(arr, axis=tuple(range(len(used))))
        return reduced_body


class ReduceArrayScan(ObjectInterpretation):
    @implements(Monoid.reduce)
    def _(self, monoid, body, streams: Streams):
        match body:
            case Term(mask_op, (value, mask), {}) if (
                _is_monoid_mask(mask_op) and mask_op.__self__ == monoid
            ):
                pass

            case _:
                return fwd()

        match mask:
            case Term(And.plus, mask_elems, {}):
                ...
            case _:
                mask_elems = (mask,)

        for i, elem in enumerate(mask_elems):
            match elem:
                case Term(_NumberTerm.__ne__, (Term(stream_op, (), {}), index)) if (
                    stream_op in streams
                    and isinstance(stream := streams[stream_op], range)
                ):
                    other_mask_elems = [e for (j, e) in enumerate(mask_elems) if i != j]
                    return monoid.plus(
                        monoid.reduce(
                            monoid.mask(
                                value, And.plus(stream_op() < index, *other_mask_elems)
                            ),
                            streams,
                        ),
                        monoid.reduce(
                            monoid.mask(
                                value, And.plus(stream_op() > index, *other_mask_elems)
                            ),
                            streams,
                        ),
                    )
                case Term(
                    (
                        _NumberTerm.__le__
                        | _NumberTerm.__ge__
                        | _NumberTerm.__lt__
                        | _NumberTerm.__gt__
                    ) as cmp_op,
                    (Term(stream_op, (), {}), index),
                ) if stream_op in streams and isinstance(
                    stream := streams[stream_op], range
                ):
                    tail_mask_elems = [e for (j, e) in enumerate(mask_elems) if i != j]
                    reverse = cmp_op in (_NumberTerm.__ge__, _NumberTerm.__gt__)
                    n = len(stream)
                    pos_value = monoid.reduce(
                        monoid.delta((stream_op(),), value),
                        {stream_op: stream},
                    )
                    # inclusive prefix (or suffix, when reverse) scan over the stream
                    inclusive = lax.associative_scan(
                        monoid.plus, pos_value, reverse=reverse
                    )
                    # The strict comparisons (`<`, `>`) need an *exclusive* scan: pad
                    # an identity element on the appropriate side and shift the result
                    # so that scan_val[k] aggregates only the positions satisfying the
                    # comparison against k. The non-strict comparisons (`<=`, `>=`) use
                    # the inclusive scan directly.
                    match cmp_op:
                        case _NumberTerm.__lt__:
                            scan_val = jnp.pad(
                                inclusive,
                                (1, 0),
                                mode="constant",
                                constant_values=monoid.identity,
                            )[:n]
                        case _NumberTerm.__gt__:
                            scan_val = jnp.pad(
                                inclusive,
                                (0, 1),
                                mode="constant",
                                constant_values=monoid.identity,
                            )[1:]
                        case _:
                            scan_val = inclusive

                    tail_body = monoid.mask(
                        jax_getitem(scan_val, (index,)), And.plus(*tail_mask_elems)
                    )

                    tail_streams = {
                        k: v for (k, v) in streams.items() if k != stream_op
                    }
                    if tail_streams:
                        return monoid.reduce(tail_body, tail_streams)
                    return tail_body
        return fwd()


def _range_stop(term: Term):
    assert term.op == jnp.arange
    if "stop" in term.kwargs:
        return term.kwargs["stop"]
    if len(term.args) < 2:
        return term.args[0]
    return term.args[1]


class ReduceDeltaSimpleRange(ObjectInterpretation):
    """Eliminate a Delta that has independent, dense index arguments.


    reduce(M, streams ∪ {v: range(N)}, delta((v(),) ++ idx', body))
    ═══════════════════════════════════════════════════════════════════════════
    bind_dims(reduce(M, streams, delta(idx', body[v() := unbind_dims(streams[v], fv)])), fv)
    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if not (isinstance(body, Term) and body.op == monoid.delta):
            return fwd()

        index, weight = body.args
        assert isinstance(index, tuple)

        if not index:
            return fwd()

        head_index, tail_index = index[0], index[1:]
        if not (isinstance(head_index, Term) and head_index.op in streams):
            return fwd()

        head_op: Operation = head_index.op
        head_stream = streams[head_op]
        if not (
            isinstance(head_stream, range)
            and head_stream.start == 0
            and head_stream.step == 1
        ):
            return fwd()

        tail_streams = {k: v for (k, v) in streams.items() if k != head_op}

        # peel the head index: substitute it into the weight (slicing direct
        # uses, materializing the rest) along a fresh named dim, but bind that
        # dim only *after* the surrounding reduce -- see the class docstring.

        fresh_op = Operation.define(head_op)

        def _jax_getitem(arr, index):
            inner_index, outer_index = [], []
            progress = False
            for i in index:
                if isinstance(i, Term) and i.op == head_op:
                    inner_index.append(
                        slice(head_stream.start, head_stream.stop, head_stream.step)
                    )
                    outer_index.append(fresh_op())
                    progress = True
                else:
                    inner_index.append(slice(None))
                    outer_index.append(i)
            if progress:
                return jax_getitem(jax_getitem(arr, inner_index), outer_index)
            return fwd(arr, index)

        slice_subst = typing.cast(Interpretation, {jax_getitem: _jax_getitem})
        sliced_weight = handler(slice_subst)(evaluate)(weight)
        sliced_streams = handler(slice_subst)(evaluate)(tail_streams)

        gather_subst = typing.cast(
            Interpretation,
            {
                head_op: deffn(
                    unbind_dims(
                        jnp.arange(
                            head_stream.start, head_stream.stop, head_stream.step
                        ),
                        fresh_op,
                    )
                )
            },
        )
        gathered_weight = handler(gather_subst)(evaluate)(sliced_weight)
        gathered_streams = handler(gather_subst)(evaluate)(sliced_streams)

        inner = (
            monoid.reduce(monoid.delta(tail_index, gathered_weight), gathered_streams)
            if gathered_streams
            else gathered_weight
        )
        return bind_dims(inner, fresh_op)


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

        # streams of the form k: range(X)
        simple_ranges = {
            k: v
            for (k, v) in streams.items()
            if isinstance(v, range) and v.start == 0 and v.step == 1
        }
        for u, u_stream in simple_ranges.items():
            if fvsof(u_stream) & stream_vars:
                continue

            for v, v_stream in streams.items():
                if (
                    isinstance(v_stream, Term)
                    and v_stream.op == jnp.arange
                    and isinstance(_range_stop(v_stream), Term)
                    and _range_stop(v_stream).op == u
                ):
                    fresh_streams = {
                        a: (u_stream if a == v else b) for (a, b) in streams.items()
                    }

                    # there are other commuting rules for delta that we do not
                    # currently include
                    if isinstance(body, Term) and body.op == monoid.delta:
                        fresh_body = monoid.delta(
                            body.args[0],
                            monoid.mask(v() < u(), body.args[1]),  # type: ignore[arg-type]
                        )
                    else:
                        fresh_body = monoid.mask(v() < u(), body)

                    return monoid.reduce(fresh_body, fresh_streams)

        return fwd()


# Cross-cutting delta rules not yet implemented:
#
# - **Delta-commuting** (DC-hoist): for any pure op ``f`` (no Scoped binders
#   that intersect a delta's index ops), push delta outward:
#       f(args..., delta(idx, body), args...)
#       ≡ delta(idx, f(args..., body, args...))
#   This normalizes delta to the outermost position so the reduce rules can
#   pattern-match ``isinstance(body, Term) and body.op == delta`` cleanly.
#   The soundness condition is mechanical via ``op.__fvs_rule__``: refuse to
#   commute when a non-delta arg's scope binds any op in the delta's idx.
#
# - **Delta-merging** (DC-merge): under a pure binary op ``f`` (or
#   generalized n-ary), merge multiple deltas when their index tuples are
#   subsequence-compatible:
#       f(delta(idx_a, v), delta(idx_b, w)) ≡ delta(idx_max, f(v, w))
#   where ``idx_max`` is the longer of ``idx_a``, ``idx_b`` and ``idx_a`` is
#   a subsequence of ``idx_b`` (or vice versa). Refuse to fire when neither
#   is a subsequence of the other, since that would silently insert an
#   outer-product broadcast.


class ContractLongestArrayStream(ObjectInterpretation):
    @implements(choose_contraction)
    def _(self, factors, streams):
        lengths = {
            k: v.shape[0] if isinstance(v, jax.Array) and v.shape else 0
            for (k, v) in streams.items()
        }
        longest = max(lengths.values())
        return fwd(
            factors, {k: v for (k, v) in streams.items() if lengths[k] == longest}
        )


class ReduceSumProductContraction(ObjectInterpretation):
    """Fast-path a sum-of-products contraction."""

    @implements(Sum.reduce)
    def _(self, body, streams: Streams):
        if not (
            isinstance(body, Term)
            and _is_monoid_plus(body.op)
            and body.op.__self__ is Product
        ):
            return fwd()

        factors = body.args
        if len(factors) != 2 or not all(
            issubclass(typeof(f), jax.Array) for f in factors
        ):
            return fwd()

        (lhs, rhs) = factors
        stream_vars = set(streams.keys())

        # a fully factored reduce only has streams that are used by all factors
        shared = fvsof(lhs) & fvsof(rhs) & stream_vars
        if shared != stream_vars:
            return fwd()

        if not all(isinstance(v, range) for v in streams.values()):
            return fwd()

        # create leading reduction dimensions
        index = tuple(k() for k in streams)
        pos_lhs = Sum.reduce(Sum.delta(index, lhs), streams)
        pos_rhs = Sum.reduce(Sum.delta(index, rhs), streams)

        dims = "".join(get_symbol(i) for i in range(len(streams)))
        contraction = jnp.einsum(f"{dims}...,{dims}...->...", pos_lhs, pos_rhs)
        return contraction


class GetitemJaxGetitem(ObjectInterpretation):
    @implements(_ArrayTerm.__getitem__)
    def _(self, arr, index):
        if (
            isinstance(arr, jax.Array)
            or issubclass(typeof(arr), jax.Array | jax.core.Tracer)
            or (
                isinstance(index, Sequence)
                and any(
                    isinstance(i, jax.Array)
                    or issubclass(typeof(i), jax.Array | jax.core.Tracer)
                    for i in index
                )
            )
        ):
            return jax_getitem(arr, index)
        return fwd()


@dataclass
class Node:
    ordinal: frozenset[str]
    children: list["Node"]
    factors: list[int]


def _build_plate_tree(factor_specs: list[str], plates: frozenset[str]) -> Node:
    # 1. ordinal (plate context) of each factor
    factor_ordinal = {
        k: frozenset(spec) & plates for k, spec in enumerate(factor_specs)
    }

    # 2. node set: every factor ordinal, the global root ∅, and every pairwise
    #    intersection. The intersection of two ordinals is the deepest context
    #    that contains both, i.e. their common ancestor -- a shared plate MUST
    #    have a node to live at, or containment can't be a tree. Closing under
    #    ∩ materializes those frame nodes (finite lattice, so it terminates).
    ordinals = set(factor_ordinal.values()) | {frozenset()}
    changed = True
    while changed:
        changed = False
        for A in list(ordinals):
            for B in list(ordinals):
                if (A & B) not in ordinals:
                    ordinals.add(A & B)
                    changed = True

    nodes = {o: Node(ordinal=o, children=[], factors=[]) for o in ordinals}

    # 3. parent of o = the largest ordinal strictly inside o ("next context out").
    #    Validity: that maximum must dominate EVERY other strict subset, i.e. the
    #    strict subsets form a chain. Two incomparable maximal subsets means o is a
    #    join over two separate branches -- not a tree -> raise.
    for o in ordinals:
        if not o:  # root ∅ has no parent
            continue
        subs = [p for p in ordinals if p < o]  # strict subsets, present as nodes
        parent = max(subs, key=len)
        offenders = [s for s in subs if not (s <= parent)]
        if offenders:
            raise ValueError(
                f"factors do not form a plate tree: context {set(o)} sits above "
                f"incomparable sub-plates {[set(s) for s in offenders]} (a join, not a nest)"
            )
        nodes[parent].children.append(nodes[o])

    # 4. hang each factor on its ordinal's node
    for k, o in factor_ordinal.items():
        nodes[o].factors.append(k)

    return nodes[frozenset()]  # the root


def _build_plate_reductions(
    plate_tree: Node,
    factors: Sequence[Mapping[tuple[int, ...], float]],
    factor_specs: Sequence[str],
    dim_index: Mapping[str, Expr],
    out_op: Mapping[str, Operation],
    plate_sizes: Mapping[str, int],
    ordinal: Mapping[str, frozenset[str]],
    parent_plates: frozenset[str] = frozenset(),
) -> Mapping[tuple[int, ...], float]:
    indexed_factors = [
        factors[f][tuple(dim_index[c] for c in factor_specs[f])]
        for f in plate_tree.factors
    ]

    masked_factors = []
    for f, factor in zip(plate_tree.factors, indexed_factors, strict=True):
        spec = factor_specs[f]
        factor_out_plates = plate_tree.ordinal & set(out_op) & set(spec)

        # Only plated output dims are delta'd per factor. Global output dims are
        # delta'd once by the outer ``out_mask`` in ``_einsum_expr``; masking
        # them here too would check the same equality (e.g. ``out_b == b``) in
        # every factor that mentions the dim as well as the outer mask.
        factor_out_dims = {
            d for d in set(ordinal) & set(out_op) & set(spec) if ordinal[d]
        }

        if factor_out_plates or factor_out_dims:
            masked_factors.append(
                Sum.plus(
                    Sum.mask(
                        factor,
                        And.plus(
                            *(dim_index[p] == out_op[p]() for p in factor_out_plates),
                            *(dim_index[d] == out_op[d]() for d in factor_out_dims),
                        ),
                    ),
                    Sum.mask(
                        factor,
                        Or.plus(
                            *(dim_index[p] != out_op[p]() for p in factor_out_plates)
                        ),
                    ),
                )
            )
        else:
            masked_factors.append(factor)

    child_reductions = (
        _build_plate_reductions(
            n,
            factors,
            factor_specs,
            dim_index,
            out_op,
            plate_sizes,
            ordinal,
            parent_plates | plate_tree.ordinal,
        )
        for n in plate_tree.children
    )
    product = Product.plus(*masked_factors, *child_reductions)
    if plate_tree.ordinal:
        plate_streams = {
            dim_index[p].op: range(plate_sizes[p])
            for p in sorted(plate_tree.ordinal - parent_plates)
        }
        return Product.reduce(product, plate_streams)
    return product


def _einsum_expr(
    subscripts: str, /, *operands: jax.Array, plates: str | None = None
) -> Term:
    if not operands:
        raise ValueError("einsum requires at least one operand")

    in_spec, out_spec, _ = opt_einsum.parser.parse_einsum_input(
        [subscripts, *(op.shape for op in operands)], shapes=True
    )
    in_specs = in_spec.split(",")

    sizes: dict[str, int] = {}
    for spec, op in zip(in_specs, operands, strict=True):
        for l, s in zip(spec, op.shape, strict=True):
            if l in sizes and sizes[l] != s:
                raise ValueError(f"Dimension {l} given sizes {s} and {sizes[l]}")
            else:
                sizes[l] = s
    for c in out_spec:
        if c not in sizes:
            raise ValueError(f"einsum: output index {c!r} not present in any input")

    plate_set = frozenset(plates or "")

    # the plate context of each sum dim: the intersection of the plate sets of
    # the inputs that mention it (so a dim that ever appears unplated is global)
    ordinal = defaultdict(lambda: plate_set)
    for spec in in_specs:
        spec_set = frozenset(spec)
        for c in spec_set - plate_set:
            ordinal[c] &= spec_set
    global_enums = {c for c, o in ordinal.items() if not o}
    plated_enums = {c: o for c, o in ordinal.items() if o}

    out_spec_set = frozenset(out_spec)
    out_plates = out_spec_set & plate_set
    for c in out_spec_set - plate_set:
        missing_plates = ordinal[c] - out_plates
        if missing_plates:
            raise ValueError(
                "It is nonsensical to preserve a plated dim without preserving "
                f"all of that dim's plates, but found {c!r} without "
                f"{','.join(sorted(missing_plates))!r}"
            )

    def dim_type(c):
        return int if c in plate_set else Array if ordinal[c] else int

    dim_op = {c: Operation.define(dim_type(c), name=c) for c in set("".join(in_specs))}
    dim_index = {
        c: dim_op[c]()
        if c in plate_set or not ordinal[c]
        else dim_op[c]()[tuple(dim_op[p]() for p in sorted(ordinal[c]))]
        for c in dim_op
    }
    arrays = [
        Operation.define(jax.Array, name=f"f{i}") for (i, _) in enumerate(operands)
    ]
    plate_tree = _build_plate_tree(in_specs, plate_set)
    out_vars = {c: Operation.define(jax.Array, name=f"out_{c}") for c in out_spec}
    reductions = _build_plate_reductions(
        plate_tree, [a() for a in arrays], in_specs, dim_index, out_vars, sizes, ordinal
    )

    # one stream of per-plate-assignment rows for each plated sum dim
    rows = {}
    for c, o in plated_enums.items():
        ps = [(p, dim_op[p]) for p in sorted(o)]
        delta_idx = tuple(op() for (_, op) in ps)
        c_streams = {op: range(sizes[p]) for (p, op) in ps}
        v = Operation.define(int, name=f"{c}_v")
        rows[c] = CartesianProduct.reduce(
            UnionM.reduce([UnionM.delta(delta_idx, v())], {v: range(sizes[c])}),
            c_streams,
        )

    streams: Streams = {dim_op[c]: range(sizes[c]) for c in global_enums} | {
        dim_op[c]: r for c, r in rows.items()
    }

    out_globals = global_enums & set(out_spec)
    out_mask = And.plus(*(out_vars[c]() == dim_index[c] for c in out_globals))
    return deffn(
        Sum.reduce(Sum.mask(reductions, out_mask), streams),
        *arrays,
        *(out_vars[c] for c in out_spec),
    ), [(out_vars[c], sizes[c]) for c in out_spec]


@jax.jit(static_argnums=(0,), static_argnames=("plates",))
def einsum(
    subscripts: str, /, *operands: jax.Array, plates: str | None = None
) -> jax.Array:
    """Evaluate an einsum expression using monoid reductions.

    Generalizes :func:`jax.numpy.einsum` with plated dimensions in the style of
    :func:`pyro.ops.contract.einsum`: indices in ``plates`` are plate
    dimensions, and reductions along plates are product reductions. A sum
    dimension that always appears together with a plate denotes a distinct
    variable for each slice of that plate; its plate context (ordinal) is the
    intersection of the plate sets of the inputs that mention it. When such a
    dimension appears in the output it must be accompanied by all of its
    plates.

    The expression is represented naively: each plated sum dimension ranges
    over a :data:`CartesianProduct` stream of per-plate-assignment rows, and
    each plated input is a :data:`Product` reduction over its plates.
    """
    expr, dims = _einsum_expr(subscripts, *operands, plates=plates)
    norm_expr = handler(NormalizeIntp)(evaluate)(expr)

    assert CartesianProduct.reduce not in fvsof(norm_expr), (
        "failed to eliminate cartesian products"
    )

    with handler(EvaluateIntp), handler(NormalizeIntp):
        assert callable(norm_expr)
        breakpoint()

        result = evaluate(
            bind_dims(
                norm_expr(
                    *operands, *(unbind_dims(jnp.arange(d), v) for (v, d) in dims)
                ),
                *(v for (v, _) in dims),
            )
        )
        assert isinstance(result, jax.typing.ArrayLike), "failed to fully evaluate"
        return result


EvaluateIntp.extend(
    SumPlusJax(),
    ProductPlusJax(),
    MinPlusJax(),
    MaxPlusJax(),
    LogSumExpPlusJax(),
    AndPlusJax(),
    OrPlusJax(),
    MaskJax(),
    ReduceSumProductContraction(),
    # ReduceArray(),
    ReduceDeltaSimpleRange(),
    GetitemJaxGetitem(),
    ReduceArrayScan(),
    PlusCastArray(),
)

NormalizeIntp.extend(
    ReduceArrayGather(),
    ReduceDependentRangeMask(),
    ContractLongestArrayStream(),
)
