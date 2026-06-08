import functools
import logging
import typing
from collections.abc import Iterable
from typing import Protocol

import jax
import jax.core
import opt_einsum

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, unbind_dims
from effectful.handlers.jax._handlers import is_eager_array
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    CartesianProduct,
    Max,
    Min,
    Monoid,
    NormalizeIntp,
    Product,
    Streams,
    Sum,
    _is_monoid_plus,
    distributes_over,
)
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
from effectful.ops.types import Expr, Interpretation, NotHandled, Operation, Term

logger = logging.getLogger(__name__)


def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    nx, dx = x.shape
    ny, dy = y.shape
    # Broadcast into (nx, ny, dx+dy), then flatten the first two axes
    x_b = jnp.broadcast_to(x[:, None, :], (nx, ny, dx))
    y_b = jnp.broadcast_to(y[None, :, :], (nx, ny, dy))
    return jnp.concatenate([x_b, y_b], axis=-1).reshape(nx * ny, dx + dy)


LogSumExp = Monoid(name="LogSumExp", identity=jnp.asarray(float("-inf")))

# ``Sum`` in log space is multiplication, which distributes over ``LogSumExp``:
#   a + logsumexp(b, c) = logsumexp(a + b, a + c)
distributes_over.register(Sum, LogSumExp)


def _jax_args(args):
    """True iff ``args`` is non-empty and every arg is a concrete
    :class:`jax.typing.ArrayLike` or named tensor. At least one argument must be
    a jax-related type.

    """
    return (
        bool(args)
        and all(is_eager_array(a) or isinstance(a, jax.typing.ArrayLike) for a in args)
        and any(is_eager_array(a) or isinstance(a, jax.Array) for a in args)
    )


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


class CartesianProductPlusJax(ObjectInterpretation):
    @implements(CartesianProduct.plus)
    def plus(self, *args):
        # Skip identity ``[()]`` args; short-circuit on zero ``[]``. Both
        # sentinels arrive as Python lists alongside jax-array factors, so
        # check for them explicitly before composing.
        if not any(isinstance(a, jax.Array) for a in args):
            return fwd()
        result = None
        for a in args:
            if a is CartesianProduct.zero:
                return CartesianProduct.zero
            if a is CartesianProduct.identity:
                continue
            if not isinstance(a, jax.Array):
                return fwd()
            result = a if result is None else cartesian_prod(result, a)
        if result is None:
            return CartesianProduct.identity
        # CartesianProduct values are streams of rows. ``cartesian_prod``
        # already lifts 1D inputs to 2D, but a single-array call seeds
        # ``result = a`` unchanged — promote so the rank invariant holds for
        # every array-path return.
        if result.ndim == 1:
            result = result[:, None]
        return result


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


class ArrayReduce(ObjectInterpretation):
    """Reduce an array body over its array/``arange`` streams.

    Substitutes every used stream into the body via :func:`_substitute_streams`
    (``arange`` streams slice their direct-index uses ``arr[v()]`` rather than
    materializing and gathering; everything else gathers), binds the resulting
    named dims positionally, and sums them out with the monoid's reductor.

    ``delta`` bodies are left to :class:`ReduceDeltaIndependent`; ``monoid.plus``
    bodies are claimed first by the higher-precedence contraction rules, with
    this rule as the dense-broadcast fallback for the ones they decline.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        reductor = ARRAY_REDUCTORS.get(monoid, None)
        if reductor is None:
            return fwd()

        if typeof(body) is not jax.Array:
            return fwd()

        # delta bodies belong to ReduceDeltaIndependent
        if isinstance(body, Term) and body.op is delta:
            return fwd()

        body_fvs = fvsof(body)
        used = [
            k
            for k, v in streams.items()
            if issubclass(typeof(v), jax.Array | jax.core.Tracer) and k in body_fvs
        ]
        if not used:
            return fwd()

        # an empty reduction axis collapses the whole reduce to the identity
        if any(_arange_length(streams[k]) == 0 for k in used):
            return monoid.identity

        tail_streams = {k: v for (k, v) in streams.items() if k not in used}

        (subst_body, subst_tail), fresh = _substitute_streams(
            (body, tail_streams), streams, used
        )
        pos_body = bind_dims(subst_body, *fresh.values())
        reduced_body = reductor(pos_body, axis=tuple(range(len(fresh))))
        if subst_tail:
            return monoid.reduce(reduced_body, subst_tail)
        return reduced_body


@Operation.define
def delta(_index: tuple[int, ...], _weight: jax.Array) -> jax.Array:
    raise NotHandled


arange = Operation.define(jnp.arange)


def _range_start(term: Term):
    assert term.op == arange
    if "start" in term.kwargs:
        return term.kwargs["start"]
    if len(term.args) < 2:
        return 0
    return term.args[0]


def _range_stop(term: Term):
    assert term.op == arange
    if "stop" in term.kwargs:
        return term.kwargs["stop"]
    if len(term.args) < 2:
        return term.args[0]
    return term.args[1]


def _range_step(term: Term):
    assert term.op == arange
    if "step" in term.kwargs:
        return term.kwargs["step"]
    if len(term.args) < 3:
        return 1
    return term.args[2]


def _is_simple_range(term: Term) -> bool:
    if term.op != arange:
        return False

    start = _range_start(term)
    step = _range_step(term)
    return (
        not isinstance(start, Term)
        and start == 0
        and not isinstance(step, Term)
        and step == 1
    )


def _arange_length(term) -> int | None:
    """Concrete length of an ``arange`` term, or ``None`` if it is not an
    ``arange`` term or any of its bounds is symbolic (a ``Term``).
    """
    if not (isinstance(term, Term) and term.op is arange):
        return None
    a, b, s = _range_start(term), _range_stop(term), _range_step(term)
    if any(isinstance(x, Term) for x in (a, b, s)):
        return None
    return len(range(int(a), int(b), int(s)))


def _substitute_streams(expr, streams, vars):
    """Substitute each op in ``vars`` into its uses within ``expr``, replacing it
    with a fresh named dimension.

    Returns ``(expr2, fresh)`` where ``fresh`` maps each var op to its fresh
    named-dim op (iteration order matches ``vars``). The fresh dims are left
    *named* -- callers decide when and how to bind them positionally.

    Substitution is context-dependent on the stream value ``streams[k]``:

    - A concrete-bounded ``arange(a, b, s)``: a bare ``v()`` used as a direct
      array index ``arr[..., v(), ...]`` is split into an inner slice
      ``arr[..., a:b:s, ...]`` plus an outer index that swaps ``v()`` for
      ``fresh_v()`` -- so the range never materializes into a gather. Any
      *remaining* ``v()`` is materialized as ``unbind_dims(jnp.arange(a, b, s),
      fresh_v)``.
    - Anything else (a concrete array, a symbolic stream term, or an ``arange``
      with non-concrete bounds): every ``v()`` becomes
      ``unbind_dims(streams[k], fresh_v)`` -- a gather.

    The two passes cannot be fused: the direct-index pass must see a bare
    ``v()`` before the materializing pass rewrites it.
    """
    vars = list(vars)
    fresh = {k: Operation.define(k) for k in vars}

    # concrete-bounded arange streams slice their direct-index uses; the rest
    # gather.
    arange_bounds: dict[Operation, tuple] = {}
    for k in vars:
        v = streams[k]
        if isinstance(v, Term) and v.op is arange:
            a, b, s = _range_start(v), _range_stop(v), _range_step(v)
            if not any(isinstance(x, Term) for x in (a, b, s)):
                arange_bounds[k] = (a, b, s)

    # PHASE 1 (direct): slice a bare range var used as an array index.
    if arange_bounds:

        def _jax_getitem(arr, index):
            inner_index, outer_index = [], []
            progress = False
            for i in index:
                if isinstance(i, Term) and i.op in arange_bounds:
                    a, b, s = arange_bounds[i.op]
                    inner_index.append(slice(a, b, s))
                    outer_index.append(fresh[i.op]())
                    progress = True
                else:
                    inner_index.append(slice(None))
                    outer_index.append(i)
            if progress:
                return jax_getitem(jax_getitem(arr, inner_index), outer_index)
            return fwd(arr, index)

        expr = handler(typing.cast(Interpretation, {jax_getitem: _jax_getitem}))(
            evaluate
        )(expr)

    # PHASE 2 (indirect): substitute any remaining uses of each var.
    subst: dict = {}
    for k in vars:
        if k in arange_bounds:
            a, b, s = arange_bounds[k]
            subst[k] = deffn(unbind_dims(jnp.arange(a, b, s), fresh[k]))
        else:
            subst[k] = deffn(unbind_dims(streams[k], fresh[k]))
    expr = handler(typing.cast(Interpretation, subst))(evaluate)(expr)

    return expr, fresh


class ReduceDeltaIndependent(ObjectInterpretation):
    """Eliminate a Delta that has independent, dense index arguments.

    ══════════════════════════════════════════════════════════════
    reduce(M, streams, delta((), body)) ≡ reduce(M, streams, body)


    reduce(M, {v: range(N)}, delta((v(),) ++ idx', body))
    ═══════════════════════════════════════════════════════
    bind_dims(body[v() := unbind_dims(streams[v], fv)], fv)


    reduce(M, streams ∪ {v: range(N)}, delta((v(),) ++ idx', body))
    ═══════════════════════════════════════════════════════════════════════════
    bind_dims(reduce(M, streams, delta(idx', body[v() := unbind_dims(streams[v], fv)])), fv)

    Not yet supported:

    - **Strided index streams** (``range(0, N, k)`` for ``k != 1``): the
      premise ``_is_simple_range`` requires ``start == 0`` and ``step == 1``.
      A strided extension would substitute ``v() := unbind_dims(jnp.arange(
      start, stop, step), fv)`` and otherwise follow the same shape — the
      change is purely in the recognised range form, the bind/unbind cycle
      below is unchanged.
    - **Non-zero start** (``range(a, b, 1)`` with ``a != 0``): same template
      as the strided case; only the recognised range form changes.
    - **Non-bare index expressions** (``delta((2*v(),), w)``,
      ``delta((f(v()),), w)``, etc.): currently requires the final index
      entry to be a bare call ``v()`` of a stream var op. Generalizing to
      arbitrary index expressions is a scatter, not a bind: materialize the
      index expression and the weight separately over ``v``, then
      ``jnp.zeros(N).at[indices].set(values)`` (for Sum; analogous for
      other monoids using ``.add``/``.min``/``.max``/...). This is a
      different leaf operation from ``bind_dims`` and warrants a sibling
      rule rather than an extension of this one.
    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if not (isinstance(body, Term) and body.op == delta):
            return fwd()

        indices, weight = body.args
        assert isinstance(indices, tuple)

        if not indices:
            return monoid.reduce(weight, streams)

        head_index, tail_indices = indices[0], indices[1:]
        if not (isinstance(head_index, Term) and head_index.op in streams):
            return fwd()

        head_op: Operation = head_index.op
        head_stream = streams[head_op]
        if not (isinstance(head_stream, Term) and _is_simple_range(head_stream)):
            return fwd()

        # peel the head index: substitute it into the weight (slicing direct
        # uses, materializing the rest) along a fresh named dim, but bind that
        # dim only *after* the surrounding reduce -- see the class docstring.
        weight2, fresh = _substitute_streams(weight, streams, [head_op])
        fresh_op = fresh[head_op]

        fresh_streams = {k: v for (k, v) in streams.items() if k != head_op}
        if tail_indices or fresh_streams:
            inner = monoid.reduce(delta(tail_indices, weight2), fresh_streams)
        else:
            inner = weight2

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
            if isinstance(v, Term) and _is_simple_range(v)
        }
        for u, u_stream in simple_ranges.items():
            if fvsof(u_stream) & stream_vars:
                continue

            for v, v_stream in simple_ranges.items():
                if (
                    isinstance(v_stream, Term)
                    and isinstance(_range_stop(v_stream), Term)
                    and _range_stop(v_stream).op == u
                ):
                    fresh_streams = {
                        a: (u_stream if a == v else b) for (a, b) in streams.items()
                    }

                    # there are other commuting rules for delta that we do not
                    # currently include
                    if isinstance(body, Term) and body.op == delta:
                        fresh_body = delta(
                            body.args[0],
                            jnp.where(v() < u(), body.args[1], monoid.identity),  # type: ignore[arg-type]
                        )
                    else:
                        fresh_body = jnp.where(v() < u(), body, monoid.identity)

                    return monoid.reduce(fresh_body, fresh_streams)

        return fwd()


class ReduceRange(ObjectInterpretation):
    """Replace concrete-range stream values with materialized ``jnp.arange``.

    reduce(M, streams ∪ {v: range(a, b, s)}, body)
    ≡ reduce(M, streams ∪ {v: jnp.arange(a, b, s)}, body)

    when ``a``, ``b``, ``s`` are concrete and ``body`` is not a delta term.
    Delegates the actual reduction to whichever handler picks up the
    materialized ``jax.Array`` streams.
    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if arange in fvsof((body, streams)):
            intp: Interpretation = {arange: jnp.arange}
            subst_body = handler(intp)(evaluate)(body)
            subst_streams = handler(intp)(evaluate)(streams)
            return monoid.reduce(subst_body, subst_streams)
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
#
# - **Empty-domain detection at the term level**: currently size-0 named
#   dims must be resolved by leaf consumers (``bind_dims``, reductors with
#   ``initial=monoid.identity``). The empty-domain check is intentionally
#   NOT a rule on its own — rewrites stay size-polymorphic and leaf ops
#   carry the burden. See the conversation in monoid.py's history for why.


class ReduceSumProductContraction(ObjectInterpretation):
    """Fast-path a sum-of-products contraction.

    A ``tensordot``-detecting variant of :class:`ArrayReduce`. Matches::

        Sum.reduce(Product.plus(A, B), streams)

    when ``A`` and ``B`` are the only factors, and emits a single
    :func:`jnp.tensordot` instead of broadcasting ``A`` and ``B`` into a dense
    ``O(|A|·|B|)`` product and summing it — the :class:`ArrayReduce` baseline.

    """

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
        if not all(k in shared for k in streams):
            return fwd()

        (lhs_subst, rhs_subst), fresh = _substitute_streams(
            (lhs, rhs), streams, streams
        )
        indexes = [fresh[k] for k in streams]

        lhs_bound = bind_dims(lhs_subst, *indexes)
        rhs_bound = bind_dims(rhs_subst, *indexes)

        axes = tuple(range(len(indexes)))
        contraction = jnp.tensordot(lhs_bound, rhs_bound, axes=(axes, axes))
        return contraction


def _named_dims(term: Expr[jax.Array]) -> tuple[Operation, ...]:
    if not (isinstance(term, Term) and term.op == jax_getitem):
        return ()
    index = term.args[1]
    assert isinstance(index, Iterable)
    return tuple(i.op for i in index if isinstance(i, Term) and not i.args)


def _reduce_axis(array, axis=None, **kwargs) -> jax.Array:
    if axis is None:
        return fwd()

    named_dims = _named_dims(array)
    if not named_dims:
        return fwd()

    bound_arr = bind_dims(array, *named_dims)

    if isinstance(axis, int):
        axis = (axis,)
    shifted_axis = tuple(a + len(named_dims) if a >= 0 else a for a in axis)

    reduced = fwd(bound_arr, axis=shifted_axis, **kwargs)
    return unbind_dims(reduced, *named_dims)


PartialEvalSingleAxisReduce: Interpretation = typing.cast(
    Interpretation,
    {
        op: _reduce_axis
        for op in [
            jnp.sum,
            jnp.prod,
            jnp.min,
            jnp.max,
            jnp.any,
            jnp.all,
            jnp.mean,
            jnp.argmax,
            logsumexp,
        ]
    },
)
"""Partial evaluator for reductions over a single axis.

More efficient than vmapping a single axis reduction over the remaining
dimensions. Multi-axis reductions are still vmapped.

"""


class PartialEvalMultiAxisReduce(ObjectInterpretation):
    @implements(jnp.tensordot)  # type: ignore[arg-type]
    def _(self, a: jax.Array, b: jax.Array, axes=2, **kwargs) -> jax.Array:
        a_shape = a.shape

        if isinstance(a_shape, Term):
            return fwd()

        # convert named dims that appear in one argument but not both to
        # positional
        named_a = _named_dims(a)
        named_b = _named_dims(b)
        distinct = set(named_a) ^ set(named_b)
        if not distinct:
            return fwd()

        distinct_a = [i for i in named_a if i in distinct]
        distinct_b = [i for i in named_b if i in distinct]
        bound_a = bind_dims(a, *distinct_a)
        bound_b = bind_dims(b, *distinct_b)

        # reduction dims shift right to accommodate new positional dims
        if isinstance(axes, int):
            a_dims = range(len(a_shape) - axes, len(a_shape))
            b_dims = range(axes)
        else:
            a_dims = axes[0]
            b_dims = axes[1]
        shifted_a_dims = tuple(d + len(distinct_a) if d >= 0 else d for d in a_dims)
        shifted_b_dims = tuple(d + len(distinct_b) if d >= 0 else d for d in b_dims)
        assert len(shifted_a_dims) == len(shifted_b_dims)

        reduced = fwd(bound_a, bound_b, axes=(shifted_a_dims, shifted_b_dims), **kwargs)

        # reindex bound named dims
        reindexed = jax_getitem(
            reduced,
            tuple(i() for i in distinct_a)
            # skip remaining positional dims in a
            + tuple(slice(None) for _ in range(len(a_shape) - len(shifted_a_dims)))
            + tuple(i() for i in distinct_b),
        )

        return reindexed


@jax.jit(static_argnums=(0,))
def einsum(subscripts: str, /, *operands: jax.Array) -> jax.Array:
    """Evaluate an einsum expression using monoid reductions."""
    if not operands:
        raise ValueError("einsum requires at least one operand")

    in_spec, out_spec, _ = opt_einsum.parser.parse_einsum_input(
        [subscripts, *(op.shape for op in operands)], shapes=True
    )
    in_specs = in_spec.split(",")

    all_letters = set(out_spec) | {c for s in in_specs for c in s}
    ops = {c: Operation.define(jax.Array, name=c) for c in all_letters}

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

    arrays = [Operation.define(jax.Array) for _ in operands]
    factors = [
        unbind_dims(arr(), *(ops[c] for c in spec))
        for arr, spec in zip(arrays, in_specs, strict=True)
    ]
    body = Product.plus(*factors)

    out_tuple = tuple(ops[c]() for c in out_spec)
    streams = {op: arange(sizes[c]) for c, op in ops.items()}
    with (
        handler(NormalizeIntp),
        handler(PartialEvalSingleAxisReduce),
        handler(PartialEvalMultiAxisReduce()),
    ):
        result = deffn(Sum.reduce(delta(out_tuple, body), streams), *arrays)(*operands)
        assert isinstance(result, jax.Array)
        return result


NormalizeIntp.extend(
    ReduceRange(),
    ArrayReduce(),
    ReduceSumProductContraction(),
    ReduceDeltaIndependent(),
    ReduceDependentRangeMask(),
    SumPlusJax(),
    ProductPlusJax(),
    MinPlusJax(),
    MaxPlusJax(),
    LogSumExpPlusJax(),
    CartesianProductPlusJax(),
)
