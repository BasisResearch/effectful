import functools
import logging
import typing
from collections.abc import Iterable
from typing import Protocol

import jax
import jax.core
import opt_einsum

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, sizesof, unbind_dims
from effectful.handlers.jax._handlers import is_eager_array
from effectful.handlers.jax.partial import (
    PartialEvalMultiAxisReduce,
    PartialEvalSingleAxisReduce,
)
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
    outer_stream,
)
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, defdata, deffn, implements
from effectful.ops.types import Interpretation, NotHandled, Operation, Term

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
    :class:`jax.Array` (no Terms).
    """
    return bool(args) and all(is_eager_array(a) for a in args)


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


ARRAY_REDUCTORS = {
    Sum: jnp.sum,
    Product: jnp.prod,
    Min: jnp.min,
    Max: jnp.max,
    LogSumExp: logsumexp,
}


class ArrayReduce(ObjectInterpretation):
    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        reductor = ARRAY_REDUCTORS.get(monoid, None)
        if reductor is None:
            return fwd()

        if typeof(body) is not jax.Array:
            return fwd()

        body_fvs = fvsof(body)
        used_array_streams = {
            k
            for k, v in streams.items()
            if issubclass(typeof(v), jax.Array | jax.core.Tracer) and k in body_fvs
        }
        if not used_array_streams:
            return fwd()

        tail_streams = {
            k: v for (k, v) in streams.items() if k not in used_array_streams
        }

        indexes = [
            (k, Operation.define(jax.Array)) for k in streams if k in used_array_streams
        ]
        subst_intp = typing.cast(
            Interpretation,
            {k: deffn(unbind_dims(streams[k], index)) for (k, index) in indexes},
        )
        subst_body = handler(subst_intp)(evaluate)(body)
        subst_tail_streams = handler(subst_intp)(evaluate)(tail_streams)
        pos_body = bind_dims(subst_body, *(i for (_, i) in indexes))
        reduced_body = reductor(pos_body, axis=tuple(py_range(len(indexes))))
        if subst_tail_streams:
            return monoid.reduce(reduced_body, subst_tail_streams)
        return reduced_body


@Operation.define
def delta(_index: tuple[int, ...], _weight: jax.Array) -> jax.Array:
    raise NotHandled


py_range = range


@Operation.define
def range(*args: int) -> Iterable[jax.Array]:
    raise NotHandled


def _range_start(term: Term):
    assert term.op == range
    if len(term.args) < 2:
        return 0
    return term.args[0]


def _range_stop(term: Term):
    assert term.op == range
    if len(term.args) < 2:
        return term.args[0]
    return term.args[1]


def _range_step(term: Term):
    assert term.op == range
    if len(term.args) < 3:
        return 1
    return term.args[2]


def _is_simple_range(term: Term) -> bool:
    if term.op != range:
        return False

    start = _range_start(term)
    step = _range_step(term)
    return (
        not isinstance(start, Term)
        and start == 0
        and not isinstance(step, Term)
        and step == 1
    )


class BindDimsMonoidPlus(ObjectInterpretation):
    @implements(bind_dims)
    def _(self, value, *names):
        if isinstance(value, Term) and _is_monoid_plus(value.op):
            return value.op(*(bind_dims(x, *names) for x in value.args))
        return fwd()


class BindDimsBindDims(ObjectInterpretation):
    @implements(bind_dims)
    def _(self, value, *names):
        if isinstance(value, Term) and value.op == bind_dims:
            return bind_dims(value.args[0], *(names + value.args[1:]))
        return fwd()


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

    The output index ``v`` is peeled from the *head* of the index tuple and
    substituted with a fresh **named** dimension ``fv`` (via ``unbind_dims``),
    but it is **not** bound to a positional axis until the surrounding reduce
    has finished. Keeping output indices named through the contraction is what
    lets downstream rules (:class:`ReduceSumProductContraction`,
    :class:`ArrayReduce`) align them by identity: if ``v`` were bound to a
    positional axis up front, a factor that does not mention ``v`` would have a
    fabricated size-1 axis that ``jnp.tensordot`` then concatenates as a
    *separate* output axis instead of broadcasting against ``v`` (e.g.
    ``einsum("ij,j->i")`` yielding ``(N, 1)`` instead of ``(N,)``). Binding
    after the reduce — in head-first peel order, so the leftmost output index
    ends up as the leading axis — avoids that. ``fv`` survives the contraction
    as a named free variable and is materialised positionally exactly once,
    here.

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

        fresh_op = Operation.define(head_op)

        # optimize direct indexing (only works for simple ranges)
        def _jax_getitem(arr, index):
            return fwd(arr, [fresh_op() if i.op == head_op else i for i in index])

        direct_weight = handler(
            typing.cast(Interpretation, {jax_getitem: _jax_getitem})
        )(evaluate)(weight)

        # substitute indirect indexing
        arange = jnp.arange(_range_stop(head_stream))
        if isinstance(arange, jax.Array) and len(arange) == 0:
            return monoid.identity
        fresh_stream = unbind_dims(arange, fresh_op)

        indirect_weight = handler(
            typing.cast(Interpretation, {head_op: deffn(fresh_stream)})
        )(evaluate)(direct_weight)

        fresh_streams = {k: v for (k, v) in streams.items() if k != head_op}
        if tail_indices or fresh_streams:
            inner = monoid.reduce(delta(tail_indices, indirect_weight), fresh_streams)
        else:
            inner = indirect_weight

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
        if isinstance(body, Term) and body.op == delta:
            return fwd()

        new_streams: dict = {}
        any_replaced = False
        for k, v in streams.items():
            if isinstance(v, Term) and v.op == range:
                new_streams[k] = jnp.arange(
                    _range_start(v), _range_stop(v), _range_step(v)
                )
                any_replaced = True
            else:
                new_streams[k] = v

        if not any_replaced:
            return fwd()
        return monoid.reduce(body, new_streams)


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


def partition(iterable, predicate):
    trues, falses = [], []
    for item in iterable:
        (trues if predicate(item) else falses).append(item)
    return trues, falses


class ReduceOrderContraction(ObjectInterpretation):
    """Reorder a large product before contraction using an ``opt_einsum`` path.

    Matches ``monoid.reduce(monoid2.plus(f1, ..., fn), streams)`` where
    ``monoid2`` distributes over ``monoid`` and there are at least three
    factors. Each factor is modelled as a tensor whose dimensions are the
    streams it reads (contracted) together with any surviving named/anonymous
    dimensions (kept in the output). An :func:`opt_einsum.contract_path` cost
    model over those sizes chooses a contraction order, and the product is
    re-emitted as a nest of pairwise reduces following that path.

    The pairwise reduces are left symbolic: :class:`ReduceSumProductContraction`
    (for ``Sum``/``Product``) or :class:`ArrayReduce` lowers each one into an
    actual contraction, and streams are substituted there rather than here.
    """

    @staticmethod
    def _stream_length(stream) -> int | None:
        if isinstance(stream, jax.Array):
            shape = stream.shape
            if len(shape) == 0:
                raise ValueError("Unexpected scalar array")
            return shape[0]
        elif isinstance(stream, Term) and stream.op == range:
            start = _range_start(stream)
            stop = _range_stop(stream)
            step = _range_step(stream)

            if (
                isinstance(start, Term)
                or isinstance(stop, Term)
                or isinstance(step, Term)
            ):
                return None

            if step > 0 and start < stop:
                length = (stop - start - 1) // step + 1
            elif step < 0 and start > stop:
                length = (start - stop - 1) // (-step) + 1
            else:
                length = 0
            assert length >= 0
            return length
        return None

    @staticmethod
    def _anon_sizes(factor, named_ops: list) -> tuple[int, ...]:
        """Sizes of ``factor``'s positional (unnamed) dimensions.

        Binding every named dimension exposes the anonymous axes as the
        trailing shape. Symbolic factors do not reduce to a concrete array, so
        they report no anonymous dimensions.
        """
        bound = bind_dims(factor, *named_ops)
        if is_eager_array(bound):
            return tuple(bound.shape[len(named_ops) :])
        return ()

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        # the body must be a monoid2.plus for a monoid2 such that
        # distributes_over(monoid2, monoid)
        if not (isinstance(body, Term) and _is_monoid_plus(body.op)):
            return fwd()

        monoid2: Monoid = body.op.__self__
        if not distributes_over(monoid2, monoid):
            return fwd()

        factors = body.args
        if len(factors) < 3:
            return fwd()
        if not all(issubclass(typeof(f), jax.Array) for f in factors):
            return fwd()

        stream_vars = set(streams.keys())
        if not stream_vars:
            return fwd()

        # grab sizes of reduction dimensions and any dimensions of the factors
        # (named or anonymous)
        #
        # for each factor, we compute an einsum approximation of it so we can
        # compute its cost
        #
        # any anonymous/named dimensions get dimension markers, but are never
        # reduced so must all appear in the output side; streams also get
        # dimension markers and are reduced so do not appear in the output side;
        # factors with no visible dimensions are treated as scalars
        dim_size: dict = {}

        def record(key, size: int | None) -> None:
            if size is not None:
                dim_size[key] = size

        factor_dims: list[list] = []
        for idx, f in enumerate(factors):
            named = sizesof(f)
            f_streams = fvsof(f) & stream_vars
            keys: list = []
            # reduced (stream) dimensions, sized by the stream itself
            for op in f_streams:
                record(op, self._stream_length(streams[op]))
                keys.append(op)
            # surviving named dimensions
            for op, size in named.items():
                record(op, size)
                if op not in f_streams:
                    keys.append(op)
            # surviving anonymous (positional) dimensions
            for pos, size in enumerate(self._anon_sizes(f, list(named.keys()))):
                anon_key = ("anon", idx, pos)
                record(anon_key, size)
                keys.append(anon_key)
            factor_dims.append(keys)

        # a cost model needs a known size for every dimension
        if any(k not in dim_size for keys in factor_dims for k in keys):
            return fwd()

        # build a string / shapes input compatible with opt_einsum.contract_path
        symbols: dict = {}

        def sym(key) -> str:
            if key not in symbols:
                symbols[key] = opt_einsum.get_symbol(len(symbols))
            return symbols[key]

        in_specs = ["".join(sym(k) for k in keys) for keys in factor_dims]

        out_keys: list = []
        seen: set = set()
        for keys in factor_dims:
            for k in keys:
                if k not in stream_vars and k not in seen:
                    seen.add(k)
                    out_keys.append(k)
        out_spec = "".join(sym(k) for k in out_keys)

        subscripts = ",".join(in_specs) + "->" + out_spec
        shapes = [tuple(dim_size[k] for k in keys) for keys in factor_dims]

        path, _ = opt_einsum.contract_path(
            subscripts, *shapes, shapes=True, optimize="auto"
        )

        # a single step is no reordering at all — defer to the greedy
        # contraction handlers rather than reproducing the same product
        if len(path) < 2:
            return fwd()

        # given a contraction path, generate a new reduce nest. streams don't
        # need to be substituted/contractions written out, since ArrayReduce and
        # ReduceSumProductContraction will do that.
        #
        # Each operand tracks the streams it still depends on; a stream is
        # reduced at the step where it stops appearing in any other operand.
        remaining: list[tuple] = [
            (f, {k for k in keys if k in stream_vars})
            for f, keys in zip(factors, factor_dims, strict=True)
        ]

        reduced: set = set()
        for step in path:
            selected = [remaining[k] for k in step]
            terms = [t for (t, _) in selected]

            for k in sorted(step, reverse=True):
                del remaining[k]

            used = set().union(*(s for (_, s) in selected))
            elsewhere = (
                set().union(*(s for (_, s) in remaining)) if remaining else set()
            )
            contract = used - elsewhere

            # using monoid2.plus directly causes an infinite loop
            combined = defdata(monoid2.plus, *terms)
            if contract:
                substreams = {s: v for (s, v) in streams.items() if s in contract}
                new_term = monoid.reduce(combined, substreams)
                reduced |= contract
            else:
                new_term = combined
            remaining.append((new_term, used - contract))

        assert len(remaining) == 1
        result = remaining[0][0]

        # streams referenced by no factor are reduced last (they scale the body)
        leftover = {s: v for (s, v) in streams.items() if s not in reduced}
        if leftover:
            result = monoid.reduce(result, leftover)
        return result


class ReduceSumProductContraction(ObjectInterpretation):
    """Fast-path a sum-of-products contraction.

    A ``tensordot``-detecting variant of :class:`ArrayReduce`. Matches::

        Sum.reduce(Product.plus(A, B), streams)

    when ``A`` and ``B`` are the only factors, and emits a single
    :func:`jnp.tensordot` instead of broadcasting ``A`` and ``B`` into a dense
    ``O(|A|·|B|)`` product and summing it — the :class:`ArrayReduce` baseline.

    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if monoid is not Sum:
            return fwd()

        if not (
            isinstance(body, Term)
            and _is_monoid_plus(body.op)
            and body.op.__self__ is Product
        ):
            return fwd()

        factors = body.args
        if len(factors) < 2 or not all(
            issubclass(typeof(f), jax.Array) for f in factors
        ):
            return fwd()

        (lhs, rhs), tail = factors[:2], factors[2:]
        stream_vars = set(streams.keys())

        lhs_idx = fvsof(lhs) & stream_vars
        rhs_idx = fvsof(rhs) & stream_vars
        tail_idx = fvsof(tail) & stream_vars

        shared = (lhs_idx & rhs_idx) - tail_idx
        contracted = [k for k in streams if k in shared]

        indexes = [Operation.define(op) for op in contracted]
        subst = typing.cast(
            Interpretation,
            {
                k: deffn(unbind_dims(streams[k], i))
                for (k, i) in zip(contracted, indexes, strict=True)
            },
        )
        tail_streams = {k: v for (k, v) in streams.items() if k not in shared}

        with handler(subst):
            lhs_subst, rhs_subst, tail_streams_subst = typing.cast(
                tuple, evaluate((lhs, rhs, tail_streams))
            )

        lhs_bound = bind_dims(lhs_subst, *indexes)
        rhs_bound = bind_dims(rhs_subst, *indexes)

        axes = tuple(py_range(len(contracted)))
        contraction = Product.plus(
            jnp.tensordot(lhs_bound, rhs_bound, axes=(axes, axes)), *factors[2:]
        )
        if tail_streams_subst:
            return Sum.reduce(contraction, tail_streams_subst)
        return contraction


def _parse_einsum_spec(
    subscripts: str, *operands: tuple[int, ...]
) -> tuple[list[str], str]:
    """Parse a numpy/jax-style einsum subscripts string.

    Supports explicit ``"ij,jk->ik"`` and implicit ``"ij,jk"`` forms; in the
    implicit case the output is each letter appearing exactly once across all
    inputs, sorted alphabetically. Does not support ellipsis ``...``.
    """
    if "..." in subscripts:
        raise NotImplementedError("einsum ellipsis not yet supported")
    if "->" in subscripts:
        in_part, out_spec = subscripts.split("->")
    else:
        in_part = subscripts
        counts: dict[str, int] = {}
        for c in in_part.replace(",", ""):
            counts[c] = counts.get(c, 0) + 1
        out_spec = "".join(sorted(c for c, n in counts.items() if n == 1))
    in_specs = in_part.split(",")
    if len(in_specs) != len(operands):
        raise ValueError(
            f"einsum: {len(in_specs)} input specs but {len(operands)} operands"
        )
    for spec, shape in zip(in_specs, operands, strict=True):
        if len(spec) != len(shape):
            raise ValueError(
                f"einsum spec {spec!r} has {len(spec)} indices but operand "
                f"has shape {shape}"
            )
    return in_specs, out_spec


def einsum(subscripts: str, /, *operands: jax.Array) -> jax.Array:
    """Evaluate an einsum expression using monoid reductions.

    Mirrors the ``jax.numpy.einsum(subscripts, *operands)`` interface for the
    subset of subscript strings that do not use ``...`` (ellipsis). The
    expression is encoded as a product of named-dim-bound operands wrapped in
    a :func:`delta` over the output indices and reduced with
    :data:`Sum.reduce` over every index; :class:`ReduceDeltaIndependent` peels
    output indices into positional axes and :class:`ArrayReduce` sums over the
    remainder.

    Repeated index letters within a single operand (``"ii"``, ``"iij"``) work
    because the unbinding substitution inside :class:`ReduceDeltaIndependent`
    threads one fresh op through every occurrence of the repeated index — the
    diagonal falls out naturally without a separate ``jnp.diagonal`` step.

    This is the *unoptimised* baseline: all operands are broadcast into a
    single intermediate before reduction. A contraction-ordering normalization
    rule (planned) reorders large products into pairwise contractions without
    requiring changes here.
    """
    if not operands:
        raise ValueError("einsum requires at least one operand")

    in_specs, out_spec = _parse_einsum_spec(subscripts, *[op.shape for op in operands])

    all_letters = set(out_spec) | {c for s in in_specs for c in s}
    ops = {c: Operation.define(jax.Array, name=c) for c in all_letters}

    sizes = {}
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
    streams = {op: range(sizes[c]) for c, op in ops.items()}
    expr = deffn(Sum.reduce(delta(out_tuple, body), streams), *arrays)
    norm_expr = handler(NormalizeIntp)(evaluate)(expr)

    print("einsum %r normalized to:\n%s" % (subscripts, norm_expr))

    @jax.jit
    def jitted_einsum(*args):
        with (
            handler(NormalizeIntp),
            handler(PartialEvalSingleAxisReduce),
            handler(PartialEvalMultiAxisReduce()),
        ):
            result = norm_expr(*args)
            return result

    return jitted_einsum(*operands)


NormalizeIntp.extend(
    ArrayReduce(),
    ReduceSumProductContraction(),
    ReduceRange(),
    ReduceDeltaIndependent(),
    ReduceOrderContraction(),
    ReduceDependentRangeMask(),
    SumPlusJax(),
    ProductPlusJax(),
    MinPlusJax(),
    MaxPlusJax(),
    LogSumExpPlusJax(),
    CartesianProductPlusJax(),
    BindDimsMonoidPlus(),
    BindDimsBindDims(),
)
