import functools
import typing
from collections.abc import Iterable
from typing import Protocol

import jax
import opt_einsum

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
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
    outer_stream,
)
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
from effectful.ops.types import Interpretation, NotHandled, Operation, Term


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
        if monoid not in ARRAY_REDUCTORS or typeof(body) is not jax.Array:
            return fwd()
        if not streams:
            return monoid.identity

        reductor = ARRAY_REDUCTORS[monoid]
        index = Operation.define(jax.Array)
        for stream_key, stream_body, streams_tail in outer_stream(streams):
            if not issubclass(typeof(stream_body), jax.Array):
                continue

            if stream_key in fvsof(body):
                with handler({stream_key: deffn(unbind_dims(stream_body, index))}):
                    eval_body = evaluate(body)
                    eval_streams_tail = evaluate(streams_tail)
                    assert isinstance(eval_streams_tail, dict)
                    reduce_tail = (
                        monoid.reduce(eval_body, eval_streams_tail)
                        if len(eval_streams_tail) > 0
                        else eval_body
                    )
                    return reductor(bind_dims(reduce_tail, index), axis=0)
            else:
                # TODO: In this case, the stream is unused in the body. The body
                # should be multiplied by the length of the stream. The current
                # behavior is not efficient.
                return fwd()

        return fwd()


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


    reduce(M, {v: range(N)}, delta(idx' ++ (v(),), body))
    ═══════════════════════════════════════════════════════
    bind_dims(body[v() := unbind_dims(streams[v], fv)], fv)


    reduce(M, streams ∪ {v: range(N)}, delta(idx' ++ (v(),), body))
    ═══════════════════════════════════════════════════════════════════════════
    reduce(M, streams, delta(idx', bind_dims(body[v() := unbind_dims(streams[v], fv)], fv)))

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

        head_indices, tail_index = indices[:-1], indices[-1]
        if not (isinstance(tail_index, Term) and tail_index.op in streams):
            return fwd()

        tail_op: Operation = tail_index.op
        tail_stream = streams[tail_op]
        if not (isinstance(tail_stream, Term) and _is_simple_range(tail_stream)):
            return fwd()

        fresh_op = Operation.define(tail_op)
        indices = jnp.arange(_range_stop(tail_stream))
        if isinstance(indices, jax.Array) and len(indices) == 0:
            return monoid.identity

        fresh_stream = unbind_dims(indices, fresh_op)
        subst_intp = typing.cast(Interpretation, {tail_op: deffn(fresh_stream)})
        fresh_body = bind_dims(handler(subst_intp)(evaluate)(weight), fresh_op)
        fresh_streams = {k: v for (k, v) in streams.items() if k != tail_op}
        if fresh_streams:
            return monoid.reduce(delta(head_indices, fresh_body), fresh_streams)
        else:
            return fresh_body


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

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if not (isinstance(body, Term) and body.op == delta):
            return fwd()

        out_indices, weight = body.args
        if not (isinstance(weight, Term) and _is_monoid_plus(weight.op)):
            return fwd()
        body_monoid = weight.op.__self__

        if not distributes_over(body_monoid, monoid):
            return fwd()

        factors = weight.args
        if len(factors) < 3:
            return fwd()

        sized_dims = {
            op: l
            for (op, stream) in streams.items()
            if (l := self._stream_length(stream)) is not None
        }

        if not all(isinstance(i, Term) and i.op in sized_dims for i in out_indices):
            return fwd()

        sized_tensor_factors, other_factors = partition(
            factors,
            lambda f: (
                issubclass(typeof(f), jax.Array)
                and (fvsof(f) & set(streams.keys())) <= set(sized_dims.keys())
            ),
        )
        factor_indices = [
            list(fvsof(f) & set(streams.keys())) for f in sized_tensor_factors
        ]

        dim_to_symbol = {
            op: opt_einsum.get_symbol(i) for (i, op) in enumerate(sized_dims.keys())
        }

        contract_str = (
            ",".join(
                "".join(dim_to_symbol[i] for i in indices) for indices in factor_indices
            )
            + "->"
            + "".join(dim_to_symbol[i.op] for i in out_indices)
        )
        contract_shapes = [tuple(sized_dims[i] for i in ix) for ix in factor_indices]
        (_, info) = opt_einsum.contract_path(
            contract_str, *contract_shapes, shapes=True
        )

        reduced_factors = list(sized_tensor_factors)
        for (fst_idx, snd_idx), reduce_dims, _, _, _ in info.contraction_list:
            fst = reduced_factors[fst_idx]
            del reduced_factors[fst_idx]

            snd = reduced_factors[snd_idx]
            del reduced_factors[snd_idx]

            pair_dims = (fvsof(fst) | fvsof(snd)) & set(sized_dims.keys())
            retained_dims = tuple(
                i() for i in pair_dims if dim_to_symbol[i] not in reduce_dims
            )
            reduced_factor = monoid.reduce(
                delta(retained_dims, body_monoid.plus(fst, snd)),
                {op: s for (op, s) in streams.items() if op in pair_dims},
            )
            breakpoint()
            reduced_factors.insert(0, reduced_factor)

        other_streams = {
            op: stream for (op, stream) in streams.items() if op not in sized_dims
        }
        if other_streams:
            result = monoid.reduce(
                body_monoid.plus(*reduced_factors, *other_factors), other_streams
            )
        else:
            result = body_monoid.plus(*reduced_factors, *other_factors)
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
        subst: Interpretation = {
            k: deffn(unbind_dims(streams[k], i))
            for (k, i) in zip(contracted, indexes, strict=True)
        }
        tail_streams = {k: v for (k, v) in streams.items() if k not in shared}

        with handler(subst):
            lhs_subst, rhs_subst, tail_streams_subst = evaluate(
                (lhs, rhs, tail_streams)
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


class Einsum(Protocol):
    def __call__(self, *args: jax.Array) -> jax.Array: ...


def einsum(subscripts: str, *operands: tuple[int, ...]) -> Einsum:
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
    in_specs, out_spec = _parse_einsum_spec(subscripts, *operands)

    all_letters = set(out_spec) | {c for s in in_specs for c in s}
    ops = {c: Operation.define(jax.Array, name=c) for c in all_letters}

    sizes = {}
    for spec, shape in zip(in_specs, operands, strict=True):
        for l, s in zip(spec, shape, strict=True):
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
    return typing.cast(Einsum, norm_expr)


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
