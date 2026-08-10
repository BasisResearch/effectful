import functools
import logging
import typing
from typing import Protocol

import jax
import jax.core
import opt_einsum
from opt_einsum import get_symbol

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
    choose_contraction,
    distributes_over,
)
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
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
    :class:`jax.typing.ArrayLike` or named tensor. At least one argument must be
    a jax-related type.

    """
    return (
        bool(args)
        and all(is_eager_array(a) or isinstance(a, jax.typing.ArrayLike) for a in args)
        and any(is_eager_array(a) or isinstance(a, jax.Array) for a in args)
    )


class PlusJaxUpcast(ObjectInterpretation):
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


class ReduceArrayGather(ObjectInterpretation):
    """M.reduce(body, {k: a} ∪ S) ≡ M.reduce(body[k := a[k']], {k': range(a.shape[0])} ∪ S)"""

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if typeof(body) is not jax.Array:
            return fwd()

        if isinstance(body, Term) and body.op is delta:
            return fwd()

        body_fvs = fvsof(body)
        stream_keys = set(streams)

        body_subst = {}
        streams_subst = {}
        range_streams = {}
        progress = False
        for k, v in streams.items():
            if is_eager_array(v) and k in body_fvs and not (fvsof(v) & stream_keys):
                kk = Operation.define(k)
                body_subst[k] = deffn(unbind_dims(v, kk))
                streams_subst[k] = kk
                range_streams[kk] = range(v.shape[0])
                progress = True
            else:
                range_streams[k] = v

        if not progress:
            return fwd()

        subst_body = handler(body_subst)(evaluate)(body)
        subst_streams = handler(streams_subst)(evaluate)(range_streams)
        return monoid.reduce(subst_body, subst_streams)


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

        pos_dims = {}
        if isinstance(body, Term):
            if body.op == delta:
                pos_dims = {
                    d.op
                    for d in body.args[0]
                    if isinstance(d, Term) and d.op in streams
                }
            elif _is_monoid_plus(body.op) and distributes_over(
                body.op.__self__, monoid
            ):
                # delegate to factorization
                return fwd()

        body_fvs = fvsof(body)
        used = {
            k
            for k, v in streams.items()
            if k in body_fvs and k not in pos_dims and isinstance(v, range)
        }
        if not used:
            return fwd()

        delta_key = tuple(k() for k in streams if k in used)
        arr = monoid.reduce(delta(delta_key, body), streams)
        reduced_body = reductor(arr, axis=tuple(range(len(used))))
        return reduced_body


@Operation.define
def delta(_index: tuple[int, ...], _weight: jax.Array) -> jax.Array:
    raise NotHandled


def _range_stop(term: Term):
    assert term.op == jnp.arange
    if "stop" in term.kwargs:
        return term.kwargs["stop"]
    if len(term.args) < 2:
        return term.args[0]
    return term.args[1]


class DeltaEmpty(ObjectInterpretation):
    """delta((), weight) ≡ weight"""

    @implements(delta)
    def _(self, index, weight):
        if not index:
            return weight
        return fwd()


class DeltaFusion(ObjectInterpretation):
    """delta(i1, delta(i2, weight)) ≡ delta(i1 ++ i2, weight)"""

    @implements(delta)
    def _(self, index, weight):
        if isinstance(weight, Term) and weight.op == delta:
            return delta(index + weight.args[0], weight.args[1])
        return fwd()


class ReduceDeltaSimpleRange(ObjectInterpretation):
    """Eliminate a Delta that has independent, dense index arguments.


    reduce(M, streams ∪ {v: range(N)}, delta((v(),) ++ idx', body))
    ═══════════════════════════════════════════════════════════════════════════
    bind_dims(reduce(M, streams, delta(idx', body[v() := unbind_dims(streams[v], fv)])), fv)
    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if not (isinstance(body, Term) and body.op == delta):
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
            monoid.reduce(delta(tail_index, gathered_weight), gathered_streams)
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
                    if isinstance(body, Term) and body.op == delta:
                        fresh_body = delta(
                            body.args[0],
                            jnp.where(v() < u(), body.args[1], monoid.identity),  # type: ignore[arg-type]
                        )
                    else:
                        fresh_body = jnp.where(v() < u(), body, monoid.identity)

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
        delta_key = tuple(k() for k in streams)
        pos_lhs = Sum.reduce(delta(delta_key, lhs), streams)
        pos_rhs = Sum.reduce(delta(delta_key, rhs), streams)

        dims = "".join(get_symbol(i) for i in range(len(streams)))
        contraction = jnp.einsum(f"{dims}...,{dims}...->...", pos_lhs, pos_rhs)
        return contraction


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
    streams = {op: range(sizes[c]) for c, op in ops.items()}
    with handler(NormalizeIntp):
        norm = deffn(Sum.reduce(delta(out_tuple, body), streams), *arrays)
        result = norm(*operands)
        assert isinstance(result, jax.Array)
        return result


NormalizeIntp.extend(
    ReduceArray(),
    ReduceSumProductContraction(),
    ReduceArrayGather(),
    ReduceDeltaSimpleRange(),
    ReduceDependentRangeMask(),
    DeltaEmpty(),
    DeltaFusion(),
    SumPlusJax(),
    ProductPlusJax(),
    MinPlusJax(),
    MaxPlusJax(),
    LogSumExpPlusJax(),
    CartesianProductPlusJax(),
    ContractLongestArrayStream(),
    PlusJaxUpcast(),
)
