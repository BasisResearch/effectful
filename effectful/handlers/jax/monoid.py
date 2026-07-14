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
from effectful.handlers.jax._handlers import is_eager_array
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
    complement,
    distributes_over,
    is_equality,
)
from effectful.ops.monoid import Union as UnionM
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, implements, ite
from effectful.ops.types import Expr, Interpretation, Operation, Term

logger = logging.getLogger(__name__)


LogSumExp = Monoid(name="LogSumExp", identity=jnp.asarray(float("-inf")))

# ``Sum`` in log space is multiplication, which distributes over ``LogSumExp``:
#   a + logsumexp(b, c) = logsumexp(a + b, a + c)
distributes_over.register(Sum, LogSumExp)

is_equality.register(jnp.equal)
for a, b in {
    (jnp.less, jnp.greater),
    (jnp.less_equal, jnp.greater_equal),
    (jnp.equal, jnp.not_equal),
}:
    complement.register(a, b)


def _jax_args(args):
    """True iff ``args`` is non-empty and every arg is a concrete
    :class:`jax.typing.ArrayLike` or named tensor.

    """
    return args and all(
        isinstance(a, jax.typing.ArrayLike) or is_eager_array(a) for a in args
    )


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


class IteJax(ObjectInterpretation):
    @implements(ite)
    def ite(self, cond, then, else_):
        if not _jax_args((cond,)):
            return fwd()
        return jnp.where(cond, then, else_)


class MaskJax(ObjectInterpretation):
    @implements(Monoid.mask)
    def mask(self, monoid, value, mask):
        if not (
            (is_eager_array(value) or not isinstance(value, Term))
            and (is_eager_array(mask) or not isinstance(value, Term))
        ):
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
        if not is_eager_array(body):
            return fwd()

        reductor = ARRAY_REDUCTORS.get(monoid, None)
        if reductor is None:
            return fwd()

        body_fvs = fvsof(body)
        used = {k for k, v in streams.items() if k in body_fvs and isinstance(v, range)}
        if not used:
            return fwd()

        index = tuple(k() for k in streams if k in used)
        arr = monoid.reduce(monoid.delta(index, body), streams)
        reduced_body = reductor(arr, axis=tuple(range(len(used))))
        return reduced_body


class ReduceArrayScan(ObjectInterpretation):
    @staticmethod
    def _disequality_to_plus(monoid, streams, stream_op, value, index, tail_mask_elems):
        return monoid.plus(
            monoid.reduce(
                monoid.mask(value, And.plus(stream_op() < index, *tail_mask_elems)),
                streams,
            ),
            monoid.reduce(
                monoid.mask(value, And.plus(stream_op() > index, *tail_mask_elems)),
                streams,
            ),
        )

    @staticmethod
    def _inequality_to_scan(
        monoid, streams, stream_op, value, index, tail_mask_elems, cmp_op
    ):
        stream = streams[stream_op]
        reverse = cmp_op in (jnp.greater_equal, jnp.greater)
        n = len(stream)
        pos_value = monoid.reduce(
            monoid.delta((stream_op(),), value), {stream_op: stream}
        )
        # inclusive prefix (or suffix, when reverse) scan over the stream
        inclusive = lax.associative_scan(monoid.plus, pos_value, reverse=reverse)
        # The strict comparisons (`<`, `>`) need an *exclusive* scan: pad
        # an identity element on the appropriate side and shift the result
        # so that scan_val[k] aggregates only the positions satisfying the
        # comparison against k. The non-strict comparisons (`<=`, `>=`) use
        # the inclusive scan directly.
        match cmp_op:
            case jnp.less:
                scan_val = jnp.pad(
                    inclusive, (1, 0), mode="constant", constant_values=monoid.identity
                )[:n]
            case jnp.greater:
                scan_val = jnp.pad(
                    inclusive, (0, 1), mode="constant", constant_values=monoid.identity
                )[1:]
            case _:
                scan_val = inclusive

        tail_body = monoid.mask(
            jax_getitem(scan_val, (index,)), And.plus(*tail_mask_elems)
        )
        tail_streams = {k: v for (k, v) in streams.items() if k != stream_op}
        if tail_streams:
            return monoid.reduce(tail_body, tail_streams)
        return tail_body

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
            tail_mask_elems = [e for (j, e) in enumerate(mask_elems) if i != j]
            match elem:
                case Term(jnp.not_equal, args, {}):
                    match args:
                        case (Term(stream_op, (), {}), index) if isinstance(
                            streams.get(stream_op, None), range
                        ):
                            return self._disequality_to_plus(
                                monoid,
                                streams,
                                stream_op,
                                value,
                                index,
                                tail_mask_elems,
                            )
                        case (index, Term(stream_op, (), {})) if isinstance(
                            streams.get(stream_op, None), range
                        ):
                            return self._disequality_to_plus(
                                monoid,
                                streams,
                                stream_op,
                                value,
                                index,
                                tail_mask_elems,
                            )
                case Term(
                    (
                        jnp.less_equal | jnp.greater_equal | jnp.less | jnp.greater
                    ) as cmp_op,
                    args,
                    {},
                ):
                    match args:
                        case (Term(stream_op, (), {}), index) if isinstance(
                            streams.get(stream_op, None), range
                        ):
                            return self._inequality_to_scan(
                                monoid,
                                streams,
                                stream_op,
                                value,
                                index,
                                tail_mask_elems,
                                cmp_op,
                            )
                        case (index, Term(stream_op, (), {})) if isinstance(
                            streams.get(stream_op, None), range
                        ):
                            return self._inequality_to_scan(
                                monoid,
                                streams,
                                stream_op,
                                value,
                                index,
                                tail_mask_elems,
                                complement.of(cmp_op),
                            )
        return fwd()


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

        gathered_body = (
            monoid.delta(tail_index, gathered_weight) if tail_index else gathered_weight
        )
        inner = (
            monoid.reduce(gathered_body, gathered_streams)
            if gathered_streams
            else gathered_weight
        )
        return bind_dims(inner, fresh_op)


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


@dataclass
class Node:
    ordinal: frozenset[str]
    children: list["Node"]
    factors: list[int]


class _EinsumBuilder:
    in_specs: Sequence[str]
    out_spec: str
    plates: frozenset[str]
    operands: Sequence[jax.Array]

    def __init__(
        self, subscripts: str, /, *operands: jax.Array, plates: str | None = None
    ):
        if not operands:
            raise ValueError("einsum requires at least one operand")

        in_spec, out_spec, _ = opt_einsum.parser.parse_einsum_input(
            [subscripts, *(op.shape for op in operands)], shapes=True
        )
        in_specs = in_spec.split(",")

        self.in_specs = in_specs
        self.out_spec = out_spec
        self.plates = frozenset(plates or "")
        self.operands = operands

        # check that the output spec preserves required plates
        out_spec_set = frozenset(self.out_spec)
        out_plates = out_spec_set & self.plates
        for c in out_spec_set - self.plates:
            missing_plates = self.ordinal[c] - out_plates
            if missing_plates:
                raise ValueError(
                    "It is nonsensical to preserve a plated dim without preserving "
                    f"all of that dim's plates, but found {c!r} without "
                    f"{','.join(sorted(missing_plates))!r}"
                )

    @functools.cached_property
    def sizes(self) -> Mapping[str, int]:
        """`sizes[d]` is the length of a dimension `d`."""
        sizes: dict[str, int] = {}
        for spec, op in zip(self.in_specs, self.operands, strict=True):
            for l, s in zip(spec, op.shape, strict=True):
                if l in sizes and sizes[l] != s:
                    raise ValueError(f"Dimension {l} given sizes {s} and {sizes[l]}")
                else:
                    sizes[l] = s
        for c in self.out_spec:
            if c not in sizes:
                raise ValueError(f"einsum: output index {c!r} not present in any input")
        return sizes

    @functools.cached_property
    def ordinal(self) -> Mapping[str, frozenset[str]]:
        """`ordinal[d]` is the plate context of a dimension `d`."""
        ordinal: dict[str, frozenset[str]] = defaultdict(lambda: self.plates)
        for spec in self.in_specs:
            spec_set = frozenset(spec)
            for c in spec_set - self.plates:
                ordinal[c] &= spec_set
        return ordinal

    @functools.cached_property
    def plate_tree(self) -> Node:
        # 1. ordinal (plate context) of each factor
        factor_ordinal = {
            k: frozenset(spec) & self.plates for k, spec in enumerate(self.in_specs)
        }

        # 2. node set: every factor ordinal, the global root ∅, and every pairwise
        #    intersection. The intersection of two ordinals is the deepest context
        #    that contains both, i.e. their common ancestor -- a shared plate MUST
        #    have a node to live at, or containment can't be a tree. Closing under
        #    ∩ materializes those frame nodes (finite lattice, so it terminates).
        ordinals: set[frozenset[str]] = set(factor_ordinal.values()) | {frozenset()}
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

    def dim_type(self, dim: str) -> type:
        return (
            int
            if dim in self.plates
            else Mapping[tuple, int]
            if self.ordinal[dim]
            else int
        )

    @functools.cached_property
    def dim_op(self) -> Mapping[str, Operation]:
        return {
            dim: Operation.define(self.dim_type(dim), name=dim)
            for dim in set("".join(self.in_specs))
        }

    @functools.cached_property
    def out_vars(self) -> Mapping[str, Operation]:
        return {c: Operation.define(jax.Array, name=f"out_{c}") for c in self.out_spec}

    @functools.cached_property
    def global_enums(self) -> frozenset[str]:
        return frozenset(c for c, o in self.ordinal.items() if not o)

    @functools.cached_property
    def arrays(self) -> list[Term]:
        return [Operation.define(jax.Array)() for _ in self.operands]

    def dim_index(self, dim: str) -> Expr:
        out_globals = self.global_enums & set(self.out_spec)

        return (
            self.out_vars[dim]()
            if dim in out_globals
            else self.dim_op[dim]()
            if dim in self.plates or not self.ordinal[dim]
            else self.dim_op[dim]()[
                tuple(self.dim_op[p]() for p in sorted(self.ordinal[dim]))
            ]
        )

    def out_mask(self, vars):
        eqs = tuple(self.dim_index(p) == self.out_vars[p]() for p in vars)
        return And.plus(*eqs)

    def _build_plate_reductions(
        self, plate_tree: Node, parent_plates: frozenset[str] = frozenset()
    ) -> Mapping[tuple[int, ...], float]:

        masked_factors = []
        for factor_idx in plate_tree.factors:
            spec = self.in_specs[factor_idx]
            factor = jax_getitem(
                self.arrays[factor_idx], tuple(self.dim_index(d) for d in spec)
            )
            # Only plated output dims are delta'd per factor. Global output dims are
            # delta'd once by the outer ``out_mask`` in ``_einsum_expr``; masking
            # them here too would check the same equality (e.g. ``out_b == b``) in
            # every factor that mentions the dim as well as the outer mask.
            factor_out_dims = frozenset(
                d
                for d in set(spec) & set(self.out_vars) - self.plates
                if self.ordinal[d]
            )

            if factor_out_dims:
                # Preserving a plated output dim requires preserving all its plates,
                # as checked in ``__init__``.
                factor_out_plates = plate_tree.ordinal & set(self.out_vars)
                masked_factors.append(
                    ite(
                        self.out_mask(factor_out_plates),
                        Sum.mask(factor, self.out_mask(factor_out_dims)),
                        factor,
                    )
                )
            else:
                masked_factors.append(factor)

        child_reductions = (
            self._build_plate_reductions(n, parent_plates | plate_tree.ordinal)
            for n in plate_tree.children
        )
        product = Product.plus(*masked_factors, *child_reductions)
        if plate_tree.ordinal:
            plate_streams = {
                self.dim_index(p).op: range(self.sizes[p])
                for p in sorted(plate_tree.ordinal - parent_plates)
            }
            return Product.reduce(product, plate_streams)
        return product

    @functools.cached_property
    def term(self):
        # one stream of per-plate-assignment rows for each plated sum dim
        rows = {}
        plated_enums = {c: o for c, o in self.ordinal.items() if o}
        for c, o in plated_enums.items():
            ps = [(p, self.dim_op[p]) for p in sorted(o)]
            delta_idx = tuple(op() for (_, op) in ps)
            c_streams = {op: range(self.sizes[p]) for (p, op) in ps}
            v = Operation.define(int, name=f"{c}_v")
            rows[c] = CartesianProduct.reduce(
                UnionM.reduce(
                    [UnionM.delta(delta_idx, v())], {v: range(self.sizes[c])}
                ),
                c_streams,
            )

        streams: Streams = {
            self.dim_op[c]: range(self.sizes[c])
            for c in self.global_enums
            if c not in self.out_spec
        } | {self.dim_op[c]: r for c, r in rows.items()}

        reductions = self._build_plate_reductions(self.plate_tree)
        reduction = Sum.reduce(reductions, streams) if streams else reductions
        return (
            deffn(
                reduction,
                *(a.op for a in self.arrays),
                *(self.out_vars[c] for c in self.out_spec),
            ),
            [(self.out_vars[c], self.sizes[c]) for c in self.out_spec],
        )


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
    expr, dims = _EinsumBuilder(subscripts, *operands, plates=plates).term

    with handler(NormalizeIntp):
        norm_expr = evaluate(expr)

    assert CartesianProduct.reduce not in fvsof(norm_expr), (
        "failed to eliminate cartesian products"
    )

    with handler(EvaluateIntp), handler(NormalizeIntp):
        assert callable(norm_expr)

        out_dims = tuple(unbind_dims(jnp.arange(d), v) for (v, d) in dims)
        result = evaluate(
            bind_dims(norm_expr(*operands, *out_dims), *(v for (v, _) in dims))
        )
        assert isinstance(result, jax.Array), "failed to fully evaluate"
        return result


EvaluateIntp.extend(
    SumPlusJax(),
    ProductPlusJax(),
    MinPlusJax(),
    MaxPlusJax(),
    LogSumExpPlusJax(),
    AndPlusJax(),
    OrPlusJax(),
    IteJax(),
    MaskJax(),
    ReduceSumProductContraction(),
    ReduceArray(),
    ReduceDeltaSimpleRange(),
    ReduceArrayScan(),
    PlusCastArray(),
)

NormalizeIntp.extend(ReduceArrayGather())
