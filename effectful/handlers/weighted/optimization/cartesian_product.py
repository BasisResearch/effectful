import functools
import itertools

import jax

import effectful.ops.weighted.reduce as ops
from effectful.handlers.jax import numpy as jnp
from effectful.handlers.jax._handlers import is_eager_array
from effectful.ops.semantics import evaluate, fvsof, fwd
from effectful.ops.syntax import (
    ObjectInterpretation,
    deffn,
    defop,
    implements,
    syntactic_eq,
)
from effectful.ops.types import Term
from effectful.ops.weighted.monoid import JaxCartesianProdMonoid, Monoid
from effectful.ops.weighted.reduce import order_streams


def unify_streams(streams1: dict, streams2: dict) -> dict | None:
    """
    Returns the interpretation that unifies
    streams2 with streams1, or None if there
    is no unifier.
    """
    if len(streams1) != len(streams2):
        return None

    streams1_order = order_streams(streams1)
    streams2_order = order_streams(streams2)

    unifier = dict(zip(streams1_order, streams2_order, strict=True))
    for k1, k2 in unifier.items():
        v1 = evaluate(streams1[k1], intp=unifier)  # type: ignore
        v2 = streams2[k2]
        if k1.__name__ != k2.__name__ or not syntactic_eq(v1, v2):
            return None
    return unifier


class ReduceDistributeCartesianProduct(ObjectInterpretation):
    """Eliminates a reduce over a cartesian product.
        ∑_x₁ ∑_x₂ ... ∑_xₙ f(x₁) ⋅ f(x₂) ... f(xₙ) = ∏_i ∑_xᵢ f(xᵢ)
    This transform is also called inversion in the lifting
    literature (e.g. [1]).

    More specifically, this transform implements the identity
    reduce(⨁, {v: reduce(×, S1, body1)} ∪ S2, reduce(⨂, S1, body2))
        = reduce(⨁, S2, reduce(⨂, S1, reduce(⨁, {v: repeat(body1)}, body2)))
            where × is the cartesian product and ⨂ distributes over ⨁.

    Note: This could be generalized to grouped inversion [2].

    [1] Braz, Rd, Eyal Amir, and Dan Roth. "Lifted first-order
    probabilistic inference." IJCAI. 2005.
    [2] Taghipour, Nima, et al. "Completeness results for lifted
    variable elimination." AISTATS. 2013.
    """

    @implements(ops.reduce)
    def reduce(self, monoid1: Monoid, streams, body):
        match body:
            case Term(ops.reduce, (monoid2, streams1, body2)):
                assert isinstance(monoid2, Monoid)

                if not monoid1.distributes_with(monoid2):
                    return fwd()
                if not all(is_eager_array(v) for v in streams1.values()):
                    return fwd()

                for k, v in streams.items():
                    match v:
                        case Term(ops.reduce, (_monoid, _streams1, body1)) if (
                            _monoid is JaxCartesianProdMonoid
                        ):
                            stream_unifier = unify_streams(streams1, _streams1)
                            if stream_unifier is None:
                                continue
                            if any(k in fvsof(v) for v in _streams1.values()):
                                continue  # another streams depends on the cartesian product stream

                            # in case cartesian stream depends on other streams
                            body1 = evaluate(body1, intp=stream_unifier)

                            size = functools.reduce(
                                lambda x, y: x * y, map(len, streams1.values())
                            )
                            cartesian_stream = {k: jnp.tile(body1, (size, 1)).T}
                            streams2 = {k2: v2 for k2, v2 in streams.items() if k != k2}

                            inner_sum_reduce = ops.reduce(
                                monoid1, cartesian_stream, body2
                            )
                            reduce = ops.reduce(monoid2, streams1, inner_sum_reduce)
                            if len(streams2) == 0:
                                return reduce
                            return ops.reduce(monoid1, streams2, reduce)
        return fwd()


class SplitCartesianProductReduce(ObjectInterpretation):
    """
    Transforms a reduce over a cartesian product into
    a reduce over separate streams. For a single cartesian
    product, this roughly corresponds to
        reduce(R, {v: reduce(×, {i: S}, body1)}, body2)
        = reduce(R, {v₁: body1/{i↦S[0]}, ..., vₙ: body1/{i↦S[n]}}, body2/{v↦(v₁, ..., vₙ)})

        (where expr/{x↦c} substitutes x with c in expr)

    This is useful because, as opposed to a single cartesian
    product stream, the individual streams can be rearranged.
    """

    @implements(ops.reduce)
    def reduce(self, monoid, streams, body):
        for stream_var, stream_arr in streams.items():
            match stream_arr:
                case Term(ops.reduce, (_monoid, cart_streams, plate_body)) if (
                    _monoid is JaxCartesianProdMonoid
                ):
                    if not all(
                        isinstance(stream, jax.Array)
                        for stream in cart_streams.values()
                    ):
                        continue  # cartesian streams need to be ground

                    cart_vals = tuple(itertools.product(*cart_streams.values()))
                    fresh_vars = [defop(stream_var)() for _ in cart_vals]
                    stream_intp = {stream_var: deffn(jnp.stack(fresh_vars))}
                    body = evaluate(body, intp=stream_intp)
                    new_streams = {
                        v.op: evaluate(plate_body, intp=stream_intp)
                        for v, vals in zip(fresh_vars, cart_vals, strict=False)
                    }
                    old_streams = {k: v for k, v in streams.items() if k != stream_var}
                    return ops.reduce(monoid, new_streams | old_streams, body)

        return fwd()
