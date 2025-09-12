import itertools
from functools import reduce

import jax
import numpy as np
from effectful.handlers.jax import numpy as jnp
from effectful.handlers.jax._handlers import is_eager_array
from effectful.ops.semantics import evaluate, fvsof, fwd
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Term

import weighted.ops.fold as ops
from weighted.handlers.jax import syntactic_eq_jax
from weighted.ops.fold import order_streams
from weighted.ops.monoid import JaxCartesianProdMonoid


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
        if k1.__name__ != k2.__name__ or not syntactic_eq_jax(v1, v2):
            return None
    return unifier


class FoldDistributeCartesianProduct(ObjectInterpretation):
    """Eliminates a fold over a cartesian product.
        ∑_x₁ ∑_x₂ ... ∑_xₙ f(x₁) ⋅ f(x₂) ... f(xₙ) = ∏_i ∑_xᵢ f(xᵢ)
    This transform is also called inversion in the lifting
    literature (e.g. [1]).

    More specifically, this transform implements the identity
    fold(⨁, {v: fold(×, S1, body1)} ∪ S2, fold(⨂, S1, body2))
        = fold(⨁, S2, fold(⨂, S1, fold(⨁, {v: repeat(body1)}, body2)))
            where × is the cartesian product and ⨂ distributes over ⨁.

    Note: This could be generalized to grouped inversion [2].

    [1] Braz, Rd, Eyal Amir, and Dan Roth. "Lifted first-order
    probabilistic inference." IJCAI. 2005.
    [2] Taghipour, Nima, et al. "Completeness results for lifted
    variable elimination." AISTATS. 2013.
    """

    @implements(ops.fold)
    def fold(self, monoid1, streams, body):
        match body:
            case Term(ops.fold, (monoid2, streams1, body2)):
                if not monoid1.distributes_with(monoid2):
                    return fwd()
                if not all(is_eager_array(v) for v in streams1.values()):
                    return fwd()

                for k, v in streams.items():
                    match v:
                        case Term(ops.fold, (_monoid, _streams1, body1)) if (
                            _monoid == JaxCartesianProdMonoid
                        ):
                            stream_unifier = unify_streams(streams1, _streams1)
                            if stream_unifier is None:
                                continue
                            if any(k in fvsof(v) for v in _streams1.values()):
                                continue  # another streams depends on the cartesian product stream

                            # in case cartesian stream depends on other streams
                            body1 = evaluate(body1, intp=stream_unifier)

                            size = reduce(lambda x, y: x * y, map(len, streams1.values()))
                            cartesian_stream = {k: jnp.tile(body1, (size, 1)).T}
                            streams2 = {k2: v2 for k2, v2 in streams.items() if k != k2}

                            inner_sum_fold = ops.fold(monoid1, cartesian_stream, body2)
                            fold = ops.fold(monoid2, streams1, inner_sum_fold)
                            if len(streams2) == 0:
                                return fold
                            return ops.fold(monoid1, streams2, fold)
        return fwd()


class SplitCartesianProductFold(ObjectInterpretation):
    """
    Transforms a fold over a cartesian product into
    a fold over separate streams. For a single cartesian
    product, this roughly corresponds to
        fold(R, {v: fold(×, {i: S}, body1)}, body2)
        = fold(R, {v₁: body1/{i↦S[0]}, ..., vₙ: body1/{i↦S[n]}}, body2/{v↦(v₁, ..., vₙ)})

        (where expr/{x↦c} substitutes x with c in expr)

    This is useful because, as opposed to a single cartesian
    product stream, the individual streams can be rearranged.
    """

    @implements(ops.fold)
    def fold(self, monoid, streams, body):
        for stream_var, stream_arr in streams.items():
            match stream_arr:
                case Term(ops.fold, (_monoid, cart_streams, plate_body)) if (
                    _monoid is JaxCartesianProdMonoid
                ):
                    if not all(
                        isinstance(stream, jax.Array) for stream in cart_streams.values()
                    ):
                        continue  # cartesian streams need to be ground

                    cart_vals = tuple(itertools.product(*cart_streams.values()))
                    fresh_vars = np.array(tuple(defop(stream_var)() for _ in cart_vals))
                    stream_intp = {stream_var: deffn(fresh_vars)}
                    body = evaluate(body, intp=stream_intp)
                    new_streams = {
                        v.op: evaluate(plate_body, intp=stream_intp)
                        for v, vals in zip(fresh_vars, cart_vals, strict=False)
                    }
                    old_streams = {k: v for k, v in streams.items() if k != stream_var}
                    return ops.fold(monoid, new_streams | old_streams, body)

        return fwd()
