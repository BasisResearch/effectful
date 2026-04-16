import functools

from scipy.cluster.hierarchy import DisjointSet

import effectful.handlers.jax.numpy as jnp
import effectful.ops.weighted.reduce as ops
from effectful.handlers.jax._handlers import is_eager_array
from effectful.handlers.weighted.optimization.utils import (
    parse_terms,
    parse_with_op,
    partition_streams,
)
from effectful.ops.semantics import fvsof, fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Term


class ReduceNoStreams(ObjectInterpretation):
    """Implements the identity
    reduce(R, ∅, body) = 0
    """

    @implements(ops.reduce)
    def reduce(self, monoid, streams, _):
        if len(streams) == 0:
            return monoid.zero
        return fwd()


class ReduceFusion(ObjectInterpretation):
    """Implements the identity
        reduce(R, S1, reduce(R, S2, body)) = reduce(R, S1 ∪ S2, body)

    This optimization fuses nested reduces with the same monoid into a single reduce
    over the product of their streams, which can be more efficient.
    """

    @implements(ops.reduce)
    def reduce(self, monoid, streams, body):
        match body:
            case Term(ops.reduce, (inner_monoid, inner_streams, inner_body)):
                if monoid == inner_monoid:
                    return ops.reduce(monoid, streams | inner_streams, inner_body)
        return fwd()


class ReduceSplit(ObjectInterpretation):
    """Implements the identity
        reduce(R, S, b1 + ... + bn) = reduce(R, S, b1) + ... + reduce(R, S, bn)

    By splitting a reduce in several terms, this transforms
    allows for individual terms to be further optimized.
    """

    @implements(ops.reduce)
    def reduce(self, monoid, streams, body):
        match body:
            case Term(monoid.add, _):
                add_terms = parse_with_op(body, monoid.add)
                new_terms = (ops.reduce(monoid, streams, t) for t in add_terms)
                return functools.reduce(monoid, new_terms, monoid.zero)
        return fwd()


class ReduceDistributeTerm(ObjectInterpretation):
    """Implements the identity
        reduce(⊕, S1 × S2, b1 ⊗ b2) = reduce(⊕, S1, b1 ⊗ reduce(⊕, S2, b2))
            where fvsof(S2) ∩ fvsof(b1) = ∅

    Note: this in more general than ReduceReorderReduction in that it
     supports free/unevaluated streams as well as dependent streams,
     but will give worse orderings (unless the einsum is hierarchical,
     which is relevant for lifting).
    """

    @implements(ops.reduce)
    def reduce(self, monoid, streams, body):
        mul, terms = parse_terms(body, monoid)
        if len(terms) < 2 or len(streams) < 2:
            return fwd()

        stream_vars = set(streams.keys())
        stream_fvs = functools.reduce(set.union, map(fvsof, streams.values()), set())
        independent_streams = stream_vars - stream_fvs
        if not independent_streams:
            return fwd()

        for term in terms:
            free_streams = independent_streams - fvsof(term)
            if len(free_streams):
                unused_streams, used_streams = partition_streams(streams, free_streams)
                t_free_fvs = [len(fvsof(t) & free_streams) > 0 for t in terms]

                inner_terms = [t for t, b in zip(terms, t_free_fvs, strict=False) if b]
                outer_terms = [
                    t for t, b in zip(terms, t_free_fvs, strict=False) if not b
                ]
                inner_body = functools.reduce(mul, inner_terms)
                inner_reduce = ops.reduce(monoid, unused_streams, inner_body)
                outer_term = functools.reduce(mul, outer_terms)
                return ops.reduce(monoid, used_streams, mul(outer_term, inner_reduce))

        return fwd()


class ReducePropagateUnusedStreams(ObjectInterpretation):
    """
    Implements the identity
        reduce(R, S × S', body) = |S'| ⋅ reduce(R, S, body)
            where fvsof(body) ∩ S' = ∅
            and `⋅` is the scalar product of the monoid addition

    To be safe, streams that are open or used by dependent
    streams are never eliminated.
    """

    @implements(ops.reduce)
    def reduce(self, monoid, streams, body):
        # A stream is redundant if it doesn't appear in the body or in another stream.
        stream_fvs = functools.reduce(set.union, map(fvsof, streams.values()), set())
        fvs = fvsof(body) | stream_fvs
        redundant_streams = {k: v for k, v in streams.items() if k not in fvs}

        constant = 1
        if not monoid.is_idempotent():
            sized_redundant_streams = {}

            for k, v in redundant_streams.items():
                try:
                    stream_len = len(v)
                except TypeError:
                    continue

                sized_redundant_streams[k] = v
                constant *= stream_len

            redundant_streams = sized_redundant_streams

        if len(redundant_streams) == 0:
            return fwd()

        used_streams = {k: v for k, v in streams.items() if k not in redundant_streams}
        new_reduce = ops.reduce(monoid, used_streams, body)
        return monoid.scalar_mul()(new_reduce, jnp.array(constant))


class ReduceReorderReduction(ObjectInterpretation):
    """
    Performs smarter reduction/multiplication ordering.
    Also known as variable elimination (in probabilistic inference)
    or contraction ordering (in tensor networks).
    """

    @implements(ops.reduce)
    def reduce(self, monoid, streams, body):
        mul, terms = parse_terms(body, monoid)

        # Only proceed when reduceing over multiple streams & tensors
        if len(terms) < 2 or len(streams) < 2:
            return fwd()
        if any(not is_eager_array(t) for t in terms):
            return fwd()

        term_vars = set.union(*map(fvsof, terms))
        out_vars = term_vars - set(streams.keys())
        # todo: support dependent streams
        var_sizes = {str(k): len(v) for k, v in streams.items()}
        in_vars = [fvsof(term.args[1]) for term in terms]
        reduce_path = _order_variables(var_sizes, in_vars, out_vars)

        for reduce_positions in reduce_path:
            reduce_terms = [terms[i] for i in reduce_positions]
            terms = [term for i, term in enumerate(terms) if i not in reduce_positions]
            reduce_out_vars, eliminated_vars = _find_contraction(
                reduce_positions, in_vars, out_vars
            )
            in_vars = [x for i, x in enumerate(in_vars) if i not in reduce_positions]
            in_vars.append(set(reduce_out_vars))

            new_body = functools.reduce(mul, reduce_terms)
            current_streams = {v: streams[v] for v in eliminated_vars}
            new_reduce = ops.reduce(monoid, current_streams, new_body)
            terms.append(new_reduce)

        assert len(terms) == 1
        return terms[0]


def _order_variables(var_sizes: dict, in_vars, out_vars, memory_limit=5000):
    """
    Greedy contraction ordering, using the numpy einsum implementation.
    Note: although this is greedy, it still scales O(n^3) in the size of terms.
    """
    from numpy._core.einsumfunc import _greedy_path  # type: ignore

    out_vars = set(map(str, out_vars))
    in_vars = [set(map(str, x)) for x in in_vars]
    var_sizes |= dict.fromkeys(out_vars, 1)  # todo: fix
    path = _greedy_path(in_vars, out_vars, var_sizes, memory_limit)
    return path


def _find_contraction(positions, in_vars, out_vars):
    """
    Given a sequence of positions to contract, return
    the variables of the newly created tensor, and what variables
    have been eliminated.
    """
    idx_contract = set()
    idx_remain = out_vars.copy()
    for ind, value in enumerate(in_vars):
        if ind in positions:
            idx_contract |= value
        else:
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = idx_contract - new_result

    return tuple(new_result), tuple(idx_removed)


class ReduceFactorization(ObjectInterpretation):
    """
    Implements factorization of independent terms.
    For example, when having two independent distributions,
    we can rewrite their marginalization as:
        ∫p(x)⋅q(y)dxdy => ∫p(x)dx ⋅ ∫q(y)dy

    More specifically, in terms of reduces we are performing:
        reduce(R, (S₁ × ... × Sₖ ) , A₁ * ... * Aₖ)
        => reduce(R, S₁, A₁) * ... * reduce(R, Sₖ, Aₖ)
        where free(Aᵢ) ∩ free(Aⱼ) ∩ S = ∅
          and free(Aᵢ) ∩ Sᵢ ≠ ∅

    (The implementation is a little more general than this, as each
    independent component can have an arbitrary number of streams.)
    """

    @staticmethod
    def _separate_factors(factors, vs) -> list[set]:
        var_sets = DisjointSet(vs)
        for factor in factors:
            factor_vars = tuple(fvsof(factor) & vs)
            for v in factor_vars[1:]:
                var_sets.merge(factor_vars[0], v)
        return var_sets.subsets()

    @implements(ops.reduce)
    def reduce(self, monoid, streams, body):
        stream_vars = fvsof(streams.keys())
        # We assume all streams are used
        if len(stream_vars - fvsof(body)) > 0:
            return fwd()

        mul, terms = parse_terms(body, monoid)
        partitions = self._separate_factors(terms, stream_vars)

        if len(partitions) < 2:
            return fwd()  # nothing to factorize :(

        new_reduces = []
        for partition in partitions:
            partition_streams = {k: v for k, v in streams.items() if k in partition}
            partition_terms = (t for t in terms if len(fvsof(t) & partition) > 0)
            partition_term = functools.reduce(mul, partition_terms)
            new_reduces.append(ops.reduce(monoid, partition_streams, partition_term))

        return functools.reduce(mul, new_reduces)
