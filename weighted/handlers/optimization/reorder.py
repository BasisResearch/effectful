from collections.abc import Sized
from functools import reduce

import effectful.handlers.numbers  # noqa: F401
import tree
from effectful.handlers.jax._handlers import is_eager_array
from effectful.handlers.numbers import mul
from effectful.ops.semantics import fvsof, fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Term
from scipy.cluster.hierarchy import DisjointSet

from weighted.handlers.optimization.utils import mul_op, parse_terms
from weighted.ops.fold import fold
from weighted.ops.semiring import (
    is_idempotent,
    scalar_mul,
)


class FoldFusion(ObjectInterpretation):
    """Implements the identity
        fold(R, S1, fold(R, S2, body)) = fold(R, S1 ∪ S2, body)

    This optimization fuses nested folds with the same semiring into a single fold
    over the product of their streams, which can be more efficient.
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        # Only proceed if body is a fold operation.
        if not (isinstance(body, Term) and body.op is fold):
            return fwd()

        # Extract the inner fold's parameters.
        inner_semiring, inner_streams, inner_body = body.args

        # Only fuse if both folds use the same semiring.
        if semiring != inner_semiring:
            return fwd()

        # Return the fused fold.
        return fold(semiring, streams | inner_streams, inner_body)


class FoldSplit(ObjectInterpretation):
    """Implements the identity
        fold(R, S, b1 + ... + bn) = fold(R, S, b1) + ... + fold(R, S, bn)

    By splitting a fold in several terms, this transforms
    allows for individual terms to be further optimized.
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        # Only proceed if body is a semiring addition.
        if not (isinstance(body, Term) and body.op is semiring.add):
            return fwd()

        # Create separate D terms for each addend.
        add_terms = parse_terms(body, semiring.add)
        new_terms = (fold(semiring, streams, t) for t in add_terms)

        # Apply fold to the new body.
        return reduce(semiring.add, new_terms, semiring.zero)


class FoldPropagateUnusedStreams(ObjectInterpretation):
    """
    Implements the identity
        fold(R, S × S', body) = |S'| ⋅ fold(R, S, body)
            where fvsof(body) ∩ S' = ∅
            and `⋅` is the scalar product of the semiring addition

    To be safe, streams that are open or used by dependent
    streams are never eliminated.
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        # A stream is redundant if it doesn't appear in the body or in another stream.
        stream_fvs = reduce(set.union, map(fvsof, streams.values()), set())
        fvs = fvsof(body) | stream_fvs
        redundant_streams = {k: v for k, v in streams.items() if k not in fvs}

        # Only proceed if there are redundant streams
        if len(redundant_streams) == 0:
            return fwd()

        if not is_idempotent(semiring.add):
            # make sure we can calculate the length of each stream
            has_size = lambda x: is_eager_array(x) or isinstance(x, Sized)
            redundant_streams = {
                k: v for k, v in redundant_streams.items() if has_size(v)
            }
            if len(redundant_streams) == 0:
                return fwd()

            constant = reduce(mul, map(len, redundant_streams.values()))
        else:
            constant = 1

        used_streams = {k: v for k, v in streams.items() if k not in redundant_streams}
        new_fold = fold(semiring, used_streams, body)
        return scalar_mul(semiring.add)(new_fold, constant)


class FoldReorderReduction(ObjectInterpretation):
    """
    Performs smarter reduction/multiplication ordering.
    Also known as pushing aggregates past joins (in probabilistic inference)
    or contraction ordering (in tensor networks).
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        semiring_mul = mul_op(semiring)
        if not (isinstance(body, Term) and body.op is semiring_mul):
            return fwd()

        terms = parse_terms(body, semiring_mul)

        # only proceed if the body multiplies multiple tensors
        if len(terms) < 2 or not all(is_eager_array(t) for t in terms):
            return fwd()

        term_vars = set.union(*[fvsof(t.args[1]) for t in terms])
        out_vars = term_vars - set(streams.keys())
        # todo: support dependent streams
        var_sizes = {str(k): len(v) for k, v in streams.items()}
        in_vars = [fvsof(term.args[1]) for term in terms]
        fold_path = _order_variables(var_sizes, in_vars, out_vars)

        if len(fold_path) == 0:
            return fwd()  # no actual folding happening

        for fold_positions in fold_path:
            fold_terms = [terms[i] for i in fold_positions]
            terms = [term for i, term in enumerate(terms) if i not in fold_positions]
            fold_out_vars, eliminated_vars = _find_contraction(
                fold_positions, in_vars, out_vars
            )
            in_vars = [x for i, x in enumerate(in_vars) if i not in fold_positions]
            in_vars.append(set(fold_out_vars))

            new_fold = reduce(semiring.mul, fold_terms)
            current_streams = {v: streams[v] for v in eliminated_vars}
            if len(current_streams) > 0:
                new_fold = fold(semiring, current_streams, new_fold)
            terms.append(new_fold)

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


class FoldFactorization(ObjectInterpretation):
    """
    Implements factorization of independent terms.
    For example, when having two independent distributions,
    we can rewrite their marginalization as:
        ∫p(x)⋅q(y)dxdy => ∫p(x)dx ⋅ ∫q(y)dy

    More specifically, in terms of folds we are performing:
        fold(R, (S₁ × ... × Sₖ ) , A₁ * ... * Aₖ)
        => fold(R, S₁, A₁) * ... * fold(R, Sₖ, Aₖ)
        where free(Aᵢ) ∩ free(Aⱼ) ∩ S = ∅
          and free(Aᵢ) ∩ Sᵢ ≠ ∅

    (The implementation is a little more general than this, as each
    independent component can have an arbitrary number of streams.)
    """

    @staticmethod
    def _separate_factors(factors, xs) -> list[set]:
        var_sets = DisjointSet(xs)
        for factor in factors:
            factor_vars = tuple(fvsof(factor) & xs)
            for v in factor_vars[1:]:
                var_sets.merge(factor_vars[0], v)
        return var_sets.subsets()

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not isinstance(body, Term):
            return fwd()

        # We assume all streams are used
        if len(set(streams.keys()) - fvsof(body)) > 0:
            return fwd()

        terms = parse_terms(body, mul_op(semiring))
        if len(terms) < 2:
            return fwd()

        stream_vars = set(tree.flatten(streams.keys()))
        partitions = self._separate_factors(terms, stream_vars)

        if len(partitions) < 2:
            return fwd()  # nothing to factorize :(

        new_folds = []
        for partition in partitions:
            partition_streams = {k: v for k, v in streams.items() if k in partition}
            partition_terms = [t for t in terms if len(fvsof(t) & partition) > 0]
            partition_term = reduce(semiring.mul, partition_terms, semiring.one)
            new_folds.append(fold(semiring, partition_streams, partition_term))

        return reduce(semiring.mul, new_folds)
