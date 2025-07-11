from functools import reduce

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numbers  # noqa: F401
import tree
from effectful.handlers.jax import bind_dims, jax_getitem
from effectful.ops.semantics import coproduct, fvsof, fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Term
from scipy.cluster.hierarchy import DisjointSet

from weighted.handlers.jax import D
from weighted.ops.fold import fold
from weighted.ops.semiring import ArgMaxAlg, ArgMinAlg, LinAlg, LogAlg, MaxAlg, MinAlg


class FoldReorderReduction(ObjectInterpretation):
    """
    Performs smarter reduction/multiplication ordering.
    Also known as pushing aggregates past joins (in probabilistic inference)
    or contraction ordering (in tensor networks).
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        # Only succeeds if body is a term of tensor multiplications.
        terms = _parse_mul_terms(body, semiring)

        # only proceed if the body multiplies multiple tensors
        if len(terms) < 2 or any(t.op != jax_getitem for t in terms):
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


def _parse_D_term(term):
    if not (isinstance(term, Term) and term.op is D):
        return fwd()

    args = term.args
    # only len 1 supported for now
    if len(args) != 1:
        return fwd()

    indices, body = args[0]
    return fvsof(indices), body


def _parse_mul_terms(value: Term, semiring):
    if value.op is _mul_op(semiring):
        return sum((_parse_mul_terms(arg, semiring) for arg in value.args), [])
    else:
        return [value]


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


class FoldFusion(ObjectInterpretation):
    """Implements the identity: fold(R, S1, fold(R, S2, body)) = fold(R, S1 x S2, body)

    This optimization fuses nested folds with the same semiring into a single fold
    over the product of their streams, which can be more efficient.
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        # Only proceed if body is a fold operation
        if not (isinstance(body, Term) and body.op is fold):
            return fwd()

        # Extract the inner fold's parameters
        inner_semiring, inner_streams, inner_body = body.args

        # Only fuse if both folds use the same semiring
        if semiring != inner_semiring:
            return fwd()

        # Return the fused fold
        return fold(semiring, streams | inner_streams, inner_body)


class FoldIndexDistributivity(ObjectInterpretation):
    """Implements the identity: fold(R, S, D((I1, X1), ..., (IN, XN))) = fold(R, S, D((I1, X1))) R.+ ... R.+ fold(R, S, D((IN, XN)))"""

    @implements(fold)
    def fold(self, semiring, streams, body):
        # Check if the body is a D term with multiple arguments (representing addition)
        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        # If there's only 0 or 1 argument, no distribution needed
        if len(body.args) <= 1:
            return fwd()

        # Create separate fold operations for each term
        results = []
        for indices, value in body.args:
            # Create a new D term with just this key-value pair
            term_body = D((indices, value))
            # Compute fold for this term
            term_result = fold(semiring, streams, term_body)
            results.append(term_result)

        # Combine results using semiring addition
        return reduce(lambda a, b: semiring.add(a, b), results, semiring.zero)


class FoldEliminateDterm(ObjectInterpretation):
    """Eliminates D-terms from a fold."""

    @implements(fold)
    def fold(self, semiring, streams, body):
        # Check if the body is a D term with one argument
        if not (isinstance(body, Term) and body.op is D and len(body.args) == 1):
            return fwd()

        indices, body = body.args[0]
        indices = [ix.op for ix in indices]
        if not all(ix in streams for ix in indices):
            return fwd()

        fresh_indices = [defop(ix, name=f"fresh_{ix}") for ix in indices]

        fresh_streams = {
            ix: jnp.expand_dims(jax_getitem(streams[ix], (fresh_ix(),)), -1)
            for ix, fresh_ix in zip(indices, fresh_indices, strict=False)
        }
        new_fold = fold(semiring, streams | fresh_streams, body)
        return bind_dims(new_fold, *fresh_indices)


class FoldAddDistributivity(ObjectInterpretation):
    """Implements the identity: fold(R, S, D((I, X1 R.+ ... R.+ XN))) = fold(R, S, D((I1, X1), ..., (IN, XN)))

    This optimization distributes fold over addition within a single index, allowing
    for parallel computation of individual terms.
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not isinstance(body, Term):
            return fwd()

        # Check if the body is a D term with a single argument
        if body.op is D and len(body.args) == 1:
            indices, value = body.args[0]

            if not (isinstance(value, Term) and value.op is semiring.add):
                return fwd()

            terms = value.args

            # Create separate D terms for each addend
            new_terms = []
            for term in terms:
                new_terms.append((indices, term))

                # Create a new body with separate terms
                new_body = D(*new_terms)

            # Apply fold to the new body
            return fold(semiring, streams, new_body)

        elif body.op is semiring.add:
            # Create separate D terms for each addend
            new_terms = []
            for term in body.args:
                new_terms.append(fold(semiring, streams, term))

            # Apply fold to the new body
            return reduce(semiring.add, new_terms, semiring.zero)

        else:
            return fwd()


def _mul_op(semiring):
    if semiring is LinAlg:
        return jnp.multiply
    elif semiring is MinAlg:
        return jnp.min
    elif semiring is MaxAlg:
        return jnp.max
    elif semiring is LogAlg:
        return jnp.add
    else:
        return None


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
    def _separate_factors(factors, xs) -> list[set[str]]:
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

        terms = _parse_mul_terms(body, semiring)
        if len(terms) < 2:
            return fwd()

        stream_vars = set(tree.flatten(streams.keys()))
        partitions = self._separate_factors(terms, stream_vars)

        if len(partitions) < 2:
            return fwd()  # nothing to factorize :(

        # We assume all streams are used
        if not all(k in set.union(*partitions) for k in stream_vars):
            return fwd()

        new_folds = []
        for partition in partitions:
            partition_streams = {k: v for k, v in streams.items() if k in partition}
            partition_terms = [t for t in terms if len(fvsof(t) & partition) > 0]
            partition_term = reduce(semiring.mul, partition_terms, semiring.one)
            new_folds.append(fold(semiring, partition_streams, partition_term))

        return reduce(semiring.mul, new_folds)


class FlipOptimizationFold(ObjectInterpretation):
    """Convert Max/ArgMax problems to Min/ArgMin by negating values.

    This handler transforms maximization problems into minimization problems
    by negating the objective function, allowing reuse of minimization algorithms.
    """

    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        # Only handle MaxAlg and ArgMaxAlg
        if semiring not in (MaxAlg, ArgMaxAlg):
            return fwd()

        # Determine the target semiring (Min for Max, ArgMin for ArgMax)
        target_semiring = MinAlg if semiring is MaxAlg else ArgMinAlg

        # Normalize the body to use D if it's not already
        if not (isinstance(body, Term) and body.op is D):
            # For ArgMaxAlg, body should be a tuple of (value, arg)
            if semiring is ArgMaxAlg:
                if not isinstance(body, tuple) or len(body) != 2:
                    return fwd()
                body = D(((), body))
            else:
                body = D(((), body))

        # For each key-value pair in the body
        new_args = []
        for indices, value in body.args:
            if semiring is MaxAlg:
                # For MaxAlg, just negate the value
                new_value = -value
            else:  # ArgMaxAlg
                # For ArgMaxAlg, negate the first element of the tuple (the value)
                # but keep the second element (the arg) unchanged
                if not (isinstance(value, tuple) and len(value) == 2):
                    raise ValueError("Expected a tuple of (value, arg) for ArgMaxAlg")
                val, arg = value
                new_value = (-val, arg)

            new_args.append((indices, new_value))

        # Create a new body with negated values
        new_body = D(*new_args)

        # Solve as a minimization problem
        result = fold(target_semiring, streams, new_body, **kwargs)

        # For MaxAlg, negate the result back
        if semiring is MaxAlg:
            if isinstance(result, dict):
                return {k: -v for k, v in result.items()}
            else:
                return -result
        else:  # ArgMaxAlg
            # For ArgMaxAlg, negate the first element of the result tuple back
            if isinstance(result, dict):
                return {k: (-v[0], v[1]) for k, v in result.items()}
            elif isinstance(result, tuple):
                return (-result[0], result[1])
            else:
                fwd()


class FoldZero(ObjectInterpretation):
    @implements(fold)
    def fold(self, semiring, streams, body):
        if (
            isinstance(body, Term)
            and body.op is D
            and all(v == semiring.zero for (_, v) in body.args)
        ):
            return semiring.zero
        if body == semiring.zero:
            return semiring.zero
        return fwd()


interpretation = reduce(
    coproduct,  # type: ignore
    [
        FoldFusion(),
        FoldIndexDistributivity(),
        FoldAddDistributivity(),
        FoldFactorization(),
    ],
)
