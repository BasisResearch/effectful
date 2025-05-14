import functools

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numbers  # noqa: F401
import tree
from effectful.ops.semantics import coproduct, fvsof, fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Term

from weighted.handlers.jax import D
from weighted.ops.fold import fold
from weighted.ops.semiring import ArgMaxAlg, ArgMinAlg, LinAlg, MaxAlg, MinAlg


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
        return functools.reduce(lambda a, b: semiring.add(a, b), results, semiring.zero)


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
            return functools.reduce(semiring.add, new_terms, semiring.zero)

        else:
            return fwd()


class FoldFactorization(ObjectInterpretation):
    """Implements the identity: fold(R, S, A * B), free(A) ∩ S = {} => A * fold(R, S, B)

    This optimization factors out terms that don't depend on the fold variables,
    which can significantly reduce computation by avoiding redundant calculations.
    """

    def _mul_op(self, semiring):
        if semiring is LinAlg:
            return jnp.multiply
        elif semiring is MinAlg:
            return jnp.min
        elif semiring is MaxAlg:
            return jnp.max
        else:
            return None

    @staticmethod
    def _separate_factors(factors, stream_vars):
        indep_factors = []
        dep_factors = []
        for f in factors:
            if len(fvsof(f) & stream_vars) == 0:
                indep_factors.append(f)
            else:
                dep_factors.append(f)

        return indep_factors, dep_factors

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not isinstance(body, Term):
            return fwd()

        # Check if the body is a D term
        if body.op is D:
            # We only handle single-term bodies for now
            if len(body.args) != 1:
                return fwd()

            indices, value = body.args[0]

            # Check if value is a multiplication operation
            if not (isinstance(value, Term) and value.op is self._mul_op(semiring)):
                return fwd()

            indep_factors, dep_factors = FoldFactorization._separate_factors(
                value.args, set(tree.flatten(streams.keys()))
            )

            if indep_factors == []:
                return fwd()

            indep_prod = functools.reduce(semiring.mul, indep_factors, semiring.one)
            dep_prod = functools.reduce(semiring.mul, dep_factors, semiring.one)
            dep_result = fold(semiring, streams, D((indices, dep_prod)))
            return semiring.mul(indep_prod, dep_result)

        elif body.op is semiring.mul:
            indep_factors, dep_factors = FoldFactorization._separate_factors(
                body.args, set(tree.flatten(streams.keys()))
            )

            if indep_factors == []:
                return fwd()

            indep_prod = functools.reduce(semiring.mul, indep_factors, semiring.one)
            dep_prod = functools.reduce(semiring.mul, dep_factors, semiring.one)
            return semiring.mul(indep_prod, fold(semiring, streams, dep_prod))
        else:
            return fwd()


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


interpretation = functools.reduce(
    coproduct,  # type: ignore
    [
        FoldFusion(),
        FoldIndexDistributivity(),
        FoldAddDistributivity(),
        FoldFactorization(),
        FoldZero(),
    ],
)
