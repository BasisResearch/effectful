import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Term

from weighted.handlers.jax import D
from weighted.ops.fold import fold
from weighted.ops.semiring import ArgMaxAlg, ArgMinAlg, MaxAlg, MinAlg


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
                return -result[0], result[1]
            else:
                return fwd()
