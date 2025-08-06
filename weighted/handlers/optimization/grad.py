import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Term

from weighted.ops.distribution import D
from weighted.ops.fold import fold
from weighted.ops.monoid import ArgMaxMonoid, ArgMinMonoid, MaxMonoid, MinMonoid


class FlipOptimizationFold(ObjectInterpretation):
    """Convert Max/ArgMax problems to Min/ArgMin by negating values.

    This handler transforms maximization problems into minimization problems
    by negating the objective function, allowing reuse of minimization algorithms.
    """

    @implements(fold)
    def fold(self, monoid, streams, body, **kwargs):
        # Only handle MaxMonoid and ArgMaxMonoid
        if monoid not in (MaxMonoid, ArgMaxMonoid):
            return fwd()

        # Determine the target monoid (Min for Max, ArgMin for ArgMax)
        target_monoid = MinMonoid if monoid is MaxMonoid else ArgMinMonoid

        # Normalize the body to use D if it's not already
        if not (isinstance(body, Term) and body.op is D):
            # For ArgMaxMonoid, body should be a tuple of (value, arg)
            if monoid is ArgMaxMonoid:
                if not isinstance(body, tuple) or len(body) != 2:
                    return fwd()
                body = D(((), body))
            else:
                body = D(((), body))

        # For each key-value pair in the body
        new_args = []
        for indices, value in body.args:
            if monoid is MaxMonoid:
                # For MaxMonoid, just negate the value
                new_value = -value
            else:  # ArgMaxMonoid
                # For ArgMaxMonoid, negate the first element of the tuple (the value)
                # but keep the second element (the arg) unchanged
                if not (isinstance(value, tuple) and len(value) == 2):
                    raise ValueError("Expected a tuple of (value, arg) for ArgMaxMonoid")
                val, arg = value
                new_value = (-val, arg)

            new_args.append((indices, new_value))

        # Create a new body with negated values
        new_body = D(*new_args)

        # Solve as a minimization problem
        result = fold(target_monoid, streams, new_body, **kwargs)

        # For MaxMonoid, negate the result back
        if monoid is MaxMonoid:
            if isinstance(result, dict):
                return {k: -v for k, v in result.items()}
            else:
                return -result
        else:  # ArgMaxMonoid
            # For ArgMaxMonoid, negate the first element of the result tuple back
            if isinstance(result, dict):
                return {k: (-v[0], v[1]) for k, v in result.items()}
            elif isinstance(result, tuple):
                return -result[0], result[1]
            else:
                return fwd()
