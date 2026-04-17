from effectful.handlers.jax.monoid import Max, Min
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Term
from effectful.ops.weighted.distribution import D
from effectful.ops.weighted.monoid import ArgMax, ArgMin, Monoid


class FlipOptimizationReduce(ObjectInterpretation):
    """Convert Max/ArgMax problems to Min/ArgMin by negating values.

    This handler transforms maximization problems into minimization problems
    by negating the objective function, allowing reuse of minimization algorithms.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, streams, body, **kwargs):
        # Only handle Max and ArgMax
        if monoid not in (Max, ArgMax):
            return fwd()

        # Determine the target monoid (Min for Max, ArgMin for ArgMax)
        target_monoid = Min if monoid is Max else ArgMin

        # Normalize the body to use D if it's not already
        if not (isinstance(body, Term) and body.op is D):
            # For ArgMax, body should be a tuple of (value, arg)
            if monoid is ArgMax:
                if not isinstance(body, tuple) or len(body) != 2:
                    return fwd()
                body = D(((), body))
            else:
                body = D(((), body))

        # For each key-value pair in the body
        new_args = []
        for indices, value in body.args:
            if monoid is Max:
                # For Max, just negate the value
                new_value = -value
            else:  # ArgMax
                # For ArgMax, negate the first element of the tuple (the value)
                # but keep the second element (the arg) unchanged
                if not (isinstance(value, tuple) and len(value) == 2):
                    raise ValueError("Expected a tuple of (value, arg) for ArgMax")
                val, arg = value
                new_value = (-val, arg)

            new_args.append((indices, new_value))

        # Create a new body with negated values
        new_body = D(*new_args)

        # Solve as a minimization problem
        result = target_monoid.reduce(streams, new_body, **kwargs)

        # For Max, negate the result back
        if monoid is Max:
            if isinstance(result, dict):
                return {k: -v for k, v in result.items()}
            else:
                return -result
        else:  # ArgMax
            # For ArgMax, negate the first element of the result tuple back
            if isinstance(result, dict):
                return {k: (-v[0], v[1]) for k, v in result.items()}
            elif isinstance(result, tuple):
                return -result[0], result[1]
            else:
                return fwd()
