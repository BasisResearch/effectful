import logging

import jax
import jax.tree as tree
import optax
from numpyro.distributions import Distribution

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax.monoid import Max, Min, Sum
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.semantics import evaluate, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Expr, Interpretation, Operation, Term
from effectful.ops.weighted.distribution import D
from effectful.ops.weighted.jax import key, reals
from effectful.ops.weighted.monoid import ArgMin, Monoid

logger = logging.getLogger(__name__)


def timed(f=None, name=None):
    from datetime import datetime, timedelta
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            start = datetime.now()
            result = func(*args, **kwds)
            elapsed = datetime.now() - start
            if elapsed > timedelta(seconds=0.1):
                print(
                    f"{func.__name__ if name is None else name} took {elapsed} time to finish"
                )
            return result

        return wrapper

    # This handles the case when called with arguments: @timed(name="something")
    if f is None:
        return decorator
    # This handles the case when called without arguments: @timed
    return decorator(f)


def _parse_body(body) -> list[tuple[tuple[Operation, ...], Expr[jax.Array]]]:
    if isinstance(body, Term) and body.op is D:
        kvs = []
        for arg in body.args:
            if not isinstance(arg, tuple) or len(arg) != 2:
                raise ValueError("Expected a tuple of (key, value)")
            k, v = arg
            if not isinstance(k, tuple):
                k = (k,)
            kvs.append((k, v))
        return kvs
    return [((), body)]


class GradientOptimizationReduce(ObjectInterpretation):
    """Handle min/argmin over reals using gradient descent.

    Notes:
    - A single empty output index is expected. Nontrivial output indexes would in
    principle allow us to represent partial optimization problems like the following:
    reduce(MinMonoid, {x: reals(), y: reals()}, {x(): f(x(), y())}) = \\lambda x. min_{y\\in R} f(x, y).

    """

    def __init__(
        self, optimizer=optax.adam, steps=1000, init=None, progress=False, **kwargs
    ):
        self.optimizer = optimizer
        self.optimizer_kwargs = kwargs
        if steps <= 0:
            raise ValueError("Expected a positive number of steps")
        self.steps = steps
        self.init = {} if init is None else init
        self.progress = progress

    @implements(Monoid.reduce)
    def reduce(self, monoid, streams, body):
        # TODO: handle mixed discrete/continuous optimization
        if not (
            monoid in (Min, ArgMin)
            and all(isinstance(v, Term) and v.op is reals for v in streams.values())
        ):
            return fwd()

        body_indices = _parse_body(body)
        if len(body_indices) <= 0:
            return jnp.array([])

        if len(body_indices) > 1:
            # TODO: handle multiple output indices
            return fwd()

        indices, value = body_indices[0]
        if indices != ():
            # TODO: handle indexed outputs
            return fwd()

        if monoid is ArgMin:
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("Expected a tuple of (value, arg) for ArgMinMonoid")
            value, arg = value

        # Initialize parameters using provided init values or zeros
        param_keys = list(streams.keys())
        param_values = []
        for k in param_keys:
            r = streams[k]
            if r.op is not reals:
                raise ValueError("Expected reals as stream values")
            shape = r.args[0] if len(r.args) > 0 else ()

            if k in self.init:
                # Use provided initialization
                init_value = self.init[k]
                if not isinstance(init_value, jax.Array):
                    init_value = jnp.array(init_value)

                # Ensure the shape matches
                if init_value.shape != shape and shape != ():
                    raise ValueError(
                        f"Init shape mismatch for {k}: expected {shape}, got {init_value.shape}"
                    )
                param = init_value
            else:
                # Default to zeros
                param = jnp.zeros(shape)
            param_values.append(param)

        loss = deffn(value, key, *param_keys)

        # we must be able to fully evaluate the loss function
        loss_value = loss(key(), *param_values)
        if not isinstance(loss_value, jax.Array):
            raise ValueError(f"Loss must evaluate to an array, but got {loss_value}")

        loss_grad = jax.jit(jax.grad(loss, argnums=range(1, len(param_values) + 1)))

        optimizer = self.optimizer(**self.optimizer_kwargs)
        opt_state = optimizer.init(param_values)

        steps_iter = range(self.steps)
        if self.progress:
            from tqdm import tqdm

            steps_iter = tqdm(steps_iter)

        keys = jax.random.split(key(), self.steps)
        for i in steps_iter:
            grads = list(loss_grad(keys[i], *param_values))
            assert all(isinstance(g, jax.Array) for g in tree.flatten(grads)[0])
            updates, opt_state = optimizer.update(grads, opt_state)
            param_values = optax.apply_updates(param_values, updates)

        final_loss = loss(key, *param_values)

        if monoid is Min:
            return final_loss

        with handler(
            {v: deffn(p) for (v, p) in zip(param_keys, param_values, strict=True)}
        ):
            final_arg = evaluate(arg)

        return final_loss, final_arg


class LikelihoodWeightingReduce(ObjectInterpretation):
    """Handle expectation computation using likelihood weighting."""

    def __init__(self, samples=1):
        self.samples = samples

    @implements(Monoid.reduce)
    def reduce(self, monoid, streams, body):
        if not (
            monoid is Sum
            and all(issubclass(typeof(v), Distribution) for v in streams.values())
        ):
            reason = "monoid" if monoid is not Sum else "streams"
            logger.debug(
                f"Skipping likelihood weighting (reason {reason}): reduce({monoid}, {streams}, {body}"
            )
            return fwd()

        sample_streams = {}
        index_streams = {}
        for k, v in streams.items():
            s = defop(jax.Array, name="sample")

            if isinstance(k, Operation):
                value = k
                samples = v.sample(key(), (self.samples,))[s()]
                sample_streams[value] = deffn(samples)
            elif isinstance(k, tuple):
                if not (len(k) == 2 and all(isinstance(i, Operation) for i in k)):
                    raise ValueError(
                        "Expected a tuple of (value, weight) for likelihood weighting"
                    )
                (value, weight) = k
                samples = v.sample(key(), (self.samples,))
                weights = v.log_prob(samples)
                weights = weights - logsumexp(weights)
                samples = jax_getitem(samples, [s()])
                weights = jax_getitem(weights, [s()])
                sample_streams[value] = deffn(samples)
                sample_streams[weight] = deffn(weights)
            else:
                raise ValueError("Unexpected key type")

            index_streams[s] = jnp.arange(self.samples)

        with handler(sample_streams):
            body = evaluate(body)

        return Sum.reduce(index_streams, body)


class PytreeMapReduce(ObjectInterpretation):
    """Map a reduce over a pytree body."""

    @implements(Monoid.reduce)
    def reduce(self, monoid, streams, body):
        if not (monoid in (Min, Max, Sum)):
            return fwd()

        body_indices = _parse_body(body)

        # Check that all values in the body have the same structure
        structure = None
        for _, t in body_indices:
            s = jax.tree.structure(t)
            if structure and structure != s:
                raise ValueError(
                    f"Found pytrees with different structures {structure} and {s} in reduce body."
                )
            structure = s

        # Do no work for trivial structures
        if structure == jax.tree.structure(0):
            return fwd()

        flat_bodies = [jax.tree.flatten(t)[0] for (_, t) in body_indices]

        flat_body = [
            monoid.reduce(
                streams,
                D(*[(k, v) for (k, _), v in zip(body_indices, vs, strict=True)]),
            )
            for vs in zip(*flat_bodies, strict=True)
        ]

        return jax.tree.unflatten(structure, flat_body)


interpretation: Interpretation = {}
