import collections.abc
import functools
import logging
import operator
from typing import Iterable

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numbers  # noqa: F401
import jax
import optax
import tree
from effectful.handlers.jax import bind_dims, jax_getitem, sizesof
from effectful.handlers.jax._handlers import _register_jax_op
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.semantics import (
    coproduct,
    evaluate,
    fvsof,
    fwd,
    handler,
    typeof,
)
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation, Term
from numpyro.distributions import Distribution

from weighted.handlers.optimization import NormalizeValueFold
from weighted.ops.fold import D, fold
from weighted.ops.semiring import ArgMaxAlg, ArgMinAlg, LinAlg, MaxAlg, MinAlg

logger = logging.getLogger(__name__)


@defop
def key() -> jax.Array:
    return jax.random.key(0)


@defop
def reals(*, shape: tuple[int, ...] = ()) -> Iterable[jax.Array]:
    raise NotImplementedError


@defop
def sample(key: jax.Array, d: Distribution, sample_shape: tuple[int]) -> jax.Array:
    if not (
        isinstance(d, Distribution)
        and (not isinstance(d, Term) or all(isinstance(a, jax.Array) for a in d.args))
    ):
        raise NotImplementedError
    return d.sample(key, sample_shape=sample_shape)


@defop
def rsample(key, d: Distribution, sample_shape: tuple[int]) -> jax.Array:
    if not (isinstance(d, Distribution) and isinstance(sample_shape, tuple)):
        raise NotImplementedError
    return d.rsample(key, sample_shape=sample_shape)


@defop
def log_prob(d: Distribution, value: jax.Array) -> jax.Array:  # todo
    if not isinstance(d, Distribution):
        raise NotImplementedError
    return _register_jax_op(d.log_prob)(value)


class DenseTensorArgFold(ObjectInterpretation):
    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        if not (
            semiring in (ArgMinAlg, ArgMaxAlg)
            and all(isinstance(s, collections.abc.Sized) for s in streams.values())
            and all(typeof(k()) is jax.Array for k in streams.keys())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) <= 0:
            return jnp.array([])

        if len(body.args) > 1:
            # todo: handle multiple output indices
            return fwd()

        indices, value = body.args[0]
        if not isinstance(value, tuple) and len(value) == 2:
            raise ValueError("Expected a tuple of (value, arg) for ArgMinAlg")
        min_value, argmin_value = value

        # Check that the output is indexed in a subset of the input indices, and
        # that there are no index transformations
        if not all(isinstance(i, Term) and i.op in streams for i in indices):
            return fwd()
        indices = [i.op for i in indices]

        old_to_fresh = {k: defop(k) for k in streams.keys()}
        fresh_to_old = {v: k for (k, v) in old_to_fresh.items()}
        indexed_streams = {
            k: deffn(jax_getitem(v, [old_to_fresh[k]()])) for k, v in streams.items()
        }
        with handler(indexed_streams):
            result = evaluate(min_value)

        result_indices = sizesof(result)
        reduction_indices = [i for i in result_indices if fresh_to_old[i] not in indices]

        result = bind_dims(result, *reduction_indices)

        # Flatten the leading len(reduction_indices) dimensions
        reduction_shape = result.shape[: len(reduction_indices)]
        flat_shape = (functools.reduce(operator.mul, reduction_shape, 1),) + result.shape[
            len(reduction_indices) :
        ]
        flat_result = jnp.reshape(result, flat_shape)

        mins = (
            jnp.min(flat_result, axis=0)
            if semiring is ArgMinAlg
            else jnp.max(flat_result, axis=0)
        )
        flat_indices = (
            jnp.argmin(flat_result, axis=0)
            if semiring is ArgMinAlg
            else jnp.argmax(flat_result, axis=0)
        )
        min_indices = jnp.unravel_index(flat_indices, reduction_shape)
        with handler(
            {
                fresh_to_old[k]: deffn(streams[fresh_to_old[k]][v])
                for k, v in zip(reduction_indices, min_indices)
            }
        ):
            argmins = evaluate(argmin_value)

        final_result = tree.map_structure(
            lambda t: bind_dims(
                t, *[i for i in result_indices if i not in reduction_indices]
            ),
            (mins, argmins),
        )
        return final_result


class DenseTensorFold(ObjectInterpretation):
    def _sum_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.sum(tensor, axis=0)
        return tensor

    def _min_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.min(tensor, axis=0)
        return tensor

    def _max_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.max(tensor, axis=0)
        return tensor

    def _get_reductor(self, semi_ring):
        if semi_ring is LinAlg:
            return self._sum_reductor
        elif semi_ring is MinAlg:
            return self._min_reductor
        elif semi_ring is MaxAlg:
            return self._max_reductor
        else:
            return None

    @implements(fold)
    def fold(self, semiring, streams, body):
        reductor = self._get_reductor(semiring)
        if not (
            reductor and all(issubclass(typeof(s), jax.Array) for s in streams.values())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) > 1:
            # todo: handle multiple output indices
            return fwd()

        indices, value = body.args[0]

        # Check that the output is indexed in a subset of the input indices, and
        # that there are no index transformations
        if not all(isinstance(i, Term) and i.op in streams for i in indices):
            return fwd()
        indices = [i.op for i in indices]

        old_to_fresh = {k: defop(k) for k in streams.keys()}
        indexed_streams = {
            k: deffn(jax_getitem(v, [old_to_fresh[k]()])) for k, v in streams.items()
        }

        with handler(indexed_streams):
            result = evaluate(value)

        fvars = fvsof(result)
        unused_streams = {
            k: v for k, v in streams.items() if old_to_fresh[k] not in fvars
        }
        if unused_streams:
            result = result * functools.reduce(
                operator.mul, (len(v) for v in unused_streams.values()), 1
            )

        result_indices = fvsof(result)
        reduction_indices = [
            old_to_fresh[i]
            for i in streams.keys()
            if old_to_fresh.get(i) in result_indices and i not in indices
        ]

        # bind and reduce indices from the streams that do not appear in the result indexing expression
        result = bind_dims(result, *reduction_indices)
        result = reductor(result, len(reduction_indices))

        # bind indices that appear in the indexing expression
        fresh_indices = [old_to_fresh[i] for i in indices]
        result = bind_dims(result, *fresh_indices)
        return result


class GradientOptimizationFold(ObjectInterpretation):
    """Handle min/argmin over reals using gradient descent.

    Notes:
    - A single empty output index is expected. Nontrivial output indexes would in
    principle allow us to represent partial optimization problems like the following:
    fold(MinAlg, {x: reals(), y: reals()}, {x(): f(x(), y())}) = \\lambda x. min_{y\\in R} f(x, y).

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

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not (
            semiring in (MinAlg, ArgMinAlg)
            and all(isinstance(v, Term) and v.op is reals for v in streams.values())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) <= 0:
            return jnp.array([])

        if len(body.args) > 1:
            # todo: handle multiple output indices
            return fwd()

        indices, value = body.args[0]
        if indices != ():
            return fwd()

        if semiring is ArgMinAlg:
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("Expected a tuple of (value, arg) for ArgMinAlg")
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
            assert all(isinstance(g, jax.Array) for g in tree.flatten(grads))
            updates, opt_state = optimizer.update(grads, opt_state)
            param_values = optax.apply_updates(param_values, updates)

        final_loss = loss(key, *param_values)

        if semiring is MinAlg:
            return final_loss

        with handler({v: deffn(p) for (v, p) in zip(param_keys, param_values)}):
            final_arg = evaluate(arg)

        return final_loss, final_arg


class LikelihoodWeightingFold(ObjectInterpretation):
    """Handle expectation computation using likelihood weighting."""

    def __init__(self, samples=1):
        self.samples = samples

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not (
            semiring is LinAlg
            and all(issubclass(typeof(v), Distribution) for v in streams.values())
        ):
            reason = "semiring" if semiring is not LinAlg else "streams"
            logger.debug(
                f"Skipping likelihood weighting (reason {reason}): fold({semiring}, {streams}, {body}"
            )
            return fwd()

        sample_streams = {}
        index_streams = {}
        for k, v in streams.items():
            s = defop(jax.Array, name="sample")

            if isinstance(k, Operation):
                value = k
                samples = sample(key(), v, (self.samples,))[s()]
                sample_streams[value] = deffn(samples)
            elif isinstance(k, tuple):
                if not (len(k) == 2 and all(isinstance(i, Operation) for i in k)):
                    raise ValueError(
                        "Expected a tuple of (value, weight) for likelihood weighting"
                    )
                (value, weight) = k
                samples = sample(key(), v, (self.samples,))
                weights = log_prob(v, samples)
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

        return fold(LinAlg, index_streams, body)


interpretation = functools.reduce(
    coproduct,
    [
        NormalizeValueFold(),
        DenseTensorArgFold(),
        DenseTensorFold(),
    ],
)
