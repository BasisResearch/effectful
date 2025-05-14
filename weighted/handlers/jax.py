import collections.abc
import functools
import logging
import operator
from collections.abc import Iterable

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numbers  # noqa: F401
import jax
import optax
import tree
from effectful.handlers.jax import bind_dims, jax_getitem, sizesof, unbind_dims
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
from effectful.ops.syntax import (
    ObjectInterpretation,
    deffn,
    defop,
    implements,
    syntactic_eq,
)
from effectful.ops.types import Operation, Term
from numpyro.distributions import Distribution

from weighted.ops.fold import fold
from weighted.ops.semiring import ArgMaxAlg, ArgMinAlg, LinAlg, MaxAlg, MinAlg

logger = logging.getLogger(__name__)


@defop
def D(*args: tuple[tuple[int, ...], jax.Array]) -> jax.Array:
    if not all(isinstance(kv, tuple) and len(kv) == 2 for kv in args):
        raise ValueError("Expected a sequence of key-value pairs")
    raise NotImplementedError


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
            and all(typeof(k()) is jax.Array for k in streams)
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

        old_to_fresh = {k: defop(k) for k in streams}
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
                for k, v in zip(reduction_indices, min_indices, strict=True)
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

        old_to_fresh = {k: defop(k) for k in streams}
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
                operator.mul, (v.shape[0] for v in unused_streams.values()), 1
            )

        result_indices = fvsof(result)
        reduction_indices = [
            old_to_fresh[i]
            for i in streams
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

        with handler(
            {v: deffn(p) for (v, p) in zip(param_keys, param_values, strict=True)}
        ):
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


class PytreeMapFold(ObjectInterpretation):
    """Map a fold over a pytree body."""

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        # Check that all values in the body have the same structure
        structure = None
        for _, t in body.args:
            s = jax.tree.structure(t)
            if structure and structure != s:
                raise ValueError(
                    f"Found pytrees with different structures {structure} and {s} in fold body."
                )
            structure = s

        # Do no work for trivial structures
        if structure == jax.tree.structure(0):
            return fwd()

        flat_bodies = [jax.tree.flatten(t)[0] for (_, t) in body.args]

        flat_body = [
            fold(
                semiring,
                streams,
                D(*[(k, v) for (k, _), v in zip(body.args, vs, strict=True)]),
            )
            for vs in zip(*flat_bodies, strict=True)
        ]

        return jax.tree.unflatten(structure, flat_body)


def scan(f, init, *args, **kwargs):
    sizes = sizesof(init)
    pos_init = bind_dims(init, *sizes.keys())

    def wrapped_f(pos_carry, x):
        carry = unbind_dims(pos_carry, *sizes.keys())
        next_carry, next_result = f(carry, x)
        return bind_dims(next_carry, *sizes.keys()), bind_dims(next_result, *sizes.keys())

    pos_carry, pos_result = jax.lax.scan(wrapped_f, pos_init, *args, **kwargs)
    carry = unbind_dims(pos_carry, *sizes.keys())
    result = jax_getitem(pos_result, [slice(None)] + [i() for i in sizes])
    return carry, result


class ScanFold(ObjectInterpretation):
    @implements(fold)
    def fold(self, semiring, streams, body):
        # FIXME: This handler does not deal with the case where some but not all
        # stream variables form a chain

        # check that the streams form a chain
        stream_vars = set(streams.keys())

        head_var, head_val = None, None
        for k, v in streams.items():
            if not (fvsof(v) & stream_vars):
                head_var, head_val = (k, v)
                break

        if head_var is None or head_val is None:
            return fwd()

        # FIXME: This algorithm is O(n^2) in the number of stream variables
        unlinked_vars = set(stream_vars)
        unlinked_vars.remove(head_var)
        chain = [(head_var, head_val)]
        while unlinked_vars:
            for var in unlinked_vars:
                free_vars = fvsof(streams[var]) & stream_vars
                if free_vars == {chain[-1][0]}:
                    chain.append((var, streams[var]))
                    unlinked_vars.remove(var)
                    break
            else:
                # Failed to find the next link in the chain
                return fwd()

        dummy_var = defop(object)
        chain_body = None
        for i in range(1, len(chain)):
            with handler({chain[i - 1][0]: dummy_var}):
                if chain_body is None:
                    chain_body = evaluate(chain[i][1])
                else:
                    if not syntactic_eq(chain_body, evaluate(chain[i][1])):
                        return fwd()

        def func(carry, _idx):
            with handler({dummy_var: lambda: carry}):
                result = evaluate(chain_body)
            return (result, result)

        _, scanned_array = scan(func, head_val, jnp.arange(len(chain) - 1))

        new_streams = {head_var: head_val}
        for i, (var, _) in enumerate(chain[1:]):
            new_streams[var] = scanned_array[i]

        return fold(semiring, new_streams, body)


class NormalizeValueFold(ObjectInterpretation):
    """Normalization rule for the body of folds."""

    @implements(fold)
    def fold(self, semiring, streams, body):
        modified_body = False
        if isinstance(body, Term) and body.op is D:
            kvs = []
            for k, v in body.args:
                if not isinstance(k, tuple):
                    k = (k,)
                    modified_body = True
                kvs.append((k, v))
            new_body = D(*kvs)
        elif isinstance(body, dict):
            modified_body = True
            new_body = D(*body.items())
        else:
            modified_body = True
            new_body = D(((), body))

        if modified_body:
            return fold(semiring, streams, new_body)
        return fwd()


interpretation = functools.reduce(
    coproduct,  # type: ignore
    [NormalizeValueFold(), DenseTensorArgFold(), DenseTensorFold()],
)
