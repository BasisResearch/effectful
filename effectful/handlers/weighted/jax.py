import functools
import logging
import operator

import jax
import jax.tree as tree
import optax
from numpyro.distributions import Distribution

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, sizesof, unbind_dims
from effectful.handlers.jax._handlers import is_eager_array
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
from effectful.ops.types import Expr, Operation, Term
from effectful.ops.weighted.distribution import D
from effectful.ops.weighted.jax import key, reals
from effectful.ops.weighted.monoid import (
    ArgMaxMonoid,
    ArgMinMonoid,
    LogSumMonoid,
    MaxMonoid,
    MinMonoid,
    ProdMonoid,
    SumMonoid,
)
from effectful.ops.weighted.reduce import order_streams, reduce

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


class DenseTensorReduce(ObjectInterpretation):
    def _sum_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.sum(tensor, axis=0)
        return tensor

    def _prod_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.prod(tensor, axis=0)
        return tensor

    def _min_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.min(tensor, axis=0)
        return tensor

    def _max_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.max(tensor, axis=0)
        return tensor

    def _argmin_reductor(self, tensor, dims):
        # Flatten the leading len(reduction_indices) dimensions
        reduction_shape = tensor.shape[:dims]
        head_shape = (functools.reduce(operator.mul, reduction_shape, 1),)
        tail_shape = tensor.shape[dims:]
        flat_shape = head_shape + tail_shape

        flat_result = jnp.reshape(tensor, flat_shape)
        mins = jnp.min(flat_result, axis=0)
        flat_indices = jnp.argmin(flat_result, axis=0)
        min_indices = jnp.unravel_index(flat_indices, reduction_shape)
        return mins, min_indices

    def _argmax_reductor(self, tensor, dims):
        # Flatten the leading len(reduction_indices) dimensions
        reduction_shape = tensor.shape[:dims]
        head_shape = (functools.reduce(operator.mul, reduction_shape, 1),)
        tail_shape = tensor.shape[dims:]
        flat_shape = head_shape + tail_shape

        flat_result = jnp.reshape(tensor, flat_shape)
        maxs = jnp.max(flat_result, axis=0)
        flat_indices = jnp.argmax(flat_result, axis=0)
        max_indices = jnp.unravel_index(flat_indices, reduction_shape)
        return maxs, max_indices

    def _logaddexp_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = logsumexp(tensor, axis=0)
        return tensor

    def _get_reductor(self, semi_ring):
        if semi_ring == SumMonoid:
            return self._sum_reductor
        if semi_ring == ProdMonoid:
            return self._prod_reductor
        elif semi_ring == MinMonoid:
            return self._min_reductor
        elif semi_ring == MaxMonoid:
            return self._max_reductor
        elif semi_ring == ArgMinMonoid:
            return self._argmin_reductor
        elif semi_ring == ArgMaxMonoid:
            return self._argmax_reductor
        elif semi_ring == LogSumMonoid:
            return self._logaddexp_reductor
        else:
            return None

    @implements(reduce)
    @timed(name="DenseTensorReduce")
    def reduce(self, monoid, streams, body):
        reductor = self._get_reductor(monoid)
        if not (
            reductor and all(issubclass(typeof(s), jax.Array) for s in streams.values())
        ):
            return fwd()

        # raises an exception if there are cyclic dependencies
        order_streams(streams)

        body_indices = _parse_body(body)

        if len(body_indices) > 1:
            # todo: handle multiple output indices
            return fwd()

        indices, value = body_indices[0]

        if monoid in (ArgMinMonoid, ArgMaxMonoid):
            if not isinstance(value, tuple) and len(value) == 2:
                raise ValueError("Expected a tuple of (value, arg) for argmin/argmax")
            value, arg = value

        # Check that the output is indexed in a subset of the input indices, and
        # that there are no index transformations
        if not all(isinstance(i, Term) and i.op in streams for i in indices):
            return fwd()
        indices = [i.op for i in indices]

        old_to_fresh = {k: defop(k, name=f"fresh_{k}") for k in streams}
        fresh_to_old = {v: k for (k, v) in old_to_fresh.items()}
        indexed_streams = {
            k: deffn(jax_getitem(v, [old_to_fresh[k]()])) for k, v in streams.items()
        }

        # add indices for streams that don't appear in body
        fvars = set(streams.keys()) - fvsof(value)
        unused_streams = tuple(v() for k, v in indexed_streams.items() if k in fvars)
        value = jax_getitem(value[*[None] * len(unused_streams)], unused_streams)

        with handler(indexed_streams):
            result_1 = evaluate(value)

        if not is_eager_array(result_1):
            return fwd()

        # bind and reduce indices from the streams that do not appear in the result indexing expression
        reduction_indices = [old_to_fresh[i] for i in streams if i not in indices]
        result_2 = bind_dims(result_1, *reduction_indices)
        result_3 = reductor(result_2, len(reduction_indices))

        # bind indices that appear in the indexing expression
        fresh_indices = [old_to_fresh[i] for i in indices]

        if monoid in (ArgMinMonoid, ArgMaxMonoid):
            result_3, min_indices = result_3

            with handler(
                {
                    fresh_to_old[k]: deffn(jax_getitem(streams[fresh_to_old[k]], [v]))
                    for k, v in zip(reduction_indices, min_indices, strict=False)
                }
            ):
                args = evaluate(arg)

            result_4 = (
                bind_dims(result_3, *fresh_indices),
                bind_dims(args, *fresh_indices),
            )
        else:
            result_4 = bind_dims(result_3, *fresh_indices)

        return result_4


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

    @implements(reduce)
    def reduce(self, monoid, streams, body):
        # TODO: handle mixed discrete/continuous optimization
        if not (
            monoid in (MinMonoid, ArgMinMonoid)
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

        if monoid is ArgMinMonoid:
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

        if monoid is MinMonoid:
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

    @implements(reduce)
    def reduce(self, monoid, streams, body):
        if not (
            monoid is SumMonoid
            and all(issubclass(typeof(v), Distribution) for v in streams.values())
        ):
            reason = "monoid" if monoid is not SumMonoid else "streams"
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

        return reduce(SumMonoid, index_streams, body)


class PytreeMapReduce(ObjectInterpretation):
    """Map a reduce over a pytree body."""

    @implements(reduce)
    def reduce(self, monoid, streams, body):
        if not (monoid in (MinMonoid, MaxMonoid, SumMonoid)):
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
            reduce(
                monoid,
                streams,
                D(*[(k, v) for (k, _), v in zip(body_indices, vs, strict=True)]),
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
        return bind_dims(next_carry, *sizes.keys()), bind_dims(
            next_result, *sizes.keys()
        )

    pos_carry, pos_result = jax.lax.scan(wrapped_f, pos_init, *args, **kwargs)
    carry = unbind_dims(pos_carry, *sizes.keys())
    result = jax_getitem(pos_result, [slice(None)] + [i() for i in sizes])
    return carry, result


class ScanReduce(ObjectInterpretation):
    def __init__(self, min_length=3):
        self.min_length = min_length

    @implements(reduce)
    def reduce(self, monoid, streams, body):
        # FIXME: This handler does not deal with the case where some but not all
        # stream variables form a chain

        if len(streams) < self.min_length:
            return fwd()

        # check that the streams form a chain
        stream_vars = set(streams.keys())
        head_var, head_val = None, None
        for k, v in streams.items():
            if not (fvsof(v) & stream_vars):
                head_var, head_val = (k, v)
                break

        if head_var is None or head_val is None or not is_eager_array(head_val):
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

        if len(chain) < 3:
            return fwd()

        dummy_var = defop(jax.Array)
        chain_body = None
        for i in range(1, len(chain)):
            with handler({chain[i - 1][0]: dummy_var}):
                if chain_body is None:
                    chain_body = evaluate(chain[i][1])
                else:
                    if not syntactic_eq(chain_body, evaluate(chain[i][1])):
                        return fwd()

        dummy_idx = defop(jax.Array, name="k")

        def func(carry, _idx):
            with handler({dummy_var: lambda: carry}):
                result = evaluate(chain_body)
                result = bind_dims(result, dummy_idx)
                result = jnp.squeeze(result, 0)
                result = unbind_dims(result, dummy_idx)
            return (result, result)

        head_val_indexed = jax_getitem(head_val, [dummy_idx()])
        _, scanned_array = scan(func, head_val_indexed, jnp.arange(len(chain) - 1))

        new_streams = {head_var: head_val}
        for i, (var, _) in enumerate(chain[1:]):
            new_streams[var] = bind_dims(scanned_array[i], dummy_idx)

        return reduce(monoid, new_streams, body)


interpretation = functools.reduce(
    coproduct,  # type: ignore
    [
        DenseTensorReduce(),
    ],
)
