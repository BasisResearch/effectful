import functools

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    CartesianProduct,
    Max,
    Min,
    Monoid,
    Product,
    Streams,
    Sum,
    outer_stream,
)
from effectful.ops.semantics import evaluate, handler, typeof
from effectful.ops.syntax import deffn
from effectful.ops.types import NotHandled, Operation


def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


LogSumExp = Monoid(name="LogSumExp", identity=jnp.asarray(float("-inf")))


@Sum.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.add, args)


@Product.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.multiply, args)


@Min.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.minimum, args)


@Max.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.maximum, args)


@LogSumExp.plus.register(jax.Array)
def _(*args):
    return functools.reduce(jnp.logaddexp, args)


@CartesianProduct.plus.register(jax.Array)
def _(*args):
    return functools.reduce(cartesian_prod, args)


ARRAY_REDUCE = {
    Sum.plus: jnp.sum,
    Product.plus: jnp.prod,
    Min.plus: jnp.min,
    Max.plus: jnp.max,
    LogSumExp.plus: logsumexp,
}


def _reduce_array_for(monoid: Monoid):
    def _reduce_array(body: jax.Array, streams: Streams):
        reductor = ARRAY_REDUCE[monoid.plus]
        index = Operation.define(jax.Array)

        if not streams:
            return monoid.identity

        # find and reduce an array stream
        for stream_key, stream_body, streams_tail in outer_stream(streams):
            if typeof(stream_body) != jax.Array:
                continue

            with handler({stream_key: deffn(unbind_dims(stream_body, index))}):
                (eval_body, eval_streams_tail) = (
                    evaluate(body),
                    evaluate(streams_tail),
                )
                assert isinstance(eval_streams_tail, dict)

                reduce_tail = (
                    monoid.reduce(eval_body, eval_streams_tail)
                    if len(eval_streams_tail) > 0
                    else eval_body
                )
                return reductor(bind_dims(reduce_tail, index), axis=0)

        raise NotHandled

    return _reduce_array


for _m in (Sum, Product, Min, Max, LogSumExp):
    _m.reduce.register(jax.Array)(_reduce_array_for(_m))
