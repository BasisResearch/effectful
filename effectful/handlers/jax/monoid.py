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
from effectful.ops.semantics import evaluate, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
from effectful.ops.types import Operation


def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


LogSumExp = Monoid("LogSumExp")


class MinKernelJax(ObjectInterpretation):
    @implements(Min.kernel)
    def kernel(self, x, y):
        if isinstance(x, jax.Array) and isinstance(y, jax.Array):
            return jnp.minimum(x, y)
        return fwd()


class MaxKernelJax(ObjectInterpretation):
    @implements(Max.kernel)
    def kernel(self, x, y):
        if isinstance(x, jax.Array) and isinstance(y, jax.Array):
            return jnp.maximum(x, y)
        return fwd()


class CartesianProductKernelJax(ObjectInterpretation):
    @implements(CartesianProduct.kernel)
    def kernel(self, x, y):
        if isinstance(x, jax.Array) and isinstance(y, jax.Array):
            return cartesian_prod(x, y)
        return fwd()


class LogSumExpKernelJax(ObjectInterpretation):
    @implements(LogSumExp.identity)
    def identity(self):
        return float("-inf")

    @implements(LogSumExp.kernel)
    def kernel(self, x, y):
        if isinstance(x, jax.Array) and isinstance(y, jax.Array):
            return jnp.logaddexp(x, y)
        return fwd()


ARRAY_REDUCE = {
    Sum.plus: jnp.sum,
    Product.plus: jnp.prod,
    Min.plus: jnp.min,
    Max.plus: jnp.max,
    LogSumExp.plus: logsumexp,
}


@Monoid.reduce.register(jax.Array)
def _reduce_array(self, body: jax.Array, streams: Streams):
    reductor = ARRAY_REDUCE[self.plus]
    index = Operation.define(jax.Array)

    if not streams:
        return self.identity

    # find and reduce an array stream
    for stream_key, stream_body, streams_tail in outer_stream(streams):
        if typeof(stream_body) != jax.Array:
            continue

        with handler({stream_key: deffn(unbind_dims(stream_body, index))}):
            (eval_body, eval_streams_tail) = evaluate(body), evaluate(streams_tail)
            assert isinstance(eval_streams_tail, dict)

            reduce_tail = (
                self.reduce(eval_body, eval_streams_tail)
                if len(eval_streams_tail) > 0
                else eval_body
            )
            return reductor(bind_dims(reduce_tail, index), axis=0)

    return self._reduce_object(body, streams)
