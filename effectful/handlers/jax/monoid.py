import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.monoid import (
    CommutativeMonoid,
    CommutativeMonoidWithZero,
    Monoid,
    Semilattice,
    Streams,
    distributes_over,
    outer_stream,
)
from effectful.ops.semantics import evaluate, handler, typeof
from effectful.ops.syntax import deffn
from effectful.ops.types import Operation


@Operation.define
def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


Sum = CommutativeMonoid(kernel=jnp.add, identity=jnp.asarray(0))
Product = CommutativeMonoidWithZero(
    kernel=jnp.multiply, identity=jnp.asarray(1), zero=jnp.asarray(0)
)
Min = Semilattice(kernel=jnp.minimum, identity=jnp.asarray(float("-inf")))
Max = Semilattice(kernel=jnp.maximum, identity=jnp.asarray(float("inf")))
LogSumExp = CommutativeMonoid(kernel=jnp.logaddexp, identity=jnp.asarray(float("-inf")))
CartesianProd = Monoid(kernel=cartesian_prod, identity=jnp.array([]))

distributes_over.register(Max.plus, Min.plus)
distributes_over.register(Min.plus, Max.plus)
distributes_over.register(Sum.plus, Min.plus)
distributes_over.register(Sum.plus, Max.plus)
distributes_over.register(Product.plus, Sum.plus)
distributes_over.register(Sum.plus, LogSumExp.plus)

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
