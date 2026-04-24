from typing import Annotated

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem
from effectful.handlers.jax._handlers import is_eager_array
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import Scoped, deffn, defop
from effectful.ops.types import Expr, Operation, Term
from effectful.ops.weighted.distribution import D
from effectful.ops.weighted.monoid import (
    ArgMax,
    ArgMin,
    Body,
    CommutativeMonoid,
    CommutativeMonoidWithZero,
    Monoid,
    Semilattice,
    Streams,
    distributes_over,
    order_streams,
)


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


class _JaxReduce:
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

    def _logaddexp_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = logsumexp(tensor, axis=0)
        return tensor

    def _get_reductor(self, semi_ring):
        if semi_ring == Sum:
            return self._sum_reductor
        if semi_ring == Product:
            return self._prod_reductor
        elif semi_ring == Min:
            return self._min_reductor
        elif semi_ring == Max:
            return self._max_reductor
        elif semi_ring == ArgMin:
            return self._argmin_reductor
        elif semi_ring == ArgMax:
            return self._argmax_reductor
        elif semi_ring == LogSumExp:
            return self._logaddexp_reductor
        else:
            return None

    @Operation.define
    def reduce[A, B, U: Body](
        self, streams: Annotated[Streams, Scoped[A]], body: Annotated[U, Scoped[A | B]]
    ) -> Annotated[U, Scoped[B]]:
        reductor = self._get_reductor(self)
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

        # Check that the output is indexed in a subset of the input indices, and
        # that there are no index transformations
        if not all(isinstance(i, Term) and i.op in streams for i in indices):
            return fwd()
        indices = [i.op for i in indices]

        old_to_fresh = {k: defop(k, name=f"fresh_{k}") for k in streams}
        indexed_streams = {
            k: deffn(jax_getitem(v, [old_to_fresh[k]()])) for k, v in streams.items()
        }

        # add indices for streams that don't appear in body
        fvars = set(streams.keys()) - fvsof(value)
        unused_streams = tuple(v() for k, v in indexed_streams.items() if k in fvars)

        value = jnp.asarray(value) if not isinstance(value, Term | jax.Array) else value
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
        result_4 = bind_dims(result_3, *fresh_indices)
        return result_4


class _JaxMonoid(_JaxReduce, Monoid[jax.Array]):
    pass


class _JaxCommutativeMonoid(_JaxReduce, CommutativeMonoid[jax.Array]):
    pass


class _JaxCommutativeMonoidWithZero(_JaxReduce, CommutativeMonoidWithZero[jax.Array]):
    pass


class _JaxSemilattice(_JaxReduce, Semilattice[jax.Array]):
    pass


class _SumMonoid(_JaxCommutativeMonoid):
    def scalar_mul(self, v: jax.Array, x: int) -> jax.Array:
        return v * x


Sum = _SumMonoid.from_binary(jnp.add, jnp.asarray(0))


class _ProductMonoid(_JaxCommutativeMonoidWithZero):
    def scalar_mul(self, v: jax.Array, x: int) -> jax.Array:
        return v**x


Product = _ProductMonoid.from_binary_with_zero(
    jnp.multiply, jnp.asarray(1), jnp.asarray(0)
)


Min = _JaxSemilattice.from_binary(jnp.minimum, jnp.asarray(float("-inf")))
Max = _JaxSemilattice.from_binary(jnp.maximum, jnp.asarray(float("inf")))
LogSumExp = _JaxSemilattice.from_binary(jnp.logaddexp, jnp.asarray(float("-inf")))


@Operation.define
def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


CartesianProd = _JaxMonoid.from_binary(cartesian_prod, jnp.array([]))

distributes_over.register(jnp.maximum, jnp.minimum)
distributes_over.register(jnp.minimum, jnp.maximum)
distributes_over.register(jnp.add, jnp.minimum)
distributes_over.register(jnp.add, jnp.maximum)
distributes_over.register(jnp.multiply, jnp.add)
distributes_over.register(jnp.add, jnp.logaddexp)

assert isinstance(Sum, CommutativeMonoid)
assert isinstance(Product, CommutativeMonoidWithZero)
assert isinstance(Min, Semilattice)
assert isinstance(Max, Semilattice)
assert isinstance(LogSumExp, Semilattice)
