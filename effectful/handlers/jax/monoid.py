from collections.abc import Callable
from typing import Annotated

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax.scipy.special import logsumexp
from effectful.ops.semantics import evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import Scoped, deffn, defop
from effectful.ops.types import Expr, Operation, Term
from effectful.ops.weighted.distribution import D
from effectful.ops.weighted.monoid import (
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


class ArrayMonoid[T](Monoid[T]):
    """Reduce array-valued bodies over array-valued streams."""

    reduce_dims: Callable[[T, int | tuple[int]], T]
    bind_dims: Operation
    unbind_dims: Operation

    def __init__(self, *args, **kwargs):
        self.reduce_dims = kwargs.pop("reduce_dims")
        self.bind_dims = kwargs.pop("bind_dims")
        self.unbind_dims = kwargs.pop("unbind_dims")
        super().__init__(*args, **kwargs)

    @Operation.define
    def reduce[A, B, U: Body](
        self, streams: Annotated[Streams, Scoped[A]], body: Annotated[U, Scoped[A | B]]
    ) -> Annotated[U, Scoped[B]]:
        if not (all(issubclass(typeof(s), jax.Array) for s in streams.values())):
            return super().reduce(streams, body)

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
            return super().reduce(streams, body)

        indices = [i.op for i in indices]

        old_to_fresh = {k: defop(k, name=f"fresh_{k}") for k in streams}
        indexed_streams = {
            k: deffn(self.unbind_dims(v, old_to_fresh[k])) for k, v in streams.items()
        }

        indexed_result = handler(indexed_streams)(evaluate)(value)
        used_indices = fvsof(indexed_result)

        # bind and reduce indices from the streams that do not appear in the result indexing expression
        reduction_indices = tuple(
            old_to_fresh[i]
            for i in streams
            if old_to_fresh[i] in used_indices and i not in indices
        )
        if reduction_indices:
            bound_result = self.bind_dims(indexed_result, *reduction_indices)
            reduced_result = self.reduce_dims(
                bound_result, tuple(range(len(reduction_indices)))
            )
        else:
            reduced_result = indexed_result

        # bind indices that appear in the indexing expression
        fresh_indices = [old_to_fresh[i] for i in indices]
        reindexed_result = self.bind_dims(reduced_result, *fresh_indices)

        unused_streams = {
            k: v for (k, v) in streams.items() if old_to_fresh[k] not in used_indices
        }
        return super().reduce(unused_streams, reindexed_result)


class _JaxArrayMonoid(ArrayMonoid[jax.Array]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bind_dims=bind_dims, unbind_dims=unbind_dims, **kwargs)


class _JaxCommutativeMonoid(_JaxArrayMonoid, CommutativeMonoid[jax.Array]):
    pass


class _JaxCommutativeMonoidWithZero(
    _JaxArrayMonoid, CommutativeMonoidWithZero[jax.Array]
):
    pass


class _JaxSemilattice(_JaxArrayMonoid, Semilattice[jax.Array]):
    pass


class _SumMonoid(_JaxCommutativeMonoid):
    def scalar_mul(self, v: jax.Array, x: int) -> jax.Array:
        return v * x


class _ProductMonoid(_JaxCommutativeMonoidWithZero):
    def scalar_mul(self, v: jax.Array, x: int) -> jax.Array:
        return v**x


def cartesian_prod(x, y):
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = jnp.repeat(x, y.shape[0], axis=0), jnp.tile(y, (x.shape[0], 1))
    return jnp.hstack([x, y])


def from_binary(cls, add, identity, reduce):
    identity = identity if identity is not None else ufunc.identity
    if identity is None:
        raise ValueError("Expected an identity element.")

    result = cls.from_binary(ufunc, identity)
    result.reduce_dims = ufunc.reduce
    return result


Sum = _SumMonoid(jnp.add, identity=jnp.asarray(0), reduce_dims=jnp.sum)
Product = _ProductMonoid(
    jnp.multiply, identity=jnp.asarray(1), zero=jnp.asarray(0), reduce_dims=jnp.prod
)
Min = _JaxSemilattice(
    jnp.minimum, identity=jnp.asarray(float("-inf")), reduce_dims=jnp.min
)
Max = _JaxSemilattice(
    jnp.maximum, identity=jnp.asarray(float("inf")), reduce_dims=jnp.max
)
LogSumExp = _JaxSemilattice(
    jnp.logaddexp, identity=jnp.asarray(float("-inf")), reduce_dims=logsumexp
)
CartesianProd = Monoid(cartesian_prod, identity=jnp.array([]))

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
