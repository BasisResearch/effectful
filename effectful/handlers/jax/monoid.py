from typing import Annotated, Callable

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


class _JaxReduceMixin:
    reduce_array: Callable[[jax.Array, int | tuple[int]], jax.Array]

    def __init__(self, *args, **kwargs):
        self.reduce_array = kwargs.pop("reduce_array")
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
            r = super().reduce
            breakpoint()
            return super().reduce(streams, body)

        # bind and reduce indices from the streams that do not appear in the result indexing expression
        reduction_indices = tuple(old_to_fresh[i] for i in streams if i not in indices)
        result_2 = bind_dims(result_1, *reduction_indices)
        result_3 = self.reduce_array(result_2, tuple(range(len(reduction_indices))))

        # bind indices that appear in the indexing expression
        fresh_indices = [old_to_fresh[i] for i in indices]
        result_4 = bind_dims(result_3, *fresh_indices)
        return result_4


class _JaxMonoid(_JaxReduceMixin, Monoid[jax.Array]):
    pass


class _JaxCommutativeMonoid(_JaxReduceMixin, CommutativeMonoid[jax.Array]):
    pass


class _JaxCommutativeMonoidWithZero(
    _JaxReduceMixin, CommutativeMonoidWithZero[jax.Array]
):
    pass


class _JaxSemilattice(_JaxReduceMixin, Semilattice[jax.Array]):
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
    result.reduce_array = ufunc.reduce
    return result


Sum = _SumMonoid(jnp.add, identity=jnp.asarray(0), reduce_array=jnp.sum)
Product = _ProductMonoid(
    jnp.multiply, identity=jnp.asarray(1), zero=jnp.asarray(0), reduce_array=jnp.prod
)
Min = _JaxSemilattice(
    jnp.minimum, identity=jnp.asarray(float("-inf")), reduce_array=jnp.min
)
Max = _JaxSemilattice(
    jnp.maximum, identity=jnp.asarray(float("inf")), reduce_array=jnp.max
)
LogSumExp = _JaxSemilattice(
    jnp.logaddexp, identity=jnp.asarray(float("-inf")), reduce_array=logsumexp
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
