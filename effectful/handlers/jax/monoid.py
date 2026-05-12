import dataclasses

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
from effectful.ops.semantics import evaluate, fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
from effectful.ops.types import Interpretation, Operation, Term


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


@dataclasses.dataclass
class Delta:
    index: tuple[int, ...]
    weight: jax.Array


@dataclasses.dataclass
class range:
    start: int
    stop: int
    step: int

    def __init__(self, *args: int):
        match args:
            case (stop,):
                self.stop = stop
            case (start, stop):
                self.start = start
                self.stop = stop
            case (start, stop, step):
                self.start = start
                self.stop = stop
                self.step = step
            case _:
                raise ValueError(f"Unexpected arguments: {args}")

    @Operation.define
    def __iter__(self):
        if (
            isinstance(self.start, Term)
            or isinstance(self.stop, Term)
            or isinstance(self.step, Term)
        ):
            raise NotHandled

        return iter(range(self.start, self.stop, self.step))


class ReduceDeltaEmpty(ObjectInterpretation):
    """Eliminate a Delta with no index.

    reduce(M, streams, Delta((), body)) ≡ reduce(M, streams, body)

    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if isinstance(body, Delta) and not body.indices:
            return monoid.reduce(body, streams)
        return fwd()


class ReduceDeltaIndependent(ObjectInterpretation):
    """Eliminate a Delta that has independent, dense index arguments.

    reduce(M, streams ∪ {v: range(N)}, Delta(idx' ++ (v(),), body))
    ═══════════════════════════════════════════════════════════════════════════
    reduce(M, streams, Delta(idx', bind_dims(body[v() := unbind_dims(streams[v], fv)], fv)))

    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if not (isinstance(body, Delta) and body.indices):
            return fwd()

        head_indices, tail_index = body.indices[:-1], body.indices[-1]
        if not (isinstance(tail_index, Term) and tail_index.op in streams):
            return fwd()

        tail_stream = streams[tail_index.op]
        fresh_op = Operation.define(tail_index.op)
        fresh_stream = unbind_dims(tail_stream, fresh_op)
        subst_intp: Interpretation = {tail_index.op: deffn(fresh_stream)}
        fresh_body = bind_dims(handler(subst_intp)(evaluate)(body), fresh_op)
        return monoid.reduce(Delta(head_indices, fresh_body))


class ReduceDependentRangeMask(ObjectInterpretation):
    """Eliminate a dependent range by masking.

    reduce(M, streams ∪ {u: range(N), v: range(u())}, body)
    ═══════════════════════════════════════════════════════════════════════════
    reduce(M, streams ∪ {u: range(N), v: range(N)}, where(v() < u(), body, M.identity))

    """

    @implements(Monoid.reduce)
    def _(self, monoid: Monoid, body, streams: Streams):
        if not (isinstance(body, Delta) and body.indices):
            return fwd()

        stream_vars = set(streams.keys())
        for u, u_stream in streams.items():
            if not (
                isinstance(u_stream, range)
                and not (fvsof(u_stream) & stream_vars)
                and not isinstance(u_stream.start, Term)
                and u_stream.start == 0
                and not isinstance(u_stream.step, Term)
                and u_stream.step == 1
            ):
                continue

            for v, v_stream in streams.items():
                if not (
                    isinstance(u_stream, range)
                    and not isinstance(v_stream.start, Term)
                    and v_stream.start == 0
                    and not isinstance(v_stream.step, Term)
                    and v_stream.step == 1
                    and isinstance(v_stream.stop, Term)
                    and v_stream.stop.op == u
                ):
                    fresh_streams = {}
                    return monoid.reduce()
                    continue

        tail_stream = streams[tail_index.op]
        fresh_op = Operation.define(tail_index.op)
        fresh_stream = unbind_dims(tail_stream, fresh_op)
        subst_intp: Interpretation = {tail_index.op: deffn(fresh_stream)}
        fresh_body = bind_dims(handler(subst_intp)(evaluate)(body), fresh_op)
        return monoid.reduce(Delta(head_indices, fresh_body))
