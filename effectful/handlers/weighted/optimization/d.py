import functools

from weighted.ops.distribution import D
from weighted.ops.reduce import reduce

from effectful.handlers.jax import bind_dims, jax_getitem
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Term


def eliminate_d(monoid, streams, indices, body):
    indices = [ix.op for ix in indices]
    fresh_indices = [defop(ix, name=f"fresh_{ix}") for ix in indices]

    fresh_streams = {
        ix: jax_getitem(streams[ix], (fresh_ix(), None))
        for ix, fresh_ix in zip(indices, fresh_indices, strict=False)
    }
    new_reduce = reduce(monoid, streams | fresh_streams, body)
    return bind_dims(new_reduce, *fresh_indices)


class ReduceEliminateDterm(ObjectInterpretation):
    """Eliminates D-terms from a reduce."""

    @implements(reduce)
    def reduce(self, monoid, streams, body):
        # Check if the body is a D term.
        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        # Check if the indices of the D term are stream variables.
        if not all(ix.op in streams for indices, _ in body.args for ix in indices):
            return fwd()

        new_reduces = (eliminate_d(monoid, streams, *args) for args in body.args)
        return functools.reduce(monoid, new_reduces)
