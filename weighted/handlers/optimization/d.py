from functools import reduce

import effectful.handlers.numbers  # noqa: F401
from effectful.handlers.jax import bind_dims, jax_getitem
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import Term

from weighted.ops.distribution import D
from weighted.ops.fold import fold


def eliminate_d(monoid, streams, indices, body):
    indices = [ix.op for ix in indices]
    fresh_indices = [defop(ix, name=f"fresh_{ix}") for ix in indices]

    fresh_streams = {
        ix: jax_getitem(streams[ix], (fresh_ix(), None))
        for ix, fresh_ix in zip(indices, fresh_indices, strict=False)
    }
    new_fold = fold(monoid, streams | fresh_streams, body)
    return bind_dims(new_fold, *fresh_indices)


class FoldEliminateDterm(ObjectInterpretation):
    """Eliminates D-terms from a fold."""

    @implements(fold)
    def fold(self, monoid, streams, body):
        # Check if the body is a D term.
        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        # Check if the indices of the D term are stream variables.
        if not all(ix.op in streams for indices, _ in body.args for ix in indices):
            return fwd()

        new_folds = (eliminate_d(monoid, streams, *args) for args in body.args)
        return reduce(monoid, new_folds)
