from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Term


class StackIndex(ObjectInterpretation):
    """
    Implements the identity
        jnp.stack([x0(), x1(), ... xn()])[i] = xi()
    """

    @implements(jax_getitem)
    def jax_getitem(self, arr, ixs):
        match ixs:
            case (ix,) if not isinstance(ix, Term):
                match arr:
                    case Term(jnp.stack, arg):
                        return arg[0][ix]
        return fwd()
