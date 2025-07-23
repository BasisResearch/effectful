from effectful.handlers.jax import numpy as jnp
from effectful.ops.types import Operation, Term

from weighted.ops.semiring import (
    LinAlg,
    LogAlg,
    MaxAlg,
    MinAlg,
)


def mul_op(semiring):
    if semiring is LinAlg:
        return jnp.multiply
    elif semiring is MinAlg:
        return jnp.min
    elif semiring is MaxAlg:
        return jnp.max
    elif semiring is LogAlg:
        return jnp.add
    else:
        return None


def parse_terms(value: Term, op: Operation) -> list[Term]:
    if isinstance(value, Term) and value.op is op:
        return sum((parse_terms(arg, op) for arg in value.args), [])
    else:
        return [value]
