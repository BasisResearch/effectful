import itertools
from functools import reduce
from operator import mul

import effectful.handlers.jax.numpy as jnp
import jax
import pytest
from effectful.handlers.jax import jax_getitem
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import deffn, defop

from tests.utils import (
    DEFAULT_TEST_FOLD_INTP,
)
from weighted.handlers.jax import D
from weighted.ops.sugar import Sum

EINSUM_EXAMPLES = [
    # vector operations
    "i->i",  # do nothing
    "i->",  # vector sum
    ",i->i",  # scalar-vector product
    "i,i->",  # inner product
    "i,j->ij",  # outer product
    "i,i->i",  # element-wise product
    # matrix operations
    "ij->ij",  # do nothing
    "ij->ji",  # matrix transpose
    "ii->",  # matrix trace
    "ii->i",  # matrix diagonal
    ",ij->ij",  # scalar-matrix product
    "ij,j->i",  # matrix-vector product
    "ij,ij->ij",  # hadamard product
    "ij,jk->ik",  # matrix-matrix product
    # composite contractions
    "ab,a->",
    "a,a,a,ab->ab",
    "ab,bc,cd->da",
    "ai->i",
    ",ai,abij->ij",
    "a,ai,bij->ij",
    "ai,abi,bci,cdi->i",
    "aij,abij,bcij->ij",
    "a,abi,bcij,cdij->ij",
]


parameterize_intp = pytest.mark.parametrize(
    "intp",
    [pytest.param(intp, id=name) for name, intp in DEFAULT_TEST_FOLD_INTP.items()],
)


def get_op(name: str) -> tuple:
    name = str(name)
    return name, defop(jax.Array, name=name)()  # type: ignore


def parse_equation(equation: str):
    symbols = tuple(sorted(set(equation) - set(" ,->")))
    inputs, outputs = equation.split("->")
    return symbols, inputs.split(","), outputs


def make_einsum_example(equation: str, sizes: tuple[int, ...] = (3, 5)):
    key = jax.random.PRNGKey(0)
    symbols, inputs, outputs = parse_equation(equation)
    dim_sizes = dict(zip(symbols, itertools.cycle(sizes)))
    operands = []
    for dims in inputs:
        shape = tuple(dim_sizes[dim] for dim in dims)
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=shape)
        operands.append(x)

    return dim_sizes, operands


def einsum_weighted(
    equation: str,
    operands: list[jax.Array],
    sizes: dict[str, tuple[int, ...]],
    fold_intp,
):
    symbols, inputs, outputs = parse_equation(equation)
    operands_map = {f"m{i}": tensor for i, tensor in enumerate(operands)}
    symbol_ops = dict(map(get_op, symbols))
    operand_ops = dict(map(get_op, operands_map))

    streams = {s.op: jnp.arange(sizes[name]) for name, s in symbol_ops.items()}
    output_indices = tuple(symbol_ops[name] for name in outputs)
    input_terms = (
        jax_getitem(operand_op, tuple(symbol_ops[s] for s in inp))
        for operand_op, inp in zip(operands_map.values(), inputs, strict=False)
    )
    input_term = reduce(mul, input_terms)

    operand_intp = {operand_ops[name].op: deffn(o) for name, o in operands_map.items()}
    body = D((output_indices, input_term))
    with handler(fold_intp), handler(operand_intp):
        return evaluate(Sum(streams, body))


@pytest.mark.parametrize("equation", EINSUM_EXAMPLES)
@parameterize_intp
def test_einsum_optimize(intp, equation: str):
    sizes, operands = make_einsum_example(equation)
    expected = jnp.einsum(equation, *operands)
    result = einsum_weighted(equation, operands, sizes, intp)
    assert jax.numpy.allclose(result, expected)
