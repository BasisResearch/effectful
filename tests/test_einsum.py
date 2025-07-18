import itertools
import operator
from functools import reduce

import effectful.handlers.jax.numpy as jnp
import jax
import pytest
from effectful.handlers.jax import jax_getitem
from effectful.ops.semantics import Operation, evaluate, handler
from effectful.ops.syntax import deffn, defop

from tests.utils import get_fold_params
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
    get_fold_params(
        "baseline_d_intp",
        "baseline_reorder_intp",
        "jax_intp",
        "jax_d_intp",
        "jax_reorder_intp",
    ),
)


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
    intp,
):
    symbols, inputs, outputs = parse_equation(equation)
    symbol_ops: dict[str, Operation] = {
        symbol: defop(jax.Array, name=symbol) for symbol in symbols
    }
    operand_ops: dict[str, Operation] = {
        f"m{i}": defop(jax.Array, name=f"m{i}") for i, _ in enumerate(operands)
    }
    streams = {op: jnp.arange(sizes[name]) for name, op in symbol_ops.items()}
    output_indices = tuple(symbol_ops[name]() for name in outputs)
    input_terms = (
        jax_getitem(operand_ops[f"m{i}"](), tuple(symbol_ops[name]() for name in inputs))
        for i, inputs in enumerate(inputs)
    )
    input_term = reduce(operator.mul, input_terms)

    operand_intp = {
        operand_ops[f"m{i}"]: deffn(tensor) for i, tensor in enumerate(operands)
    }
    body = D((output_indices, input_term))
    with handler(intp), handler(operand_intp):
        return evaluate(Sum(streams, body))


@pytest.mark.parametrize("equation", EINSUM_EXAMPLES)
@parameterize_intp
def test_einsum_optimize(intp, equation: str):
    sizes, operands = make_einsum_example(equation)
    expected = jnp.einsum(equation, *operands)
    result = einsum_weighted(equation, operands, sizes, intp)
    assert jax.numpy.allclose(result, expected)
