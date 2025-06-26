import functools
import itertools
import operator

import effectful.handlers.jax.numpy as jnp
import jax
import pytest
from effectful.handlers.jax import jax_getitem
from effectful.ops.semantics import Operation, handler
from effectful.ops.syntax import defop

from weighted.handlers.jax import D
from weighted.handlers.jax import interpretation as jax_intp
from weighted.ops.sugar import Sum

# taken from https://github.com/pyro-ppl/funsor/blob/master/test/test_einsum.py
EINSUM_EXAMPLES = [
    "a,b->",
    "ab,a->",
    "a,a->",
    "a,a->a",
    "a,a,a,ab->ab",
    "ab->ba",
    "ab,bc,cd->da",
    "i->i",
    ",i->i",
    "ai->i",
    ",ai,abij->ij",
    "a,ai,bij->ij",
    "ai,abi,bci,cdi->i",
    "aij,abij,bcij->ij",
    "a,abi,bcij,cdij->ij",
]


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
    equation: str, operands: list[jax.Array], sizes: dict[str, tuple[int, ...]]
):
    symbols, inputs, outputs = parse_equation(equation)
    ops: dict[str, Operation] = {
        symbol: defop(jax.Array, name=symbol) for symbol in symbols
    }
    streams = {op: jnp.arange(sizes[name]) for name, op in ops.items()}
    output_indices = tuple(ops[name]() for name in outputs)
    input_terms = (
        jax_getitem(tensor, tuple(ops[name]() for name in inputs))
        for tensor, inputs in zip(operands, inputs, strict=False)
    )
    input_term = functools.reduce(operator.mul, input_terms)

    with handler(jax_intp):
        return Sum(streams, D((output_indices, input_term)))


@pytest.mark.parametrize("equation", EINSUM_EXAMPLES)
def test_einsum(equation: str):
    sizes, operands = make_einsum_example(equation)
    expected = jnp.einsum(equation, *operands)
    result = einsum_weighted(equation, operands, sizes)
    assert result.shape == expected.shape
    assert jnp.allclose(result, expected)
