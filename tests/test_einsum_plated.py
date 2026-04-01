import functools
import itertools
import operator

import jax
import pyro.ops.contract
import pytest
from torch import tensor
from weighted.handlers.optimization import ReduceDistributeTerm, ReduceReorderReduction
from weighted.handlers.optimization.cartesian_product import (
    ReduceDistributeCartesianProduct,
    SplitCartesianProductReduce,
)
from weighted.ops.reduce import BaselineReduce
from weighted.ops.sugar import CartesianProd, Prod, Sum

from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import deffn, defop

PLATED_EINSUM_EXAMPLES = [
    ("i->", "i"),
    (",i->", "i"),
    ("ai->", "i"),
    ("a,ai->", "i"),
    ("a,abij,bi->", "ij"),
    ("ai,abi,bci,cdi->", "i"),
]

base_intp = BaselineReduce()
lift_intp = functools.reduce(
    coproduct,  # type: ignore
    (
        BaselineReduce(),
        ReduceDistributeCartesianProduct(),
        ReduceDistributeTerm(),
    ),
)
ground_intp = functools.reduce(
    coproduct,  # type: ignore
    (
        BaselineReduce(),
        ReduceReorderReduction(),
        SplitCartesianProductReduce(),
    ),
)


parameterize_intp = pytest.mark.parametrize(
    "intp",
    [
        pytest.param(base_intp, id="base"),
        pytest.param(lift_intp, id="lift"),
        pytest.param(ground_intp, id="ground"),
    ],
)


def parse_equation(equation: str, plates: str):
    symbols = tuple(sorted(set(equation) - set(" ,->") - set(plates)))
    inputs2, outputs = equation.split("->")
    assert len(outputs) == 0
    inputs = inputs2.split(",")

    # for each var symbol, which plates are shared
    shared_vars = {
        s: set.intersection(*(set(inp) for inp in inputs if s in inp)) for s in symbols
    }
    plate_func = {s: tuple(p for p in plates if p in shared_vars[s]) for s in symbols}
    # separate var dims and plate dims per input operand
    inputs_split = [
        (
            "".join(x for x in inp if x not in plates),
            "".join(x for x in inp if x in plates),
        )
        for inp in inputs
    ]
    return symbols, inputs_split, outputs, plate_func


def make_plated_einsum_example(equation: str, plates: str, sizes=(2, 3)):
    key = jax.random.PRNGKey(0)
    symbols, inputs, outputs, plate_func = parse_equation(equation, plates)
    var_sizes = dict(zip(symbols, itertools.cycle(reversed(sizes))))
    plate_sizes = dict(zip(plates, itertools.cycle(sizes)))

    operands = []
    for dims, plts in inputs:
        var_shape = tuple(var_sizes[d] for d in dims)
        plate_shape = tuple(plate_sizes[p] for p in plts)
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=var_shape + plate_shape)
        operands.append(x)

    return var_sizes, plate_sizes, operands


def einsum_plated(
    equation, plate_symbols, operands, var_sizes, plate_sizes, reduce_intp
):
    symbols, inputs, outputs, plate_func = parse_equation(equation, plate_symbols)
    symbols = tuple(var_sizes.keys())
    symbol_ops = {s: defop(jax.Array, name=s) for s in symbols}
    plate_ops = {p: defop(jax.Array, name=p) for p in plate_sizes}
    operand_ops = {
        f"m{i}": defop(jax.Array, name=f"m{i}") for i in range(len(operands))
    }

    input_terms = []
    for (dims, plts), operand_op in zip(inputs, operand_ops.values(), strict=False):
        var_ixs = tuple(
            jax_getitem(symbol_ops[d](), tuple(plate_ops[p2]() for p2 in plate_func[d]))
            if plate_func[d]
            else symbol_ops[d]()
            for d in dims
        )
        plate_ixs = tuple(plate_ops[p]() for p in plts)
        term = jax_getitem(operand_op(), var_ixs + plate_ixs)
        plate_streams = {plate_ops[p]: jnp.arange(plate_sizes[p]) for p in plts}
        if len(plate_streams):
            term = Prod(plate_streams, term)
        input_terms.append(term)

    var_streams = {}
    for s in symbols:
        plate_streams = {
            plate_ops[p]: jnp.arange(plate_sizes[p]) for p in plate_func[s]
        }
        var_stream = jnp.arange(var_sizes[s])
        var_streams[symbol_ops[s]] = CartesianProd(plate_streams, var_stream)

    expr = functools.reduce(operator.mul, input_terms)
    if len(var_streams):
        expr = Sum(var_streams, expr)

    operand_intp = {
        operand_ops[name]: deffn(arr)
        for name, arr in zip(operand_ops, operands, strict=False)
    }

    with handler(operand_intp), handler(reduce_intp):
        return evaluate(expr)


@parameterize_intp
@pytest.mark.parametrize("equation,plate_symbols", PLATED_EINSUM_EXAMPLES)
def test_plated_einsum_optimize(intp, equation: str, plate_symbols: str):
    var_sizes, plate_sizes, operands = make_plated_einsum_example(
        equation, plate_symbols
    )
    result = einsum_plated(
        equation, plate_symbols, operands, var_sizes, plate_sizes, intp
    )

    torch_operands = (tensor(arr) for arr in operands)
    expected = pyro.ops.contract.einsum(equation, *torch_operands, plates=plate_symbols)
    expected = expected[0].numpy()

    assert jax.numpy.allclose(result, expected)
