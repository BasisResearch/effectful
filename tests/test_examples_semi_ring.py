import random
import typing

from docs.source.semi_ring import Dict, Field, Let, SemiRingDict, Sum, eager, ops, opt
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop, trace
from effectful.ops.types import Term

S = typing.TypeVar("S")
T = typing.TypeVar("T")


@trace
def add1(v: int) -> int:
    return v + 1


def test_simple_sum():
    x = defop(SemiRingDict[str, T], name="x")
    y = defop(SemiRingDict[str, T], name="y")
    k = defop(str, name="k")
    v = defop(int, name="v")

    with handler(eager):
        e = Sum(Dict(("a", 1), ("b", 2)), k, v, Dict(("v", v())))
        assert e["v"] == 3

    with handler(eager):
        e = Let(Dict(("a", 1), ("b", 2)), x, Field(x(), "b"))
        assert e == 2

    with handler(eager):
        e = Sum(Dict(("a", 1), ("b", 2)), k, v, Dict((k(), add1(add1(v())))))
        assert e["a"] == 3
        assert e["b"] == 4

    with handler(eager), handler(opt):
        e = Let(
            Dict(("a", 1), ("b", 2)),
            x,
            Let(
                Sum(x(), k, v, Dict((k(), add1(v())))),
                y,
                Sum(y(), k, v, Dict((k(), add1(v())))),
            ),
        )
        assert e["a"] == 3
        assert e["b"] == 4


def fusion_test(d):
    x = defop(SemiRingDict[S, T], name="x")
    y = defop(SemiRingDict[S, T], name="y")
    k = defop(str, name="k")
    v = defop(int, name="v")

    return (
        Let(
            d,
            x,
            Let(
                Sum(x(), k, v, Dict((k(), add1(v())))),
                y,
                Sum(y(), k, v, Dict((k(), add1(v())))),
            ),
        ),
        (x, y, k, v),
    )


def make_dict(n):
    return Dict(*[(i, random.randint(1, 10)) for i in range(n)])


def test_fusion_term():
    dvar = defop(SemiRingDict[str, T], name="dvar")
    with handler(eager), handler(opt):
        result, (x, _, k, v) = fusion_test(dvar())

    match result:
        case Term(ops.Sum, (_, _, _, Term(ops.Dict))):
            pass
        case _:
            assert False


def test_fusion_unopt(benchmark):
    @benchmark
    def run():
        with handler(eager):
            return fusion_test(make_dict(100))


def test_fusion_opt(benchmark):
    @benchmark
    def run():
        with handler(eager), handler(opt):
            return fusion_test(make_dict(100))
