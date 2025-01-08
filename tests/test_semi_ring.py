import random

from docs.source.semi_ring import App, Dict, Field, Let, Sum, eager, opt
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop, defterm


@defterm
def add1(v: int) -> int:
    return v + 1


def test_simple_sum():
    x = defop(object)
    y = defop(object)
    k = defop(object)
    v = defop(object)

    with handler(eager):
        e = Sum(Dict("a", 1, "b", 2), k, v, Dict("v", v()))
        assert e["v"] == 3

    with handler(eager):
        e = Let(Dict("a", 1, "b", 2), x, Field(x(), "b"))
        assert e == 2

    with handler(eager):
        e = Sum(Dict("a", 1, "b", 2), k, v, Dict(k(), App(add1, App(add1, v()))))
        assert e["a"] == 3
        assert e["b"] == 4

    with handler(eager), handler(opt):
        e = Let(
            Dict("a", 1, "b", 2),
            x,
            Let(
                Sum(x(), k, v, Dict(k(), App(add1, v()))),
                y,
                Sum(y(), k, v, Dict(k(), App(add1, v()))),
            ),
        )
        assert e["a"] == 3
        assert e["b"] == 4


def fusion_test(d):
    x = defop(object)
    y = defop(object)
    k = defop(object)
    v = defop(object)

    return (
        Let(
            d,
            x,
            Let(
                Sum(x(), k, v, Dict(k(), App(add1, v()))),
                y,
                Sum(y(), k, v, Dict(k(), App(add1, v()))),
            ),
        ),
        (x, y, k, v),
    )


def make_dict(n):
    kv = []
    for i in range(n):
        kv.append(i)
        kv.append(random.randint(1, 10))
    return Dict(*kv)


def test_fusion_term():
    d = defop(object)
    with handler(opt):
        result, (x, _, k, v) = fusion_test(d)
    assert result == Let(
        d,
        x,
        Sum(x(), k, v, Dict(k(), App(add1, App(add1, v())))),
    )


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
