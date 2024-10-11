import collections.abc
import random
import types

from effectful.internals.sugar import NoDefaultRule, gensym
from effectful.ops.core import Operation, Term, evaluate
from effectful.ops.handler import fwd, handler


# https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be
class SemiRingDict(collections.abc.Mapping):
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __repr__(self):
        return f"SemiRingDict({repr(self._d)})"

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash


@Operation
def Add(x, y):
    raise NoDefaultRule


@Operation
def Sum(e1, k, v, e2):
    raise NoDefaultRule


@Operation
def Let(x, e1, e2):
    raise NoDefaultRule


@Operation
def Record(**kwargs):
    raise NoDefaultRule


@Operation
def Dict(*contents):
    raise NoDefaultRule


@Operation
def Field(record, key):
    raise NoDefaultRule


@Operation
def App(f, x):
    raise NoDefaultRule


@Operation
def Multiply(e1, e2):
    raise NoDefaultRule


@Operation
def Add(e1, e2):
    raise NoDefaultRule


ops = types.SimpleNamespace()
ops.Add = Add
ops.Sum = Sum
ops.Let = Let
ops.Record = Record
ops.Dict = Dict
ops.Field = Field
ops.App = App
ops.Multiply = Multiply
ops.Add = Add


def is_value(v):
    return not isinstance(v, (Operation, Term))


def eager_multiply(e1, e2):
    match e1, e2:
        case (int() | float()), (int() | float()):
            return e1 * e2
        case (int() | float()), _:
            return Multiply(e2, e1)
        case SemiRingDict(), (int() | float() | SemiRingDict()):
            return SemiRingDict({k: Multiply(v, e2) for k, v in e1.items()})
        case _:
            return fwd(None)


def eager_add_(e1, e2):
    match e1, e2:
        case (int() | float()), (int() | float()):
            return e1 + e2
        case (int() | float()), _:
            return Add(e2, e1)
        case SemiRingDict(), (int() | float()):
            return SemiRingDict({k: Add(v, e2) for k, v in e1.items()})
        case SemiRingDict(), SemiRingDict():
            new_dict = e1._d.copy()
            for key, value in e2.items():
                if key in new_dict:
                    new_dict[key] = Add(new_dict[key], value)
                else:
                    new_dict[key] = value
            return SemiRingDict(new_dict)
        case _:
            return fwd(None)


def eager_add(e1, e2):
    ret = eager_add_(e1, e2)
    print()
    print("Add", e1, e2, ret)
    return ret


def eager_dict(*contents):
    if not all(is_value(v) for v in contents):
        return fwd(None)

    if len(contents) % 2 != 0:
        raise ValueError("Dict requires an even number of arguments")

    kv = []
    for i in range(0, len(contents), 2):
        if not is_value(contents[i]):
            return fwd(None)
        kv.append((contents[i], contents[i + 1]))
    return SemiRingDict(kv)


def eager_record(**kwargs):
    if all(is_value(v) for v in kwargs.values()):
        return dict(**kwargs)
    return fwd(None)


def eager_app(f, x):
    if is_value(x):
        return f(x)
    return fwd(None)


def eager_field(r, k):
    match r, k:
        case dict(), str():
            return r[k]
        case SemiRingDict(), _ if is_value(k):
            return r[k]
        case _:
            return fwd(None)


def eager_sum(e1, k, v, e2):
    match e1, e2:
        case SemiRingDict(), b2:
            new_d = SemiRingDict()
            for key, value in e1.items():
                with handler({k: lambda: key, v: lambda: value}):
                    new_v = evaluate(b2)
                new_d = Add(new_d, new_v)
            return new_d
        case _:
            return fwd(None)


def eager_let(e1, x, e2):
    match e1, e2:
        case SemiRingDict(), Term():
            return handler({x: lambda: e1})(evaluate)(e2)
        case _, SemiRingDict():
            return e2
        case _:
            return fwd(None)


def loop_factorization(e1, e2):
    """
    sum(x in e1) e2 * f(x) -> e2 * sum(x in e1) f(x)
    sum(x in e1) f(x) * e2 -> (sum(x in e1) f(x)) * e2
    """
    pass


def vertical_fusion(e1, x, e2):
    match e1, e2:
        case (
            Term(
                ops.Sum,
                (
                    e_sum,
                    k1,
                    v1,
                    Term(ops.Dict, (Term(k1a), Term(ops.App, (f1, Term(v1a))))),
                ),
            ),
            Term(
                ops.Sum,
                (
                    Term(xa),
                    k2,
                    v2,
                    Term(ops.Dict, (Term(k2a), Term(ops.App, (f2, Term(v2a))))),
                ),
            ),
        ) if k1 == k1a and v1 == v1a and x == xa and k2 == k2a and v2 == v2a:
            return Sum(e_sum, k2, v2, Dict(k2(), App(f2, App(f1, v2()))))
        case _:
            return fwd(None)


free = {
    Add: Add.__default_rule__,
    Sum: Sum.__default_rule__,
    Let: Let.__default_rule__,
    Record: Record.__default_rule__,
    Dict: Dict.__default_rule__,
    Field: Field.__default_rule__,
    App: App.__default_rule__,
}

eager = {
    Add: eager_add,
    Dict: eager_dict,
    Record: eager_record,
    Sum: eager_sum,
    Field: eager_field,
    Let: eager_let,
    App: eager_app,
    Multiply: eager_multiply,
    Add: eager_add,
}

opt = {
    Let: vertical_fusion,
}


def test_multiply():
    with handler(free), handler(eager):
        # Scalar multiplication
        d1 = Dict("a", 2, "b", 3)
        result = Multiply(d1, 2)
        assert result["a"] == 4
        assert result["b"] == 6

        # Dictionary multiplication
        d2 = Dict("x", 4, "y", 5)
        result = Multiply(d1, d2)
        assert result["a"]["x"] == 8  # 2 * 4
        assert result["a"]["y"] == 10  # 2 * 5
        assert result["b"]["x"] == 12  # 3 * 4
        assert result["b"]["y"] == 15  # 3 * 5

        # Associativity
        d3 = Dict("p", 2, "q", 3)
        result1 = Multiply(Multiply(d1, d2), d3)
        result2 = Multiply(d1, Multiply(d2, d3))
        assert result1 == result2

        # Distributivity
        result1 = Multiply(d1, Sum(d2, gensym(object), gensym(object), d3))
        result2 = Sum(
            Multiply(d1, d2), gensym(object), gensym(object), Multiply(d1, d3)
        )
        assert result1 == result2


# def test_multiply_free():
#     with handler(free), handler(eager):
#         a = gensym(object)
#         d = Dict("x", 5, "y", a)
#         result = Multiply(d, d)
#         assert result["x"]["x"] == 25
#         assert result["x"]["y"] == Multiply(a, 5)
#         assert result["y"] == Multiply(a, d)


def test_simple_sum():
    x = gensym(object)
    y = gensym(object)
    k = gensym(object)
    v = gensym(object)

    with handler(free), handler(eager):
        e = Sum(Dict("a", 1, "b", 2), k, v, Dict("v", v()))
        print(e)
        assert e["v"] == 3

    with handler(free), handler(eager):
        e = Let(Dict("a", 1, "b", 2), x, Field(x(), "b"))
        assert e == 2

    def add1(v):
        return Add(v, 1)

    with handler(free), handler(eager):
        e = Sum(Dict("a", 1, "b", 2), k, v, Dict(k(), App(add1, App(add1, v()))))
        assert e["a"] == 3
        assert e["b"] == 4

    with handler(free), handler(eager), handler(opt):
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


def add1(v):
    return Add(v, 1)


def fusion_test(d):
    x = gensym(object)
    y = gensym(object)
    k = gensym(object)
    v = gensym(object)

    return Let(
        d,
        x,
        Let(
            Sum(x(), k, v, Dict(k(), App(add1, v()))),
            y,
            Sum(y(), k, v, Dict(k(), App(add1, v()))),
        ),
    )


def make_dict(n):
    kv = []
    for i in range(n):
        kv.append(i)
        kv.append(random.randint(1, 10))
    return Dict(*kv)


def test_fusion_unopt(benchmark):
    @benchmark
    def run():
        with handler(free), handler(eager):
            return fusion_test(make_dict(100))


def test_fusion_opt(benchmark):
    @benchmark
    def run():
        with handler(free), handler(eager), handler(opt):
            return fusion_test(make_dict(100))
