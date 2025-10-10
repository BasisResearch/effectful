from effectful.internals.product_n import argsof, productN
from effectful.ops.semantics import apply, coproduct, evaluate, handler
from effectful.ops.syntax import defop
from effectful.ops.types import Interpretation, NotHandled


def test_simul_analysis():
    @defop
    def plus1(x: int) -> int:
        raise NotImplementedError

    @defop
    def plus2(x: int) -> int:
        raise NotImplementedError

    @defop
    def times(x: int, y: int) -> int:
        raise NotImplementedError

    x, y = defop(int, name="x"), defop(int, name="y")

    typ = defop(Interpretation, name="typ")
    value = defop(Interpretation, name="value")

    type_rules = {
        plus1: lambda x: int,
        plus2: lambda x: int,
        times: lambda x, y: int,
        x: lambda: int,
        y: lambda: int,
    }

    def plus1_value(x):
        return x + 1

    def plus2_value(x):
        return plus1(plus1(x))

    def times_value(x, y):
        if typ() is int and argsof(typ)[0][0] is int:
            return x * y
        raise TypeError("unexpected type!")

    value_rules = {
        plus1: plus1_value,
        plus2: plus2_value,
        times: times_value,
        x: lambda: 3,
        y: lambda: 4,
    }

    analysisN = productN({typ: type_rules, value: value_rules})

    def f1():
        v1 = x()  # {typ: lambda: int, val: lambda: 3}
        v2 = y()  # {typ: lambda: int, val: lambda: 4}
        v3 = plus2(v1)  # {typ: lambda: int, val: lambda: 5}
        v4 = times(v2, v3)  # {typ: lambda: int, val: lambda: 20}
        v5 = plus1(v4)  # {typ: lambda: int, val: lambda: 21}
        return v5  # {typ: lambda: int, val: lambda: 21}

    with handler(analysisN):
        i = f1()
        t = i.values(typ)
        v = i.values(value)
        assert t is int
        assert v == 21


def test_simul_analysis_apply():
    @defop
    def plus1[T](x: T) -> T:
        raise NotHandled

    @defop
    def plus2[T](x: T) -> T:
        raise NotHandled

    @defop
    def times[T](x: T, y: T) -> T:
        raise NotHandled

    x, y = defop(int, name="x"), defop(int, name="y")

    typ = defop(Interpretation, name="typ")
    value = defop(Interpretation, name="value")

    def apply_type(op, *a, **k):
        return op.__type_rule__(*a, **k)

    type_rules = {apply: apply_type}

    def plus1_value(x):
        return x + 1

    def plus2_value(x):
        return plus1(plus1(x))

    def times_value(x, y):
        if typ() is int and argsof(typ)[0][0] is int:
            return x * y
        raise TypeError("unexpected type!")

    value_rules = {
        plus1: plus1_value,
        plus2: plus2_value,
        times: times_value,
        x: lambda: 3,
        y: lambda: 4,
    }

    analysisN = productN({typ: type_rules, value: value_rules})

    def f1():
        v1 = x()  # {typ: lambda: int, val: lambda: 3}
        v2 = y()  # {typ: lambda: int, val: lambda: 4}
        v3 = plus2(v1)  # {typ: lambda: int, val: lambda: 5}
        v4 = times(v2, v3)  # {typ: lambda: int, val: lambda: 20}
        v5 = plus1(v4)  # {typ: lambda: int, val: lambda: 21}
        return v5  # {typ: lambda: int, val: lambda: 21}

    with handler(analysisN):
        i = f1()
        t = i.values(typ)
        v = i.values(value)
        assert t is int
        assert v == 21


def test_productN_distributive():
    """Test that productN distributes over coproducts."""

    @defop
    def add[T](x: T, y: T) -> T:
        raise NotHandled

    x = defop(object, name="x")
    i = defop(object, name="i")
    s = defop(object, name="s")

    intp1 = {add: lambda x, y: x + y}
    intp2 = {x: lambda: 1}
    intp3 = {x: lambda: "a"}

    term = add(x(), x())

    prod_intp1 = productN({i: coproduct(intp2, intp1), s: coproduct(intp3, intp1)})
    prod_intp2 = coproduct(
        productN({i: intp2, s: intp3}), productN({i: intp1, s: intp1})
    )
    result1 = evaluate(term, intp=prod_intp1)
    result2 = evaluate(term, intp=prod_intp2)

    assert result1.values(i) == result2.values(i) == 2
    assert result1.values(s) == result2.values(s) == "aa"
