from typing import Callable

from effectful.ops.syntax import syntactic_eq
from effectful.ops.types import Operation
from effectful.ops.weighted.monoid import (
    CommutativeMonoid,
    IdempotentMonoid,
    Max,
    Min,
    Product,
    Sum,
)


def define_vars(*names, typ=int):
    if len(names) == 1:
        return Operation.define(typ, name=names[0])
    return tuple(Operation.define(typ, name=n) for n in names)


def test_plus_single():
    x = define_vars("x")
    assert syntactic_eq(Sum.plus(x()), x())


def test_plus_identity():
    x = define_vars("x")
    assert syntactic_eq(Sum.plus(x(), Sum.identity), x())
    assert syntactic_eq(Sum.plus(Sum.identity, x()), x())


def test_plus_plus():
    (x, y, z) = define_vars("x", "y", "z")
    assert syntactic_eq(Sum.plus(x(), Sum.plus(y(), z())), Sum.plus(x(), y(), z()))
    assert syntactic_eq(Sum.plus(Sum.plus(x(), y()), z()), Sum.plus(x(), y(), z()))


def test_plus_sequence():
    (a, b, c, d) = define_vars("a", "b", "c", "d")
    assert syntactic_eq(
        Sum.plus([a(), b()], [c(), d()]), [Sum.plus(a(), c()), Sum.plus(b(), d())]
    )


def test_plus_mapping():
    (a, b, c, d) = define_vars("a", "b", "c", "d")
    assert syntactic_eq(
        Sum.plus({"x": a(), "y": b()}, {"x": c(), "z": d()}),
        {"x": Sum.plus(a(), c()), "y": b(), "z": d()},
    )


def test_plus_distributes():
    (a, b, c, d) = define_vars("a", "b", "c", "d")
    assert syntactic_eq(
        Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d())),
        Sum.plus(
            Product.plus(a(), c()),
            Product.plus(a(), d()),
            Product.plus(b(), c()),
            Product.plus(b(), d()),
        ),
    )


def test_plus_distributes_multiple():
    (a, b, c, d) = define_vars("a", "b", "c", "d")
    assert syntactic_eq(
        Sum.plus(
            Min.plus(a(), b()),
            Min.plus(c(), d()),
            Max.plus(a(), b()),
            Max.plus(c(), d()),
        ),
        Sum.plus(
            Min.plus(
                Sum.plus(a(), c()),
                Sum.plus(a(), d()),
                Sum.plus(b(), c()),
                Sum.plus(b(), d()),
            ),
            Max.plus(
                Sum.plus(a(), c()),
                Sum.plus(a(), d()),
                Sum.plus(b(), c()),
                Sum.plus(b(), d()),
            ),
        ),
    )


def test_plus_idempotent():
    (a, b, identity) = define_vars("a", "b", "identity")

    IdMonoid = IdempotentMonoid(
        kernel=Operation.define(Callable[[int, int], int]), identity=identity()
    )

    assert syntactic_eq(IdMonoid.plus(a(), a(), b()), IdMonoid.plus(a(), b()))
    assert syntactic_eq(IdMonoid.plus(a(), b(), a()), IdMonoid.plus(a(), b(), a()))
    assert syntactic_eq(
        IdMonoid.plus(a(), b(), a(), b(), b(), a(), a()),
        IdMonoid.plus(a(), b(), a(), b(), a()),
    )


def test_plus_commutative_idempotent():
    (a, b) = define_vars("a", "b")

    assert syntactic_eq(Min.plus(a(), a(), b()), Min.plus(a(), b()))
    assert syntactic_eq(Min.plus(b(), a(), b()), Min.plus(b(), a()))
    assert syntactic_eq(Min.plus(a(), b(), a(), b(), b(), a(), a()), Min.plus(a(), b()))


def test_plus_zero():
    a = define_vars("a")
    assert syntactic_eq(Product.plus(a(), Product.zero), Product.zero)
    assert syntactic_eq(Product.plus(Product.zero, a()), Product.zero)


def test_reduce_body_int():
    a, b, c, x = define_vars("a", "b", "c", "x")
    assert syntactic_eq(Sum.reduce({x: [a(), b(), c()]}, x()), Sum.plus(a(), b(), c()))


def test_reduce_body_sequence():
    x = Operation.define(int)
    X = Operation.define(list[int])
    f, g = define_vars("f", "g", typ=Callable[[int], int])

    assert syntactic_eq(
        Sum.reduce({x: X()}, [f(x()), g(x())]),
        [Sum.reduce({x: X()}, f(x())), Sum.reduce({x: X()}, g(x()))],
    )


def test_reduce_body_sequence_2():
    x, y = define_vars("x", "y")
    X, Y = define_vars("X", "Y", typ=list[int])
    f, g = define_vars("f", "g", typ=Callable[[int], int])

    assert syntactic_eq(
        Sum.reduce({x: X(), y: Y()}, [f(x()), g(y())]),
        [
            Sum.reduce({x: X(), y: Y()}, [f(x())]),
            Sum.reduce({x: X(), y: Y()}, [g(y())]),
        ],
    )


def test_reduce_body_mapping():
    x = Operation.define(int)
    X = Operation.define(list[int])
    f, g = define_vars("f", "g", typ=Callable[[int], int])

    assert syntactic_eq(
        Sum.reduce({x: X()}, {"a": f(x()), "b": g(x())}),
        {"a": Sum.reduce({x: X()}, f(x())), "b": Sum.reduce({x: X()}, [g(x())])},
    )


def test_reduce_no_streams():
    a = define_vars("a")
    assert syntactic_eq(Sum.reduce({}, a()), Sum.identity)


def test_reduce_empty():
    a, b, c = define_vars("a", "b", "c")
    A, C = define_vars("A", "C", typ=list[int])
    assert syntactic_eq(Sum.reduce({a: A(), b: [], c: C(a())}, c()), Sum.identity)


def test_reduce_plus():
    a, b, A, B = define_vars("a", "b", "A", "B")
    assert syntactic_eq(
        Sum.reduce({a: A(), b: B()}, Sum.plus(a(), b())),
        Sum.plus(Sum.reduce({a: A()}, a()), Sum.reduce({b: B()}, b())),
    )


def test_reduce_reduce():
    a, b = define_vars("a", "b")
    A = Operation.define(list[int])
    f = Operation.define(Callable[[int], int])
    g = Operation.define(Callable[[int, int], int])

    assert syntactic_eq(
        Sum.reduce({a: A()}, Sum.reduce({b: f(A())}, g(a(), b()))),
        Sum.reduce({a: A(), b: f(A())}, g(a(), b())),
    )


def test_reduce_idempotent():
    a, b = define_vars("a", "b")
    A = Operation.define(list[int])
    assert syntactic_eq(Min.reduce({a: A()}, b()), b())


# def test_reduce_distr():
#     a, b = define_vars("a", "b")
#     A, B = define_vars("A", "B", typ=list[int])
#     assert syntactic_eq(Product.reduce({a: A()}, Sum({b: B(a())}, b())), ??)
