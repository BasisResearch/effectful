from effectful.internals.mgu import mgu
from effectful.ops.syntax import defop, syntactic_eq


def test_mgu():
    x = defop(int, name="x")
    y = defop(int, name="y")
    z = defop(int, name="z")

    match = mgu((x(), 2))
    assert syntactic_eq(match[x](), 2)

    match = mgu((x(), y()))
    assert syntactic_eq(match[x](), y())

    match = mgu((2, 2))
    assert match == {}

    match = mgu((2, 3))
    assert match is None

    match = mgu((x() + y(), z()))
    assert match is None

    match = mgu((x(), y() + 2))
    assert syntactic_eq(match[x](), y() + 2)

    match = mgu(((x() + y()) / 2, (z() + 3) / 2))
    assert syntactic_eq(match[x](), z())
    assert syntactic_eq(match[y](), 3)
