import pytest

import effectful.handlers.numbers  # noqa: F401
from effectful.internals.mgu import unify
from effectful.ops.syntax import defop, syntactic_eq


def test_mgu():
    x = defop(int, name="x")
    y = defop(int, name="y")
    z = defop(int, name="z")

    with unify(2, 2):
        pass

    with unify(x(), 2):
        assert syntactic_eq(x(), 2)

    with unify(x() + 2, y() + 2):
        assert syntactic_eq(x(), y())

    with unify(x(), y() + 2):
        assert syntactic_eq(x(), y() + 2)

    with unify((x() + y()) / 2, (z() + 3) / 2):
        assert syntactic_eq(x(), z())
        assert syntactic_eq(y(), 3)


@pytest.mark.xfail(raises=NotImplementedError)
def test_mgu_fail_ground():
    with unify(2, 3):
        pass


@pytest.mark.xfail(raises=NotImplementedError)
def test_mgu_fail():
    x = defop(int, name="x")
    y = defop(int, name="y")

    with unify(x() + 2, y()):
        pass


def test_unify_twice():
    x = defop(int, name="x")
    y = defop(int, name="y")

    with unify(x() / 2, y() / 2), unify(y(), 3):
        assert syntactic_eq(x(), 3)
        assert syntactic_eq(y(), 3)


@pytest.mark.xfail(raises=NotImplementedError)
def test_unify_twice_fail():
    x = defop(int, name="x")

    with unify(x(), 2), unify(x(), 3):
        pass
