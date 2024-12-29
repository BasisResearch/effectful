from typing import Annotated, Callable, TypeVar

from effectful.ops.semantics import fvsof
from effectful.ops.syntax import Bound, NoDefaultRule, defop
from effectful.ops.types import Operation, Term


def test_always_fresh():
    x = defop(int)
    y = defop(int)
    assert x != y

    x = defop(int, name="x")
    y = defop(int, name="y")
    assert x != y
    assert x.__name__ == "x"
    assert y.__name__ == "y"

    x1 = defop(int, name="x")
    x2 = defop(int, name="x")
    assert x1 != x2
    assert x1.__name__ == "x"
    assert x2.__name__ == "x"


def f(x: int) -> int:
    return x


def test_gensym_operation():
    def g(x: int) -> int:
        return x

    assert defop(f) != f != defop(f)

    assert defop(f) != defop(g) != defop(f)

    assert defop(f).__name__ == f.__name__
    assert defop(f, name="f2").__name__ == "f2"


def test_gensym_operation_2():
    @defop
    def op(x: int) -> int:
        return x

    # passing an operation to gensym should return a new operation, but preserve the name
    g_op = defop(op)
    assert g_op != op
    assert g_op.__name__ == op.__name__

    # the new operation should have no default rule
    f_sym = defop(f)
    t = f_sym(0)
    assert isinstance(t, Term)
    assert t.op == f_sym
    assert t.args == (0,)


def test_gensym_annotations():
    """Test that gensym respects annotations."""
    S, T = TypeVar("S"), TypeVar("T")

    @defop
    def Lam(var: Annotated[Operation[[], S], Bound()], body: T) -> Callable[[S], T]:
        raise NoDefaultRule

    x = defop(int)
    y = defop(int)
    lam = defop(Lam)

    assert x not in fvsof(Lam(x, x()))
    assert y in fvsof(Lam(x, y()))

    # binding annotations must be preserved for ctxof to work properly
    assert x not in fvsof(lam(x, x()))
    assert y in fvsof(lam(x, y()))
