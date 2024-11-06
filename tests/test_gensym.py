from typing import Annotated, Callable, TypeVar

from effectful.internals.sugar import gensym, Bound, NoDefaultRule
from effectful.ops.core import Operation, Term, ctxof


def test_always_fresh():
    x = gensym(int)
    y = gensym(int)
    assert x != y

    x = gensym(int, name="x")
    y = gensym(int, name="y")
    assert x != y
    assert x.__name__ == "x"
    assert y.__name__ == "y"

    x1 = gensym(int, name="x")
    x2 = gensym(int, name="x")
    assert x1 != x2
    assert x1.__name__ == "x"
    assert x2.__name__ == "x"


def f(x: int) -> int:
    return x


def test_gensym_operation():
    def g(x: int) -> int:
        return x

    assert gensym(f) != f != Operation(f)

    assert gensym(f) != gensym(g) != gensym(f)

    assert gensym(f).__name__ == f.__name__
    assert gensym(f, name="f2").__name__ == "f2"


def test_gensym_operation_2():
    @Operation
    def op(x: int) -> int:
        return x

    # passing an operation to gensym should return a new operation, but preserve the name
    g_op = gensym(op)
    assert g_op != op
    assert g_op.__name__ == op.__name__

    # the new operation should have no default rule
    f_sym = gensym(f)
    t = f_sym(0)
    assert isinstance(t, Term)
    assert t.op == f_sym
    assert t.args == (0,)


def test_gensym_annotations():
    """Test that gensym respects annotations."""
    S, T = TypeVar("S"), TypeVar("T")

    @Operation
    def Lam(var: Annotated[Operation[[], S], Bound()], body: T) -> Callable[[S], T]:
        raise NoDefaultRule

    x = gensym(int)
    y = gensym(int)
    lam = gensym(Lam)

    assert x not in ctxof(Lam(x, x()))
    assert y in ctxof(Lam(x, y()))

    # binding annotations must be preserved for ctxof to work properly
    assert x not in ctxof(lam(x, x()))
    assert y in ctxof(lam(x, y()))
