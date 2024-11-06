from effectful.internals.sugar import gensym
from effectful.ops.core import Operation, Term


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


def test_gensym_operation():
    def f(x: int) -> int:
        return x

    def g(x: int) -> int:
        return x

    @Operation
    def op(x: int) -> int:
        return x

    assert gensym(f) != f != Operation(f)

    assert gensym(f) != gensym(g) != gensym(f)

    assert gensym(f).__name__ == f.__name__
    assert gensym(f, name="f2").__name__ == "f2"

    assert gensym(op) != op
    assert gensym(op).__name__ == op.__name__

    f_sym = gensym(f)
    t = f_sym(0)
    assert isinstance(t, Term)
    assert t.op == f_sym
    assert t.args == (0,)
