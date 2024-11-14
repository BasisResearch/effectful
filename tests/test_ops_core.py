import pytest

from effectful.internals.sugar import NoDefaultRule, Operation, gensym
from effectful.ops.core import ctxof, evaluate, Term
from effectful.ops.handler import handler


@pytest.mark.xfail
def test_evaluate():
    @Operation
    def Nested(*args, **kwargs):
        raise NoDefaultRule

    x = gensym(object)
    y = gensym(object)
    t = Nested([{"a": y()}, x(), (x(), y())], x(), arg1={"b": x()})

    with handler({x: lambda: 1, y: lambda: 2}):
        assert evaluate(t) == Nested([{1: 2}, 1, (1, 2)], 1, arg1={2: 1})


def test_evaluate_2():
    x = gensym(int, name="x")
    y = gensym(int, name="y")
    t = x() + y()
    assert isinstance(t, Term)
    with handler({x: lambda: 1, y: lambda: 2}):
        assert evaluate(t) == 3


def test_operation_metadata():
    """Test that Operation instances preserve decorated function metadata."""

    def f(x):
        """Docstring for f"""
        return x + 1

    f_op = Operation(f)
    ff_op = Operation(f)

    assert f.__doc__ == f_op.__doc__
    assert f.__name__ == f_op.__name__
    assert hash(f) == hash(f_op)
    assert f_op == ff_op


def test_ctxof():
    x = gensym(object)
    y = gensym(object)

    @Operation
    def Nested(*args, **kwargs):
        raise NoDefaultRule

    assert ctxof(Nested(x(), y())).keys() >= {x, y}
    assert ctxof(Nested([x()], y())).keys() >= {x, y}
    assert ctxof(Nested([x()], [y()])).keys() >= {x, y}
    assert ctxof(Nested((x(), y()))).keys() >= {x, y}
