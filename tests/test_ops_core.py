from effectful.internals.sugar import Operation, gensym
from effectful.ops.core import evaluate
from effectful.ops.handler import handler


def test_evaluate():
    @Operation
    def Nested(*args, **kwargs):
        pass

    free = {Nested: Nested.__free_rule__}

    x = gensym(object)
    y = gensym(object)
    with handler(free):
        t = Nested([{x: y}, x, (x, y)], x, arg1={y: x})

    with handler({x: lambda: 1, y: lambda: 2}):
        assert evaluate(t) == Nested([{1: 2}, 1, (1, 2)], 1, arg1={2: 1})


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
