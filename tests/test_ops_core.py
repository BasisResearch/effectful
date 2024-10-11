from effectful.internals.sugar import NoDefaultRule, Operation, gensym
from effectful.ops.core import ctxof, evaluate
from effectful.ops.handler import handler


def test_evaluate():
    @Operation
    def Nested(*args, **kwargs):
        raise NoDefaultRule

    x = gensym(object)
    y = gensym(object)
    t = Nested([{"a": y()}, x(), (x(), y())], x(), arg1={"b": x()})

    with handler({x: lambda: 1, y: lambda: 2}):
        assert evaluate(t) == Nested([{"a": 2}, 1, (1, 2)], 1, arg1={"b": 1})


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
