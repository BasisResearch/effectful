from effectful.internals.sugar import Operation, Term, gensym
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
