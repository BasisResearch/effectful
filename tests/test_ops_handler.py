import contextlib
import itertools
import logging
from typing import TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_result, value_or_result
from effectful.internals.sugar import ObjectInterpretation, implements
from effectful.ops.core import Interpretation, Operation, define
from effectful.ops.handler import coproduct, fwd, handler
from effectful.ops.interpreter import interpreter

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@define(Operation)
def plus_1(x: int) -> int:
    return x + 1


@define(Operation)
def plus_2(x: int) -> int:
    return x + 2


@define(Operation)
def times_plus_1(x: int, y: int) -> int:
    return x * y + 1


def defaults(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(value_or_result(op.default)) for op in ops}


def times_n_handler(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda v, *args, **kwargs: fwd(v) * n) for op in ops}


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_affine_continuation_compose(op, args):
    def f():
        return op(*args)

    h_twice = {op: bind_result(lambda v, *a, **k: fwd(fwd(v)))}

    assert (
        interpreter(defaults(op))(f)()
        == interpreter(coproduct(defaults(op), h_twice))(f)()
    )


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    intp1 = coproduct(h0, coproduct(h1, h2))
    intp2 = coproduct(coproduct(h0, h1), h2)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_commute_orthogonal(op, args, n1, n2):
    def f():
        return op(*args) + new_op(*args)

    new_op = define(Operation)(lambda *args: op(*args) + 3)

    h0 = defaults(op, new_op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, new_op)

    intp1 = coproduct(h0, h1, h2)
    intp2 = coproduct(h0, h2, h1)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_handler_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    expected = interpreter(coproduct(h0, h1, h2))(f)()

    with handler(h0), handler(h1), handler(h2):
        assert f() == expected

    with handler(coproduct(h0, h1)), handler(h2):
        assert f() == expected

    with handler(h0), handler(coproduct(h1, h2)):
        assert f() == expected


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_stop_without_fwd(op, args, n, depth):
    def f():
        return op(*args)

    expected = f()

    with contextlib.ExitStack() as stack:
        for _ in range(depth):
            stack.enter_context(handler(times_n_handler(n, op)))

        stack.enter_context(handler(defaults(op)))

        assert f() == expected


def test_sugar_subclassing():
    class ScaleBy(ObjectInterpretation):
        def __init__(self, scale):
            self._scale = scale

        @implements(plus_1)
        def plus_1(self, v):
            return v + self._scale

        @implements(plus_2)
        def plus_2(self, v):
            return v + 2 * self._scale

    class ScaleAndShiftBy(ScaleBy):
        def __init__(self, scale, shift):
            super().__init__(scale)
            self._shift = shift

        @implements(plus_1)
        def plus_1(self, v):
            return super().plus_1(v) + self._shift

        # plus_2 inhereted from ScaleBy

    with handler(ScaleBy(4)):
        assert plus_1(4) == 8
        assert plus_2(4) == 12

    with handler(ScaleAndShiftBy(4, 1)):
        assert plus_1(4) == 9
        assert plus_2(4) == 12


def test_lazy_1():
    from effectful.ops.core import JUDGEMENTS, Term, TypeInContext, gensym

    @Operation
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    JUDGEMENTS[Add] = lambda x, y: TypeInContext({**(x.context if isinstance(x, TypeInContext) else {}), **(y.context if isinstance(y, TypeInContext) else {})}, int)

    def eager_add(x, y):
        if not isinstance(x, Term) and not isinstance(y, Term):
            return x + y
        else:
            return fwd(None)

    eager = {Add: eager_add}

    def simplify_add(x, y):
        if isinstance(x, Term) and not isinstance(y, Term):
            # x + c -> c + x
            return Add(y, x)
        elif isinstance(y, Term) and y.op == Add:
            # a + (b + c) -> (a + b) + c
            return Add(Add(x, y.args[0]), y.args[1])
        else:
            return fwd(None)

    simplify_assoc_commut = {Add: simplify_add}

    def unit_add(x, y):
        if x == zero:
            return y
        elif y == zero:
            return x
        else:
            return fwd(None)

    simplify_unit = {Add: unit_add}

    lazy = {Add: lambda x, y: Term(Add, (x, y), {})}
    mixed = coproduct(lazy, eager)
    simplified = coproduct(simplify_assoc_commut, simplify_unit)
    mixed_simplified = coproduct(mixed, simplified)

    x_, y_, z_ = gensym(int), gensym(int), gensym(int)
    x, y, z = x_(), y_(), z_()
    zero, one, two, three, four = 0, 1, 2, 3, 4

    with interpreter(eager):
        assert Add(one, two) == three

    with interpreter(lazy):
        assert Add(one, two) == Term(Add, (one, two), {})
        assert Add(one, Add(two, three)) == Term(Add, (one, Term(Add, (two, three), {})), {})
        assert Add(x, y) == Term(Add, (x, y), {})
        assert Add(x, one) != Add(y, one)

    with interpreter(mixed):
        assert Add(one, two) == three
        assert Add(Add(one, two), x) == Term(Add, (three, x), {})

    with interpreter(mixed_simplified):
        assert Add(one, two) == three
        assert Add(three, x) == Add(x, three)
        assert Add(Add(one, two), x) == Add(x, three)
        assert Add(one, Add(x, two)) == Add(x, three)
        assert Add(Add(one, Add(y, one)), Add(one, Add(x, one))) == Add(Add(y, x), four)

        assert Add(one, Add(Add(x, y), two)) == Add(Add(x, y), three)
        assert Add(one, Add(Add(x, Add(y, one)), one)) == Add(Add(x, y), three)

        assert Add(Add(Add(x, x), Add(x, x)), Add(Add(x, x), Add(x, x))) == \
            Add(Add(Add(Add(Add(Add(Add(x, x), x), x), x), x), x), x) == \
            Add(x, Add(x, Add(x, Add(x, Add(x, Add(x, Add(x, x)))))))

        assert Add(x, zero) == x


def test_bind_with_handler():
    import functools
    from effectful.ops.core import Term, TypeInContext, evaluate, gensym, BINDINGS, JUDGEMENTS

    @Operation
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    @Operation
    def App(f: Term, arg: Term) -> Term:
        raise NotImplementedError

    @Operation
    def Lam(var: Operation, body: Term) -> Term:
        raise NotImplementedError
    
    JUDGEMENTS[Lam] = lambda var, body: TypeInContext({v: t for v, t in body.context.items() if v != var}, body.type)
    JUDGEMENTS[App] = lambda f, arg: TypeInContext({**f.context, **(arg.context if isinstance(arg, TypeInContext) else {})}, f.type)
    JUDGEMENTS[Add] = lambda x, y: TypeInContext({**(x.context if isinstance(x, TypeInContext) else {}), **(y.context if isinstance(y, TypeInContext) else {})}, int)

    def alpha_lam(var: Operation, body: Term):
        """alpha reduction"""
        if not getattr(var, "_fresh", False):
            mangled_var = gensym()
            mangled_var._fresh = True
            return Lam(mangled_var, interpreter({var: mangled_var})(evaluate)(body))
        else:
            return fwd(None)

    BINDINGS[Lam] = alpha_lam

    def eager_add(x, y):
        """integer addition"""
        if not isinstance(x, Term) and not isinstance(y, Term):
            return x + y
        else:
            return fwd(None)

    def eager_app(f: Term, arg: Term | int):
        """beta reduction"""
        if f.op == Lam:
            var, body = f.args
            return interpreter({var: lambda: arg})(evaluate)(body)
        else:
            return fwd(None)

    def eta_lam(var: Operation, body: Term):
        """eta reduction"""
        if var not in interpreter(JUDGEMENTS)(evaluate)(body).context:
            return body
        else:
            return fwd(None)

    free = {op: functools.partial(lambda op, *a, **k: Term(op, a, k), op) for op in (Add, App, Lam)}
    free_alpha = coproduct(free, BINDINGS)
    eager = {Add: eager_add, App: eager_app, Lam: eta_lam}
    eager_mixed = coproduct(free_alpha, eager)

    x, y, z = gensym(), gensym(), gensym()
    zero, one, two, three = 0, 1, 2, 3

    with interpreter(eager_mixed):
        f1 = Lam(x, Add(x(), one))
        assert interpreter({x: lambda: one})(evaluate)(f1) == f1
        assert interpreter({y: lambda: one})(evaluate)(f1) == f1
        assert App(f1, one) == two
        assert Lam(y, f1) == f1

        f2 = Lam(x, Lam(y, Add(x(), y())))
        assert App(App(f2, one), two) == three
        assert Lam(y, f2) == f2

        app2 = Lam(z, Lam(x, Lam(y, App(App(z(), x()), y()))))
        assert App(App(App(app2, f2), one), two) == three

        compose = Lam(x, Lam(y, Lam(z, App(x(), App(y(), z())))))
        f1_twice = App(App(compose, f1), f1)
        assert App(f1_twice, one) == three
