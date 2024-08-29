import functools
import logging
from typing import TypeVar

from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_continuation
from effectful.ops.core import (
    BINDINGS,
    JUDGEMENTS,
    Operation,
    Term,
    TypeInContext,
    evaluate,
    gensym,
)
from effectful.ops.handler import coproduct, handler
from effectful.ops.interpreter import interpreter

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def test_lazy_1():

    @Operation
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    JUDGEMENTS[Add] = lambda x, y: TypeInContext(
        {
            **(x.context if isinstance(x, TypeInContext) else {}),
            **(y.context if isinstance(y, TypeInContext) else {}),
        },
        int,
    )

    @bind_continuation
    def eager_add(fwd, x, y):
        if not isinstance(x, Term) and not isinstance(y, Term):
            return x + y
        else:
            return fwd(None, x, y)

    eager = {Add: eager_add}

    @bind_continuation
    def simplify_add(fwd, x, y):
        if isinstance(x, Term) and not isinstance(y, Term):
            # x + c -> c + x
            return Add(y, x)
        elif isinstance(y, Term) and y.op == Add:
            # a + (b + c) -> (a + b) + c
            return Add(Add(x, y.args[0]), y.args[1])
        else:
            return fwd(None, x, y)

    simplify_assoc_commut = {Add: simplify_add}

    @bind_continuation
    def unit_add(fwd, x, y):
        if x == zero:
            return y
        elif y == zero:
            return x
        else:
            return fwd(None, x, y)

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
        assert Add(one, Add(two, three)) == Term(
            Add, (one, Term(Add, (two, three), {})), {}
        )
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

        assert (
            Add(Add(Add(x, x), Add(x, x)), Add(Add(x, x), Add(x, x)))
            == Add(Add(Add(Add(Add(Add(Add(x, x), x), x), x), x), x), x)
            == Add(x, Add(x, Add(x, Add(x, Add(x, Add(x, Add(x, x)))))))
        )

        assert Add(x, zero) == x


def test_bind_with_handler():

    @Operation
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    @Operation
    def App(f: Term, arg: Term) -> Term:
        raise NotImplementedError

    @Operation
    def Lam(var: Operation, body: Term) -> Term:
        raise NotImplementedError

    JUDGEMENTS[Lam] = lambda var, body: TypeInContext(
        {v: t for v, t in body.context.items() if v != var}, body.type
    )
    JUDGEMENTS[App] = lambda f, arg: TypeInContext(
        {**f.context, **(arg.context if isinstance(arg, TypeInContext) else {})}, f.type
    )
    JUDGEMENTS[Add] = lambda x, y: TypeInContext(
        {
            **(x.context if isinstance(x, TypeInContext) else {}),
            **(y.context if isinstance(y, TypeInContext) else {}),
        },
        int,
    )

    @bind_continuation
    def alpha_lam(fwd, var: Operation, body: Term):
        """alpha reduction"""
        if not getattr(var, "_fresh", False):
            mangled_var = gensym(object)
            setattr(mangled_var, "_fresh", True)
            return Lam(mangled_var, handler({var: mangled_var})(evaluate)(body))
        else:
            return fwd(None, var, body)

    BINDINGS[Lam] = alpha_lam

    @bind_continuation
    def eager_add(fwd, x, y):
        """integer addition"""
        if not isinstance(x, Term) and not isinstance(y, Term):
            return x + y
        else:
            return fwd(None, x, y)

    @bind_continuation
    def eager_app(fwd, f: Term, arg: Term | int):
        """beta reduction"""
        if f.op == Lam:
            var, body = f.args
            return handler({var: lambda: arg})(evaluate)(body)  # type: ignore
        else:
            return fwd(None, f, arg)

    @bind_continuation
    def eta_lam(fwd, var: Operation, body: Term):
        """eta reduction"""
        if var not in interpreter(JUDGEMENTS)(evaluate)(body).context:
            return body
        else:
            return fwd(None, var, body)

    free = {
        op: functools.partial(lambda op, *a, **k: Term(op, a, k), op)
        for op in (Add, App, Lam)
    }
    free_alpha = coproduct(free, BINDINGS)
    eager = {Add: eager_add, App: eager_app, Lam: eta_lam}
    eager_mixed = coproduct(free_alpha, eager)

    x, y, z = gensym(object), gensym(object), gensym(object)
    zero, one, two, three = 0, 1, 2, 3

    with interpreter(eager_mixed):
        f1 = Lam(x, Add(x(), one))
        assert handler({x: lambda: one})(evaluate)(f1) == f1
        assert handler({y: lambda: one})(evaluate)(f1) == f1
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