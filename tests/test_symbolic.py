import functools
import logging
from typing import TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter
from effectful.ops.core import Operation, Term, apply, evaluate, gensym
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def test_lazy_1():

    @Operation
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    def eager_add(x, y):
        match x, y:
            case int(_), int(_):
                return x + y
            case _:
                return fwd(None)

    eager = {Add: eager_add}

    def simplify_add(x, y):
        match x, y:
            case Term(_, _, _), int(_):
                return Add(y, x)
            case _, Term(_, (a, b), ()):
                return Add(Add(x, a), b)
            case _:
                return fwd(None)

    simplify_assoc_commut = {Add: simplify_add}

    def unit_add(x, y):
        match x, y:
            case _, 0:
                return x
            case 0, _:
                return y
            case _:
                return fwd(None)

    simplify_unit = {Add: unit_add}

    lazy = {Add: lambda x, y: Term(Add, (x, y), ())}
    mixed = coproduct(lazy, eager)
    simplified = coproduct(simplify_assoc_commut, simplify_unit)
    mixed_simplified = coproduct(mixed, simplified)

    x_, y_ = gensym(int), gensym(int)
    x, y = x_(), y_()
    zero, one, two, three, four = 0, 1, 2, 3, 4

    with interpreter(eager):
        assert Add(one, two) == three

    with interpreter(lazy):
        assert Add(one, two) == Term(Add, (one, two), ())
        assert Add(one, Add(two, three)) == Term(
            Add, (one, Term(Add, (two, three), ())), ()
        )
        assert Add(x, y) == Term(Add, (x, y), ())
        assert Add(x, one) != Add(y, one)

    with interpreter(mixed):
        assert Add(one, two) == three
        assert Add(Add(one, two), x) == Term(Add, (three, x), ())

    with interpreter(mixed_simplified):
        assert Add(one, two) == three
        assert Add(three, x) == Add(x, three)
        assert Add(Add(one, two), x) == Add(x, three)
        assert Add(one, Add(x, two)) == Add(x, three)
        assert Add(Add(x, one), two) == Add(x, three)
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

    def alpha_lam(var: Operation, body: Term):
        """alpha reduction"""
        mangled_var = gensym(object)
        rename = interpreter(
            {var: mangled_var, apply: lambda op, *a, **k: Term(op, a, tuple(k.items()))}
        )(evaluate)
        return fwd(None, mangled_var, rename(body))

    def eager_add(x, y):
        """integer addition"""
        match x, y:
            case int(_), int(_):
                return x + y
            case _:
                return fwd(None)

    def eager_app(f: Term, arg: Term | int):
        """beta reduction"""
        match f, arg:
            case Term(op, (var, body), ()), _ if op == Lam:
                return handler({var: lambda: arg})(evaluate)(body)  # type: ignore
            case _:
                return fwd(None)

    def _fvs(term) -> set[Operation]:
        match term:
            case Term(op, (var, body), ()) if op == Lam:
                return _fvs(body) - {var}  # type: ignore
            case Term(op, args, kwargs):
                return set().union(
                    *(_fvs(a) for a in (op, *args, *(v for _, v in kwargs)))
                )
            case op if isinstance(op, Operation):
                return {op}
            case _:
                return set()

    def eta_lam(var: Operation, body: Term):
        """eta reduction"""
        if var not in _fvs(body):
            return body
        else:
            return fwd(None)

    free = {
        op: functools.partial(lambda op, *a, **k: Term(op, a, tuple(k.items())), op)
        for op in (Add, App, Lam)
    }
    eager = {Add: eager_add, App: eager_app, Lam: eta_lam}
    eager_mixed = coproduct(coproduct(free, {Lam: alpha_lam}), eager)

    x, y, z = gensym(object), gensym(object), gensym(object)
    one, two, three = 1, 2, 3

    with interpreter(eager_mixed):
        f1 = Lam(x, Add(x(), one))
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
