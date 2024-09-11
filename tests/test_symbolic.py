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
    return fwd(None, mangled_var, handler({var: mangled_var})(evaluate)(body))


def eta_lam(var: Operation, body: Term):
    """eta reduction"""

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

    if var not in _fvs(body):
        return body
    else:
        return fwd(None)


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


free = {
    apply: lambda _, op, *a, **k: Term(op, a, tuple(k.items())),
    Add: lambda x, y: Term(Add, (x, y), ()),
    App: lambda f, arg: Term(App, (f, arg), ()),
    Lam: lambda var, body: Term(Lam, (var, body), ()),
}
lazy = coproduct(
    {Lam: alpha_lam},
    {Lam: eta_lam},
)
eager = {
    Add: eager_add,
    App: eager_app,
}


def test_lambda_calculus_1():

    x, y, z = gensym(object), gensym(object), gensym(object)

    with handler(free), handler(lazy), handler(eager):
        f1 = Lam(x, Add(x(), 1))
        assert App(f1, 1) == 2
        assert Lam(y, f1) == f1

        f2 = Lam(x, Lam(y, Add(x(), y())))
        assert App(App(f2, 1), 2) == 3
        assert Lam(y, f2) == f2

        app2 = Lam(z, Lam(x, Lam(y, App(App(z(), x()), y()))))
        assert App(App(App(app2, f2), 1), 2) == 3

        compose = Lam(x, Lam(y, Lam(z, App(x(), App(y(), z())))))
        f1_twice = App(App(compose, f1), f1)
        assert App(f1_twice, 1) == 3
