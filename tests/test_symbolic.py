import logging
import weakref
from typing import Callable, Type, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter
from effectful.ops.core import Context, Operation, Term, evaluate, gensym
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


CTXOF_RULES: weakref.WeakKeyDictionary[Operation, Callable[..., Context]]
CTXOF_RULES = weakref.WeakKeyDictionary()


def gensym_(t: Type[T]) -> Operation[[], T]:
    op = gensym(t)
    CTXOF_RULES[op] = lambda: set()
    return op


def ctxof(term: Term[T]) -> set[Operation[..., T]]:

    _scope = set()

    def make_scope_rule(op, ctxof_rule):

        def scope_rule(*args, **kwargs):
            bound = ctxof_rule(*args, **kwargs)
            _scope.add(op)
            for v in bound:
                _scope.remove(v)
            return fwd(None)

        return scope_rule

    with interpreter({op: make_scope_rule(op, rule) for op, rule in CTXOF_RULES.items()}):  # type: ignore
        evaluate(term)

    return _scope


@Operation
def Add(x: int, y: int) -> int:
    raise NotImplementedError


@Operation
def App(f: Term, arg: Term) -> Term:
    raise NotImplementedError


@Operation
def Lam(var: Operation, body: Term) -> Term:
    raise NotImplementedError


CTXOF_RULES[Add] = lambda x, y: set()
CTXOF_RULES[App] = lambda f, arg: set()
CTXOF_RULES[Lam] = lambda var, body: {var}


def alpha_lam(var: Operation, body: Term):
    """alpha reduction"""
    mangled_var = gensym_(object)
    return fwd(None, mangled_var, handler({var: mangled_var})(evaluate)(body))


def eta_lam(var: Operation, body: Term):
    """eta reduction"""
    if var not in ctxof(body):  # typeof(body).context:
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

    x, y, z = gensym_(object), gensym_(object), gensym_(object)

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
