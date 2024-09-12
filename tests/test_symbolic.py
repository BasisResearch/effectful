import dataclasses
import logging
import weakref
from typing import Callable, Generic, Type, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter
from effectful.ops.core import Context, Operation, Term, evaluate, gensym
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@dataclasses.dataclass(frozen=True, eq=True, repr=True, unsafe_hash=True)
class TypeInContext(Generic[T]):
    context: Context[T, Type[T]]
    type: Type[T]


TYPEOF_RULES: weakref.WeakKeyDictionary[Operation, Callable[..., TypeInContext]]
TYPEOF_RULES = weakref.WeakKeyDictionary()


def gensym_(t: Type[T]) -> Operation[[], T]:
    op = gensym(t)
    TYPEOF_RULES[op] = lambda: TypeInContext({op: t}, t)
    return op


def typeof(term: Term[T]) -> TypeInContext[T]:
    with interpreter(TYPEOF_RULES):
        return evaluate(term)  # type: ignore


@Operation
def Add(x: int, y: int) -> int:
    raise NotImplementedError


@Operation
def App(f: Term, arg: Term) -> Term:
    raise NotImplementedError


@Operation
def Lam(var: Operation, body: Term) -> Term:
    raise NotImplementedError


TYPEOF_RULES[Add] = lambda x, y: TypeInContext(
    {
        **(x.context if isinstance(x, TypeInContext) else {}),
        **(y.context if isinstance(y, TypeInContext) else {}),
    },
    int,
)

TYPEOF_RULES[App] = lambda f, arg: TypeInContext(
    {**f.context, **(arg.context if isinstance(arg, TypeInContext) else {})}, f.type
)

TYPEOF_RULES[Lam] = lambda var, body: TypeInContext(
    {v: t for v, t in body.context.items() if v != var}, body.type
)


def alpha_lam(var: Operation, body: Term):
    """alpha reduction"""
    mangled_var = gensym_(object)
    return fwd(None, mangled_var, handler({var: mangled_var})(evaluate)(body))


def eta_lam(var: Operation, body: Term):
    """eta reduction"""
    if var not in typeof(body).context:
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
