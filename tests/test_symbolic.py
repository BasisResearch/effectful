import functools
import logging
import operator
from typing import Generic, TypeVar

import wrapt
from typing_extensions import ParamSpec

from effectful.internals.runtime import get_runtime, interpreter
from effectful.ops.core import Operation, Term, TypeInContext, apply, evaluate, gensym
from effectful.ops.handler import coproduct, fwd, handler

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


def test_lazy_addition():

    class Box(Generic[T], wrapt.ObjectProxy):
        __wrapped__: Term[T] | T

        def __add__(self, other: T | Term[T] | "Box[T]") -> "Box[T]":
            return type(self)(
                Add(
                    self if not isinstance(self, Box) else self.__wrapped__,
                    other if not isinstance(other, Box) else other.__wrapped__,
                )
            )

        def __radd__(self, other: T | Term[T] | "Box[T]") -> "Box[T]":
            return type(self)(
                Add(
                    other if not isinstance(other, Box) else other.__wrapped__,
                    self if not isinstance(self, Box) else self.__wrapped__,
                )
            )

    Add = Operation(operator.add)

    def eager_add(x, y):
        match x, y:
            case int(x_), int(y_):
                return x_ + y_
            case _:
                return fwd(None)

    eager = {Add: eager_add}

    def simplify_add(x, y):
        match x, y:
            case Term(_, _, _), int(_):
                return y + Box(x)
            case _, Term(_, (a, b), ()):
                return (Box(x) + a) + b
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
    x, y = Box(x_()), Box(y_())
    one, two, three = Box(1), Box(2), Box(3)

    with interpreter(eager):
        assert one + two == three

    with interpreter(lazy):
        assert one + two == Term(Add, (one, two), ())
        assert Add(one, Add(two, three)) == Term(
            Add, (one, Term(Add, (two, three), ())), ()
        )
        assert x + y == Term(Add, (x, y), ())
        assert not (x + 1 == y + 1)

    with interpreter(mixed):
        assert one + two == three
        assert (one + two) + x == 3 + x

    with interpreter(mixed_simplified):
        assert one + two == three
        assert (one + two) + x == x + 3

        assert 3 + x == x + 3
        assert 1 + (x + 2) == x + 3
        assert (x + 1) + 2 == x + 3
        assert (1 + (y + 1)) + (1 + (x + 1)) == (y + x) + 4

        assert 1 + ((x + y) + 2) == (x + y) + 3
        assert 1 + ((x + (y + 1)) + 1) == (x + y) + 3

        assert (
            ((x + x) + (x + x)) + ((x + x) + (x + x))
            == x + (x + (x + (x + (x + (x + (x + x))))))
            == ((((((x + x) + x) + x) + x) + x) + x) + x
        )

        assert x + 0 == 0 + x == x


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

    get_runtime()._JUDGEMENTS[Lam] = lambda var, body: TypeInContext(
        {v: t for v, t in body.context.items() if v != var}, body.type
    )
    get_runtime()._JUDGEMENTS[App] = lambda f, arg: TypeInContext(
        {**f.context, **(arg.context if isinstance(arg, TypeInContext) else {})}, f.type
    )
    get_runtime()._JUDGEMENTS[Add] = lambda x, y: TypeInContext(
        {
            **(x.context if isinstance(x, TypeInContext) else {}),
            **(y.context if isinstance(y, TypeInContext) else {}),
        },
        int,
    )

    def alpha_lam(var: Operation, body: Term):
        """alpha reduction"""
        mangled_var = gensym(object)
        rename = interpreter(
            {var: mangled_var, apply: lambda op, *a, **k: Term(op, a, tuple(k.items()))}
        )(evaluate)
        return fwd(None, mangled_var, rename(body))

    get_runtime()._BINDINGS[Lam] = alpha_lam

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

    def eta_lam(var: Operation, body: Term):
        """eta reduction"""
        if var not in interpreter(get_runtime()._JUDGEMENTS)(evaluate)(body).context:
            return body
        else:
            return fwd(None)

    free = {
        op: functools.partial(lambda op, *a, **k: Term(op, a, tuple(k.items())), op)
        for op in (Add, App, Lam)
    }
    eager = {Add: eager_add, App: eager_app, Lam: eta_lam}
    eager_mixed = coproduct(free, eager)

    x, y, z = gensym(object), gensym(object), gensym(object)
    one, two, three = 1, 2, 3

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
