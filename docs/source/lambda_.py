import functools
import operator
from typing import Annotated, Callable, TypeVar

from typing_extensions import ParamSpec

from effectful.handlers.numbers import add
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler
from effectful.ops.syntax import Bound, NoDefaultRule, Scoped, defop
from effectful.ops.types import Expr, Interpretation, Operation, Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@defop
def App(f: Callable[[S], T], arg: S) -> T:
    raise NoDefaultRule


@defop
def Lam(var: Annotated[Operation[[], S], Bound()], body: T) -> Callable[[S], T]:
    raise NoDefaultRule


@defop
def Let(
    var: Annotated[Operation[[], S], Bound(0)],
    val: Annotated[S, Scoped(1)],
    body: Annotated[T, Scoped(0)],
) -> T:
    raise NoDefaultRule


def beta_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    """integer addition"""
    match x, y:
        case int(), int():
            return x + y
        case _:
            return fwd()


def beta_app(f: Expr[Callable[[S], T]], arg: Expr[S]) -> Expr[T]:
    """beta reduction"""
    match f, arg:
        case Term(op, (var, body)), _ if op == Lam:
            return handler({var: lambda: arg})(evaluate)(body)  # type: ignore
        case _:
            return fwd()


def beta_let(var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """let binding"""
    return handler({var: lambda: val})(evaluate)(body)  # type: ignore


def eta_lam(var: Operation[[], S], body: Expr[T]) -> Expr[Callable[[S], T]] | Expr[T]:
    """eta reduction"""
    if var not in fvsof(body):  # type: ignore
        return body
    else:
        return fwd()


def eta_let(var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """eta reduction"""
    if var not in fvsof(body):  # type: ignore
        return body
    else:
        return fwd()


def commute_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case Term(), int():
            return y + x  # type: ignore
        case _:
            return fwd()


def assoc_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case _, Term(op, (a, b)) if op == add:
            return (x + a) + b  # type: ignore
        case _:
            return fwd()


def unit_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case _, 0:
            return x
        case 0, _:
            return y
        case _:
            return fwd()


def sort_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    match x, y:
        case Term(vx, ()), Term(vy, ()) if id(vx) > id(vy):
            return y + x  # type: ignore
        case Term(add_, (a, Term(vx, ()))), Term(vy, ()) if add_ == add and id(vx) > id(
            vy
        ):
            return (a + vy()) + vx()  # type: ignore
        case _:
            return fwd()


eta_rules: Interpretation = {
    Lam: eta_lam,
    Let: eta_let,
}
beta_rules: Interpretation = {
    add: beta_add,
    App: beta_app,
    Let: beta_let,
}
commute_rules: Interpretation = {
    add: commute_add,
}
assoc_rules: Interpretation = {
    add: assoc_add,
}
unit_rules: Interpretation = {
    add: unit_add,
}
sort_rules: Interpretation = {
    add: sort_add,
}

eager_mixed = functools.reduce(
    coproduct,
    (
        eta_rules,
        beta_rules,
        commute_rules,
        assoc_rules,
        unit_rules,
        sort_rules,
    ),
)

if __name__ == "__main__":
    x, y = defop(int, name="x"), defop(int, name="y")

    with handler(eager_mixed):
        f2 = Lam(x, Lam(y, (x() + y())))

        assert App(App(f2, 1), 2) == 3
        assert Lam(y, f2) == f2
