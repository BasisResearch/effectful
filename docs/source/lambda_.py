import functools
from typing import Annotated, Callable

from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler
from effectful.ops.syntax import Scoped, defdata, defop, syntactic_eq
from effectful.ops.types import Expr, Interpretation, NotHandled, Operation, Term

add = defdata.dispatch(int).__add__


@defop
def App[S, T](f: Callable[[S], T], arg: S) -> T:
    raise NotHandled


@defop
def Lam[S, T, A](
    var: Annotated[Operation[[], S], Scoped[A]], body: Annotated[T, Scoped[A]]
) -> Callable[[S], T]:
    raise NotHandled


@defop
def Let[S, T, A](
    var: Annotated[Operation[[], S], Scoped[A]],
    val: S,
    body: Annotated[T, Scoped[A]],
) -> T:
    raise NotHandled


def beta_add(x: Expr[int], y: Expr[int]) -> Expr[int]:
    """integer addition"""
    match x, y:
        case int(), int():
            return x + y
        case _:
            return fwd()


def beta_app[S, T](f: Expr[Callable[[S], T]], arg: Expr[S]) -> Expr[T]:
    """beta reduction"""
    match f, arg:
        case Term(op, (var, body)), _ if op == Lam:
            return handler({var: lambda: arg})(evaluate)(body)
        case _:
            return fwd()


def beta_let[S, T](var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """let binding"""
    return handler({var: lambda: val})(evaluate)(body)


def eta_lam[S, T](
    var: Operation[[], S], body: Expr[T]
) -> Expr[Callable[[S], T]] | Expr[T]:
    """eta reduction"""
    if var not in fvsof(body):
        return body
    else:
        return fwd()


def eta_let[S, T](var: Operation[[], S], val: Expr[S], body: Expr[T]) -> Expr[T]:
    """eta reduction"""
    if var not in fvsof(body):
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
    if syntactic_eq(y, 0):
        return x
    elif syntactic_eq(x, 0):
        return y
    else:
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
