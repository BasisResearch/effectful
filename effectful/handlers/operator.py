import functools
import operator
from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec

from effectful.ops.syntax import NoDefaultRule, defop, syntactic_eq
from effectful.ops.types import Expr, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


OPERATORS: dict[Callable[..., Any], Operation[..., Any]] = {}


def register_syntax_op(syntax_fn: Callable[P, T]):
    def register_syntax_op_fn(syntax_op_fn: Callable[P, T]):
        OPERATORS[syntax_fn] = defop(syntax_op_fn)
        return OPERATORS[syntax_fn]

    return register_syntax_op_fn


def create_arithmetic_binop_rule(op):
    def rule(x: T, y: T) -> T:
        if not isinstance(x, Term) and not isinstance(y, Term):
            return op(x, y)
        raise NoDefaultRule

    # Note: functools.wraps would be better, but it does not preserve type
    # annotations
    rule.__name__ = op.__name__
    return rule


def create_arithmetic_unop_rule(op):
    def rule(x: T) -> T:
        if not isinstance(x, Term):
            return op(x)
        raise NoDefaultRule

    rule.__name__ = op.__name__
    return rule


def create_generic_rule(op):
    @functools.wraps(op)
    def rule(*args, **kwargs):
        if not any(isinstance(a, Term) for a in args) and not any(
            isinstance(a, Term) for a in kwargs.values()
        ):
            return op(*args, **kwargs)

        raise NoDefaultRule

    return rule


ARITHMETIC_BINOPS = (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.matmul,
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.or_,
    operator.xor,
)
ARITHMETIC_UNOPS = (
    operator.neg,
    operator.pos,
    operator.abs,
    operator.invert,
)
OTHER_OPS = (
    operator.not_,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.is_,
    operator.is_not,
    operator.contains,
    operator.index,
    operator.getitem,
    operator.setitem,
    operator.delitem,
    # TODO handle these builtin functions
    # getattr,
    # setattr,
    # delattr,
    # len,
    # iter,
    # next,
    # reversed,
)

for op in ARITHMETIC_BINOPS:
    register_syntax_op(op)(create_arithmetic_binop_rule(op))

for op in ARITHMETIC_UNOPS:  # type: ignore
    register_syntax_op(op)(create_arithmetic_unop_rule(op))

for op in OTHER_OPS:  # type: ignore
    register_syntax_op(op)(create_generic_rule(op))


@register_syntax_op(operator.eq)
def _eq_op(a: Expr[T], b: Expr[T]) -> Expr[bool]:
    """Default implementation of equality for terms. As a special case, equality defaults to syntactic equality rather
    than producing a free term.

    """
    return syntactic_eq(a, b)


@register_syntax_op(operator.ne)
def _ne_op(a: T, b: T) -> bool:
    return OPERATORS[operator.not_](OPERATORS[operator.eq](a, b))  # type: ignore
