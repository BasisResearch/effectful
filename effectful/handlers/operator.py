import functools
import numbers
import operator
from typing import Any, Callable, Generic, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.base_impl import _BaseTerm
from effectful.ops.syntax import NoDefaultRule, defdata, defop, syntactic_eq
from effectful.ops.types import Expr, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
_T_Number = TypeVar("_T_Number", bound=numbers.Number)


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


@defdata.register(numbers.Number)
class NumberTerm(Generic[_T_Number], _BaseTerm[_T_Number]):

    #######################################################################
    # arithmetic binary operators
    #######################################################################
    def __add__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.add](self, other)  # type: ignore

    def __sub__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.sub](self, other)  # type: ignore

    def __mul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mul](self, other)  # type: ignore

    def __truediv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.truediv](self, other)  # type: ignore

    def __floordiv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.floordiv](self, other)  # type: ignore

    def __mod__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mod](self, other)  # type: ignore

    def __pow__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.pow](self, other)  # type: ignore

    def __matmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.matmul](self, other)  # type: ignore

    #######################################################################
    # unary operators
    #######################################################################
    def __neg__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.neg](self)  # type: ignore

    def __pos__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.pos](self)  # type: ignore

    def __abs__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.abs](self)  # type: ignore

    def __invert__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.invert](self)  # type: ignore

    #######################################################################
    # comparisons
    #######################################################################
    def __ne__(self, other) -> bool:
        return OPERATORS[operator.ne](self, other)  # type: ignore

    def __lt__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.lt](self, other)  # type: ignore

    def __le__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.le](self, other)  # type: ignore

    def __gt__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.gt](self, other)  # type: ignore

    def __ge__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.ge](self, other)  # type: ignore

    #######################################################################
    # bitwise operators
    #######################################################################
    def __and__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.and_](self, other)  # type: ignore

    def __or__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.or_](self, other)  # type: ignore

    def __xor__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.xor](self, other)  # type: ignore

    def __rshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.rshift](self, other)  # type: ignore

    def __lshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.lshift](self, other)  # type: ignore

    #######################################################################
    # reflected operators
    #######################################################################
    def __radd__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.add](other, self)  # type: ignore

    def __rsub__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.sub](other, self)  # type: ignore

    def __rmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mul](other, self)  # type: ignore

    def __rtruediv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.truediv](other, self)  # type: ignore

    def __rfloordiv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.floordiv](other, self)  # type: ignore

    def __rmod__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mod](other, self)  # type: ignore

    def __rpow__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.pow](other, self)  # type: ignore

    def __rmatmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.matmul](other, self)  # type: ignore

    # bitwise
    def __rand__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.and_](other, self)  # type: ignore

    def __ror__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.or_](other, self)  # type: ignore

    def __rxor__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.xor](other, self)  # type: ignore

    def __rrshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.rshift](other, self)  # type: ignore

    def __rlshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.lshift](other, self)  # type: ignore
