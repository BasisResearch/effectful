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


def _register_syntax_op(syntax_fn: Callable[P, T]):
    def register_syntax_op_fn(syntax_op_fn: Callable[P, T]):
        OPERATORS[syntax_fn] = defop(syntax_op_fn)
        return OPERATORS[syntax_fn]

    return register_syntax_op_fn


def _create_arithmetic_binop_rule(op):
    def rule(x: T, y: T) -> T:
        if not isinstance(x, Term) and not isinstance(y, Term):
            return op(x, y)
        raise NoDefaultRule

    # Note: functools.wraps would be better, but it does not preserve type
    # annotations
    rule.__name__ = op.__name__
    return rule


def _create_arithmetic_unop_rule(op):
    def rule(x: T) -> T:
        if not isinstance(x, Term):
            return op(x)
        raise NoDefaultRule

    rule.__name__ = op.__name__
    return rule


def _create_generic_rule(op):
    @functools.wraps(op)
    def rule(*args, **kwargs):
        if not any(isinstance(a, Term) for a in args) and not any(
            isinstance(a, Term) for a in kwargs.values()
        ):
            return op(*args, **kwargs)

        raise NoDefaultRule

    return rule


_ARITHMETIC_BINOPS = (
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
_ARITHMETIC_UNOPS = (
    operator.neg,
    operator.pos,
    operator.abs,
    operator.invert,
)
_OTHER_OPS = (
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

for op in _ARITHMETIC_BINOPS:
    _register_syntax_op(op)(_create_arithmetic_binop_rule(op))

for op in _ARITHMETIC_UNOPS:  # type: ignore
    _register_syntax_op(op)(_create_arithmetic_unop_rule(op))

for op in _OTHER_OPS:  # type: ignore
    _register_syntax_op(op)(_create_generic_rule(op))


@_register_syntax_op(operator.eq)
def _eq_op(a: Expr[T], b: Expr[T]) -> Expr[bool]:
    """Default implementation of equality for terms. As a special case, equality defaults to syntactic equality rather
    than producing a free term.

    """
    return syntactic_eq(a, b)


@_register_syntax_op(operator.ne)
def _ne_op(a: T, b: T) -> bool:
    return OPERATORS[operator.not_](OPERATORS[operator.eq](a, b))


@defdata.register(numbers.Number)
class _NumberTerm(Generic[_T_Number], _BaseTerm[_T_Number]):

    #######################################################################
    # arithmetic binary operators
    #######################################################################
    def __add__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.add](self, other)

    def __sub__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.sub](self, other)

    def __mul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mul](self, other)

    def __truediv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.truediv](self, other)

    def __floordiv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.floordiv](self, other)

    def __mod__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mod](self, other)

    def __pow__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.pow](self, other)

    def __matmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.matmul](self, other)

    #######################################################################
    # unary operators
    #######################################################################
    def __neg__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.neg](self)

    def __pos__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.pos](self)

    def __abs__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.abs](self)

    def __invert__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.invert](self)

    #######################################################################
    # comparisons
    #######################################################################
    def __ne__(self, other) -> bool:
        return OPERATORS[operator.ne](self, other)

    def __lt__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.lt](self, other)

    def __le__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.le](self, other)

    def __gt__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.gt](self, other)

    def __ge__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.ge](self, other)

    #######################################################################
    # bitwise operators
    #######################################################################
    def __and__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.and_](self, other)

    def __or__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.or_](self, other)

    def __xor__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.xor](self, other)

    def __rshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.rshift](self, other)

    def __lshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.lshift](self, other)

    #######################################################################
    # reflected operators
    #######################################################################
    def __radd__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.add](other, self)

    def __rsub__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.sub](other, self)

    def __rmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mul](other, self)

    def __rtruediv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.truediv](other, self)

    def __rfloordiv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.floordiv](other, self)

    def __rmod__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mod](other, self)

    def __rpow__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.pow](other, self)

    def __rmatmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.matmul](other, self)

    # bitwise
    def __rand__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.and_](other, self)

    def __ror__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.or_](other, self)

    def __rxor__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.xor](other, self)

    def __rrshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.rshift](other, self)

    def __rlshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.lshift](other, self)
