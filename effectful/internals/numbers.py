import numbers
import operator
from typing import Generic, TypeVar

from effectful.internals.base_impl import BaseTerm, as_data_register
from effectful.internals.operator import OPERATORS
from effectful.ops.types import Expr

_T_Number = TypeVar("_T_Number", bound=numbers.Number)


@as_data_register(numbers.Number)
class NumberTerm(Generic[_T_Number], BaseTerm[_T_Number]):

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
