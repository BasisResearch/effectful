"""
This module provides a term representation for numbers and operations on them.
"""

import numbers
import operator
from typing import Any, TypeVar

from typing_extensions import ParamSpec

from effectful.ops.syntax import defdata, defop, syntactic_eq
from effectful.ops.types import Expr, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")

T_Number = TypeVar("T_Number", bound=numbers.Number)


@defdata.register(numbers.Number)
@numbers.Number.register
class _NumberTerm(Term[numbers.Number]):
    def __init__(
        self, op: Operation[..., numbers.Number], *args: Expr, **kwargs: Expr
    ) -> None:
        self._op = op
        self._args = args
        self._kwargs = kwargs

    @property
    def op(self) -> Operation[..., numbers.Number]:
        return self._op

    @property
    def args(self) -> tuple:
        return self._args

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    def __hash__(self):
        return hash((self.op, tuple(self.args), tuple(self.kwargs.items())))


# Complex specific methods
@defop
def eq(x: T_Number, y: T_Number) -> bool:
    if not any(isinstance(a, Term) for a in (x, y)):
        return operator.eq(x, y)
    else:
        return syntactic_eq(x, y)


def _wrap_cmp(op):
    def _wrapped_op(x: T_Number, y: T_Number) -> bool:
        if not any(isinstance(a, Term) for a in (x, y)):
            return op(x, y)
        else:
            raise NotImplementedError

    _wrapped_op.__name__ = op.__name__
    return _wrapped_op


def _wrap_binop(op):
    def _wrapped_op(x: T_Number, y: T_Number) -> T_Number:
        if not any(isinstance(a, Term) for a in (x, y)):
            return op(x, y)
        else:
            raise NotImplementedError

    _wrapped_op.__name__ = op.__name__
    return _wrapped_op


def _wrap_unop(op):
    def _wrapped_op(x: T_Number) -> T_Number:
        if not isinstance(x, Term):
            return op(x)
        else:
            raise NotImplementedError

    _wrapped_op.__name__ = op.__name__
    return _wrapped_op


add = defop(_wrap_binop(operator.add))
neg = defop(_wrap_unop(operator.neg))
pos = defop(_wrap_unop(operator.pos))
sub = defop(_wrap_binop(operator.sub))
mul = defop(_wrap_binop(operator.mul))
truediv = defop(_wrap_binop(operator.truediv))
pow = defop(_wrap_binop(operator.pow))
abs = defop(_wrap_unop(operator.abs))


@defdata.register(numbers.Complex)
@numbers.Complex.register
class _ComplexTerm(_NumberTerm, Term[numbers.Complex]):
    def __bool__(self) -> bool:
        raise ValueError("Cannot convert term to bool")

    def __add__(self, other: Any) -> numbers.Real:
        return add(self, other)

    def __radd__(self, other: Any) -> numbers.Real:
        return add(other, self)

    def __neg__(self):
        return neg(self)

    def __pos__(self):
        return pos(self)

    def __sub__(self, other: Any) -> numbers.Real:
        return sub(self, other)

    def __rsub__(self, other: Any) -> numbers.Real:
        return sub(other, self)

    def __mul__(self, other: Any) -> numbers.Real:
        return mul(self, other)

    def __rmul__(self, other: Any) -> numbers.Real:
        return mul(other, self)

    def __truediv__(self, other: Any) -> numbers.Real:
        return truediv(self, other)

    def __rtruediv__(self, other: Any) -> numbers.Real:
        return truediv(other, self)

    def __pow__(self, other: Any) -> numbers.Real:
        return pow(self, other)

    def __rpow__(self, other: Any) -> numbers.Real:
        return pow(other, self)

    def __abs__(self) -> numbers.Real:
        return abs(self)

    def __eq__(self, other: Any) -> bool:
        return eq(self, other)


# Real specific methods
floordiv = defop(_wrap_binop(operator.floordiv))
mod = defop(_wrap_binop(operator.mod))
lt = defop(_wrap_cmp(operator.lt))
le = defop(_wrap_cmp(operator.le))
gt = defop(_wrap_cmp(operator.gt))
ge = defop(_wrap_cmp(operator.ge))


@defdata.register(numbers.Real)
@numbers.Real.register
class _RealTerm(_ComplexTerm, Term[numbers.Real]):
    # Real specific methods
    def __float__(self) -> float:
        raise ValueError("Cannot convert term to float")

    def __trunc__(self) -> numbers.Integral:
        raise NotImplementedError

    def __floor__(self) -> numbers.Integral:
        raise NotImplementedError

    def __ceil__(self) -> numbers.Integral:
        raise NotImplementedError

    def __round__(self, ndigits=None) -> numbers.Integral:
        raise NotImplementedError

    def __floordiv__(self, other):
        return floordiv(self, other)

    def __rfloordiv__(self, other):
        return floordiv(other, self)

    def __mod__(self, other):
        return mod(self, other)

    def __rmod__(self, other):
        return mod(other, self)

    def __lt__(self, other):
        return lt(self, other)

    def __le__(self, other):
        return le(self, other)


@defdata.register(numbers.Rational)
@numbers.Rational.register
class _RationalTerm(_RealTerm, Term[numbers.Rational]):
    @property
    def numerator(self):
        raise NotImplementedError

    @property
    def denominator(self):
        raise NotImplementedError


# Integral specific methods
index = defop(_wrap_unop(operator.index))
lshift = defop(_wrap_binop(operator.lshift))
rshift = defop(_wrap_binop(operator.rshift))
and_ = defop(_wrap_binop(operator.and_))
xor = defop(_wrap_binop(operator.xor))
or_ = defop(_wrap_binop(operator.or_))
invert = defop(_wrap_unop(operator.invert))


@defdata.register(numbers.Integral)
@numbers.Integral.register
class _IntegralTerm(_RationalTerm, Term[numbers.Integral]):
    # Integral specific methods
    def __int__(self) -> int:
        raise ValueError("Cannot convert term to int")

    def __index__(self) -> numbers.Integral:
        return index(self)

    def __pow__(self, exponent: Any, modulus=None) -> numbers.Integral:
        return pow(self, exponent)

    def __lshift__(self, other):
        return lshift(self, other)

    def __rlshift__(self, other):
        return lshift(other, self)

    def __rshift__(self, other):
        return rshift(self, other)

    def __rrshift__(self, other):
        return rshift(other, self)

    def __and__(self, other):
        return and_(self, other)

    def __rand__(self, other):
        return and_(other, self)

    def __xor__(self, other):
        return xor(self, other)

    def __rxor__(self, other):
        return xor(other, self)

    def __or__(self, other):
        return or_(self, other)

    def __ror__(self, other):
        return or_(other, self)

    def __invert__(self):
        return invert(self)
