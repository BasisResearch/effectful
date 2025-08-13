"""
This module provides a term representation for numbers and operations on them.
"""

import numbers

from effectful.ops.syntax import defdata

add = defdata.dispatch(numbers.Number).__add__
neg = defdata.dispatch(numbers.Number).__neg__
pos = defdata.dispatch(numbers.Number).__pos__
sub = defdata.dispatch(numbers.Number).__sub__
mul = defdata.dispatch(numbers.Number).__mul__
truediv = defdata.dispatch(numbers.Number).__truediv__
pow = defdata.dispatch(numbers.Number).__pow__
abs = defdata.dispatch(numbers.Number).__abs__
floordiv = defdata.dispatch(numbers.Number).__floordiv__
mod = defdata.dispatch(numbers.Number).__mod__
eq = defdata.dispatch(numbers.Number).__eq__
lt = defdata.dispatch(numbers.Number).__lt__
le = defdata.dispatch(numbers.Number).__le__
gt = defdata.dispatch(numbers.Number).__gt__
ge = defdata.dispatch(numbers.Number).__ge__
index = defdata.dispatch(numbers.Number).__index__
lshift = defdata.dispatch(numbers.Number).__lshift__
rshift = defdata.dispatch(numbers.Number).__rshift__
and_ = defdata.dispatch(numbers.Number).__and__
xor = defdata.dispatch(numbers.Number).__xor__
or_ = defdata.dispatch(numbers.Number).__or__
invert = defdata.dispatch(numbers.Number).__invert__
