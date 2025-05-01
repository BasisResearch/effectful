import collections.abc
import dataclasses
import itertools
from typing import Any, Callable, Generic, ParamSpec, TypeVar

import effectful.handlers.numbers  # noqa: F401
import tree
from effectful.ops.syntax import defop
from effectful.ops.types import Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
A = TypeVar("A")
B = TypeVar("B")


@dataclasses.dataclass
class Semiring(Generic[T]):
    add: Callable[[T, T], T]
    mul: Callable[[T, T], T]
    zero: T
    one: T
    name: str | None = None

    def __init__(self, add, mul, zero, one, name=None):
        self.add = add
        self.mul = mul
        self.zero = zero
        self.one = one
        self.name = name

    def __str__(self):
        if self.name is None:
            return repr(self)
        return self.name


# Semiring laws:
# (R, +) is a commutative monoid with identity 0
# (R, *) is a monoid with identity 1
# a * 0 = 0 = 0 * a
# a * (b + c) = (a * b) + (a * c)
# (b + c) * a = (b * a) + (c * a)

# actually a near-semiring
StreamAlg: Semiring[collections.abc.Generator] = Semiring(
    add=lambda a, b: (v for v in itertools.chain(a, b)),
    mul=lambda a, b: ((v1, v2) for (v1, v2) in itertools.product(a, b)),
    zero=(),
    one=(),  # note: empty tuple is not a valid identity for multiplication
    name="StreamAlg",
)


@defop
def add(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotImplementedError
    return a + b


@defop
def mul(a, b):
    if a == 1:
        return b
    if b == 1:
        return a
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotImplementedError
    return a * b


@defop
def arg_min(a, b):
    if isinstance(a, tuple) and a[0] is float("inf"):
        return b
    if isinstance(b, tuple) and b[0] is float("inf"):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotImplementedError
    return a if a[0] < b[0] else b


@defop
def arg_max(a, b):
    if isinstance(a, tuple) and a[0] is float("-inf"):
        return b
    if isinstance(b, tuple) and b[0] is float("-inf"):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotImplementedError
    return a if a[0] > b[0] else b


LinAlg: Semiring[float] = Semiring(add, mul, 0.0, 1.0, "LinAlg")

MinAlg: Semiring[float] = Semiring(min, mul, float("inf"), 1.0, "MinAlg")

MaxAlg: Semiring[float] = Semiring(max, mul, float("-inf"), 1.0, "MaxAlg")

ArgMinAlg: Semiring[tuple[float, Any]] = Semiring(
    arg_min, mul, (float("inf"), None), (1.0, None), "ArgMinAlg"
)

ArgMaxAlg: Semiring[tuple[float, Any]] = Semiring(
    arg_max, mul, (float("-inf"), None), (1.0, None), "ArgMaxAlg"
)
