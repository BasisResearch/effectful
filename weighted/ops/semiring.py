import collections.abc
import dataclasses
import itertools
import numbers
from collections.abc import Callable
from typing import Any

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numbers  # noqa: F401
import tree
from effectful.handlers.jax._handlers import is_eager_array
from effectful.ops.syntax import defop
from effectful.ops.types import Term


@dataclasses.dataclass
class Semiring[T]:
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
def add[T](a: T, b: T) -> T:
    if isinstance(a, numbers.Number) and a == 0:
        return b
    if isinstance(b, numbers.Number) and b == 0:
        return a
    if any(isinstance(x, Term) and not is_eager_array(x) for x in (a, b)):
        raise NotImplementedError
    return a + b  # type: ignore


@defop
def mul[T](a: T, b: T) -> T:
    if (isinstance(a, numbers.Number) and a == 0) or (
        isinstance(b, numbers.Number) and b == 0
    ):
        return 0  # type: ignore
    if isinstance(a, numbers.Number) and a == 1:
        return b
    if isinstance(b, numbers.Number) and b == 1:
        return a
    if any(isinstance(x, Term) and not is_eager_array(x) for x in (a, b)):
        raise NotImplementedError
    return a * b  # type: ignore


@defop
def min[T](a: T, b: T) -> T:
    if isinstance(a, numbers.Number) and a == float("inf"):
        return b
    if isinstance(b, numbers.Number) and b == float("inf"):
        return a
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotImplementedError
    return a if a < b else b  # type: ignore


@defop
def max[T](a: T, b: T) -> T:
    if isinstance(a, numbers.Number) and a == float("-inf"):
        return b
    if isinstance(b, numbers.Number) and b == float("-inf"):
        return a
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotImplementedError
    return a if a > b else b  # type: ignore


@defop
def arg_min(a, b):
    if isinstance(a, tuple) and isinstance(a[0], numbers.Number) and a[0] == float("inf"):
        return b
    if isinstance(b, tuple) and isinstance(b[0], numbers.Number) and b[0] == float("inf"):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotImplementedError
    return a if a[0] < b[0] else b


@defop
def arg_max(a, b):
    if (
        isinstance(a, tuple)
        and isinstance(a[0], numbers.Number)
        and a[0] == float("-inf")
    ):
        return b
    if (
        isinstance(b, tuple)
        and isinstance(b[0], numbers.Number)
        and b[0] == float("-inf")
    ):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotImplementedError
    return a if a[0] > b[0] else b


@defop
def logaddexp(a, b):
    if isinstance(a, numbers.Number) and a == float("-inf"):
        return b
    if isinstance(b, numbers.Number) and b == float("-inf"):
        return a
    return jnp.logaddexp(a, b)


def is_idempotent(monoid) -> bool:
    """
    Check whether a monoid is idempotent.

    A monoid (A, ⊕, e) is idempotent if
        ∀x ∈ A: x ⊕ x = x
    """
    return monoid in (min, max, arg_min, arg_max)


def scalar_mul(monoid):
    """
    Returns the scalar multiplication w.r.t. a monoid.

    The scalar multiplication of a monoid (A, ⊕, e) is
    a function (⋅): A × ℕ → A, inductively defined as
        a⋅0 = e
        a⋅n = a⋅(n-1) + a

    Warning: scalar multiplication is not commutative,
        the scalar is always the second argument.
    """
    if is_idempotent(monoid):
        return lambda x, _: x
    if monoid is add:
        return mul
    if monoid is logaddexp:
        return add
    if monoid is mul:
        return pow
    raise ValueError(f"Unknown monoid {monoid}")


LinAlg: Semiring[float] = Semiring(add, mul, 0.0, 1.0, "LinAlg")

LogAlg: Semiring[float] = Semiring(logaddexp, add, float("-inf"), 0.0, "LogAlg")

MinAlg: Semiring[float] = Semiring(min, mul, float("inf"), 1.0, "MinAlg")

MaxAlg: Semiring[float] = Semiring(max, mul, float("-inf"), 1.0, "MaxAlg")

ArgMinAlg: Semiring[tuple[float, Any]] = Semiring(
    arg_min, mul, (float("inf"), None), (1.0, None), "ArgMinAlg"
)

ArgMaxAlg: Semiring[tuple[float, Any]] = Semiring(
    arg_max, mul, (float("-inf"), None), (1.0, None), "ArgMaxAlg"
)
